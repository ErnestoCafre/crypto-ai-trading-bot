"""Tests for the target-exposure (Donchian ensemble) backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.target_engine import TargetExposureEngine
from trading_bot.core.strategies.donchian_ensemble import exposure_series


def _daily(n: int = 900, seed: int = 1, drift: float = 0.001, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.03, n)))
    # Opens differ from the previous close so next-open execution matters.
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.01, n))
    index = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def _script_returns(close: pd.Series, cost: float) -> pd.Series:
    """Research-script P&L: w decided at t held over t+1, cost x |dw|."""
    w = exposure_series(close)["exposure"]
    held = w.shift(1).fillna(0.0)
    turnover = w.diff().abs().fillna(w.abs())
    return held * close.pct_change().fillna(0.0) - turnover.shift(1).fillna(0) * cost


class TestParityWithResearchScript:
    def test_close_execution_without_costs_is_exact(self) -> None:
        df = _daily(seed=2, drift=0.002)
        res = TargetExposureEngine(
            rebalance_threshold=0.0,
            execution="close",
            fee_pct=0.0,
            slippage_pct=0.0,
            min_history=0,
        ).run({"X/USDT": df})
        engine = res.equity_curve.pct_change().fillna(0.0)
        np.testing.assert_allclose(
            engine.to_numpy(), _script_returns(df["close"], 0.0).to_numpy(), atol=1e-12
        )

    def test_close_execution_with_costs_matches_metrics(self) -> None:
        df = _daily(seed=2, drift=0.002)
        res = TargetExposureEngine(
            rebalance_threshold=0.0,
            execution="close",
            fee_pct=0.1,
            slippage_pct=0.05,
            min_history=0,
        ).run({"X/USDT": df})
        script = _script_returns(df["close"], 0.0015)
        sharpe = script.mean() / script.std() * np.sqrt(365)
        assert res.metrics()["sharpe_ratio"] == pytest.approx(sharpe, abs=0.02)


class TestExecution:
    def test_next_open_trades_previous_target_at_open(self) -> None:
        df = _daily(seed=4, drift=0.003)
        engine = TargetExposureEngine(
            rebalance_threshold=0.0, fee_pct=0.0, slippage_pct=0.0, min_history=0
        )
        res = engine.run({"X/USDT": df})
        target = engine.targets(df)
        first = int(np.argmax(target.to_numpy() > 0))
        buy_day = df.index[first + 1]
        assert res.traded_notional.iloc[: first + 1].sum() == 0
        assert res.traded_notional.loc[buy_day] == pytest.approx(
            100_000 * target.iloc[first]
        )

    def test_threshold_cuts_fills(self) -> None:
        df = _daily(seed=5, drift=0.002, n=1200)

        def fills(threshold: float) -> float:
            res = TargetExposureEngine(rebalance_threshold=threshold).run(
                {"X/USDT": df}
            )
            return res.metrics()["fills"]

        assert fills(0.0) > fills(0.05) > fills(0.10) > 0

    def test_live_warmup_holds_cash(self) -> None:
        df = _daily(n=600, seed=6, drift=0.004)
        res = TargetExposureEngine().run({"X/USDT": df})  # min_history = 450
        assert res.traded_notional.iloc[:450].sum() == 0
        assert res.equity_curve.iloc[449] == pytest.approx(100_000)

    def test_costs_charged_per_side_on_traded_notional(self) -> None:
        df = _daily(seed=7, drift=0.002)
        free = TargetExposureEngine(fee_pct=0.0, slippage_pct=0.0).run({"X": df})
        paid = TargetExposureEngine(fee_pct=0.1, slippage_pct=0.0).run({"X": df})
        assert paid.fees == pytest.approx(0.001 * paid.traded_notional.sum(), rel=1e-9)
        assert free.fees == 0.0
        assert paid.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]


class TestPortfolio:
    def test_allocation_caps_each_sleeve(self) -> None:
        data = {
            "A/USDT": _daily(seed=8, drift=0.003),
            "B/USDT": _daily(seed=9, drift=0.003),
        }
        res = TargetExposureEngine(allocation=0.25, rebalance_threshold=0.0).run(data)
        values = res.exposure * 0.25  # share of equity per pair
        assert (values.sum(axis=1) <= 0.5 + 1e-6).all()
        assert (res.exposure.to_numpy() <= 1.0 + 0.05).all()

    def test_pair_listed_later_trades_once_it_has_history(self) -> None:
        late = _daily(n=700, seed=10, drift=0.003, start="2019-07-01")
        data = {"A/USDT": _daily(seed=11), "B/USDT": late}
        res = TargetExposureEngine(allocation=0.5).run(data)
        assert res.exposure["B/USDT"].loc[: late.index[449]].sum() == 0
        assert len(res.equity_curve) == 900

    def test_missing_bar_does_not_liquidate_the_pair(self) -> None:
        df = _daily(seed=14, drift=0.003, n=1200)
        engine = TargetExposureEngine(
            allocation=0.5, rebalance_threshold=0.0, fee_pct=0.0, slippage_pct=0.0
        )
        full = engine.run({"A/USDT": df, "B/USDT": df})
        held = full.exposure["B/USDT"]
        day = held[held > 0.2].index[100]
        gap = day - pd.Timedelta(1, unit="D")
        res = engine.run({"A/USDT": df, "B/USDT": df.drop(index=gap)})
        # No decision on the missing day: B keeps its position into ``day``.
        assert res.exposure.loc[day, "B/USDT"] > 0.1

    def test_partial_sells_record_realised_trades(self) -> None:
        df = _daily(seed=12, drift=0.002, n=1500)
        res = TargetExposureEngine().run({"X/USDT": df})
        reasons = {t.exit_reason for t in res.trades}
        assert reasons <= {"rebalance", "exit"}
        assert "exit" in reasons
        assert all(t.exit_time >= t.entry_time for t in res.trades)

    def test_period_metrics_anchor_on_previous_close(self) -> None:
        df = _daily(seed=13, drift=0.002, n=1200)
        res = TargetExposureEngine().run({"X/USDT": df})
        m = res.metrics("2020-06-01", "2020-12-31")
        eq = res.equity_curve.loc["2020-05-31":"2020-12-31"]
        assert m["total_return_pct"] == pytest.approx(
            (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        )
        assert {"avg_exposure_pct", "turnover_per_year", "fills"} <= set(m)

    def test_rejects_bad_parameters(self) -> None:
        with pytest.raises(ValueError):
            TargetExposureEngine(allocation=0)
        with pytest.raises(ValueError):
            TargetExposureEngine(rebalance_threshold=-0.1)
        with pytest.raises(ValueError):
            TargetExposureEngine(execution="vwap")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Kill-switch simulation (audit M-4)
# ---------------------------------------------------------------------------

CRASH_DAY = 20


def _trend_with_crash(n: int = 30, dip: float = 0.10) -> pd.DataFrame:
    """Steady uptrend (fully invested) with one day whose low dips ``dip``
    below its open and closes back up."""
    close = 100 * 1.01 ** np.arange(n)
    open_ = np.concatenate([[100.0], close[:-1]])
    low = np.minimum(open_, close) * 0.999
    low[CRASH_DAY] = open_[CRASH_DAY] * (1 - dip)
    index = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": close, "low": low, "close": close, "volume": 1.0},
        index=index,
    )


def _risk_engine(**kwargs) -> TargetExposureEngine:
    return TargetExposureEngine(
        lookbacks=(2,),
        vol_target=0.0,  # falsy: no volatility scaling, w = 1 in the trend
        vol_window=2,
        min_history=0,
        rebalance_threshold=0.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        **kwargs,
    )


class TestKillSwitchSimulation:
    def _run(self, **kwargs):
        df = _trend_with_crash()
        res = _risk_engine(**kwargs).run({"X/USDT": df})
        return df, res, df.index[CRASH_DAY]

    def test_disabled_by_default(self) -> None:
        _, res, day = self._run()
        assert res.risk_events == []
        assert res.exposure.loc[day, "X/USDT"] > 0.99

    def test_liquidate_sells_where_the_limit_is_crossed(self) -> None:
        df, res, day = self._run(max_daily_loss_pct=3.0, daily_loss_action="liquidate")
        assert res.risk_events == [(day, "daily_liquidate")]
        assert res.exposure.loc[day, "X/USDT"] == 0.0
        sell = next(t for t in res.trades if t.exit_reason == "daily_loss_limit")
        # Fully invested: the portfolio is 3 % down when the price is 3 % down.
        assert sell.exit_price == pytest.approx(df["open"].iloc[CRASH_DAY] * 0.97)
        # The next open rebuys: the strategy still wants full exposure.
        assert res.exposure.iloc[CRASH_DAY + 1]["X/USDT"] > 0.99

    def test_block_increases_keeps_the_position(self) -> None:
        _, res, day = self._run(
            max_daily_loss_pct=3.0, daily_loss_action="block_increases"
        )
        assert res.risk_events == [(day, "daily_block_increases")]
        assert res.exposure.loc[day, "X/USDT"] > 0.99

    def test_halve_sells_half(self) -> None:
        _, res, day = self._run(max_daily_loss_pct=3.0, daily_loss_action="halve")
        assert res.risk_events == [(day, "daily_halve")]
        assert res.exposure.loc[day, "X/USDT"] == pytest.approx(0.5, abs=0.02)

    def test_off_ignores_the_daily_limit(self) -> None:
        _, res, _ = self._run(max_daily_loss_pct=3.0, daily_loss_action="off")
        assert res.risk_events == []

    def test_drawdown_halts_for_good(self) -> None:
        df, res, day = self._run(
            max_daily_loss_pct=3.0,
            max_drawdown_pct=5.0,
            daily_loss_action="block_increases",
        )
        # Same path: the daily limit at -3 %, then the HALT at -5 %.
        assert res.risk_events == [(day, "daily_block_increases"), (day, "halted")]
        sell = next(t for t in res.trades if t.exit_reason == "kill_switch")
        assert sell.exit_price == pytest.approx(df["open"].iloc[CRASH_DAY] * 0.95)
        assert (res.exposure.iloc[CRASH_DAY:]["X/USDT"] == 0.0).all()  # no re-entry
        assert res.metrics()["halted"] == 1.0

    def test_liquidation_first_leaves_nothing_to_halt(self) -> None:
        _, res, day = self._run(
            max_daily_loss_pct=3.0, max_drawdown_pct=5.0, daily_loss_action="liquidate"
        )
        assert res.risk_events == [(day, "daily_liquidate")]

    def test_shallow_dip_triggers_nothing(self) -> None:
        df = _trend_with_crash(dip=0.02)
        res = _risk_engine(max_daily_loss_pct=3.0, max_drawdown_pct=5.0).run({"X": df})
        assert res.risk_events == []

    def test_metrics_count_events_in_the_window(self) -> None:
        _, res, day = self._run(max_daily_loss_pct=3.0, daily_loss_action="halve")
        assert res.metrics()["daily_loss_events"] == 1.0
        assert (
            res.metrics(start=day + pd.Timedelta(1, unit="D"))["daily_loss_events"]
            == 0.0
        )

    def test_rejects_bad_risk_parameters(self) -> None:
        with pytest.raises(ValueError):
            TargetExposureEngine(max_daily_loss_pct=0.0)
        with pytest.raises(ValueError):
            TargetExposureEngine(max_drawdown_pct=150.0)
        with pytest.raises(ValueError):
            TargetExposureEngine(daily_loss_action="panic")  # type: ignore[arg-type]


def test_pipeline_simulates_the_bots_kill_switch(monkeypatch) -> None:
    from trading_bot.backtest.pipeline import build_target_engine
    from trading_bot.config.settings import Settings

    for key in Settings.model_fields:
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    params = (
        build_target_engine(settings=settings)
        .run({"X": _trend_with_crash()})
        .parameters
    )
    assert params["max_drawdown_pct"] == settings.MAX_DRAWDOWN_PCT
    assert params["max_daily_loss_pct"] == settings.MAX_DAILY_LOSS_PCT
    assert params["daily_loss_action"] == "block_increases"

    off = build_target_engine(settings=settings, kill_switch=False)
    params = off.run({"X": _trend_with_crash()}).parameters
    assert params["max_drawdown_pct"] is None and params["max_daily_loss_pct"] is None
