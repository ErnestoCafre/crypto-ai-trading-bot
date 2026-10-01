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
