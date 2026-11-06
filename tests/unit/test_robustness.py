"""Tests for the robustness checks (random starts, slippage stress)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.robustness import (
    RandomStartResult,
    RobustnessCriteria,
    RobustnessReport,
    StressPoint,
    random_starts,
    random_starts_target,
    sample_starts,
    slippage_stress,
)
from trading_bot.backtest.target_engine import TargetExposureEngine
from trading_bot.core.models import Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy


def _hourly(n: int = 400, drift: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 1000 + np.cumsum(rng.normal(drift, 2, n))
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 3,
            "low": close - 3,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def _daily(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.r_[100, close[:-1]],
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


class _AlwaysLong(BaseStrategy):
    @property
    def name(self) -> str:
        return "always_long"

    def get_required_lookback(self) -> int:
        return 10

    def generate_signal(self, ohlcv: pd.DataFrame, ai_signals: object = None) -> Signal:
        return Signal(
            type=SignalType.LONG,
            pair="X/USDT",
            confidence=0.9,
            strength=1.0,
            source="t",
            metadata={},
            timestamp=datetime.now(UTC),
        )


class TestSampleStarts:
    def test_deterministic_sorted_and_in_bounds(self) -> None:
        a = sample_starts(1000, 100, 20, 50, seed=1)
        assert a == sample_starts(1000, 100, 20, 50, seed=1)
        assert a == sorted(a)
        assert len(set(a)) == 20
        assert min(a) >= 50 and max(a) <= 900

    def test_other_seed_other_starts(self) -> None:
        assert sample_starts(1000, 100, 20, 50, 1) != sample_starts(
            1000, 100, 20, 50, 2
        )

    def test_more_samples_than_starts_repeats(self) -> None:
        starts = sample_starts(20, 10, 30, 5, seed=0)
        assert len(starts) == 30 and set(starts) <= set(range(5, 11))

    def test_window_must_fit(self) -> None:
        with pytest.raises(ValueError, match="not enough data"):
            sample_starts(100, 80, 5, 50, seed=0)


class TestRandomStarts:
    def test_each_window_trades_flat_from_its_start(self) -> None:
        ohlcv = _hourly()
        engine = BacktestEngine(_AlwaysLong(), fee_pct=0.0, slippage_pct=0.0)
        res = random_starts(
            engine, ohlcv, "X/USDT", n_samples=5, window_bars=50, seed=7
        )
        assert len(res.samples) == 5 and res.window_bars == 50
        for ts, m in zip(res.starts, res.samples, strict=True):
            assert ts >= ohlcv.index[engine.required_lookback]
            # always long on a rising series: one round trip, closed at the end
            assert m["total_trades"] == 1
            assert m["total_return_pct"] > 0
        summary = res.summary()
        assert summary["profitable_share"] == 1.0
        assert summary["trades_median"] == 1

    def test_window_defaults_to_a_quarter(self) -> None:
        engine = BacktestEngine(_AlwaysLong())
        res = random_starts(engine, _hourly(410), "X/USDT", n_samples=2)
        assert res.window_bars == (410 - 10) // 4


class TestTargetEngineStart:
    def test_cash_until_start_then_trades(self) -> None:
        data = {"A/USDT": _daily(500, 1)}
        engine = TargetExposureEngine(lookbacks=(5, 10), vol_window=10)
        start = data["A/USDT"].index[300]
        res = engine.run(data, start=start)
        before = res.equity_curve.loc[: data["A/USDT"].index[299]]
        assert (before == 100_000.0).all()
        assert all(t.entry_time >= start.to_pydatetime() for t in res.trades)
        assert res.parameters["start"] == str(start)

    def test_no_start_unchanged(self) -> None:
        data = {"A/USDT": _daily(300, 2)}
        engine = TargetExposureEngine(lookbacks=(5, 10), vol_window=10)
        a, b = engine.run(data), engine.run(data, start=None)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)

    def test_random_starts_target(self) -> None:
        data = {"A/USDT": _daily(400, 3), "B/USDT": _daily(400, 4)}
        engine = TargetExposureEngine(lookbacks=(5, 10), vol_window=10, allocation=0.5)
        res = random_starts_target(engine, data, n_samples=4, window_days=60, seed=1)
        assert len(res.samples) == 4
        assert all(ts >= data["A/USDT"].index[engine.min_history] for ts in res.starts)
        assert all("sharpe_ratio" in m and "fills" in m for m in res.samples)


class TestStressAndVerdict:
    def test_stress_multiplies_slippage(self) -> None:
        seen: list[float] = []

        def evaluate(slip: float) -> dict[str, float]:
            seen.append(slip)
            return {"sharpe_ratio": 1.0 - slip}

        points = slippage_stress(evaluate, 0.05, (1, 2, 3))
        assert seen == pytest.approx([0.05, 0.10, 0.15])
        assert [p.multiplier for p in points] == [1.0, 2.0, 3.0]

    def test_stress_needs_positive_base(self) -> None:
        with pytest.raises(ValueError):
            slippage_stress(lambda s: {}, 0.0)

    @staticmethod
    def _report(returns: list[float], stress_sharpe: float) -> RobustnessReport:
        samples = [
            {"total_return_pct": r, "sharpe_ratio": r, "max_drawdown_pct": 1.0}
            for r in returns
        ]
        rs = RandomStartResult([], 10, 0, samples)
        stress = [StressPoint(2.0, 0.1, {"sharpe_ratio": stress_sharpe})]
        return RobustnessReport(rs, stress, RobustnessCriteria())

    def test_pass(self) -> None:
        passed, reasons = self._report([1, 2, 3, -1], 0.5).verdict()
        assert passed, reasons

    def test_fail_on_profitable_share(self) -> None:
        passed, reasons = self._report([1, -2, -3, -1], 0.5).verdict()
        assert not passed and "FAIL" in reasons[0]

    def test_fail_on_stress(self) -> None:
        passed, reasons = self._report([1, 2, 3], -0.1).verdict()
        assert not passed and "FAIL" in reasons[1]

    def test_missing_checks_do_not_pass(self) -> None:
        passed, reasons = RobustnessReport().verdict()
        assert not passed
        assert any("not run" in r for r in reasons)

    def test_as_dict_is_serialisable(self) -> None:
        import json

        json.dumps(self._report([1, 2], 1.0).as_dict(), default=str)
