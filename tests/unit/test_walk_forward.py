"""Tests for WalkForwardValidator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.walk_forward import WalkForwardValidator
from trading_bot.core.models import Signal
from trading_bot.core.strategies.base import BaseStrategy


def _make_ohlcv(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 30, n),
            "high": close + abs(rng.normal(0, 150, n)),
            "low": close - abs(rng.normal(0, 150, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 200, n)),
        },
        index=dates,
    )


class _NeverTradeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "never"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        return None


class TestSplits:
    def test_correct_number_of_splits(self) -> None:
        wf = WalkForwardValidator(n_splits=5, train_ratio=0.8)
        data = _make_ohlcv(500)
        folds = wf.split(data)
        assert len(folds) == 5

    def test_no_overlap_train_test(self) -> None:
        """Train and test sets should not overlap in time."""
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.8)
        data = _make_ohlcv(300)
        folds = wf.split(data)

        for train, test in folds:
            # Last train index should be before first test index
            train_end = train.index[-1]
            test_start = test.index[0]
            assert train_end < test_start

    def test_expanding_window_grows(self) -> None:
        """Expanding window should increase train size across folds."""
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.7, expanding=True)
        data = _make_ohlcv(300)
        folds = wf.split(data)

        train_sizes = [len(train) for train, _ in folds]
        # Expanding: each train should be >= previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_gap_bars_respected(self) -> None:
        """With gap_bars > 0, there should be a gap between train and test."""
        gap = 5
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.8, gap_bars=gap)
        data = _make_ohlcv(500)
        folds = wf.split(data)

        for train, test in folds:
            train_end_idx = data.index.get_loc(train.index[-1])
            test_start_idx = data.index.get_loc(test.index[0])
            actual_gap = test_start_idx - train_end_idx - 1
            assert actual_gap >= gap


class TestWalkForwardRun:
    def test_run_returns_result(self) -> None:
        wf = WalkForwardValidator(n_splits=3)
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        ohlcv = _make_ohlcv(300)
        result = wf.run(engine, ohlcv, "BTC/USDT")
        assert len(result.fold_results) == 3

    def test_aggregate_metrics_populated(self) -> None:
        wf = WalkForwardValidator(n_splits=2)
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        ohlcv = _make_ohlcv(200)
        result = wf.run(engine, ohlcv, "BTC/USDT")
        assert "total_trades" in result.aggregate_metrics

    def test_oos_equity_curve_concatenated(self) -> None:
        wf = WalkForwardValidator(n_splits=3)
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        ohlcv = _make_ohlcv(300)
        result = wf.run(engine, ohlcv, "BTC/USDT")
        # OOS equity should have some data points
        assert len(result.oos_equity_curve) > 0


class _AlwaysLong(BaseStrategy):
    @property
    def name(self) -> str:
        return "always_long"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        from datetime import UTC, datetime

        from trading_bot.core.models import SignalType

        return Signal(
            type=SignalType.LONG,
            pair="BTC/USDT",
            confidence=1.0,
            strength=1.0,
            source="t",
            metadata={},
            timestamp=datetime.now(UTC),
        )


class TestOutOfSample:
    def test_fold_results_cover_test_window_only(self) -> None:
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.7)
        ohlcv = _make_ohlcv(300)
        result = wf.run(BacktestEngine(strategy=_AlwaysLong()), ohlcv, "BTC/USDT")

        for (_, test), fold in zip(wf.split(ohlcv), result.fold_results, strict=True):
            assert fold.equity_curve.index.equals(test.index)
            assert all(t.entry_time >= test.index[0] for t in fold.trades)
            assert fold.start_date == test.index[0]

    def test_oos_curve_is_chained_without_jumps(self) -> None:
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.7)
        ohlcv = _make_ohlcv(300)
        result = wf.run(BacktestEngine(strategy=_AlwaysLong()), ohlcv, "BTC/USDT")

        curve = result.oos_equity_curve
        assert curve.index.is_monotonic_increasing
        assert curve.index.is_unique
        # Each fold's first bar continues from the previous fold's last level.
        for prev, nxt in zip(
            result.fold_results, result.fold_results[1:], strict=False
        ):
            end_prev = curve.loc[prev.equity_curve.index[-1]]
            start_next = curve.loc[nxt.equity_curve.index[0]]
            raw_ratio = nxt.equity_curve.iloc[0] / prev.equity_curve.iloc[0]
            assert start_next == pytest.approx(end_prev * raw_ratio, rel=1e-9)
        # Aggregate return is the compounded OOS return.
        compounded = np.prod(
            [
                r.equity_curve.iloc[-1] / r.equity_curve.iloc[0]
                for r in result.fold_results
            ]
        )
        assert result.aggregate_metrics["total_return_pct"] == pytest.approx(
            (compounded - 1) * 100, rel=1e-9
        )
        assert result.aggregate_metrics["total_trades"] == sum(
            len(r.trades) for r in result.fold_results
        )

    def test_sliding_window_keeps_train_size(self) -> None:
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.7, expanding=False)
        sizes = {f.train_end - f.train_start for f in wf.folds(300)}
        assert sizes == {210}
