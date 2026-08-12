"""Tests for WalkForwardValidator."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
