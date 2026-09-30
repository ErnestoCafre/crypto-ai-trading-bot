"""Walk-forward validation for backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trading_bot.backtest.data_types import (
    BacktestResult,
    BacktestTrade,
    WalkForwardResult,
)
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """Positional bounds of one fold: ``[start, end)`` for train and test."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardValidator:
    """Walk-forward validation with expanding or sliding windows.

    Splits data into *n_splits* train/test folds. Each fold's test set
    comes **after** its training set in time order. An optional ``gap_bars``
    parameter inserts a gap between train and test to avoid leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.8,
        expanding: bool = True,
        gap_bars: int = 0,
    ) -> None:
        self._n_splits = n_splits
        self._train_ratio = train_ratio
        self._expanding = expanding
        self._gap_bars = gap_bars

    def folds(self, n: int) -> list[Fold]:
        """Fold bounds for a series of *n* bars.

        Expanding window: train always starts from the first bar.
        Sliding window: train keeps the size of the first fold's train set.
        """
        test_total_bars = int(n * (1 - self._train_ratio))
        if test_total_bars < self._n_splits:
            # Not enough data for requested splits
            test_per_fold = max(1, test_total_bars)
            actual_splits = max(1, test_total_bars)
        else:
            test_per_fold = test_total_bars // self._n_splits
            actual_splits = self._n_splits

        initial_train_end = n - test_total_bars
        train_size = max(0, initial_train_end - self._gap_bars)

        result: list[Fold] = []
        for fold_i in range(actual_splits):
            test_start = initial_train_end + fold_i * test_per_fold
            test_end = test_start + test_per_fold
            if fold_i == actual_splits - 1:
                test_end = n  # last fold gets remaining bars

            train_end = max(0, test_start - self._gap_bars)
            train_start = 0 if self._expanding else max(0, train_end - train_size)

            if train_end > train_start and test_end > test_start:
                result.append(Fold(train_start, train_end, test_start, test_end))
        return result

    def split(self, data: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return a list of (train, test) DataFrames."""
        return [
            (
                data.iloc[f.train_start : f.train_end],
                data.iloc[f.test_start : f.test_end],
            )
            for f in self.folds(len(data))
        ]

    def run(
        self,
        engine: BacktestEngine,
        ohlcv: pd.DataFrame,
        pair: str,
    ) -> WalkForwardResult:
        """Run walk-forward validation with fixed strategy parameters.

        For each fold the engine sees ``train + gap + test`` but only trades
        and reports the test window: train bars are indicator warm-up. Fold
        metrics are therefore purely out-of-sample. For parameter search use
        :meth:`StrategyOptimizer.walk_forward`, which optimizes on each train
        window before testing.
        """
        results = [
            run_fold(engine, ohlcv, fold, pair) for fold in self.folds(len(ohlcv))
        ]
        return summarize_folds(results)


def run_fold(
    engine: BacktestEngine, ohlcv: pd.DataFrame, fold: Fold, pair: str
) -> BacktestResult:
    """Backtest one fold's test window, using everything before it as warm-up."""
    window = ohlcv.iloc[fold.train_start : fold.test_end]
    return engine.run(window, pair, trade_from=fold.test_start - fold.train_start)


def chain_equity(curves: list[pd.Series]) -> pd.Series:
    """Concatenate per-fold OOS equity, rebasing each onto the previous end.

    Every fold starts from the initial capital; rebasing turns the folds
    into one compounded curve without level jumps between them.
    """
    chained: list[pd.Series] = []
    level: float | None = None
    for curve in curves:
        if curve.empty:
            continue
        start = float(curve.iloc[0])
        scaled = curve if level is None or start == 0 else curve * (level / start)
        chained.append(scaled)
        level = float(scaled.iloc[-1])
    if not chained:
        return pd.Series(dtype=float, name="equity")
    return pd.concat(chained).rename("equity")


def summarize_folds(
    fold_results: list[BacktestResult],
    parameter_stability: dict[str, list[float]] | None = None,
) -> WalkForwardResult:
    """Aggregate fold results into OOS metrics over the chained equity curve."""
    oos_curve = chain_equity([r.equity_curve for r in fold_results])
    oos_trades: list[BacktestTrade] = [t for r in fold_results for t in r.trades]
    aggregate = (
        MetricsCalculator.calculate(oos_trades, oos_curve) if fold_results else {}
    )
    if fold_results:
        aggregate["n_folds"] = float(len(fold_results))

    stability = dict(parameter_stability or {})
    for key in ("sharpe_ratio", "win_rate", "profit_factor", "total_return_pct"):
        stability[key] = [r.metrics.get(key, 0.0) for r in fold_results]

    return WalkForwardResult(
        fold_results=fold_results,
        aggregate_metrics=aggregate,
        oos_equity_curve=oos_curve,
        parameter_stability=stability,
    )
