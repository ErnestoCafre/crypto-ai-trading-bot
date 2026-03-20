"""Walk-forward validation for backtesting."""

from __future__ import annotations

import logging

import pandas as pd

from backtest.data_types import BacktestResult, WalkForwardResult
from backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)


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

    def split(self, data: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return a list of (train, test) DataFrames.

        Expanding window: train always starts from the first bar.
        Sliding window: train has a fixed size.
        """
        n = len(data)
        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        # Total test portion
        test_total_bars = int(n * (1 - self._train_ratio))
        if test_total_bars < self._n_splits:
            # Not enough data for requested splits
            test_per_fold = max(1, test_total_bars)
            actual_splits = max(1, test_total_bars)
        else:
            test_per_fold = test_total_bars // self._n_splits
            actual_splits = self._n_splits

        # Build folds
        initial_train_end = n - test_total_bars

        for fold_i in range(actual_splits):
            test_start = initial_train_end + fold_i * test_per_fold
            test_end = test_start + test_per_fold
            if fold_i == actual_splits - 1:
                test_end = n  # last fold gets remaining bars

            # Apply gap
            train_end = max(0, test_start - self._gap_bars)

            if self._expanding:
                train_start = 0
            else:
                # Sliding window: fixed train size
                train_size = initial_train_end
                train_start = max(
                    0,
                    train_end - test_per_fold - self._gap_bars - train_size,
                )

            train = data.iloc[train_start:train_end]
            test = data.iloc[test_start:test_end]

            if len(train) > 0 and len(test) > 0:
                folds.append((train, test))

        return folds

    def run(
        self,
        engine: BacktestEngine,
        ohlcv: pd.DataFrame,
        pair: str,
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Each fold runs the backtest engine on the test set only (train set
        is available for strategy warm-up via lookback).
        """
        folds = self.split(ohlcv)
        fold_results: list[BacktestResult] = []
        oos_equities: list[pd.Series] = []

        for train_df, test_df in folds:
            # Combine train + test so strategy has lookback context
            combined = pd.concat([train_df, test_df])
            result = engine.run(combined, pair)
            fold_results.append(result)

            if len(result.equity_curve) > 0:
                # Only keep OOS (test) portion of equity
                test_len = len(test_df)
                oos_eq = result.equity_curve.iloc[-test_len:]
                oos_equities.append(oos_eq)

        # Aggregate metrics across folds
        aggregate = self._aggregate_metrics(fold_results)

        # Concatenate OOS equity curves
        oos_curve = pd.concat(oos_equities) if oos_equities else pd.Series(dtype=float)

        # Parameter stability
        param_stability: dict[str, list[float]] = {}
        for key in ("sharpe_ratio", "win_rate", "profit_factor"):
            param_stability[key] = [r.metrics.get(key, 0.0) for r in fold_results]

        return WalkForwardResult(
            fold_results=fold_results,
            aggregate_metrics=aggregate,
            oos_equity_curve=oos_curve,
            parameter_stability=param_stability,
        )

    @staticmethod
    def _aggregate_metrics(
        results: list[BacktestResult],
    ) -> dict[str, float]:
        """Average metrics across folds."""
        if not results:
            return {}

        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.metrics.keys())

        agg: dict[str, float] = {}
        for key in all_keys:
            values = [r.metrics.get(key, 0.0) for r in results]
            agg[key] = float(sum(values) / len(values))

        return agg
