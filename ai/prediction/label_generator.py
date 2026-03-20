"""Label generation for ML training — forward returns, direction, triple barrier."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LabelGenerator:
    """Generate training labels from OHLCV data."""

    @staticmethod
    def forward_return(
        df: pd.DataFrame, horizon: int = 1, column: str = "close"
    ) -> pd.Series:
        """Compute forward return over *horizon* bars.

        ``return_t = close_{t+horizon} / close_t - 1``
        The last *horizon* values will be NaN.
        """
        return df[column].shift(-horizon) / df[column] - 1

    @staticmethod
    def direction_label(
        forward_returns: pd.Series, threshold: float = 0.0
    ) -> pd.Series:
        """Convert forward returns to binary direction labels.

        1 if return > threshold, 0 otherwise.
        """
        return (forward_returns > threshold).astype(int)

    @staticmethod
    def triple_barrier_label(
        df: pd.DataFrame,
        profit_take: float = 0.02,
        stop_loss: float = 0.01,
        max_holding: int = 24,
    ) -> pd.Series:
        """Triple-barrier labeling.

        For each bar, look ahead up to *max_holding* bars:
        - If close crosses profit_take first → 1
        - If close crosses -stop_loss first → -1
        - If neither within max_holding → 0

        Returns a Series of {-1, 0, 1}.
        """
        close = df["close"].values
        n = len(close)
        labels = np.zeros(n, dtype=int)

        for i in range(n):
            entry = close[i]
            if entry == 0:
                continue

            label = 0
            end = min(i + max_holding + 1, n)
            for j in range(i + 1, end):
                ret = (close[j] - entry) / entry
                if ret >= profit_take:
                    label = 1
                    break
                if ret <= -stop_loss:
                    label = -1
                    break
            labels[i] = label

        return pd.Series(labels, index=df.index)
