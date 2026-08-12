"""ML data pipeline — feature/label alignment and time-series splits."""

from __future__ import annotations

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.label_generator import LabelGenerator


class MLDataPipeline:
    """Prepare OHLCV data for ML training: features + labels, time-series split."""

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        label_generator: LabelGenerator | None = None,
    ) -> None:
        self._fe = feature_engine or FeatureEngine()
        self._lg = label_generator or LabelGenerator()

    def prepare(
        self,
        ohlcv: pd.DataFrame,
        label_horizon: int = 1,
        label_threshold: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build features and labels, drop NaN rows, return aligned pair.

        Returns:
            (features DataFrame, labels Series) — same index, no NaN.
        """
        features = self._fe.build_features(ohlcv)

        forward_ret = self._lg.forward_return(ohlcv, horizon=label_horizon)
        labels = self._lg.direction_label(forward_ret, threshold=label_threshold)

        # Align and drop NaN
        combined = features.copy()
        combined["__label__"] = labels
        combined = combined.dropna()

        y = combined["__label__"].astype(int)
        x = combined.drop(columns=["__label__"])

        return x, y

    @staticmethod
    def time_series_split(
        features: pd.DataFrame,
        labels: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ]:
        """Split into train/val/test sets in chronological order (no shuffle).

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        x_train = features.iloc[:train_end]
        y_train = labels.iloc[:train_end]
        x_val = features.iloc[train_end:val_end]
        y_val = labels.iloc[train_end:val_end]
        x_test = features.iloc[val_end:]
        y_test = labels.iloc[val_end:]

        return x_train, y_train, x_val, y_val, x_test, y_test
