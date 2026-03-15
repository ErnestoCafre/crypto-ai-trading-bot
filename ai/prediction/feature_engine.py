"""Feature engineering stub."""

from __future__ import annotations

import pandas as pd


class FeatureEngine:
    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
