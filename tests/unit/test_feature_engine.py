"""Tests for ai.prediction.feature_engine.FeatureEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.feature_engine import FeatureEngine


@pytest.fixture()
def fe() -> FeatureEngine:
    return FeatureEngine()


class TestBuildFeatures:
    def test_returns_enriched_dataframe(
        self, fe: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fe.build_features(sample_ohlcv)
        assert len(result.columns) > len(sample_ohlcv.columns)

    def test_preserves_ohlcv_columns(
        self, fe: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fe.build_features(sample_ohlcv)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_known_indicators_present(
        self, fe: FeatureEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = fe.build_features(sample_ohlcv)
        expected = [
            FeatureEngine.RSI_14,
            FeatureEngine.MACD_12_26_9,
            FeatureEngine.EMA_9,
            FeatureEngine.SMA_20,
            FeatureEngine.ADX_14,
            FeatureEngine.OBV,
        ]
        for col in expected:
            assert col in result.columns, f"Missing indicator column: {col}"

    def test_no_nan_after_dropna_with_enough_rows(
        self, fe: FeatureEngine, sample_ohlcv_250: pd.DataFrame
    ) -> None:
        result = fe.build_features(sample_ohlcv_250).dropna()
        assert len(result) > 0
        assert not result.isna().any().any()

    def test_short_dataframe_does_not_crash(self, fe: FeatureEngine) -> None:
        rng = np.random.default_rng(0)
        short = pd.DataFrame(
            {
                "open": rng.normal(100, 1, 5),
                "high": rng.normal(101, 1, 5),
                "low": rng.normal(99, 1, 5),
                "close": rng.normal(100, 1, 5),
                "volume": rng.normal(500, 50, 5),
            }
        )
        result = fe.build_features(short)
        assert len(result) == 5

    def test_missing_column_raises_value_error(self, fe: FeatureEngine) -> None:
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5]})
        with pytest.raises(ValueError, match="missing columns"):
            fe.build_features(df)
