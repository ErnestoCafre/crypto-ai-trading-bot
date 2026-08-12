"""Tests for AdvancedFeatureEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.feature_engine_advanced import AdvancedFeatureEngine


@pytest.fixture()
def engine() -> AdvancedFeatureEngine:
    return AdvancedFeatureEngine()


@pytest.fixture()
def ohlcv_100() -> pd.DataFrame:
    """Synthetic 100-bar OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    n = 100
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
    )


class TestReturnFeatures:
    def test_return_columns_present(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        df = engine.build_features(ohlcv_100)
        for col in [
            AdvancedFeatureEngine.LOG_RETURN,
            AdvancedFeatureEngine.ROLLING_RETURN_5,
            AdvancedFeatureEngine.ROLLING_RETURN_20,
        ]:
            assert col in df.columns, f"Missing column: {col}"


class TestRollingStatFeatures:
    def test_stat_columns_present(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        df = engine.build_features(ohlcv_100)
        for col in [
            AdvancedFeatureEngine.ROLLING_SKEW_20,
            AdvancedFeatureEngine.ROLLING_KURT_20,
            AdvancedFeatureEngine.LONGEST_STRIKE_BELOW_MEAN_20,
        ]:
            assert col in df.columns, f"Missing column: {col}"


class TestVolatilityFeatures:
    def test_volatility_columns_present(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        df = engine.build_features(ohlcv_100)
        for col in [
            AdvancedFeatureEngine.REALIZED_VOL_20,
            AdvancedFeatureEngine.PARKINSON_VOL_20,
            AdvancedFeatureEngine.GARMAN_KLASS_VOL_20,
        ]:
            assert col in df.columns, f"Missing column: {col}"


class TestVolumeFeatures:
    def test_volume_columns_present(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        df = engine.build_features(ohlcv_100)
        for col in [
            AdvancedFeatureEngine.VOLUME_RATIO,
            AdvancedFeatureEngine.VOLUME_ZSCORE_20,
        ]:
            assert col in df.columns, f"Missing column: {col}"


class TestPriceFeatures:
    def test_price_columns_present(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        df = engine.build_features(ohlcv_100)
        for col in [
            AdvancedFeatureEngine.PRICE_DIST_SMA20,
            AdvancedFeatureEngine.HIGH_LOW_RANGE_PCT,
            AdvancedFeatureEngine.CLOSE_POSITION_IN_RANGE,
        ]:
            assert col in df.columns, f"Missing column: {col}"


class TestBasePreservation:
    def test_base_features_preserved(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        """Base FeatureEngine columns should still be present."""
        from trading_bot.ai.prediction.feature_engine import FeatureEngine

        df = engine.build_features(ohlcv_100)
        # At minimum RSI should be present from base engine
        assert FeatureEngine.RSI_14 in df.columns


class TestShortDataFrame:
    def test_short_dataframe_no_crash(self, engine: AdvancedFeatureEngine) -> None:
        """Very short DataFrames should not raise."""
        short_df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [98, 99],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )
        df = engine.build_features(short_df)
        assert len(df) == 2


class TestGetFeatureNames:
    def test_feature_names_returns_list(self, engine: AdvancedFeatureEngine) -> None:
        names = engine.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 14  # 3 returns + 3 stats + 3 vol + 2 volume + 3 price

    def test_feature_names_all_present_in_output(
        self, engine: AdvancedFeatureEngine, ohlcv_100: pd.DataFrame
    ) -> None:
        df = engine.build_features(ohlcv_100)
        for name in engine.get_feature_names():
            assert name in df.columns, f"Feature {name} not in output DataFrame"


class TestMissingColumns:
    def test_missing_columns_raises(self, engine: AdvancedFeatureEngine) -> None:
        bad_df = pd.DataFrame({"close": [100, 200], "volume": [10, 20]})
        with pytest.raises(ValueError, match="missing columns"):
            engine.build_features(bad_df)
