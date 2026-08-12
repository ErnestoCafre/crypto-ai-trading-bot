"""Tests for data.storage module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.storage import OHLCVRepository


@pytest.fixture()
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 10
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 42000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        },
        index=timestamps,
    ).rename_axis("timestamp")


class TestParquetCache:
    """Tests for parquet save/load roundtrip."""

    def test_save_and_load_parquet(
        self, sample_ohlcv_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        import trading_bot.data.storage as storage_mod

        original_cache = storage_mod.CACHE_DIR
        storage_mod.CACHE_DIR = tmp_path
        try:
            path = OHLCVRepository.save_parquet(
                sample_ohlcv_df, "binance", "BTC/USDT", "1h"
            )
            assert path.exists()
            assert path.suffix == ".parquet"

            loaded = OHLCVRepository.load_parquet("binance", "BTC/USDT", "1h")
            assert loaded is not None
            assert len(loaded) == len(sample_ohlcv_df)
            pd.testing.assert_frame_equal(loaded, sample_ohlcv_df, check_freq=False)
        finally:
            storage_mod.CACHE_DIR = original_cache

    def test_load_parquet_returns_none_when_missing(self, tmp_path: Path) -> None:
        import trading_bot.data.storage as storage_mod

        original_cache = storage_mod.CACHE_DIR
        storage_mod.CACHE_DIR = tmp_path
        try:
            result = OHLCVRepository.load_parquet("binance", "BTC/USDT", "1h")
            assert result is None
        finally:
            storage_mod.CACHE_DIR = original_cache

    def test_save_parquet_creates_directory_structure(
        self, sample_ohlcv_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        import trading_bot.data.storage as storage_mod

        original_cache = storage_mod.CACHE_DIR
        storage_mod.CACHE_DIR = tmp_path
        try:
            OHLCVRepository.save_parquet(sample_ohlcv_df, "binance", "ETH/USDT", "4h")
            expected_dir = tmp_path / "binance" / "ETHUSDT"
            assert expected_dir.exists()
            assert (expected_dir / "4h.parquet").exists()
        finally:
            storage_mod.CACHE_DIR = original_cache


class TestDFToRecords:
    """Tests for internal _df_to_records helper."""

    def test_converts_df_to_records(self, sample_ohlcv_df: pd.DataFrame) -> None:
        from trading_bot.data.storage import _df_to_records

        records = _df_to_records(sample_ohlcv_df, "binance", "BTC/USDT", "1h")

        assert len(records) == len(sample_ohlcv_df)
        assert records[0]["exchange"] == "binance"
        assert records[0]["pair"] == "BTC/USDT"
        assert records[0]["timeframe"] == "1h"
        assert isinstance(records[0]["open"], float)
        assert isinstance(records[0]["timestamp"], datetime)
