"""Integration tests for the data pipeline.

These tests require a running PostgreSQL instance and optionally
a network connection to Binance.

Run with: uv run pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trading_bot.data.storage import OHLCVRepository
from trading_bot.db.base import Base


@pytest_asyncio.fixture
async def db_session():
    """Create an in-memory SQLite session for testing.

    Note: For full integration tests with PostgreSQL, set DATABASE_URL
    environment variable and use the real async_session_factory.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
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


@pytest.mark.integration
class TestOHLCVPipeline:
    """Integration tests for OHLCV save/load pipeline."""

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(
        self, db_session: AsyncSession, sample_df: pd.DataFrame
    ) -> None:
        repo = OHLCVRepository(db_session)

        rows = await repo.save_ohlcv(sample_df, "binance", "BTC/USDT", "1h")
        assert rows == 10

        loaded = await repo.load_ohlcv("binance", "BTC/USDT", "1h")
        assert len(loaded) == 10
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]

    @pytest.mark.asyncio
    async def test_get_latest_timestamp(
        self, db_session: AsyncSession, sample_df: pd.DataFrame
    ) -> None:
        repo = OHLCVRepository(db_session)
        await repo.save_ohlcv(sample_df, "binance", "BTC/USDT", "1h")

        latest = await repo.get_latest_timestamp("binance", "BTC/USDT", "1h")
        assert latest is not None

    @pytest.mark.asyncio
    async def test_get_latest_timestamp_empty(self, db_session: AsyncSession) -> None:
        repo = OHLCVRepository(db_session)
        latest = await repo.get_latest_timestamp("binance", "BTC/USDT", "1h")
        assert latest is None

    @pytest.mark.asyncio
    async def test_load_ohlcv_with_time_range(
        self, db_session: AsyncSession, sample_df: pd.DataFrame
    ) -> None:
        repo = OHLCVRepository(db_session)
        await repo.save_ohlcv(sample_df, "binance", "BTC/USDT", "1h")

        since = datetime(2024, 1, 1, 3, 0, tzinfo=UTC)
        until = datetime(2024, 1, 1, 7, 0, tzinfo=UTC)
        loaded = await repo.load_ohlcv(
            "binance", "BTC/USDT", "1h", since=since, until=until
        )
        assert len(loaded) <= 4

    @pytest.mark.asyncio
    async def test_load_empty_table(self, db_session: AsyncSession) -> None:
        repo = OHLCVRepository(db_session)
        loaded = await repo.load_ohlcv("binance", "BTC/USDT", "1h")
        assert loaded.empty


@pytest.mark.integration
class TestCCXTIntegration:
    """Integration tests requiring network access to Binance."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_fetch_small_ohlcv(self) -> None:
        """Fetch 10 candles from Binance (real network call)."""
        from trading_bot.data.market.ccxt_fetcher import CCXTFetcher

        since = datetime.now(UTC) - timedelta(hours=12)

        async with CCXTFetcher(exchange_id="binance", sandbox=False) as fetcher:
            df = await fetcher.fetch_ohlcv("BTC/USDT", "1h", since=since, limit=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 10
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
