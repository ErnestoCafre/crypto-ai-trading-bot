"""OHLCV repository — persistence layer for market data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from trading_bot.db.models import OHLCVData

CACHE_DIR = Path("var/cache")


class OHLCVRepository:
    """Repository pattern for OHLCV data: PostgreSQL + parquet cache."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_ohlcv(
        self,
        df: pd.DataFrame,
        exchange: str,
        pair: str,
        timeframe: str,
        chunk_size: int = 5000,
    ) -> int:
        """Bulk insert OHLCV data with ON CONFLICT DO NOTHING.

        Returns the number of rows inserted.
        """
        if df.empty:
            return 0

        rows_inserted = 0
        records = _df_to_records(df, exchange, pair, timeframe)

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = text(
                """
                INSERT INTO ohlcv_data (exchange, pair, timeframe, timestamp,
                                        open, high, low, close, volume)
                VALUES (:exchange, :pair, :timeframe, :timestamp,
                        :open, :high, :low, :close, :volume)
                ON CONFLICT (exchange, pair, timeframe, timestamp) DO NOTHING
                """
            )
            await self._session.execute(stmt, chunk)
            rows_inserted += len(chunk)

        await self._session.commit()
        logger.debug(
            "Saved {} rows for {} {} {} to PostgreSQL",
            rows_inserted,
            exchange,
            pair,
            timeframe,
        )
        return rows_inserted

    async def load_ohlcv(
        self,
        exchange: str,
        pair: str,
        timeframe: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from PostgreSQL into a DataFrame."""
        stmt = (
            select(OHLCVData)
            .where(OHLCVData.exchange == exchange)
            .where(OHLCVData.pair == pair)
            .where(OHLCVData.timeframe == timeframe)
        )
        if since:
            stmt = stmt.where(OHLCVData.timestamp >= since)
        if until:
            stmt = stmt.where(OHLCVData.timestamp < until)
        stmt = stmt.order_by(OHLCVData.timestamp)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        data = [
            {
                "timestamp": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    async def get_latest_timestamp(
        self,
        exchange: str,
        pair: str,
        timeframe: str,
    ) -> datetime | None:
        """Get the most recent timestamp for incremental downloads."""
        stmt = (
            select(OHLCVData.timestamp)
            .where(OHLCVData.exchange == exchange)
            .where(OHLCVData.pair == pair)
            .where(OHLCVData.timeframe == timeframe)
            .order_by(OHLCVData.timestamp.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return row

    @staticmethod
    def save_parquet(
        df: pd.DataFrame,
        exchange: str,
        pair: str,
        timeframe: str,
    ) -> Path:
        """Save DataFrame as parquet file in cache directory."""
        pair_clean = pair.replace("/", "")
        cache_path = CACHE_DIR / exchange / pair_clean
        cache_path.mkdir(parents=True, exist_ok=True)

        file_path = cache_path / f"{timeframe}.parquet"
        df.to_parquet(file_path)
        logger.debug("Saved parquet: {}", file_path)
        return file_path

    @staticmethod
    def load_parquet(
        exchange: str,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Load DataFrame from parquet cache. Returns None if not found."""
        pair_clean = pair.replace("/", "")
        file_path = CACHE_DIR / exchange / pair_clean / f"{timeframe}.parquet"

        if not file_path.exists():
            return None

        df = pd.read_parquet(file_path)
        logger.debug("Loaded parquet: {} ({} rows)", file_path, len(df))
        return df


def _df_to_records(
    df: pd.DataFrame,
    exchange: str,
    pair: str,
    timeframe: str,
) -> list[dict[str, object]]:
    """Convert DataFrame to list of dicts for bulk insert."""
    records: list[dict[str, object]] = []
    for ts, row in df.iterrows():
        records.append(
            {
                "exchange": exchange,
                "pair": pair,
                "timeframe": timeframe,
                "timestamp": ts.to_pydatetime(),  # type: ignore[union-attr]
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return records
