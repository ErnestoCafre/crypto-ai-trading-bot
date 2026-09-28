"""Download historical OHLCV data, news, and Fear & Greed Index.

Usage:
    uv run python scripts/download_historical.py
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker

from trading_bot.config import get_settings
from trading_bot.config.logging import setup_logging
from trading_bot.data.market.candles import split_closed_candles
from trading_bot.data.market.ccxt_fetcher import CCXTFetcher
from trading_bot.data.news.cryptocompare import CryptoCompareNews
from trading_bot.data.onchain.fear_greed import FearGreedIndex
from trading_bot.data.storage import OHLCVRepository
from trading_bot.db.base import async_session_factory


async def download_ohlcv(
    fetcher: CCXTFetcher,
    session_factory: async_sessionmaker,
    exchange: str,
    pair: str,
    timeframe: str,
    since: datetime,
    semaphore: asyncio.Semaphore,
) -> int:
    """Download OHLCV for one pair/timeframe combination."""
    async with semaphore, session_factory() as session:
        repo = OHLCVRepository(session)

        # Incremental: resume from last known timestamp
        latest = await repo.get_latest_timestamp(exchange, pair, timeframe)
        start = latest + timedelta(milliseconds=1) if latest else since

        logger.info(
            "Downloading {} {} {} from {}",
            pair,
            timeframe,
            exchange,
            start.isoformat(),
        )

        df = await fetcher.fetch_ohlcv(pair, timeframe, since=start)
        # Never persist the forming candle: inserts are ON CONFLICT DO NOTHING
        # and resume after the last stored timestamp, so its partial values
        # would never be corrected.
        df, _ = split_closed_candles(df, timeframe, datetime.now(UTC))

        if df.empty:
            logger.info("No new data for {} {} {}", pair, timeframe, exchange)
            return 0

        # Save to PostgreSQL
        rows = await repo.save_ohlcv(df, exchange, pair, timeframe)

        # Save parquet cache
        OHLCVRepository.save_parquet(df, exchange, pair, timeframe)

        logger.info(
            "Saved {} candles for {} {} {}",
            rows,
            pair,
            timeframe,
            exchange,
        )
        return rows


async def download_news(api_key: str = "") -> int:
    """Download last 48h of crypto news."""
    news_fetcher = CryptoCompareNews(api_key=api_key)
    try:
        articles = await news_fetcher.fetch_latest(limit=100)
        logger.info("Downloaded {} news articles", len(articles))
        return len(articles)
    finally:
        await news_fetcher.close()


async def download_fear_greed() -> int:
    """Download last year of Fear & Greed Index."""
    fgi = FearGreedIndex()
    try:
        df = await fgi.fetch_history(days=365)
        logger.info("Downloaded {} days of Fear & Greed data", len(df))
        return len(df)
    finally:
        await fgi.close()


async def main() -> None:
    setup_logging()
    settings = get_settings()
    t0 = time.monotonic()

    exchange = settings.EXCHANGE_ID
    pairs = settings.TRADING_PAIRS
    timeframes = ["1h", "4h", "1d"]
    since = datetime.now(UTC) - timedelta(days=5 * 365)

    logger.info(
        "Starting historical download: {} pairs x {} timeframes on {}",
        len(pairs),
        len(timeframes),
        exchange,
    )

    # Always use real data (not sandbox) for historical downloads
    fetcher = CCXTFetcher(
        exchange_id=exchange,
        api_key=settings.EXCHANGE_API_KEY,
        api_secret=settings.EXCHANGE_API_SECRET,
        sandbox=False,
        # Multi-year pagination legitimately takes minutes; retries still bound
        # each page, so no total timeout here.
        timeout_s=None,
    )

    semaphore = asyncio.Semaphore(5)
    total_candles = 0

    async with fetcher:
        # Build work items: pairs x timeframes
        # Each task gets its own DB session (AsyncSession is not task-safe)
        tasks = [
            download_ohlcv(
                fetcher, async_session_factory, exchange, pair, tf, since, semaphore
            )
            for pair in pairs
            for tf in timeframes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                logger.error("Download task failed: {}", r)
            else:
                total_candles += int(r)

    # Download news and Fear & Greed in parallel
    news_count, fgi_count = await asyncio.gather(
        download_news(api_key=settings.CRYPTOCOMPARE_API_KEY),
        download_fear_greed(),
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Download complete in {:.1f}s — {} candles, {} articles, {} FGI days",
        elapsed,
        total_candles,
        news_count,
        fgi_count,
    )


if __name__ == "__main__":
    asyncio.run(main())
