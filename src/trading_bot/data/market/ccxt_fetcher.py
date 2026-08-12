"""CCXT multi-exchange market data fetcher."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import TracebackType

import ccxt.async_support as ccxt_async
import pandas as pd
from loguru import logger

from trading_bot.data.market.base import BaseFetcher


class CCXTFetcher(BaseFetcher):
    """Fetches OHLCV and ticker data via CCXT (any supported exchange)."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = True,
    ) -> None:
        exchange_cls = getattr(ccxt_async, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        config: dict[str, object] = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret

        self._exchange: ccxt_async.Exchange = exchange_cls(config)
        if sandbox:
            self._exchange.set_sandbox_mode(True)
        self._exchange_id = exchange_id

    # -- async context manager --

    async def __aenter__(self) -> CCXTFetcher:
        await self._exchange.load_markets()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._exchange.close()

    # -- BaseFetcher implementation --

    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV with automatic pagination."""
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000) if until else None
        batch_limit = min(limit, 1000) if limit else 1000

        all_rows: list[list[float]] = []
        total_fetched = 0

        while True:
            try:
                batch = await self._exchange.fetch_ohlcv(
                    pair, timeframe, since=since_ms, limit=batch_limit
                )
            except ccxt_async.RateLimitExceeded:
                logger.warning("Rate limit hit, sleeping 10s")
                await asyncio.sleep(10)
                continue
            except ccxt_async.NetworkError as exc:
                logger.warning("Network error: {} — retrying in 5s", exc)
                await asyncio.sleep(5)
                continue

            if not batch:
                break

            for row in batch:
                ts = int(row[0])
                if until_ms and ts >= until_ms:
                    break
                all_rows.append(row)
            else:
                # No break → process next page
                total_fetched += len(batch)
                if limit and total_fetched >= limit:
                    break
                if len(batch) < batch_limit:
                    break
                since_ms = int(batch[-1][0]) + 1
                rate_limit_ms = getattr(self._exchange, "rateLimit", 1000)
                await asyncio.sleep(rate_limit_ms / 1000)
                continue

            # Inner break → until reached
            total_fetched += len(batch)
            break

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        if limit:
            df = df.head(limit)

        logger.debug(
            "Fetched {} candles for {} {} from {}",
            len(df),
            pair,
            timeframe,
            self._exchange_id,
        )
        return df

    async def fetch_ticker(self, pair: str) -> dict[str, object]:
        """Fetch current ticker for a pair."""
        ticker = await self._exchange.fetch_ticker(pair)
        return {
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "last": ticker.get("last"),
            "volume_24h": ticker.get("quoteVolume"),
            "timestamp": ticker.get("timestamp"),
        }

    async def get_available_pairs(self) -> list[str]:
        """Return all active trading pairs on the exchange."""
        if not self._exchange.markets:
            await self._exchange.load_markets()
        return [
            symbol
            for symbol, market in self._exchange.markets.items()
            if market.get("active", True)
        ]
