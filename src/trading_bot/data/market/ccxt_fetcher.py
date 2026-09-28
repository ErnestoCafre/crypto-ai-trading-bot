"""CCXT multi-exchange market data fetcher."""

from __future__ import annotations

import asyncio
import random
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
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        timeout_s: float | None = 60.0,
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
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._timeout_s = timeout_s

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
        """Fetch OHLCV from ``since`` with automatic pagination.

        ``limit`` keeps the most recent ``limit`` candles; it does not stop
        pagination early. The result may end with the still-forming candle:
        use ``candles.split_closed_candles`` before computing signals.
        Transient errors are retried with backoff up to ``max_retries``
        times, then re-raised; the whole call is bounded by ``timeout_s``
        (raises TimeoutError).
        """
        if self._timeout_s is None:
            return await self._fetch_ohlcv(pair, timeframe, since, until, limit)
        return await asyncio.wait_for(
            self._fetch_ohlcv(pair, timeframe, since, until, limit),
            timeout=self._timeout_s,
        )

    async def _fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: datetime,
        until: datetime | None,
        limit: int | None,
    ) -> pd.DataFrame:
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000) if until else None
        batch_limit = min(limit, 1000) if limit else 1000

        all_rows: list[list[float]] = []

        while True:
            batch = await self._fetch_page(pair, timeframe, since_ms, batch_limit)
            if not batch:
                break

            for row in batch:
                ts = int(row[0])
                if until_ms and ts >= until_ms:
                    break
                all_rows.append(row)
            else:
                # No break → process next page
                if len(batch) < batch_limit:
                    break
                since_ms = int(batch[-1][0]) + 1
                rate_limit_ms = getattr(self._exchange, "rateLimit", 1000)
                await asyncio.sleep(rate_limit_ms / 1000)
                continue

            # Inner break → until reached
            break

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        df = df[~df.index.duplicated(keep="last")]
        if limit:
            df = df.tail(limit)

        logger.debug(
            "Fetched {} candles for {} {} from {}",
            len(df),
            pair,
            timeframe,
            self._exchange_id,
        )
        return df

    async def _fetch_page(
        self, pair: str, timeframe: str, since_ms: int, limit: int
    ) -> list[list[float]]:
        """One fetch_ohlcv call with bounded exponential backoff + jitter."""
        attempt = 0
        while True:
            try:
                rows: list[list[float]] = await self._exchange.fetch_ohlcv(
                    pair, timeframe, since=since_ms, limit=limit
                )
                return rows
            except ccxt_async.NetworkError as exc:  # incl. RateLimitExceeded
                if attempt >= self._max_retries:
                    logger.error(
                        "{} {}: giving up after {} retries: {}",
                        pair,
                        timeframe,
                        attempt,
                        exc,
                    )
                    raise
                cap = min(self._backoff_max_s, self._backoff_base_s * 2**attempt)
                delay = cap / 2 + random.uniform(0, cap / 2)
                if isinstance(exc, ccxt_async.RateLimitExceeded):
                    rate_limit_s = getattr(self._exchange, "rateLimit", 1000) / 1000
                    delay = max(delay, rate_limit_s)
                attempt += 1
                logger.warning(
                    "{} {}: {} — retry {}/{} in {:.1f}s",
                    pair,
                    timeframe,
                    type(exc).__name__,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

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
