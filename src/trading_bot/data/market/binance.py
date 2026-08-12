"""Binance data fetcher — thin wrapper around CCXTFetcher."""

from __future__ import annotations

from datetime import datetime
from types import TracebackType

import pandas as pd

from trading_bot.data.market.base import BaseFetcher
from trading_bot.data.market.ccxt_fetcher import CCXTFetcher


class BinanceFetcher(BaseFetcher):
    """Binance-specific fetcher delegating to CCXT."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = True,
    ) -> None:
        self._fetcher = CCXTFetcher(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
        )

    async def __aenter__(self) -> BinanceFetcher:
        await self._fetcher.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._fetcher.__aexit__(exc_type, exc_val, exc_tb)

    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        return await self._fetcher.fetch_ohlcv(pair, timeframe, since, until, limit)

    async def fetch_ticker(self, pair: str) -> dict[str, object]:
        return await self._fetcher.fetch_ticker(pair)

    async def get_available_pairs(self) -> list[str]:
        return await self._fetcher.get_available_pairs()
