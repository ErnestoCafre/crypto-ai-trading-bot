"""CoinGecko data fetcher stub."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.market.base import BaseFetcher


class CoinGeckoFetcher(BaseFetcher):
    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    async def fetch_ticker(self, pair: str) -> dict[str, object]:
        raise NotImplementedError

    async def get_available_pairs(self) -> list[str]:
        raise NotImplementedError
