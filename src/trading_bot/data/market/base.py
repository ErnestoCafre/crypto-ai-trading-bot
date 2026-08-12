"""Abstract base class for market data fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class BaseFetcher(ABC):
    @abstractmethod
    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    async def fetch_ticker(self, pair: str) -> dict[str, object]: ...

    @abstractmethod
    async def get_available_pairs(self) -> list[str]: ...
