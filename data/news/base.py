"""Abstract base class for news fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from core.models import NewsArticle


class BaseNewsFetcher(ABC):
    @abstractmethod
    async def fetch_latest(
        self,
        limit: int = 50,
        categories: list[str] | None = None,
    ) -> list[NewsArticle]: ...

    @abstractmethod
    async def fetch_by_pair(
        self,
        pair: str,
        since: datetime,
        limit: int = 20,
    ) -> list[NewsArticle]: ...
