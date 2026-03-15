"""CryptoCompare news fetcher stub."""

from __future__ import annotations

from datetime import datetime

from core.models import NewsArticle
from data.news.base import BaseNewsFetcher


class CryptoCompareNews(BaseNewsFetcher):
    async def fetch_latest(
        self,
        limit: int = 50,
        categories: list[str] | None = None,
    ) -> list[NewsArticle]:
        raise NotImplementedError

    async def fetch_by_pair(
        self,
        pair: str,
        since: datetime,
        limit: int = 20,
    ) -> list[NewsArticle]:
        raise NotImplementedError
