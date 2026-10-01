"""Abstract base class for news fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from trading_bot.core.models import NewsArticle

# News category (base asset) -> trading pair
CATEGORY_TO_PAIR: dict[str, str] = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "BNB": "BNB/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "DOGE": "DOGE/USDT",
    "AVAX": "AVAX/USDT",
    "DOT": "DOT/USDT",
    "LINK": "LINK/USDT",
}


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

    async def fetch_for_pairs(
        self,
        pairs: list[str],
        since: datetime,
        limit_per_pair: int = 20,
    ) -> dict[str, list[NewsArticle]]:
        """News for several pairs; providers override it with one request."""
        return {
            pair: await self.fetch_by_pair(pair, since=since, limit=limit_per_pair)
            for pair in pairs
        }

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


def pair_category(pair: str) -> str:
    """News category of a pair's base asset: "BTC/USDT" -> "BTC"."""
    return pair.split("/")[0].upper()


def split_by_pair(
    articles: list[NewsArticle],
    pairs: list[str],
    since: datetime,
    limit_per_pair: int,
) -> dict[str, list[NewsArticle]]:
    """Group articles by the pairs whose category they carry (newest first)."""
    since_ts = since.timestamp()
    recent = sorted(
        (a for a in articles if a.published_at.timestamp() >= since_ts),
        key=lambda a: a.published_at,
        reverse=True,
    )
    result: dict[str, list[NewsArticle]] = {}
    for pair in pairs:
        category = pair_category(pair)
        matches = [a for a in recent if category in {c.upper() for c in a.categories}]
        result[pair] = matches[:limit_per_pair]
    return result
