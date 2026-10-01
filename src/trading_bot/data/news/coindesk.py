"""CoinDesk Data API news fetcher (successor of CryptoCompare min-api).

Endpoint "Latest Articles", ``GET /news/v1/article/list`` on
https://data-api.coindesk.com — parameters and response fields as published
in the API's OpenAPI spec (``/info/v1/openapi``; human docs at
https://developers.coindesk.com/documentation/data-api/news). An API key is
required; it goes in the ``Authorization: Apikey <key>`` header.

The free tier is a fixed lifetime call quota, so the bot makes one request
per sentiment cycle for all pairs (``fetch_for_pairs``) and splits the
articles by category in memory.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trading_bot.core.models import NewsArticle
from trading_bot.data._http import create_http_client, retry_with_backoff
from trading_bot.data.news.base import (
    CATEGORY_TO_PAIR,
    BaseNewsFetcher,
    pair_category,
    split_by_pair,
)

BASE_URL = "https://data-api.coindesk.com"
ARTICLE_LIST_PATH = "/news/v1/article/list"
MAX_LIMIT = 100  # API bound for ``limit``
_MAX_BODY_CHARS = 2000


class CoinDeskNews(BaseNewsFetcher):
    """Fetches crypto news from the CoinDesk Data API."""

    def __init__(self, api_key: str = "", **client_kwargs: Any) -> None:
        headers = {"User-Agent": "trading-bot/1.0"}
        if api_key:
            headers["Authorization"] = f"Apikey {api_key}"
        self._client = create_http_client(
            base_url=BASE_URL, headers=headers, **client_kwargs
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def _fetch_articles(
        self,
        limit: int,
        categories: list[str] | None = None,
        lang: str = "EN",
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {
            "lang": lang,
            "limit": max(1, min(limit, MAX_LIMIT)),
        }
        if categories:
            # OpenAPI: style=form, explode=false -> comma-separated.
            params["categories"] = ",".join(categories)
        resp = await self._client.get(ARTICLE_LIST_PATH, params=params)
        resp.raise_for_status()
        payload = resp.json()
        err = payload.get("Err") or {}
        if err:
            # The response schema carries "Err" next to "Data"; log it if set.
            logger.warning("CoinDesk news error: {}", err.get("message", err))
        data = payload.get("Data") or []
        return data if isinstance(data, list) else []

    async def fetch_latest(
        self,
        limit: int = 50,
        categories: list[str] | None = None,
    ) -> list[NewsArticle]:
        """Latest articles (single request, at most 100)."""
        items = await self._fetch_articles(limit, categories)
        articles = [self._parse_article(item) for item in items]
        logger.debug("CoinDesk: fetched {} articles", len(articles))
        return articles

    async def fetch_for_pairs(
        self,
        pairs: list[str],
        since: datetime,
        limit_per_pair: int = 20,
    ) -> dict[str, list[NewsArticle]]:
        """One request for all pairs, split by category in memory."""
        categories = sorted({pair_category(p) for p in pairs})
        articles = await self.fetch_latest(limit=MAX_LIMIT, categories=categories)
        return split_by_pair(articles, pairs, since, limit_per_pair)

    async def fetch_by_pair(
        self,
        pair: str,
        since: datetime,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """News related to one trading pair."""
        by_pair = await self.fetch_for_pairs([pair], since, limit_per_pair=limit)
        return by_pair[pair]

    @staticmethod
    def _parse_article(item: dict[str, Any]) -> NewsArticle:
        """Parse a ``NEWS_ARTICLES`` item into a NewsArticle."""
        categories = [
            str(c.get("CATEGORY") or c.get("NAME") or "").strip()
            for c in item.get("CATEGORY_DATA") or []
            if isinstance(c, dict)
        ]
        categories = [c for c in categories if c]
        source_data = item.get("SOURCE_DATA") or {}
        body = str(item.get("BODY") or "")[:_MAX_BODY_CHARS]
        return NewsArticle(
            title=str(item.get("TITLE") or ""),
            body=body,
            source=str(source_data.get("NAME") or ""),
            url=str(item.get("URL") or ""),
            published_at=datetime.fromtimestamp(
                int(item.get("PUBLISHED_ON") or 0), tz=UTC
            ),
            categories=categories,
            related_pairs=[
                CATEGORY_TO_PAIR[c.upper()]
                for c in categories
                if c.upper() in CATEGORY_TO_PAIR
            ],
            article_id=str(item.get("ID") or item.get("GUID") or ""),
        )
