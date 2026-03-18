"""CryptoCompare news fetcher."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger

from core.models import NewsArticle
from data._http import create_http_client, retry_with_backoff
from data.news.base import BaseNewsFetcher

BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# Map CryptoCompare categories to trading pairs
_CATEGORY_TO_PAIR: dict[str, str] = {
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

_PAIR_TO_CATEGORIES: dict[str, list[str]] = {
    v: [k] for k, v in _CATEGORY_TO_PAIR.items()
}


class CryptoCompareNews(BaseNewsFetcher):
    """Fetches crypto news from CryptoCompare API."""

    def __init__(self, api_key: str = "") -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["authorization"] = f"Apikey {api_key}"
        self._client = create_http_client(headers=headers)

    async def close(self) -> None:
        await self._client.aclose()

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def _fetch_page(
        self,
        lang: str = "EN",
        lTs: int | None = None,  # noqa: N803
        categories: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        params: dict[str, str | int] = {"lang": lang}
        if lTs is not None:
            params["lTs"] = lTs
        if categories:
            params["categories"] = categories
        resp = await self._client.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def fetch_latest(
        self,
        limit: int = 50,
        categories: list[str] | None = None,
    ) -> list[NewsArticle]:
        """Fetch latest news articles, paginating if needed."""
        cat_str = ",".join(categories) if categories else None
        articles: list[NewsArticle] = []
        last_ts: int | None = None

        while len(articles) < limit:
            data = await self._fetch_page(lTs=last_ts, categories=cat_str)
            news_list = data.get("Data", [])
            if not news_list:
                break

            for item in news_list:
                if len(articles) >= limit:
                    break
                articles.append(self._parse_article(item))

            last_ts = news_list[-1].get("published_on")
            if last_ts is None or len(news_list) < 20:
                break

        logger.debug("CryptoCompare: fetched {} articles", len(articles))
        return articles

    async def fetch_by_pair(
        self,
        pair: str,
        since: datetime,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news related to a specific trading pair."""
        pair_cats = _PAIR_TO_CATEGORIES.get(pair)
        if not pair_cats:
            base = pair.split("/")[0]
            pair_cats = [base]

        all_articles = await self.fetch_latest(limit=limit * 2, categories=pair_cats)

        since_ts = since.timestamp()
        filtered = [a for a in all_articles if a.published_at.timestamp() >= since_ts]
        return filtered[:limit]

    @staticmethod
    def _parse_article(item: dict) -> NewsArticle:  # type: ignore[type-arg]
        """Parse a CryptoCompare news item into NewsArticle."""
        categories_raw = item.get("categories", "")
        categories = [c.strip() for c in categories_raw.split("|") if c.strip()]

        related_pairs: list[str] = []
        for cat in categories:
            pair = _CATEGORY_TO_PAIR.get(cat)
            if pair:
                related_pairs.append(pair)

        body = item.get("body", "")
        if len(body) > 2000:
            body = body[:2000]

        return NewsArticle(
            title=item.get("title", ""),
            body=body,
            source=item.get("source", ""),
            url=item.get("url", ""),
            published_at=datetime.fromtimestamp(item.get("published_on", 0), tz=UTC),
            categories=categories,
            related_pairs=related_pairs,
        )
