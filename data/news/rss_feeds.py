"""RSS feed news fetcher for crypto news sites."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from loguru import logger

from core.models import NewsArticle
from data._http import create_http_client, retry_with_backoff
from data.news.base import BaseNewsFetcher

# Major crypto RSS feeds
_DEFAULT_FEEDS: list[str] = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.theblock.co/rss.xml",
]

# Keywords to match pairs
_PAIR_KEYWORDS: dict[str, list[str]] = {
    "BTC/USDT": ["bitcoin", "btc"],
    "ETH/USDT": ["ethereum", "eth"],
    "SOL/USDT": ["solana", "sol"],
    "BNB/USDT": ["bnb", "binance coin"],
    "XRP/USDT": ["xrp", "ripple"],
}


class RSSFeedFetcher(BaseNewsFetcher):
    """Fetches and parses crypto news from RSS feeds."""

    def __init__(self, feeds: list[str] | None = None) -> None:
        self._feeds = feeds or _DEFAULT_FEEDS
        self._client = create_http_client()

    async def close(self) -> None:
        await self._client.aclose()

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def _fetch_feed(self, url: str) -> str:
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.text

    async def fetch_latest(
        self,
        limit: int = 50,
        categories: list[str] | None = None,
    ) -> list[NewsArticle]:
        """Fetch latest articles from all configured RSS feeds."""
        articles: list[NewsArticle] = []

        for feed_url in self._feeds:
            try:
                xml_text = await self._fetch_feed(feed_url)
                feed_articles = self._parse_feed(xml_text, feed_url)
                articles.extend(feed_articles)
            except Exception:
                logger.warning("Failed to fetch RSS feed: {}", feed_url)
                continue

        # Sort by published date, newest first
        articles.sort(key=lambda a: a.published_at, reverse=True)

        if categories:
            kw_lower = [c.lower() for c in categories]
            articles = [
                a
                for a in articles
                if any(kw in a.title.lower() or kw in a.body.lower() for kw in kw_lower)
            ]

        logger.debug(
            "RSS: fetched {} articles from {} feeds",
            len(articles),
            len(self._feeds),
        )
        return articles[:limit]

    async def fetch_by_pair(
        self,
        pair: str,
        since: datetime,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news matching a specific trading pair."""
        keywords = _PAIR_KEYWORDS.get(pair)
        if not keywords:
            base = pair.split("/")[0].lower()
            keywords = [base]

        all_articles = await self.fetch_latest(limit=limit * 3)

        since_ts = since.timestamp()
        filtered = []
        for article in all_articles:
            if article.published_at.timestamp() < since_ts:
                continue
            text = (article.title + " " + article.body).lower()
            if any(kw in text for kw in keywords):
                filtered.append(article)

        return filtered[:limit]

    @staticmethod
    def _parse_feed(xml_text: str, feed_url: str) -> list[NewsArticle]:
        """Parse RSS XML into NewsArticle list."""
        articles: list[NewsArticle] = []
        try:
            root = ET.fromstring(xml_text)  # noqa: S314
        except ET.ParseError:
            logger.warning("Failed to parse XML from {}", feed_url)
            return articles

        # Handle both RSS 2.0 (<channel><item>) and Atom (<entry>)
        items = root.findall(".//item")
        if not items:
            # Try Atom format
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//atom:entry", ns)

        for item in items:
            title = _text(item, "title") or ""
            description = _text(item, "description") or _text(item, "summary") or ""
            link = _text(item, "link") or _text(item, "guid") or ""
            pub_date_str = _text(item, "pubDate") or _text(item, "published") or ""

            if len(description) > 2000:
                description = description[:2000]

            published_at = _parse_date(pub_date_str)

            # Detect related pairs from title + description
            related_pairs = _detect_pairs(title + " " + description)

            articles.append(
                NewsArticle(
                    title=title,
                    body=description,
                    source=feed_url,
                    url=link,
                    published_at=published_at,
                    categories=[],
                    related_pairs=related_pairs,
                )
            )

        return articles


def _text(element: ET.Element, tag: str) -> str | None:
    """Extract text from a child element."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _parse_date(date_str: str) -> datetime:
    """Parse RSS date string to datetime."""
    if not date_str:
        return datetime.now(UTC)
    try:
        return parsedate_to_datetime(date_str).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return datetime.now(UTC)


def _detect_pairs(text: str) -> list[str]:
    """Detect trading pairs mentioned in text."""
    text_lower = text.lower()
    found: list[str] = []
    for pair, keywords in _PAIR_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(pair)
    return found
