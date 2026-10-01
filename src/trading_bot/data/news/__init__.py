"""News data fetchers."""

from __future__ import annotations

from loguru import logger

from trading_bot.config.settings import Settings
from trading_bot.data.news.base import BaseNewsFetcher
from trading_bot.data.news.coindesk import CoinDeskNews
from trading_bot.data.news.cryptocompare import CryptoCompareNews
from trading_bot.data.news.rss_feeds import RSSFeedFetcher

__all__ = [
    "BaseNewsFetcher",
    "CoinDeskNews",
    "CryptoCompareNews",
    "RSSFeedFetcher",
    "build_news_fetcher",
]


def build_news_fetcher(settings: Settings) -> BaseNewsFetcher | None:
    """Client for NEWS_PROVIDER, or None (with a warning) without an API key.

    Both CoinDesk and the legacy CryptoCompare API answer HTTP 401 to
    unauthenticated calls, so without a key there is no news source.
    """
    provider = str(settings.NEWS_PROVIDER).lower()
    if provider == "cryptocompare":
        key, key_name = settings.CRYPTOCOMPARE_API_KEY, "CRYPTOCOMPARE_API_KEY"
    else:
        key, key_name = settings.COINDESK_API_KEY, "COINDESK_API_KEY"
    if not key:
        logger.warning(
            "{} is empty: the {} news API rejects unauthenticated calls "
            "(HTTP 401). News disabled; Fear & Greed still used.",
            key_name,
            provider,
        )
        return None
    if provider == "cryptocompare":
        return CryptoCompareNews(api_key=key)
    return CoinDeskNews(api_key=key)
