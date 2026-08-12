"""News data fetchers."""

from trading_bot.data.news.cryptocompare import CryptoCompareNews
from trading_bot.data.news.rss_feeds import RSSFeedFetcher

__all__ = ["CryptoCompareNews", "RSSFeedFetcher"]
