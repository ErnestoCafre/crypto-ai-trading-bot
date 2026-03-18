"""News data fetchers."""

from data.news.cryptocompare import CryptoCompareNews
from data.news.rss_feeds import RSSFeedFetcher

__all__ = ["CryptoCompareNews", "RSSFeedFetcher"]
