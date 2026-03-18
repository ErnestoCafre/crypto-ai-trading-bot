"""Market data fetchers."""

from data.market.binance import BinanceFetcher
from data.market.ccxt_fetcher import CCXTFetcher
from data.market.coingecko import CoinGeckoFetcher

__all__ = ["BinanceFetcher", "CCXTFetcher", "CoinGeckoFetcher"]
