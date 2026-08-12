"""Market data fetchers."""

from trading_bot.data.market.binance import BinanceFetcher
from trading_bot.data.market.ccxt_fetcher import CCXTFetcher
from trading_bot.data.market.coingecko import CoinGeckoFetcher

__all__ = ["BinanceFetcher", "CCXTFetcher", "CoinGeckoFetcher"]
