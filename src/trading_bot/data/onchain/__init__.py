"""On-chain data fetchers."""

from trading_bot.data.onchain.fear_greed import FearGreedIndex
from trading_bot.data.onchain.glassnode import GlassnodeFetcher

__all__ = ["FearGreedIndex", "GlassnodeFetcher"]
