"""On-chain data fetchers."""

from data.onchain.fear_greed import FearGreedIndex
from data.onchain.glassnode import GlassnodeFetcher

__all__ = ["FearGreedIndex", "GlassnodeFetcher"]
