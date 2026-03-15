"""Glassnode on-chain data fetcher stub."""

from __future__ import annotations


class GlassnodeFetcher:
    def fetch(self, metric: str) -> dict[str, object]:
        raise NotImplementedError
