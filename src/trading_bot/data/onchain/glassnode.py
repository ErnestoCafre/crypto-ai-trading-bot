"""Glassnode on-chain data fetcher — deferred.

Glassnode API requires a paid subscription for most useful metrics.
Free tier only provides a very limited set of endpoints.

When implementing:
- Use httpx with API key auth (header: x-api-key)
- Endpoints: /v1/metrics/{category}/{metric}
- Useful metrics: sopr, nupl, mvrv, exchange_balance
- Rate limit depends on plan tier
"""

from __future__ import annotations


class GlassnodeFetcher:
    """Stub — Glassnode API requires paid subscription for useful metrics."""

    async def fetch(self, metric: str) -> dict[str, object]:
        raise NotImplementedError(
            "Glassnode API requires a paid subscription. "
            "See https://docs.glassnode.com for pricing."
        )
