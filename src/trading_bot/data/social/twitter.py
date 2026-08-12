"""Twitter/X data fetcher — deferred.

The X API v2 requires a paid plan ($100/month minimum for Basic tier).
This fetcher is intentionally left as a stub until API access is provisioned.

When implementing:
- Use httpx with Bearer token auth (TWITTER_BEARER_TOKEN from Settings)
- Endpoint: GET /2/tweets/search/recent
- Rate limit: 450 requests / 15 min (Basic)
- Parse tweet text, author, metrics, created_at
"""

from __future__ import annotations


class TwitterFetcher:
    """Stub — X API requires paid subscription ($100/month Basic tier)."""

    async def fetch(self, query: str) -> list[dict[str, object]]:
        raise NotImplementedError(
            "Twitter/X API requires a paid plan. "
            "Set TWITTER_BEARER_TOKEN and implement this fetcher."
        )
