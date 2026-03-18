"""Reddit data fetcher — JSON API, no auth required."""

from __future__ import annotations

from loguru import logger

from data._http import create_http_client, retry_with_backoff

_DEFAULT_SUBREDDITS = ["CryptoCurrency", "Bitcoin", "ethereum", "solana"]


class RedditFetcher:
    """Fetches hot posts from crypto subreddits via public JSON API."""

    def __init__(self, subreddits: list[str] | None = None) -> None:
        self._subreddits = subreddits or _DEFAULT_SUBREDDITS
        self._client = create_http_client(
            headers={"User-Agent": "trading-bot/1.0 (crypto research)"}
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry_with_backoff(max_retries=2, base_delay=2.0)
    async def _fetch_subreddit(
        self, subreddit: str, limit: int = 25
    ) -> list[dict[str, object]]:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        resp = await self._client.get(url, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()

        posts: list[dict[str, object]] = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            posts.append(
                {
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", "")[:1000],
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "subreddit": subreddit,
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                }
            )
        return posts

    async def fetch(
        self, subreddit: str | None = None, limit: int = 25
    ) -> list[dict[str, object]]:
        """Fetch hot posts from one or all configured subreddits."""
        if subreddit:
            return await self._fetch_subreddit(subreddit, limit)

        all_posts: list[dict[str, object]] = []
        for sub in self._subreddits:
            try:
                posts = await self._fetch_subreddit(sub, limit)
                all_posts.extend(posts)
            except Exception:
                logger.warning("Failed to fetch r/{}", sub)
                continue

        # Sort by score descending
        all_posts.sort(key=lambda p: int(str(p.get("score", 0))), reverse=True)
        logger.debug("Reddit: fetched {} posts", len(all_posts))
        return all_posts
