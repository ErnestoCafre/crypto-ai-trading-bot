"""Unit tests for the data.social module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from trading_bot.data.social.reddit import RedditFetcher
from trading_bot.data.social.twitter import TwitterFetcher


@pytest.mark.asyncio
async def test_twitter_fetcher_raises_not_implemented() -> None:
    fetcher = TwitterFetcher()
    with pytest.raises(NotImplementedError) as exc_info:
        await fetcher.fetch("BTC")
    assert "paid plan" in str(exc_info.value)


@pytest.mark.asyncio
async def test_reddit_fetcher_parses_posts() -> None:
    fetcher = RedditFetcher(subreddits=["Bitcoin"])

    mock_json = {
        "data": {
            "children": [
                {
                    "data": {
                        "title": "Bitcoin reaches new milestone",
                        "selftext": "Details about the recent surge in adoption.",
                        "score": 450,
                        "num_comments": 85,
                        "created_utc": 1700000000.0,
                        "permalink": "/r/Bitcoin/comments/xyz123/",
                    }
                }
            ]
        }
    }

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = lambda: None
    mock_resp.json = lambda: mock_json

    with patch.object(
        fetcher._client, "get", new_callable=AsyncMock, return_value=mock_resp
    ):
        posts = await fetcher.fetch("Bitcoin", limit=5)
        assert len(posts) == 1
        assert posts[0]["title"] == "Bitcoin reaches new milestone"
        assert posts[0]["score"] == 450
        assert posts[0]["subreddit"] == "Bitcoin"
        assert "xyz123" in str(posts[0]["url"])

    await fetcher.close()
