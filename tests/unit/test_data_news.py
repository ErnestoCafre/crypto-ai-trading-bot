"""Tests for data.news fetchers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from trading_bot.core.models import NewsArticle

# ---------------------------------------------------------------------------
# CryptoCompareNews
# ---------------------------------------------------------------------------


class TestCryptoCompareNews:
    """Tests for CryptoCompareNews fetcher."""

    def _make_fetcher(self) -> object:
        from trading_bot.data.news.cryptocompare import CryptoCompareNews

        fetcher = CryptoCompareNews()
        return fetcher

    def _sample_api_response(self, count: int = 3) -> dict:  # type: ignore[type-arg]
        return {
            "Data": [
                {
                    "title": f"Bitcoin hits new high #{i}",
                    "body": f"Article body about BTC #{i}" * 10,
                    "source": "CoinDesk",
                    "url": f"https://example.com/article-{i}",
                    "published_on": 1704067200 + i * 3600,
                    "categories": "BTC|Trading",
                }
                for i in range(count)
            ]
        }

    @pytest.mark.asyncio
    async def test_parse_article(self) -> None:
        from trading_bot.data.news.cryptocompare import CryptoCompareNews

        item = {
            "title": "Bitcoin Rally",
            "body": "BTC surges past 50k",
            "source": "CoinDesk",
            "url": "https://example.com/btc-rally",
            "published_on": 1704067200,
            "categories": "BTC|ETH|Trading",
        }

        article = CryptoCompareNews._parse_article(item)

        assert isinstance(article, NewsArticle)
        assert article.title == "Bitcoin Rally"
        assert article.source == "CoinDesk"
        assert "BTC" in article.categories
        assert "BTC/USDT" in article.related_pairs

    @pytest.mark.asyncio
    async def test_fetch_latest(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._fetch_page = AsyncMock(return_value=self._sample_api_response(3))  # type: ignore[attr-defined]

        articles = await fetcher.fetch_latest(limit=3)  # type: ignore[union-attr]

        assert len(articles) == 3
        assert all(isinstance(a, NewsArticle) for a in articles)

    @pytest.mark.asyncio
    async def test_fetch_latest_pagination(self) -> None:
        fetcher = self._make_fetcher()
        page1 = self._sample_api_response(20)
        page2 = self._sample_api_response(5)
        fetcher._fetch_page = AsyncMock(side_effect=[page1, page2])  # type: ignore[attr-defined]

        articles = await fetcher.fetch_latest(limit=25)  # type: ignore[union-attr]

        assert len(articles) == 25
        assert fetcher._fetch_page.call_count == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_category_to_pair_mapping(self) -> None:
        from trading_bot.data.news.cryptocompare import CryptoCompareNews

        item = {
            "title": "ETH Update",
            "body": "Ethereum upgrade",
            "source": "Source",
            "url": "https://example.com",
            "published_on": 1704067200,
            "categories": "ETH",
        }

        article = CryptoCompareNews._parse_article(item)
        assert "ETH/USDT" in article.related_pairs

    @pytest.mark.asyncio
    async def test_body_truncation(self) -> None:
        from trading_bot.data.news.cryptocompare import CryptoCompareNews

        item = {
            "title": "Long Article",
            "body": "x" * 5000,
            "source": "Source",
            "url": "https://example.com",
            "published_on": 1704067200,
            "categories": "",
        }

        article = CryptoCompareNews._parse_article(item)
        assert len(article.body) == 2000


# ---------------------------------------------------------------------------
# RSSFeedFetcher
# ---------------------------------------------------------------------------


class TestRSSFeedFetcher:
    """Tests for RSSFeedFetcher."""

    SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Crypto News</title>
        <item>
          <title>Bitcoin breaks $50k</title>
          <description>BTC surges past 50k mark today</description>
          <link>https://example.com/btc-50k</link>
          <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Ethereum DeFi growth</title>
          <description>Ethereum ecosystem sees massive DeFi growth</description>
          <link>https://example.com/eth-defi</link>
          <pubDate>Mon, 01 Jan 2024 11:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>"""

    @pytest.mark.asyncio
    async def test_parse_rss_xml(self) -> None:
        from trading_bot.data.news.rss_feeds import RSSFeedFetcher

        articles = RSSFeedFetcher._parse_feed(
            self.SAMPLE_RSS, "https://example.com/rss"
        )

        assert len(articles) == 2
        assert articles[0].title == "Bitcoin breaks $50k"
        assert articles[0].url == "https://example.com/btc-50k"
        assert "BTC/USDT" in articles[0].related_pairs

    @pytest.mark.asyncio
    async def test_fetch_latest_aggregates_feeds(self) -> None:
        from trading_bot.data.news.rss_feeds import RSSFeedFetcher

        fetcher = RSSFeedFetcher(feeds=["https://feed1.com", "https://feed2.com"])
        fetcher._fetch_feed = AsyncMock(return_value=self.SAMPLE_RSS)

        articles = await fetcher.fetch_latest(limit=10)

        assert len(articles) <= 10
        assert fetcher._fetch_feed.call_count == 2
        await fetcher.close()

    @pytest.mark.asyncio
    async def test_fetch_by_pair_filters(self) -> None:
        from trading_bot.data.news.rss_feeds import RSSFeedFetcher

        fetcher = RSSFeedFetcher(feeds=["https://feed1.com"])
        fetcher._fetch_feed = AsyncMock(return_value=self.SAMPLE_RSS)

        articles = await fetcher.fetch_by_pair(
            "BTC/USDT",
            since=datetime(2020, 1, 1, tzinfo=UTC),
        )

        # Only Bitcoin article should match
        assert all(
            "btc" in a.title.lower() or "bitcoin" in a.title.lower() for a in articles
        )
        await fetcher.close()

    @pytest.mark.asyncio
    async def test_invalid_xml_handled(self) -> None:
        from trading_bot.data.news.rss_feeds import RSSFeedFetcher

        articles = RSSFeedFetcher._parse_feed("not xml at all", "https://bad.com")
        assert articles == []


# ---------------------------------------------------------------------------
# CoinDeskNews (Data API, /news/v1/article/list)
# ---------------------------------------------------------------------------


def _coindesk_item(
    i: int, categories: list[str], published_on: int | None = None
) -> dict:  # type: ignore[type-arg]
    """Item shaped as the NEWS_ARTICLES schema of the Data API OpenAPI spec."""
    return {
        "TYPE": "121",
        "ID": 1000 + i,
        "GUID": f"guid-{i}",
        "PUBLISHED_ON": published_on
        if published_on is not None
        else int(datetime.now(UTC).timestamp()) - i * 60,
        "TITLE": f"Headline {i}",
        "BODY": f"Body {i}",
        "URL": f"https://example.com/{i}",
        "SOURCE_DATA": {"NAME": "CoinDesk", "SOURCE_KEY": "coindesk"},
        "CATEGORY_DATA": [
            {"TYPE": "122", "ID": j, "NAME": c, "CATEGORY": c}
            for j, c in enumerate(categories)
        ],
    }


class TestCoinDeskNews:
    def _fetcher(self, handler):  # type: ignore[no-untyped-def]
        import httpx

        from trading_bot.data.news.coindesk import CoinDeskNews

        return CoinDeskNews(api_key="k", transport=httpx.MockTransport(handler))

    def test_parse_article(self) -> None:
        from trading_bot.data.news.coindesk import CoinDeskNews

        item = _coindesk_item(1, ["BTC", "MARKET"], published_on=1704067200)
        item["BODY"] = "x" * 5000
        article = CoinDeskNews._parse_article(item)

        assert article.article_id == "1001"
        assert article.title == "Headline 1"
        assert article.source == "CoinDesk"
        assert article.categories == ["BTC", "MARKET"]
        assert article.related_pairs == ["BTC/USDT"]  # MARKET is no pair
        assert article.published_at == datetime(2024, 1, 1, tzinfo=UTC)
        assert len(article.body) == 2000

    @pytest.mark.asyncio
    async def test_one_request_for_all_pairs(self) -> None:
        import httpx

        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "Data": [
                        _coindesk_item(0, ["BTC"]),
                        _coindesk_item(1, ["ETH", "BTC"]),
                        _coindesk_item(2, ["SOL"]),
                        _coindesk_item(3, ["BTC"], published_on=1),  # too old
                    ],
                    "Err": {},
                },
            )

        fetcher = self._fetcher(handler)
        since = datetime.now(UTC) - timedelta(hours=24)
        by_pair = await fetcher.fetch_for_pairs(
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"], since=since
        )
        await fetcher.close()

        assert len(requests) == 1
        req = requests[0]
        assert req.url.path == "/news/v1/article/list"
        assert req.url.params["categories"] == "BTC,DOGE,ETH,SOL"
        assert req.url.params["limit"] == "100"
        assert req.url.params["lang"] == "EN"
        assert req.headers["Authorization"] == "Apikey k"
        assert [a.title for a in by_pair["BTC/USDT"]] == ["Headline 0", "Headline 1"]
        assert [a.title for a in by_pair["ETH/USDT"]] == ["Headline 1"]
        assert [a.title for a in by_pair["SOL/USDT"]] == ["Headline 2"]
        assert by_pair["DOGE/USDT"] == []

    @pytest.mark.asyncio
    async def test_fetch_by_pair_keeps_interface(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"Data": [_coindesk_item(0, ["ETH"])]})

        fetcher = self._fetcher(handler)
        articles = await fetcher.fetch_by_pair(
            "ETH/USDT", since=datetime.now(UTC) - timedelta(hours=1), limit=5
        )
        await fetcher.close()
        assert [a.article_id for a in articles] == ["1000"]

    @pytest.mark.asyncio
    async def test_unauthorized_is_not_retried(self) -> None:
        import httpx

        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                401, json={"Data": {}, "Err": {"message": "API key required"}}
            )

        fetcher = self._fetcher(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await fetcher.fetch_latest()
        await fetcher.close()
        assert calls == 1


class TestCryptoCompareForPairs:
    @pytest.mark.asyncio
    async def test_cryptocompare_single_request(self) -> None:
        from trading_bot.data.news.cryptocompare import CryptoCompareNews

        now = int(datetime.now(UTC).timestamp())
        fetcher = CryptoCompareNews()
        fetcher._fetch_page = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "Data": [
                    {
                        "id": "7",
                        "title": "ETH news",
                        "body": "b",
                        "published_on": now,
                        "categories": "ETH|Trading",
                    }
                ]
            }
        )
        by_pair = await fetcher.fetch_for_pairs(
            ["BTC/USDT", "ETH/USDT"], since=datetime.now(UTC) - timedelta(hours=1)
        )
        fetcher._fetch_page.assert_awaited_once_with(categories="BTC,ETH")
        assert by_pair["BTC/USDT"] == []
        assert by_pair["ETH/USDT"][0].article_id == "7"
