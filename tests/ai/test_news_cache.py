"""Tests for the per-article news sentiment cache."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading_bot.ai.sentiment.news_cache import NewsSentimentCache, article_key
from trading_bot.core.models import NewsArticle, SentimentResult

NOW = datetime(2026, 9, 30, 12, tzinfo=UTC)


def _article(title: str, age_h: float = 1.0, article_id: str = "") -> NewsArticle:
    return NewsArticle(
        title=title,
        body=f"{title} body",
        source="s",
        url="u",
        published_at=NOW - timedelta(hours=age_h),
        article_id=article_id,
    )


def _result(score: float) -> SentimentResult:
    return SentimentResult(
        score=score, confidence=0.9, label="x", model_used="m", analyzed_text=""
    )


def _fill(cache: NewsSentimentCache, by_pair: dict[str, list[NewsArticle]]) -> int:
    pending = cache.missing(by_pair)
    for key, (article, pairs) in pending.items():
        cache.store(key, article, _result(0.5), pairs)
    return len(pending)


def test_key_uses_provider_id_then_content_hash() -> None:
    assert article_key(_article("a", article_id="42")) == "id:42"
    assert article_key(_article("a")) == article_key(_article("a", age_h=5))
    assert article_key(_article("a")) != article_key(_article("b"))


def test_each_article_scored_once_across_cycles_and_pairs() -> None:
    cache = NewsSentimentCache()
    shared = _article("shared", article_id="1")
    by_pair = {"BTC/USDT": [shared], "ETH/USDT": [shared, _article("eth")]}

    assert _fill(cache, by_pair) == 2  # shared article deduplicated
    assert _fill(cache, by_pair) == 0  # next cycle: nothing new
    assert len(cache.results_for("BTC/USDT", NOW)) == 1
    assert len(cache.results_for("ETH/USDT", NOW)) == 2


def test_aggregate_uses_every_live_result_not_only_this_fetch() -> None:
    cache = NewsSentimentCache()
    _fill(cache, {"BTC/USDT": [_article("old", age_h=10)]})
    _fill(cache, {"BTC/USDT": [_article("new", age_h=0.1)]})
    assert cache.headlines_for("BTC/USDT", now=NOW) == ["new", "old"]


def test_entries_expire_24h_after_publication() -> None:
    cache = NewsSentimentCache(ttl=timedelta(hours=24))
    _fill(cache, {"BTC/USDT": [_article("a", age_h=23), _article("b", age_h=2)]})

    later = NOW + timedelta(hours=2)
    assert cache.headlines_for("BTC/USDT", now=later) == ["b"]
    assert cache.purge(later) == 1
    assert len(cache) == 1


def test_tag_adds_pairs_to_cached_articles() -> None:
    cache = NewsSentimentCache()
    art = _article("x", article_id="9")
    _fill(cache, {"BTC/USDT": [art]})
    cache.tag({"SOL/USDT": [art]})
    assert cache.missing({"SOL/USDT": [art]}) == {}
    assert len(cache.results_for("SOL/USDT", NOW)) == 1
