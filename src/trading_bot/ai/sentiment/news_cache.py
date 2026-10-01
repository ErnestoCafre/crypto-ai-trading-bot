"""Per-article sentiment cache: each news article is analysed once.

The sentiment job runs every few minutes over a 24 h news window, so without
a cache the same article would be re-scored on every cycle. Entries expire
``ttl`` after the article was published; a pair's aggregate uses every entry
still alive for it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from trading_bot.core.models import NewsArticle, SentimentResult


def article_key(article: NewsArticle) -> str:
    """Provider id, or a hash of title + body when the source has none."""
    if article.article_id:
        return f"id:{article.article_id}"
    digest = hashlib.sha256(f"{article.title}\n{article.body}".encode()).hexdigest()
    return f"sha256:{digest}"


@dataclass
class _Entry:
    result: SentimentResult
    title: str
    published_at: datetime
    pairs: set[str] = field(default_factory=set)


class NewsSentimentCache:
    """Sentiment results keyed by article, with a time-to-live."""

    def __init__(self, ttl: timedelta = timedelta(hours=24)) -> None:
        self._ttl = ttl
        self._entries: dict[str, _Entry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def tag(self, articles_by_pair: dict[str, list[NewsArticle]]) -> None:
        """Attach pairs to already-cached articles (an article can gain pairs)."""
        for pair, articles in articles_by_pair.items():
            for article in articles:
                entry = self._entries.get(article_key(article))
                if entry is not None:
                    entry.pairs.add(pair)

    def missing(
        self, articles_by_pair: dict[str, list[NewsArticle]]
    ) -> dict[str, tuple[NewsArticle, set[str]]]:
        """Uncached articles (deduplicated across pairs) with their pairs."""
        pending: dict[str, tuple[NewsArticle, set[str]]] = {}
        for pair, articles in articles_by_pair.items():
            for article in articles:
                key = article_key(article)
                if key in self._entries:
                    continue
                pending.setdefault(key, (article, set()))[1].add(pair)
        return pending

    def store(
        self,
        key: str,
        article: NewsArticle,
        result: SentimentResult,
        pairs: set[str],
    ) -> None:
        self._entries[key] = _Entry(
            result=result,
            title=article.title,
            published_at=article.published_at,
            pairs=set(pairs),
        )

    def purge(self, now: datetime | None = None) -> int:
        """Drop expired entries; returns how many were removed."""
        cutoff = (now or datetime.now(UTC)) - self._ttl
        expired = [k for k, e in self._entries.items() if e.published_at < cutoff]
        for key in expired:
            del self._entries[key]
        return len(expired)

    def _live(self, pair: str, now: datetime | None) -> list[_Entry]:
        cutoff = (now or datetime.now(UTC)) - self._ttl
        entries = [
            e
            for e in self._entries.values()
            if pair in e.pairs and e.published_at >= cutoff
        ]
        return sorted(entries, key=lambda e: e.published_at, reverse=True)

    def results_for(
        self, pair: str, now: datetime | None = None
    ) -> list[SentimentResult]:
        return [e.result for e in self._live(pair, now)]

    def headlines_for(
        self, pair: str, limit: int = 10, now: datetime | None = None
    ) -> list[str]:
        """Most recent live headlines for a pair."""
        return [e.title for e in self._live(pair, now) if e.title][:limit]
