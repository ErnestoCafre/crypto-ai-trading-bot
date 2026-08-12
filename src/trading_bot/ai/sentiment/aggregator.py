"""Sentiment aggregator — weighted combination of multiple sentiment sources."""

from __future__ import annotations

from datetime import UTC, datetime

from trading_bot.core.models import AggregatedSentiment, SentimentResult


class SentimentAggregator:
    """Combine sentiment results from multiple sources using weighted averaging.

    Classifies results by ``model_used`` field into source categories:
    - news: models containing "bert", "news", "crypto"
    - social: models containing "reddit", "twitter", "social"
    - fear_greed: models containing "fear", "greed"
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "news": 0.4,
        "social": 0.3,
        "fear_greed": 0.3,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or dict(self.DEFAULT_WEIGHTS)

    def aggregate(
        self,
        results: list[SentimentResult],
        pair: str = "BTC/USDT",
        time_window: str = "1h",
    ) -> AggregatedSentiment:
        """Aggregate multiple sentiment results into a single score."""
        if not results:
            return AggregatedSentiment(
                pair=pair,
                overall_score=0.0,
                news_score=0.0,
                social_score=0.0,
                fear_greed_score=0.0,
                num_sources_analyzed=0,
                time_window=time_window,
                timestamp=datetime.now(UTC),
            )

        # Classify results by source
        news_scores: list[float] = []
        social_scores: list[float] = []
        fg_scores: list[float] = []

        for r in results:
            source = self._classify_source(r.model_used)
            if source == "news":
                news_scores.append(r.score)
            elif source == "social":
                social_scores.append(r.score)
            elif source == "fear_greed":
                fg_scores.append(r.score)
            else:
                news_scores.append(r.score)  # default to news

        news_avg = _safe_mean(news_scores)
        social_avg = _safe_mean(social_scores)
        fg_avg = _safe_mean(fg_scores)

        # Weighted overall score — only use sources that have data
        weighted_sum = 0.0
        total_weight = 0.0
        for source, avg_score in [
            ("news", news_avg),
            ("social", social_avg),
            ("fear_greed", fg_avg),
        ]:
            scores_map = {
                "news": news_scores,
                "social": social_scores,
                "fear_greed": fg_scores,
            }
            if scores_map.get(source):
                w = self._weights.get(source, 0.0)
                weighted_sum += avg_score * w
                total_weight += w

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        return AggregatedSentiment(
            pair=pair,
            overall_score=overall,
            news_score=news_avg,
            social_score=social_avg,
            fear_greed_score=fg_avg,
            num_sources_analyzed=len(results),
            time_window=time_window,
            timestamp=datetime.now(UTC),
        )

    @staticmethod
    def _classify_source(model_used: str) -> str:
        """Classify a model name into a source category."""
        lower = model_used.lower()
        if any(kw in lower for kw in ("reddit", "twitter", "social")):
            return "social"
        if any(kw in lower for kw in ("fear", "greed")):
            return "fear_greed"
        # Default: news (covers "bert", "crypto", "news", LLM models, etc.)
        return "news"


def _safe_mean(values: list[float]) -> float:
    """Return mean of values, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0
