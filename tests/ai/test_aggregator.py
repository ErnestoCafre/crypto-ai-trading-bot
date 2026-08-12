"""Tests for SentimentAggregator."""

from __future__ import annotations

import pytest

from trading_bot.ai.sentiment.aggregator import SentimentAggregator
from trading_bot.core.models import AggregatedSentiment, SentimentResult


def _result(score: float, model: str) -> SentimentResult:
    return SentimentResult(
        score=score,
        confidence=0.8,
        label="positive" if score > 0 else "negative",
        model_used=model,
        analyzed_text="test",
    )


class TestEmptyResults:
    def test_empty_gives_neutral(self) -> None:
        agg = SentimentAggregator()
        result = agg.aggregate([])
        assert isinstance(result, AggregatedSentiment)
        assert result.overall_score == 0.0
        assert result.num_sources_analyzed == 0


class TestSingleSource:
    def test_single_news_source(self) -> None:
        agg = SentimentAggregator()
        result = agg.aggregate([_result(0.8, "CryptoBERT")])
        assert result.overall_score == pytest.approx(0.8, abs=0.01)
        assert result.news_score == pytest.approx(0.8, abs=0.01)
        assert result.num_sources_analyzed == 1


class TestMultipleSources:
    def test_weighted_combination(self) -> None:
        agg = SentimentAggregator()
        results = [
            _result(0.8, "CryptoBERT"),  # news
            _result(0.6, "reddit_sentiment"),  # social
            _result(0.4, "fear_greed_index"),  # fear_greed
        ]
        output = agg.aggregate(results)
        assert output.num_sources_analyzed == 3
        assert output.news_score == pytest.approx(0.8, abs=0.01)
        assert output.social_score == pytest.approx(0.6, abs=0.01)
        assert output.fear_greed_score == pytest.approx(0.4, abs=0.01)

        # Weighted: 0.8*0.4 + 0.6*0.3 + 0.4*0.3 = 0.32 + 0.18 + 0.12 = 0.62
        assert output.overall_score == pytest.approx(0.62, abs=0.01)

    def test_num_sources_correct(self) -> None:
        agg = SentimentAggregator()
        results = [_result(0.5, "news"), _result(-0.3, "reddit")]
        output = agg.aggregate(results)
        assert output.num_sources_analyzed == 2
