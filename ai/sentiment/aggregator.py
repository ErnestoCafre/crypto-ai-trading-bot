"""Sentiment aggregator stub."""

from __future__ import annotations

from core.models import AggregatedSentiment, SentimentResult


class SentimentAggregator:
    def aggregate(self, results: list[SentimentResult]) -> AggregatedSentiment:
        raise NotImplementedError
