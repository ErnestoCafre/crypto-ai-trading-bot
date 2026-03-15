"""CryptoBERT sentiment analyzer stub."""

from __future__ import annotations

from ai.sentiment.base import BaseSentimentAnalyzer
from core.models import SentimentResult


class CryptoBERTAnalyzer(BaseSentimentAnalyzer):
    def analyze(self, text: str) -> SentimentResult:
        raise NotImplementedError

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError
