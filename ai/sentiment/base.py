"""Abstract base class for sentiment analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import SentimentResult


class BaseSentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> SentimentResult: ...

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...
