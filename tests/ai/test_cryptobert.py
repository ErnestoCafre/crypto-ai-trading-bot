"""Tests for CryptoBERTAnalyzer (mocked pipeline — never downloads model in CI)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai.sentiment.cryptobert import CryptoBERTAnalyzer
from core.models import SentimentResult


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    """Return a mock transformers pipeline."""
    pipe = MagicMock()
    pipe.return_value = [{"label": "Positive", "score": 0.85}]
    return pipe


@pytest.fixture()
def analyzer(mock_pipeline: MagicMock) -> CryptoBERTAnalyzer:
    a = CryptoBERTAnalyzer()
    a._pipeline = mock_pipeline
    return a


class TestAnalyze:
    def test_returns_sentiment_result(self, analyzer: CryptoBERTAnalyzer) -> None:
        result = analyzer.analyze("Bitcoin is going to the moon!")
        assert isinstance(result, SentimentResult)

    def test_score_in_range(self, analyzer: CryptoBERTAnalyzer) -> None:
        result = analyzer.analyze("BTC breaks new ATH")
        assert -1.0 <= result.score <= 1.0

    def test_confidence_in_range(self, analyzer: CryptoBERTAnalyzer) -> None:
        result = analyzer.analyze("BTC breaks new ATH")
        assert 0.0 <= result.confidence <= 1.0

    def test_positive_label_gives_positive_score(
        self, analyzer: CryptoBERTAnalyzer
    ) -> None:
        result = analyzer.analyze("Crypto market surges")
        assert result.score > 0

    def test_negative_label(self) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "Negative", "score": 0.9}]
        a = CryptoBERTAnalyzer()
        a._pipeline = pipe
        result = a.analyze("Bitcoin crashes hard")
        assert result.score < 0

    def test_empty_text_handled(self, analyzer: CryptoBERTAnalyzer) -> None:
        result = analyzer.analyze("")
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.label == "neutral"


class TestBatch:
    def test_batch_count(self, analyzer: CryptoBERTAnalyzer) -> None:
        texts = ["text 1", "text 2", "text 3"]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SentimentResult)


class TestIsLoaded:
    def test_not_loaded_initially(self) -> None:
        a = CryptoBERTAnalyzer()
        assert not a.is_loaded()

    def test_loaded_after_pipeline_set(self, analyzer: CryptoBERTAnalyzer) -> None:
        assert analyzer.is_loaded()
