"""Tests for LLMSentimentAnalyzer (mocked API calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from trading_bot.ai.sentiment.llm_sentiment import LLMSentimentAnalyzer
from trading_bot.core.models import SentimentResult


def _make_analyzer_with_mock(response_json: dict) -> LLMSentimentAnalyzer:
    """Create an analyzer with a mocked _call_llm method."""
    a = LLMSentimentAnalyzer(provider="anthropic", api_key="test-key")
    a._call_llm = MagicMock(return_value=json.dumps(response_json))  # type: ignore[method-assign]
    return a


class TestAnalyze:
    def test_returns_sentiment_result(self) -> None:
        a = _make_analyzer_with_mock(
            {"score": 0.7, "confidence": 0.85, "label": "bullish"}
        )
        result = a.analyze("Bitcoin is surging")
        assert isinstance(result, SentimentResult)
        assert result.score == pytest.approx(0.7)
        assert result.confidence == pytest.approx(0.85)
        assert result.label == "bullish"

    def test_prompt_sent_to_llm(self) -> None:
        a = _make_analyzer_with_mock(
            {"score": 0.5, "confidence": 0.5, "label": "neutral"}
        )
        a.analyze("Test text")
        a._call_llm.assert_called_once_with("Test text")  # type: ignore[union-attr]

    def test_malformed_response_handled(self) -> None:
        a = LLMSentimentAnalyzer(provider="anthropic", api_key="test-key")
        a._call_llm = MagicMock(return_value="not valid json")  # type: ignore[method-assign]
        result = a.analyze("Some text")
        # Should return neutral defaults, not raise
        assert result.score == 0.0
        assert result.label == "neutral"

    def test_empty_text_returns_neutral(self) -> None:
        a = _make_analyzer_with_mock({"score": 0, "confidence": 0, "label": "neutral"})
        result = a.analyze("")
        assert result.score == 0.0
        assert result.label == "neutral"
