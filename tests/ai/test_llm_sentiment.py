"""Tests for LLMSentimentAnalyzer (mocked API calls)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx2
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


class TestAnthropicClient:
    """The sync anthropic 1.x client end to end, over a mocked httpx2 transport."""

    @staticmethod
    def _analyzer(handler: Any) -> LLMSentimentAnalyzer:
        a = LLMSentimentAnalyzer(provider="anthropic", api_key="test-key")
        a._client = anthropic.Anthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(
                transport=httpx2.MockTransport(handler)
            ),
        )
        return a

    @staticmethod
    def _message(content: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": content,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }

    def test_request_and_parsed_result(self) -> None:
        requests: list[httpx2.Request] = []
        verdict = {"score": 0.6, "confidence": 0.9, "label": "bullish"}

        def handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            body = self._message([{"type": "text", "text": json.dumps(verdict)}])
            return httpx2.Response(200, json=body)

        result = self._analyzer(handler).analyze("x" * 3000)

        assert result.label == "bullish"
        assert result.score == pytest.approx(0.6)
        assert result.model_used == "anthropic/claude-haiku-4-5"
        (request,) = requests
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        sent = json.loads(request.content)
        assert sent["model"] == "claude-haiku-4-5"
        assert sent["max_tokens"] == 256
        assert sent["system"].startswith("You are a crypto market sentiment")
        assert sent["messages"] == [{"role": "user", "content": "x" * 2000}]

    def test_leading_thinking_block_is_skipped(self) -> None:
        verdict = {"score": -0.4, "confidence": 0.7, "label": "bearish"}

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = self._message(
                [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": json.dumps(verdict)},
                ]
            )
            return httpx2.Response(200, json=body)

        result = self._analyzer(handler).analyze("ETH dumps")

        assert result.label == "bearish"
        assert result.score == pytest.approx(-0.4)

    def test_api_error_returns_neutral(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            error = {"type": "overloaded_error", "message": "Overloaded"}
            return httpx2.Response(529, json={"type": "error", "error": error})

        result = self._analyzer(handler).analyze("BTC news")

        assert result.label == "neutral"
        assert result.confidence == 0.0
