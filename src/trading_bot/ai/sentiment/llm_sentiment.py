"""LLM-based sentiment analyzer — Claude/OpenAI API for nuanced analysis."""

from __future__ import annotations

import json
import logging
from typing import Any

from trading_bot.ai.sentiment.base import BaseSentimentAnalyzer
from trading_bot.core.models import SentimentResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a crypto market sentiment analyzer. "
    "Analyze the following text and return a JSON object with:\n"
    '- "score": float from -1.0 (very bearish) to 1.0 (very bullish)\n'
    '- "confidence": float from 0.0 to 1.0\n'
    '- "label": one of "bullish", "bearish", or "neutral"\n'
    "Return ONLY valid JSON, no other text."
)


class LLMSentimentAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analysis using LLM APIs (Anthropic or OpenAI).

    Not wired into TradingOrchestrator: its calls are outside the daily LLM
    budget (SentimentCostTracker). Route them through it before enabling.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5",
        api_key: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single text for sentiment using an LLM."""
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label="neutral",
                model_used=f"{self._provider}/{self._model}",
                analyzed_text=text,
            )

        try:
            response = self._call_llm(text)
            return self._parse_response(response, text)
        except Exception:
            logger.exception("LLM sentiment analysis failed")
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label="neutral",
                model_used=f"{self._provider}/{self._model}",
                analyzed_text=text[:200],
            )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze multiple texts sequentially."""
        return [self.analyze(t) for t in texts]

    def is_loaded(self) -> bool:
        """Return True — LLM is always available if API key is set."""
        return bool(self._api_key)

    def _call_llm(self, text: str) -> str:
        """Call the LLM API and return the raw response text."""
        if self._provider == "anthropic":
            return self._call_anthropic(text)
        elif self._provider == "openai":
            return self._call_openai(text)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    def _call_anthropic(self, text: str) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text[:2000]}],
        )
        return response.content[0].text  # type: ignore[no-any-return]

    def _call_openai(self, text: str) -> str:
        import openai

        if self._client is None:
            self._client = openai.OpenAI(api_key=self._api_key)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text[:2000]},
            ],
            max_tokens=256,
        )
        return response.choices[0].message.content  # type: ignore[no-any-return]

    def _parse_response(self, response: str, original_text: str) -> SentimentResult:
        """Parse JSON from LLM response."""
        try:
            data = json.loads(response)
            score = float(data.get("score", 0.0))
            confidence = float(data.get("confidence", 0.0))
            label = str(data.get("label", "neutral"))
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("Failed to parse LLM response: %s", response[:200])
            score = 0.0
            confidence = 0.0
            label = "neutral"

        return SentimentResult(
            score=max(-1.0, min(1.0, score)),
            confidence=max(0.0, min(1.0, confidence)),
            label=label,
            model_used=f"{self._provider}/{self._model}",
            analyzed_text=original_text[:200],
        )
