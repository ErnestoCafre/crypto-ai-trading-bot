"""Sentiment analyst agent — LLM-powered sentiment analysis."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from ai.agents.base import BaseAgent
from ai.agents.prompts.sentiment import SENTIMENT_ANALYST_PROMPT
from core.models import AgentAnalysis

logger = logging.getLogger(__name__)

_VALID_RECOMMENDATIONS = {"LONG", "SHORT", "NEUTRAL"}


class SentimentAnalystAgent(BaseAgent):
    """Analyses market sentiment via an LLM and returns a recommendation."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        api_key: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    @property
    def name(self) -> str:
        return "sentiment_analyst"

    async def analyze(self, context: dict[str, object]) -> AgentAnalysis:
        prompt = self._build_prompt(context)
        t0 = time.perf_counter()
        raw = await self._call_llm_async(prompt)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        tokens_used = self._estimate_tokens(prompt, raw)
        return self._parse_response(raw, latency_ms, tokens_used)

    def _build_prompt(self, context: dict[str, object]) -> str:
        headlines_raw = context.get("headlines", [])
        if isinstance(headlines_raw, list) and headlines_raw:
            headlines_str = "\n".join(
                f"  {i + 1}. {h}" for i, h in enumerate(headlines_raw)
            )
        else:
            headlines_str = "  No recent headlines available."

        return SENTIMENT_ANALYST_PROMPT.format(
            pair=context.get("pair", "UNKNOWN"),
            sentiment_score=context.get("sentiment_score", 0.0),
            fear_greed=context.get("fear_greed", "N/A"),
            headlines=headlines_str,
        )

    async def _call_llm_async(self, prompt: str) -> str:
        if self._provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self._provider == "openai":
            return await self._call_openai(prompt)
        raise ValueError(f"Unknown provider: {self._provider}")

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt[:4000]}],
        )
        return response.content[0].text  # type: ignore[no-any-return]

    async def _call_openai(self, prompt: str) -> str:
        import openai

        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt[:4000]}],
            max_tokens=512,
        )
        return response.choices[0].message.content  # type: ignore[no-any-return]

    def _parse_response(
        self, raw: str, latency_ms: int, tokens_used: int
    ) -> AgentAnalysis:
        try:
            cleaned = re.sub(r"```json\s*", "", raw)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
            data = json.loads(cleaned)

            recommendation = str(data.get("recommendation", "NEUTRAL")).upper()
            if recommendation not in _VALID_RECOMMENDATIONS:
                recommendation = "NEUTRAL"

            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            reasoning = str(data.get("reasoning", ""))
            key_factors = list(data.get("key_factors", []))
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            logger.warning("Failed to parse sentiment agent response: %s", raw[:200])
            recommendation = "NEUTRAL"
            confidence = 0.0
            reasoning = "Parse error"
            key_factors = []

        return AgentAnalysis(
            agent_name=self.name,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            raw_response=raw,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _estimate_tokens(prompt: str, response: str) -> int:
        return (len(prompt) + len(response)) // 4
