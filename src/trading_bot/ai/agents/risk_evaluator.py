"""Risk evaluator agent — LLM-powered risk assessment."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from trading_bot.ai.agents.base import BaseAgent
from trading_bot.ai.agents.prompts.risk import RISK_EVALUATOR_PROMPT
from trading_bot.core.models import AgentAnalysis

logger = logging.getLogger(__name__)

_VALID_RECOMMENDATIONS = {"LONG", "SHORT", "NEUTRAL"}


class RiskEvaluatorAgent(BaseAgent):
    """Evaluates portfolio risk via an LLM and returns a recommendation."""

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
        return "risk_evaluator"

    async def analyze(self, context: dict[str, object]) -> AgentAnalysis:
        prompt = self._build_prompt(context)
        t0 = time.perf_counter()
        raw = await self._call_llm_async(prompt)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        tokens_used = self._estimate_tokens(prompt, raw)
        return self._parse_response(raw, latency_ms, tokens_used)

    def _build_prompt(self, context: dict[str, object]) -> str:
        pv = context.get("portfolio_value", 0.0)
        portfolio_val = float(pv) if isinstance(pv, (int, float, str)) else 0.0
        return RISK_EVALUATOR_PROMPT.format(
            pair=context.get("pair", "UNKNOWN"),
            portfolio_value=portfolio_val,
            drawdown_pct=context.get("drawdown_pct", 0.0),
            open_positions=context.get("open_positions", 0),
            max_risk_pct=context.get("max_risk_pct", 1.0),
            max_position_pct=context.get("max_position_pct", 10.0),
            max_drawdown_pct=context.get("max_drawdown_pct", 15.0),
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
            logger.warning("Failed to parse risk agent response: %s", raw[:200])
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
