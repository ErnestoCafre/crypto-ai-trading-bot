"""Base class for LLM agents: provider calls, token accounting and parsing.

Subclasses only define ``name`` and ``_build_prompt``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from trading_bot.ai.pricing import estimate_cost
from trading_bot.core.models import AgentAnalysis

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 512
MAX_PROMPT_CHARS = 4000
# Pre-call budgeting has no tokenizer: English averages ~4 chars/token, so
# 3 chars/token overstates the prompt, which is the safe side for a budget.
_WORST_CASE_CHARS_PER_TOKEN = 3
# Input the prompt text does not show (message framing, the structured
# output schema): a margin for the worst case, not a measured figure.
_REQUEST_OVERHEAD_TOKENS = 500

_VALID_RECOMMENDATIONS = {"LONG", "SHORT", "NEUTRAL"}

# Structured output schema (Anthropic output_config.format). Numeric bounds
# are not supported by structured outputs; confidence is clamped on parse.
AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendation": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["recommendation", "confidence", "reasoning", "key_factors"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LLMCompletion:
    text: str
    input_tokens: int
    output_tokens: int


class BaseAgent(ABC):
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = DEFAULT_LLM_MODEL,
        api_key: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._client: Any = None
        # Cleared if the model rejects output_config (structured outputs).
        self._structured_output = True

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _build_prompt(self, context: dict[str, object]) -> str: ...

    async def analyze(self, context: dict[str, object]) -> AgentAnalysis:
        prompt = self._build_prompt(context)
        if len(prompt) > MAX_PROMPT_CHARS:
            logger.warning(
                "%s: prompt truncated from %d to %d chars",
                self.name,
                len(prompt),
                MAX_PROMPT_CHARS,
            )
            prompt = prompt[:MAX_PROMPT_CHARS]

        t0 = time.perf_counter()
        completion = await self._call_llm_async(prompt)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        cost = estimate_cost(
            self._model, completion.input_tokens, completion.output_tokens
        )
        logger.info(
            "%s: %s %d in + %d out tokens = $%.6f",
            self.name,
            self._model,
            completion.input_tokens,
            completion.output_tokens,
            cost,
        )
        return self._parse_response(completion, latency_ms, cost)

    def estimate_max_cost_usd(self, context: dict[str, object]) -> float:
        """Worst-case cost of one call: prompt estimate + full output budget."""
        chars = min(len(self._build_prompt(context)), MAX_PROMPT_CHARS)
        input_tokens = -(-chars // _WORST_CASE_CHARS_PER_TOKEN)
        input_tokens += _REQUEST_OVERHEAD_TOKENS
        return estimate_cost(self._model, input_tokens, MAX_OUTPUT_TOKENS)

    async def _call_llm_async(self, prompt: str) -> LLMCompletion:
        if self._provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self._provider == "openai":
            return await self._call_openai(prompt)
        raise ValueError(f"Unknown provider: {self._provider}")

    async def _call_anthropic(self, prompt: str) -> LLMCompletion:
        import anthropic

        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._structured_output:
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": AGENT_OUTPUT_SCHEMA}
            }
        try:
            response = await self._client.messages.create(**request)
        except anthropic.BadRequestError:
            if "output_config" not in request:
                raise
            # Not every model supports structured outputs: retry with the
            # prompt's JSON instructions. Only a successful retry proves the
            # 400 came from output_config; otherwise the error propagates and
            # structured outputs stay on.
            del request["output_config"]
            response = await self._client.messages.create(**request)
            logger.warning(
                "%s: %s rejected structured outputs — using prompt-only JSON",
                self.name,
                self._model,
            )
            self._structured_output = False
        if response.stop_reason == "refusal":
            logger.warning("%s: model refused the request", self.name)
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        # Cache reads/writes are billed as input too; counting them at the
        # full input rate can only overstate spend.
        input_tokens = (
            int(usage.input_tokens)
            + int(getattr(usage, "cache_creation_input_tokens", None) or 0)
            + int(getattr(usage, "cache_read_input_tokens", None) or 0)
        )
        return LLMCompletion(text, input_tokens, int(usage.output_tokens))

    async def _call_openai(self, prompt: str) -> LLMCompletion:
        import openai

        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        if usage is None:
            logger.warning(
                "%s: OpenAI response without usage — charging the worst case",
                self.name,
            )
            input_tokens = -(-len(prompt) // _WORST_CASE_CHARS_PER_TOKEN)
            return LLMCompletion(text, input_tokens, MAX_OUTPUT_TOKENS)
        return LLMCompletion(
            text, int(usage.prompt_tokens), int(usage.completion_tokens)
        )

    def _parse_response(
        self, completion: LLMCompletion, latency_ms: int, cost_usd: float
    ) -> AgentAnalysis:
        raw = completion.text
        try:
            # Structured outputs return bare JSON; other providers may wrap it.
            cleaned = re.sub(r"```json\s*", "", raw)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
            data = json.loads(cleaned)

            recommendation = str(data.get("recommendation", "NEUTRAL")).upper()
            if recommendation not in _VALID_RECOMMENDATIONS:
                recommendation = "NEUTRAL"

            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            reasoning = str(data.get("reasoning", ""))
            key_factors = [str(f) for f in data.get("key_factors", [])]
        except (json.JSONDecodeError, TypeError, KeyError, ValueError, AttributeError):
            logger.warning("Failed to parse %s response: %s", self.name, raw[:200])
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
            tokens_used=completion.input_tokens + completion.output_tokens,
            latency_ms=latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=cost_usd,
        )
