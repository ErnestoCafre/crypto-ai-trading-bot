"""Budgeted, asynchronous LLM news sentiment (SENTIMENT_PROVIDER=llm).

Adds to :class:`LLMSentimentAnalyzer` (same prompt and parsing) what the bot
needs to run it inside its event loop:

* async provider clients with a per-call timeout and bounded retries;
* the daily LLM budget: a worst-case pre-check before every call and the
  real cost (from the response's token usage) recorded afterwards, in the
  same :class:`SentimentCostTracker` the LLM agents use;
* failures return None instead of a neutral score, so the article stays out
  of the per-article cache and is retried next cycle.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from trading_bot.ai.pricing import estimate_cost
from trading_bot.ai.sentiment.cost_tracker import SentimentCostTracker
from trading_bot.ai.sentiment.llm_sentiment import _SYSTEM_PROMPT, LLMSentimentAnalyzer
from trading_bot.core.models import SentimentResult

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 256
MAX_TEXT_CHARS = 2000
# No tokenizer before the call: 3 chars/token overstates English text, and
# the overhead covers message framing — the safe side for a budget.
_WORST_CASE_CHARS_PER_TOKEN = 3
_REQUEST_OVERHEAD_TOKENS = 100


@dataclass(frozen=True)
class _Completion:
    text: str
    input_tokens: int
    output_tokens: int


class AsyncLLMSentimentAnalyzer(LLMSentimentAnalyzer):
    """LLM sentiment that respects the shared daily LLM budget."""

    def __init__(
        self,
        cost_tracker: SentimentCostTracker,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5",
        api_key: str = "",
        timeout_s: float = 15.0,
        max_retries: int = 1,
        concurrency: int = 4,
    ) -> None:
        super().__init__(provider=provider, model=model, api_key=api_key)
        self._tracker = cost_tracker
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._concurrency = max(1, concurrency)
        self._async_client: Any = None
        # Serialises budget check + spend so concurrent calls cannot all pass
        # the pre-check against the same remaining budget.
        self._budget_lock = asyncio.Lock()
        self._reserved = 0.0

    @property
    def model_used(self) -> str:
        return f"{self._provider}/{self._model}"

    def estimate_max_cost_usd(self, text: str) -> float:
        chars = len(_SYSTEM_PROMPT) + min(len(text), MAX_TEXT_CHARS)
        input_tokens = -(-chars // _WORST_CASE_CHARS_PER_TOKEN)
        return estimate_cost(
            self._model, input_tokens + _REQUEST_OVERHEAD_TOKENS, MAX_OUTPUT_TOKENS
        )

    async def analyze_batch_async(
        self, texts: list[str]
    ) -> list[SentimentResult | None]:
        """Score ``texts``; None where the call failed or the budget ran out."""
        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(text: str) -> SentimentResult | None:
            async with semaphore:
                return await self.analyze_async(text)

        return list(await asyncio.gather(*(one(t) for t in texts)))

    async def analyze_async(self, text: str) -> SentimentResult | None:
        if not text or not text.strip():
            return None
        worst_case = self.estimate_max_cost_usd(text)
        async with self._budget_lock:
            if not self._tracker.can_spend(self._reserved + worst_case):
                self._tracker.mark_exhausted()
                logger.warning("LLM sentiment skipped: daily LLM budget exhausted")
                return None
            self._reserved += worst_case
        completion: _Completion | None = None
        try:
            completion = await self._complete(text[:MAX_TEXT_CHARS])
        except Exception:
            logger.exception("LLM sentiment call failed")
        finally:
            async with self._budget_lock:
                self._reserved -= worst_case
                if completion is not None:
                    cost = estimate_cost(
                        self._model, completion.input_tokens, completion.output_tokens
                    )
                    self._tracker.record_cost(cost)
                else:
                    # Unknown usage (error or timeout after sending): assume
                    # the request was billed at its worst case.
                    self._tracker.record_cost(worst_case)
        if completion is None:
            return None
        result = self._parse_response(completion.text, text)
        return result if result.confidence > 0 else None

    async def _complete(self, text: str) -> _Completion:
        if self._provider == "anthropic":
            return await self._complete_anthropic(text)
        if self._provider == "openai":
            return await self._complete_openai(text)
        raise ValueError(f"Unknown provider: {self._provider}")

    async def _complete_anthropic(self, text: str) -> _Completion:
        import anthropic

        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self._timeout_s,
                max_retries=self._max_retries,
            )
        response = await self._async_client.messages.create(
            model=self._model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        # A leading thinking block has no .text: join the text blocks only.
        out = "".join(b.text for b in response.content if b.type == "text")
        usage = response.usage
        input_tokens = (
            int(usage.input_tokens)
            + int(getattr(usage, "cache_creation_input_tokens", None) or 0)
            + int(getattr(usage, "cache_read_input_tokens", None) or 0)
        )
        return _Completion(out, input_tokens, int(usage.output_tokens))

    async def _complete_openai(self, text: str) -> _Completion:
        import openai

        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout_s,
                max_retries=self._max_retries,
            )
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        out = response.choices[0].message.content or ""
        usage = response.usage
        if usage is None:
            input_tokens = -(-(len(_SYSTEM_PROMPT) + len(text)) // 3)
            return _Completion(out, input_tokens, MAX_OUTPUT_TOKENS)
        return _Completion(out, int(usage.prompt_tokens), int(usage.completion_tokens))
