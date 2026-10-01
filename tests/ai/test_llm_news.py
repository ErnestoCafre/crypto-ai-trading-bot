"""Tests for the budgeted asynchronous LLM news sentiment analyzer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from trading_bot.ai.pricing import estimate_cost
from trading_bot.ai.sentiment.cost_tracker import EXHAUSTED, SentimentCostTracker
from trading_bot.ai.sentiment.llm_news import AsyncLLMSentimentAnalyzer


def _response(text: str, input_tokens: int = 120, output_tokens: int = 20):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=0,
        ),
    )


def _analyzer(tracker: SentimentCostTracker, create: AsyncMock):
    analyzer = AsyncLLMSentimentAnalyzer(
        tracker, provider="anthropic", model="claude-haiku-4-5", api_key="k"
    )
    analyzer._async_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return analyzer


async def test_scores_and_records_real_cost() -> None:
    tracker = SentimentCostTracker(5.0)
    create = AsyncMock(
        return_value=_response('{"score": 0.6, "confidence": 0.8, "label": "bullish"}')
    )
    result = await _analyzer(tracker, create).analyze_async("BTC ETF approved")

    assert result is not None
    assert result.score == pytest.approx(0.6)
    assert result.model_used == "anthropic/claude-haiku-4-5"
    assert tracker.daily_spent_usd == pytest.approx(
        estimate_cost("claude-haiku-4-5", 120, 20)
    )
    assert create.await_args.kwargs["max_tokens"] == 256


async def test_budget_exhausted_skips_the_call() -> None:
    tracker = SentimentCostTracker(0.0)
    create = AsyncMock()
    results = await _analyzer(tracker, create).analyze_batch_async(["a", "b"])

    assert results == [None, None]
    create.assert_not_awaited()
    assert EXHAUSTED in tracker.pop_alerts()


async def test_concurrent_calls_never_exceed_the_budget() -> None:
    import asyncio

    probe = _analyzer(SentimentCostTracker(1.0), AsyncMock())
    worst = probe.estimate_max_cost_usd("x" * 100)
    tracker = SentimentCostTracker(worst * 2.5)  # two worst cases fit
    in_flight = peak = 0

    async def slow_create(**_: object):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _response('{"score": 0.1, "confidence": 0.5, "label": "neutral"}')

    analyzer = _analyzer(tracker, AsyncMock(side_effect=slow_create))
    results = await analyzer.analyze_batch_async(["x" * 100] * 4)

    assert peak == 2  # the other two were refused by the pre-check
    assert sum(r is not None for r in results) == 2
    assert tracker.daily_spent_usd <= tracker.max_daily_usd


async def test_api_error_returns_none_and_charges_worst_case() -> None:
    tracker = SentimentCostTracker(5.0)
    analyzer = _analyzer(tracker, AsyncMock(side_effect=RuntimeError("503")))
    assert await analyzer.analyze_async("text") is None
    assert tracker.daily_spent_usd == pytest.approx(
        analyzer.estimate_max_cost_usd("text")
    )


async def test_unparseable_answer_is_not_a_score() -> None:
    tracker = SentimentCostTracker(5.0)
    analyzer = _analyzer(tracker, AsyncMock(return_value=_response("no json")))
    assert await analyzer.analyze_async("text") is None
