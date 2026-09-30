"""Orchestrator — parallel LLM agent execution with consensus."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import UTC, datetime

from trading_bot.ai.agents.base import DEFAULT_LLM_MODEL, BaseAgent
from trading_bot.ai.sentiment.cost_tracker import SentimentCostTracker
from trading_bot.core.models import AgentAnalysis, OrchestratorDecision

logger = logging.getLogger(__name__)

_AGENT_TIMEOUT_S = 15.0
_TIMED_OUT = object()


class Orchestrator:
    """Run multiple LLM agents in parallel and apply consensus voting."""

    def __init__(
        self,
        agents: list[BaseAgent],
        cost_tracker: SentimentCostTracker,
        provider: str = "anthropic",
        model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self._agents = agents
        self._cost_tracker = cost_tracker
        self._provider = provider
        self._model = model

    async def run(
        self, pair: str, context: dict[str, object]
    ) -> OrchestratorDecision | None:
        """Execute all agents, apply consensus, return decision or None."""
        # Budget check against the worst case: every agent uses its whole
        # prompt and output budget.
        max_costs = [agent.estimate_max_cost_usd(context) for agent in self._agents]
        if not self._cost_tracker.can_spend(sum(max_costs)):
            logger.warning(
                "LLM budget exceeded (spent $%.4f today, next run up to $%.4f) "
                "— skipping agent analysis",
                self._cost_tracker.daily_spent_usd,
                sum(max_costs),
            )
            self._cost_tracker.mark_exhausted()
            return None

        # Run agents in parallel
        results = await asyncio.gather(
            *[self._safe_analyze(agent, context) for agent in self._agents]
        )

        # A timed-out call may still be billed: charge its worst case.
        timed_out_cost = sum(
            cost
            for result, cost in zip(results, max_costs, strict=True)
            if result is _TIMED_OUT
        )
        analyses: list[AgentAnalysis] = [
            r for r in results if isinstance(r, AgentAnalysis)
        ]
        actual_cost = sum(a.cost_usd for a in analyses) + timed_out_cost
        if actual_cost > 0:
            self._cost_tracker.record_cost(actual_cost)

        if not analyses:
            logger.warning("All agents failed for %s", pair)
            return None

        # Consensus
        recommendation, confidence, consensus = self._apply_consensus(analyses)

        total_tokens = sum(a.tokens_used for a in analyses)
        total_latency = max((a.latency_ms for a in analyses), default=0)

        reasoning_parts = [f"{a.agent_name}: {a.recommendation}" for a in analyses]
        reasoning = f"Consensus={consensus}. Votes: {', '.join(reasoning_parts)}"

        return OrchestratorDecision(
            pair=pair,
            final_recommendation=recommendation,
            confidence=confidence,
            agent_analyses=analyses,
            consensus=consensus,
            reasoning=reasoning,
            total_tokens=total_tokens,
            total_cost_usd=actual_cost,
            total_latency_ms=total_latency,
            timestamp=datetime.now(UTC),
        )

    async def _safe_analyze(
        self, agent: BaseAgent, context: dict[str, object]
    ) -> AgentAnalysis | object | None:
        """Run a single agent with timeout; _TIMED_OUT or None on failure."""
        try:
            return await asyncio.wait_for(
                agent.analyze(context), timeout=_AGENT_TIMEOUT_S
            )
        except TimeoutError:
            logger.warning(
                "Agent %s timed out after %.0fs", agent.name, _AGENT_TIMEOUT_S
            )
            return _TIMED_OUT
        except Exception:
            logger.exception("Agent %s failed", agent.name)
            return None

    @staticmethod
    def _apply_consensus(
        analyses: list[AgentAnalysis],
    ) -> tuple[str, float, bool]:
        """Determine majority recommendation with confidence scaling."""
        votes = Counter(a.recommendation for a in analyses)
        total = len(analyses)
        majority_rec, majority_count = votes.most_common(1)[0]
        avg_confidence = sum(a.confidence for a in analyses) / total

        if majority_count == total:
            # Unanimous
            return majority_rec, avg_confidence * 1.0, True
        elif majority_count >= 2 and total >= 2:
            # Majority (2/3 or 2/2)
            return majority_rec, avg_confidence * 0.75, True
        else:
            # No agreement
            return "NEUTRAL", 0.0, False
