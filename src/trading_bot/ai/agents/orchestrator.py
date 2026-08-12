"""Orchestrator — parallel LLM agent execution with consensus."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import UTC, datetime

from trading_bot.ai.agents.base import BaseAgent
from trading_bot.ai.sentiment.cost_tracker import SentimentCostTracker
from trading_bot.core.models import AgentAnalysis, OrchestratorDecision

logger = logging.getLogger(__name__)

# Estimated tokens per agent call (~800 prompt + response)
_ESTIMATED_TOKENS_PER_AGENT = 800
# Haiku rate: $0.25 per 1M input + $1.25 per 1M output ≈ $0.00025 per 1K tokens
_COST_PER_1K_TOKENS = 0.00025
_AGENT_TIMEOUT_S = 15.0


class Orchestrator:
    """Run multiple LLM agents in parallel and apply consensus voting."""

    def __init__(
        self,
        agents: list[BaseAgent],
        cost_tracker: SentimentCostTracker,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._agents = agents
        self._cost_tracker = cost_tracker
        self._provider = provider
        self._model = model

    async def run(
        self, pair: str, context: dict[str, object]
    ) -> OrchestratorDecision | None:
        """Execute all agents, apply consensus, return decision or None."""
        # Budget check
        estimated_cost = (
            len(self._agents) * _ESTIMATED_TOKENS_PER_AGENT / 1000 * _COST_PER_1K_TOKENS
        )
        if not self._cost_tracker.can_spend(estimated_cost):
            logger.warning("LLM budget exceeded — skipping agent analysis")
            return None

        # Run agents in parallel
        results = await asyncio.gather(
            *[self._safe_analyze(agent, context) for agent in self._agents]
        )

        # Filter failures
        analyses: list[AgentAnalysis] = [r for r in results if r is not None]
        if not analyses:
            logger.warning("All agents failed for %s", pair)
            return None

        # Consensus
        recommendation, confidence, consensus = self._apply_consensus(analyses)

        # Record cost
        total_tokens = sum(a.tokens_used for a in analyses)
        actual_cost = total_tokens / 1000 * _COST_PER_1K_TOKENS
        self._cost_tracker.record_cost(actual_cost)

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
    ) -> AgentAnalysis | None:
        """Run a single agent with timeout; return None on failure."""
        try:
            return await asyncio.wait_for(
                agent.analyze(context), timeout=_AGENT_TIMEOUT_S
            )
        except TimeoutError:
            logger.warning(
                "Agent %s timed out after %.0fs", agent.name, _AGENT_TIMEOUT_S
            )
            return None
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
