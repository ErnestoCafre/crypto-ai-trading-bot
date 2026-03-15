"""Risk evaluator agent stub."""

from __future__ import annotations

from ai.agents.base import BaseAgent
from core.models import AgentAnalysis


class RiskEvaluatorAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "risk_evaluator"

    async def analyze(self, context: dict[str, object]) -> AgentAnalysis:
        raise NotImplementedError
