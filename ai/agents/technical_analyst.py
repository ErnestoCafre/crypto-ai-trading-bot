"""Technical analyst agent stub."""

from __future__ import annotations

from ai.agents.base import BaseAgent
from core.models import AgentAnalysis


class TechnicalAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "technical_analyst"

    async def analyze(self, context: dict[str, object]) -> AgentAnalysis:
        raise NotImplementedError
