"""Sentiment analyst agent stub."""

from __future__ import annotations

from ai.agents.base import BaseAgent
from core.models import AgentAnalysis


class SentimentAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "sentiment_analyst"

    async def analyze(self, context: dict[str, object]) -> AgentAnalysis:
        raise NotImplementedError
