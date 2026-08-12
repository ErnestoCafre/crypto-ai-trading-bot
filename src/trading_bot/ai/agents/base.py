"""Abstract base class for LLM agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trading_bot.core.models import AgentAnalysis


class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def analyze(self, context: dict[str, object]) -> AgentAnalysis: ...
