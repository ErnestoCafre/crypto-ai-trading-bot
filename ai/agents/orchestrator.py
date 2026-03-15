"""Orchestrator stub."""

from __future__ import annotations

from core.models import OrchestratorDecision


class Orchestrator:
    def run(self, pair: str, context: dict[str, object]) -> OrchestratorDecision:
        raise NotImplementedError
