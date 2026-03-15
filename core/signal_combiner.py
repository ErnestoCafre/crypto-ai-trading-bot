"""Signal combiner stub."""

from __future__ import annotations

from core.models import AISignals, CombinedSignal, Signal


class SignalCombiner:
    def combine(
        self,
        technical: Signal | None,
        ai: AISignals | None,
        weights: dict[str, float],
    ) -> CombinedSignal:
        raise NotImplementedError
