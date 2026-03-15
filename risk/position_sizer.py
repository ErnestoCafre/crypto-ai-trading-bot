"""Position sizer stub."""

from __future__ import annotations

from core.models import PositionSize


class PositionSizer:
    def calculate(
        self,
        pair: str,
        signal_confidence: float,
        portfolio_value: float,
    ) -> PositionSize:
        raise NotImplementedError
