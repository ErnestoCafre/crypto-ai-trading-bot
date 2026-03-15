"""Stop-loss manager stub."""

from __future__ import annotations


class StopLossManager:
    def calculate_stop(self, pair: str, side: str, entry_price: float) -> float:
        raise NotImplementedError
