"""AI confidence filter — gates trading on combined signal confidence."""

from __future__ import annotations

from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import CombinedSignal


class AIConfidenceFilter:
    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._min_confidence = s.MIN_CONFIDENCE_THRESHOLD

    def should_trade(self, signal: CombinedSignal) -> bool:
        """Return True if the signal is confident enough to trade.

        In Phase 2 (no AI), this is a pass-through when ai_signals is None.
        """
        if signal.ai_signals is None:
            return True

        return signal.combined_confidence >= self._min_confidence
