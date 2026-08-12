"""AI confidence filter — gates trading on combined signal confidence."""

from __future__ import annotations

from trading_bot.config.settings import get_settings
from trading_bot.core.models import CombinedSignal


class AIConfidenceFilter:
    def should_trade(self, signal: CombinedSignal) -> bool:
        """Return True if the signal is confident enough to trade.

        In Phase 2 (no AI), this is a pass-through when ai_signals is None.
        """
        if signal.ai_signals is None:
            return True

        settings = get_settings()
        return signal.combined_confidence >= settings.MIN_CONFIDENCE_THRESHOLD
