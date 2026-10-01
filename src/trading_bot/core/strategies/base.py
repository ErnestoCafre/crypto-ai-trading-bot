"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd

from trading_bot.core.models import AISignals, Signal


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> Signal | None: ...

    @abstractmethod
    def get_required_lookback(self) -> int: ...


def graded_confidence(margins: Iterable[float]) -> float:
    """Entry confidence on a scale shared by every strategy.

    0.5 means the entry conditions are only just met; each margin measures
    how far past its threshold one condition is, already scaled so that 1.0
    is a strong reading. The result is 0.5 + 0.5 x mean(clipped margins), so
    a qualifying technical signal always clears MIN_CONFIDENCE_THRESHOLD=0.5
    on its own and AI sources decide whether it stays above it.
    """
    clipped = [min(1.0, max(0.0, m)) for m in margins if pd.notna(m)]
    if not clipped:
        return 0.5
    return 0.5 + 0.5 * sum(clipped) / len(clipped)
