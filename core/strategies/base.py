"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.models import AISignals, Signal


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
