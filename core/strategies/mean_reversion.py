"""Mean-reversion strategy stub."""

from __future__ import annotations

import pandas as pd

from core.models import AISignals, Signal
from core.strategies.base import BaseStrategy


class MeanReversion(BaseStrategy):
    @property
    def name(self) -> str:
        return "mean_reversion"

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> Signal | None:
        raise NotImplementedError

    def get_required_lookback(self) -> int:
        raise NotImplementedError
