"""Mean-reversion strategy — oversold/overbought extremes + Bollinger Bands."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy


class MeanReversion(BaseStrategy):
    def __init__(self, pair: str, feature_engine: FeatureEngine | None = None) -> None:
        self._pair = pair
        self._fe = feature_engine or FeatureEngine()

    @property
    def name(self) -> str:
        return "mean_reversion"

    def get_required_lookback(self) -> int:
        return 50

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> Signal | None:
        df = self._fe.build_features(ohlcv)
        last = df.iloc[-1]

        required = [
            FeatureEngine.RSI_14,
            FeatureEngine.BBL_20,
            FeatureEngine.BBU_20,
        ]
        for col in required:
            if col not in df.columns or pd.isna(last.get(col)):
                return None

        rsi = float(last[FeatureEngine.RSI_14])
        close = float(last["close"])
        bbl = float(last[FeatureEngine.BBL_20])
        bbu = float(last[FeatureEngine.BBU_20])

        # --- CLOSE: RSI returned to mean zone ---
        if 40 <= rsi <= 60:
            return Signal(
                type=SignalType.CLOSE,
                pair=self._pair,
                confidence=1.0 - abs(rsi - 50) / 10.0,
                strength=0.5,
                source=self.name,
                metadata={"reason": "mean_reversion_close"},
                timestamp=datetime.now(UTC),
            )

        # --- LONG: oversold ---
        if rsi < 30 and close < bbl:
            confidence = (30 - rsi) / 30.0
            return Signal(
                type=SignalType.LONG,
                pair=self._pair,
                confidence=confidence,
                strength=abs(close - bbl) / bbl if bbl != 0 else 0.5,
                source=self.name,
                metadata={"reason": "oversold_long"},
                timestamp=datetime.now(UTC),
            )

        # --- SHORT: overbought ---
        if rsi > 70 and close > bbu:
            confidence = (rsi - 70) / 30.0
            return Signal(
                type=SignalType.SHORT,
                pair=self._pair,
                confidence=confidence,
                strength=abs(close - bbu) / bbu if bbu != 0 else 0.5,
                source=self.name,
                metadata={"reason": "overbought_short"},
                timestamp=datetime.now(UTC),
            )

        return None
