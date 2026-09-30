"""Mean-reversion strategy — oversold/overbought extremes + Bollinger Bands."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy


class MeanReversion(BaseStrategy):
    """Fade RSI extremes outside the Bollinger Bands; exit near the mean.

    Thresholds are constructor parameters so they can be optimized; the
    defaults are the production values.
    """

    def __init__(
        self,
        pair: str,
        feature_engine: FeatureEngine | None = None,
        *,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_exit_low: float = 40.0,
        rsi_exit_high: float = 60.0,
    ) -> None:
        if not rsi_oversold < rsi_exit_low <= rsi_exit_high < rsi_overbought:
            raise ValueError(
                "RSI thresholds must satisfy "
                "oversold < exit_low <= exit_high < overbought"
            )
        self._pair = pair
        self._fe = feature_engine or FeatureEngine()
        self._oversold = rsi_oversold
        self._overbought = rsi_overbought
        self._exit_low = rsi_exit_low
        self._exit_high = rsi_exit_high

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
        if self._exit_low <= rsi <= self._exit_high:
            mid = (self._exit_low + self._exit_high) / 2
            half_band = (self._exit_high - self._exit_low) / 2 or 1.0
            return Signal(
                type=SignalType.CLOSE,
                pair=self._pair,
                confidence=1.0 - abs(rsi - mid) / half_band,
                strength=0.5,
                source=self.name,
                metadata={"reason": "mean_reversion_close"},
                timestamp=datetime.now(UTC),
            )

        # --- LONG: oversold ---
        if rsi < self._oversold and close < bbl:
            confidence = (self._oversold - rsi) / self._oversold
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
        if rsi > self._overbought and close > bbu:
            confidence = (rsi - self._overbought) / (100.0 - self._overbought)
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
