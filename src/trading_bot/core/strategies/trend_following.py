"""Trend-following strategy — EMA alignment, RSI, MACD, ADX."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy


class TrendFollowing(BaseStrategy):
    def __init__(self, pair: str, feature_engine: FeatureEngine | None = None) -> None:
        self._pair = pair
        self._fe = feature_engine or FeatureEngine()

    @property
    def name(self) -> str:
        return "trend_following"

    def get_required_lookback(self) -> int:
        return 200

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> Signal | None:
        df = self._fe.build_features(ohlcv)
        last = df.iloc[-1]

        required = [
            FeatureEngine.EMA_9,
            FeatureEngine.EMA_21,
            FeatureEngine.EMA_50,
            FeatureEngine.RSI_14,
            FeatureEngine.MACD_12_26_9,
            FeatureEngine.MACDs_12_26_9,
            FeatureEngine.ADX_14,
        ]
        for col in required:
            if col not in df.columns or pd.isna(last.get(col)):
                return None

        ema9 = float(last[FeatureEngine.EMA_9])
        ema21 = float(last[FeatureEngine.EMA_21])
        ema50 = float(last[FeatureEngine.EMA_50])
        rsi = float(last[FeatureEngine.RSI_14])
        macd = float(last[FeatureEngine.MACD_12_26_9])
        macd_signal = float(last[FeatureEngine.MACDs_12_26_9])
        adx = float(last[FeatureEngine.ADX_14])

        # --- CLOSE conditions (any) ---
        close_conds = [ema9 < ema21, rsi < 40]
        if any(close_conds):
            return Signal(
                type=SignalType.CLOSE,
                pair=self._pair,
                confidence=sum(close_conds) / len(close_conds),
                strength=min(adx / 50.0, 1.0),
                source=self.name,
                metadata={"reason": "close_conditions"},
                timestamp=datetime.now(UTC),
            )

        # --- LONG conditions (all) ---
        long_subs = [
            ema9 > ema21,
            ema21 > ema50,
            rsi > 50,
            macd > macd_signal,
            adx > 25,
        ]
        if all(long_subs):
            return Signal(
                type=SignalType.LONG,
                pair=self._pair,
                confidence=sum(long_subs) / len(long_subs),
                strength=min(adx / 50.0, 1.0),
                source=self.name,
                metadata={"reason": "trend_long"},
                timestamp=datetime.now(UTC),
            )

        return None
