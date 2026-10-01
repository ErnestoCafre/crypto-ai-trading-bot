"""Mean-reversion strategy — oversold/overbought extremes + Bollinger Bands."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy, graded_confidence


class MeanReversion(BaseStrategy):
    """Fade RSI extremes outside the Bollinger Bands; exit near the mean.

    Thresholds are constructor parameters so they can be optimized; the
    defaults are the production values. LONG/SHORT confidence is graded (see
    :func:`graded_confidence`) on the same scale as TrendFollowing: RSI 20
    points past its extreme and a close 1 ATR outside the band each count as
    a strong reading. Before this, (30 - RSI) / 30 needed RSI <= 15 to clear
    MIN_CONFIDENCE_THRESHOLD, so the strategy never traded.
    """

    # Margin scales: how far past each threshold counts as a strong reading.
    RSI_STRONG_POINTS = 20.0
    BAND_STRONG_ATR = 1.0

    def __init__(
        self,
        pair: str,
        feature_engine: FeatureEngine | None = None,
        *,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_exit_low: float = 40.0,
        rsi_exit_high: float = 60.0,
        enable_entries: bool = True,
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
        # False: only exits (CLOSE, and SHORT, which closes a spot position).
        self._enable_entries = enable_entries

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
        atr = float(last.get(FeatureEngine.ATR_14, float("nan")))

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
        if self._enable_entries and rsi < self._oversold and close < bbl:
            margins = [(self._oversold - rsi) / self.RSI_STRONG_POINTS]
            if atr > 0:
                margins.append((bbl - close) / atr / self.BAND_STRONG_ATR)
            confidence = graded_confidence(margins)
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
            margins = [(rsi - self._overbought) / self.RSI_STRONG_POINTS]
            if atr > 0:
                margins.append((close - bbu) / atr / self.BAND_STRONG_ATR)
            confidence = graded_confidence(margins)
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
