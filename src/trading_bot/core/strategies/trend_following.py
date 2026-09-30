"""Trend-following strategy — EMA alignment, RSI, MACD, ADX."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy


class TrendFollowing(BaseStrategy):
    """LONG when EMAs are stacked, momentum is up and the trend is strong.

    Thresholds are constructor parameters so they can be optimized; the
    defaults are the production values.
    """

    def __init__(
        self,
        pair: str,
        feature_engine: FeatureEngine | None = None,
        *,
        ema_fast: int = 9,
        ema_mid: int = 21,
        ema_slow: int = 50,
        rsi_entry_min: float = 50.0,
        rsi_exit_below: float = 40.0,
        adx_min: float = 25.0,
    ) -> None:
        if not ema_fast < ema_mid < ema_slow:
            raise ValueError("EMA periods must satisfy fast < mid < slow")
        self._pair = pair
        self._ema_periods = (ema_fast, ema_mid, ema_slow)
        self._fe = feature_engine or FeatureEngine(extra_ema_periods=self._ema_periods)
        self._rsi_entry_min = rsi_entry_min
        self._rsi_exit_below = rsi_exit_below
        self._adx_min = adx_min

    @property
    def name(self) -> str:
        return "trend_following"

    def get_required_lookback(self) -> int:
        return max(200, 4 * self._ema_periods[2])

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> Signal | None:
        df = self._fe.build_features(ohlcv)
        cols = [FeatureEngine.ema_column(p) for p in self._ema_periods]
        missing = {
            col: p
            for col, p in zip(cols, self._ema_periods, strict=True)
            if col not in df.columns
        }
        if missing and len(df) >= max(missing.values()):
            # Shared engine without these periods: slow path, only for
            # non-default parameters.
            df = df.assign(**{col: df.ta.ema(length=p) for col, p in missing.items()})
        last = df.iloc[-1]

        fast_col, mid_col, slow_col = cols
        required = [
            fast_col,
            mid_col,
            slow_col,
            FeatureEngine.RSI_14,
            FeatureEngine.MACD_12_26_9,
            FeatureEngine.MACDs_12_26_9,
            FeatureEngine.ADX_14,
        ]
        for col in required:
            if col not in df.columns or pd.isna(last.get(col)):
                return None

        ema9 = float(last[fast_col])
        ema21 = float(last[mid_col])
        ema50 = float(last[slow_col])
        rsi = float(last[FeatureEngine.RSI_14])
        macd = float(last[FeatureEngine.MACD_12_26_9])
        macd_signal = float(last[FeatureEngine.MACDs_12_26_9])
        adx = float(last[FeatureEngine.ADX_14])

        # --- CLOSE conditions (any) ---
        close_conds = [ema9 < ema21, rsi < self._rsi_exit_below]
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
            rsi > self._rsi_entry_min,
            macd > macd_signal,
            adx > self._adx_min,
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
