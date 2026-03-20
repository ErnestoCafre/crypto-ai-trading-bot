"""ATR-based stop-loss manager."""

from __future__ import annotations

import pandas as pd

from ai.prediction.feature_engine import FeatureEngine


class StopLossManager:
    def __init__(self, atr_multiplier: float = 2.0, default_pct: float = 0.02) -> None:
        self._atr_mult = atr_multiplier
        self._default_pct = default_pct
        self._fe = FeatureEngine()

    def calculate_stop(
        self,
        pair: str,
        side: str,
        entry_price: float,
        ohlcv: pd.DataFrame | None = None,
        atr_value: float | None = None,
    ) -> float:
        """Return the stop-loss price for a trade.

        Priority: explicit *atr_value* > computed from *ohlcv* > percentage fallback.
        """
        atr = atr_value

        if atr is None and ohlcv is not None:
            df = self._fe.build_features(ohlcv)
            col = FeatureEngine.ATR_14
            if col in df.columns:
                last_atr = df[col].iloc[-1]
                if pd.notna(last_atr):
                    atr = float(last_atr)

        if atr is None:
            atr = entry_price * self._default_pct

        if side == "buy":
            stop = entry_price - atr * self._atr_mult
        else:
            stop = entry_price + atr * self._atr_mult

        return max(0.0, stop)
