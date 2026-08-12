"""Market regime detection — ADX + ATR-based classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeClassification:
    regime: MarketRegime
    confidence: float
    adx_value: float
    volatility_percentile: float
    trend_strength: float
    timestamp: datetime


class RegimeDetector:
    """Classify market regime using ADX + ATR percentile."""

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        volatility_high_percentile: float = 80.0,
        lookback_period: int = 100,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        self._adx_threshold = adx_trending_threshold
        self._vol_pct_threshold = volatility_high_percentile
        self._lookback = lookback_period
        self._fe = feature_engine or FeatureEngine()

    def classify(self, ohlcv: pd.DataFrame) -> RegimeClassification:
        """Classify the current market regime from OHLCV data.

        Logic:
        - VOLATILE: ATR percentile > volatility_high_percentile (regardless of ADX)
        - TRENDING: ADX > adx_trending_threshold AND ATR percentile <= threshold
        - RANGING: ADX <= threshold AND ATR percentile <= threshold
        """
        df = self._fe.build_features(ohlcv)

        # Get ADX value
        adx_col = FeatureEngine.ADX_14
        has_adx = adx_col in df.columns and pd.notna(df[adx_col].iloc[-1])
        adx = float(df[adx_col].iloc[-1]) if has_adx else 20.0

        # Get ATR percentile over lookback
        atr_col = FeatureEngine.ATR_14
        if atr_col in df.columns:
            atr_series = df[atr_col].dropna()
            lookback_data = atr_series.tail(self._lookback)
            if len(lookback_data) > 1:
                current_atr = float(atr_series.iloc[-1])
                vol_percentile = float(
                    (lookback_data < current_atr).sum() / len(lookback_data) * 100
                )
            else:
                vol_percentile = 50.0
        else:
            vol_percentile = 50.0

        # Classification
        if vol_percentile > self._vol_pct_threshold:
            regime = MarketRegime.VOLATILE
        elif adx > self._adx_threshold:
            regime = MarketRegime.TRENDING
        else:
            regime = MarketRegime.RANGING

        # Confidence: normalized distance from thresholds
        confidence = self._compute_confidence(adx, vol_percentile, regime)

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            adx_value=adx,
            volatility_percentile=vol_percentile,
            trend_strength=min(adx / 50.0, 1.0),
            timestamp=datetime.now(UTC),
        )

    def classify_series(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Return a Series of MarketRegime values, one per row.

        Uses an expanding window approach: for each bar i, classify using ohlcv[:i+1].
        Min 30 bars needed; earlier bars default to RANGING.
        """
        n = len(ohlcv)
        regimes = []
        min_bars = 30

        for i in range(n):
            if i < min_bars - 1:
                regimes.append(MarketRegime.RANGING)
            else:
                try:
                    classification = self.classify(ohlcv.iloc[: i + 1])
                    regimes.append(classification.regime)
                except Exception:
                    regimes.append(MarketRegime.RANGING)

        return pd.Series(regimes, index=ohlcv.index)

    def _compute_confidence(
        self, adx: float, vol_percentile: float, regime: MarketRegime
    ) -> float:
        """Compute confidence as normalized distance from classification thresholds."""
        if regime == MarketRegime.VOLATILE:
            # How far above the volatility threshold
            denom = 100 - self._vol_pct_threshold
            raw = (vol_percentile - self._vol_pct_threshold) / denom
            return float(np.clip(0.5 + raw * 0.5, 0.0, 1.0))
        elif regime == MarketRegime.TRENDING:
            # How far above the ADX threshold
            raw = (adx - self._adx_threshold) / self._adx_threshold
            return float(np.clip(0.5 + raw * 0.5, 0.0, 1.0))
        else:  # RANGING
            # How far below ADX threshold + how far below vol threshold
            adx_dist = (self._adx_threshold - adx) / self._adx_threshold
            vol_dist = (
                self._vol_pct_threshold - vol_percentile
            ) / self._vol_pct_threshold
            raw = (adx_dist + vol_dist) / 2
            return float(np.clip(0.5 + raw * 0.5, 0.0, 1.0))
