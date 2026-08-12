"""Strategy selector — routes to the appropriate strategy based on market regime."""

from __future__ import annotations

import logging

import pandas as pd

from trading_bot.core.models import AISignals, Signal
from trading_bot.core.regime import MarketRegime, RegimeClassification, RegimeDetector
from trading_bot.core.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategySelector:
    """Select and run the appropriate strategy based on the detected market regime.

    Routing:
    - TRENDING -> trend_strategy
    - RANGING -> reversion_strategy
    - VOLATILE -> either strategy with reduced sizing (or None if confidence too low)
    """

    def __init__(
        self,
        trend_strategy: BaseStrategy,
        reversion_strategy: BaseStrategy,
        regime_detector: RegimeDetector,
        volatile_sizing_factor: float = 0.5,
    ) -> None:
        self._trend = trend_strategy
        self._reversion = reversion_strategy
        self._detector = regime_detector
        self._volatile_factor = volatile_sizing_factor

    def select(self, ohlcv: pd.DataFrame) -> tuple[BaseStrategy, RegimeClassification]:
        """Return the strategy to use and the regime classification."""
        classification = self._detector.classify(ohlcv)

        if classification.regime == MarketRegime.TRENDING:
            return self._trend, classification
        elif classification.regime == MarketRegime.RANGING:
            return self._reversion, classification
        else:  # VOLATILE
            # Use trend strategy in volatile regime but with reduced sizing
            return self._trend, classification

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> tuple[Signal | None, RegimeClassification]:
        """Generate a signal using the regime-appropriate strategy.

        In VOLATILE regime, signal strength is reduced by volatile_sizing_factor.
        """
        strategy, classification = self.select(ohlcv)
        signal = strategy.generate_signal(ohlcv, ai_signals)

        # Reduce strength in volatile regime
        if signal is not None and classification.regime == MarketRegime.VOLATILE:
            signal = Signal(
                type=signal.type,
                pair=signal.pair,
                confidence=signal.confidence * self._volatile_factor,
                strength=signal.strength * self._volatile_factor,
                source=signal.source,
                metadata={
                    **signal.metadata,
                    "regime": "volatile",
                    "sizing_reduced": True,
                },
                timestamp=signal.timestamp,
            )

        return signal, classification
