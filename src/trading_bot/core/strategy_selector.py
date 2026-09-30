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

    Routing (``use_regime=True``):
    - TRENDING -> trend_strategy
    - RANGING -> reversion_strategy
    - VOLATILE -> trend_strategy; the signal carries
      ``metadata["sizing_factor"] = volatile_sizing_factor`` so the position
      is taken with reduced risk. Confidence is left untouched: scaling it
      would push most signals under MIN_CONFIDENCE_THRESHOLD and block the
      entry instead of shrinking it.

    With ``use_regime=False`` the trend strategy always runs at full size;
    the regime is still classified for reporting (e.g. the LLM context).
    """

    def __init__(
        self,
        trend_strategy: BaseStrategy,
        reversion_strategy: BaseStrategy,
        regime_detector: RegimeDetector,
        volatile_sizing_factor: float = 0.5,
        use_regime: bool = True,
    ) -> None:
        self._trend = trend_strategy
        self._reversion = reversion_strategy
        self._detector = regime_detector
        self._volatile_factor = volatile_sizing_factor
        self._use_regime = use_regime

    def select(self, ohlcv: pd.DataFrame) -> tuple[BaseStrategy, RegimeClassification]:
        """Return the strategy to use and the regime classification."""
        classification = self._detector.classify(ohlcv)

        if not self._use_regime:
            return self._trend, classification
        if classification.regime == MarketRegime.RANGING:
            return self._reversion, classification
        # TRENDING, and VOLATILE with reduced sizing (see generate_signal)
        return self._trend, classification

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: AISignals | None = None,
    ) -> tuple[Signal | None, RegimeClassification]:
        """Generate a signal using the regime-appropriate strategy.

        In VOLATILE regime the signal is tagged with the sizing factor.
        """
        strategy, classification = self.select(ohlcv)
        signal = strategy.generate_signal(ohlcv, ai_signals)

        if (
            signal is not None
            and self._use_regime
            and classification.regime == MarketRegime.VOLATILE
        ):
            signal = Signal(
                type=signal.type,
                pair=signal.pair,
                confidence=signal.confidence,
                strength=signal.strength,
                source=signal.source,
                metadata={
                    **signal.metadata,
                    "regime": "volatile",
                    "sizing_factor": self._volatile_factor,
                },
                timestamp=signal.timestamp,
            )

        return signal, classification
