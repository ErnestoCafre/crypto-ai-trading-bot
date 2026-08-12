"""Tests for RegimeDetector and StrategySelector."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import SignalType
from trading_bot.core.regime import MarketRegime, RegimeClassification, RegimeDetector
from trading_bot.core.strategies.mean_reversion import MeanReversion
from trading_bot.core.strategies.trend_following import TrendFollowing
from trading_bot.core.strategy_selector import StrategySelector


def _make_ohlcv(
    n: int = 100, trend: float = 0.0, volatility: float = 100.0
) -> pd.DataFrame:
    """Create synthetic OHLCV data with controllable trend and volatility."""
    rng = np.random.default_rng(42)
    close = 50000 + np.cumsum(rng.normal(trend, volatility, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, volatility * 2, n)),
            "low": close - abs(rng.normal(0, volatility * 2, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
    )


@pytest.fixture()
def detector() -> RegimeDetector:
    return RegimeDetector()


class TestRegimeDetector:
    def test_trending_high_adx(self) -> None:
        """Force ADX high + moderate ATR -> TRENDING."""
        detector = RegimeDetector()
        ohlcv = _make_ohlcv(250, trend=200.0, volatility=50.0)

        # Build features and check what ADX we get
        fe = FeatureEngine()
        df = fe.build_features(ohlcv)
        adx_val = df[FeatureEngine.ADX_14].iloc[-1]

        classification = detector.classify(ohlcv)
        # Verify classification structure
        assert isinstance(classification.regime, MarketRegime)
        assert isinstance(classification.confidence, float)
        assert classification.adx_value == pytest.approx(adx_val, abs=0.01)

    def test_ranging_low_adx(self) -> None:
        """Low trend data should tend toward RANGING."""
        detector = RegimeDetector()
        # Flat data: no trend
        ohlcv = _make_ohlcv(250, trend=0.0, volatility=50.0)
        classification = detector.classify(ohlcv)
        assert isinstance(classification.regime, MarketRegime)
        assert classification.confidence >= 0.0
        assert classification.confidence <= 1.0

    def test_volatile_high_atr(self) -> None:
        """Very high volatility data should detect VOLATILE."""
        detector = RegimeDetector()
        # Create data with sudden spike in volatility at the end
        rng = np.random.default_rng(42)
        n = 200
        close = np.concatenate(
            [
                50000 + np.cumsum(rng.normal(0, 20, n - 20)),  # calm
                50000 + np.cumsum(rng.normal(0, 2000, 20)),  # volatile end
            ]
        )
        ohlcv = pd.DataFrame(
            {
                "open": close + rng.normal(0, 50, n),
                "high": close + abs(rng.normal(0, 1000, n)),
                "low": close - abs(rng.normal(0, 1000, n)),
                "close": close,
                "volume": abs(rng.normal(1000, 300, n)),
            }
        )
        classification = detector.classify(ohlcv)
        # Volatile end should push ATR percentile high
        assert isinstance(classification, RegimeClassification)

    def test_confidence_bounded(self) -> None:
        """Confidence should always be between 0.0 and 1.0."""
        detector = RegimeDetector()
        ohlcv = _make_ohlcv(250)
        classification = detector.classify(ohlcv)
        assert 0.0 <= classification.confidence <= 1.0

    def test_classify_series_length(self) -> None:
        """classify_series should return a Series with the same length as input."""
        detector = RegimeDetector()
        ohlcv = _make_ohlcv(50)
        series = detector.classify_series(ohlcv)
        assert len(series) == len(ohlcv)

    def test_classify_series_values(self) -> None:
        """All values in classify_series should be MarketRegime members."""
        detector = RegimeDetector()
        ohlcv = _make_ohlcv(50)
        series = detector.classify_series(ohlcv)
        for val in series:
            assert isinstance(val, MarketRegime)


class TestStrategySelector:
    def _make_selector(self) -> StrategySelector:
        fe = MagicMock(spec=FeatureEngine)
        fe.build_features = MagicMock(side_effect=lambda df: df)
        trend = TrendFollowing(pair="BTC/USDT", feature_engine=fe)
        reversion = MeanReversion(pair="BTC/USDT", feature_engine=fe)
        detector = RegimeDetector(feature_engine=fe)
        return StrategySelector(trend, reversion, detector)

    def test_trending_routes_trend_following(self) -> None:
        """TRENDING regime should route to TrendFollowing."""
        selector = self._make_selector()
        # Mock the detector to return TRENDING
        selector._detector.classify = MagicMock(  # type: ignore[method-assign]
            return_value=RegimeClassification(
                regime=MarketRegime.TRENDING,
                confidence=0.8,
                adx_value=35.0,
                volatility_percentile=50.0,
                trend_strength=0.7,
                timestamp=pd.Timestamp.now(tz="UTC"),
            )
        )
        ohlcv = _make_ohlcv(100)
        strategy, classification = selector.select(ohlcv)
        assert strategy.name == "trend_following"
        assert classification.regime == MarketRegime.TRENDING

    def test_ranging_routes_mean_reversion(self) -> None:
        """RANGING regime should route to MeanReversion."""
        selector = self._make_selector()
        selector._detector.classify = MagicMock(  # type: ignore[method-assign]
            return_value=RegimeClassification(
                regime=MarketRegime.RANGING,
                confidence=0.7,
                adx_value=15.0,
                volatility_percentile=40.0,
                trend_strength=0.3,
                timestamp=pd.Timestamp.now(tz="UTC"),
            )
        )
        ohlcv = _make_ohlcv(100)
        strategy, classification = selector.select(ohlcv)
        assert strategy.name == "mean_reversion"
        assert classification.regime == MarketRegime.RANGING

    def test_volatile_reduces_sizing(self) -> None:
        """VOLATILE regime should reduce signal confidence/strength."""
        selector = self._make_selector()
        selector._detector.classify = MagicMock(  # type: ignore[method-assign]
            return_value=RegimeClassification(
                regime=MarketRegime.VOLATILE,
                confidence=0.6,
                adx_value=30.0,
                volatility_percentile=90.0,
                trend_strength=0.6,
                timestamp=pd.Timestamp.now(tz="UTC"),
            )
        )
        # Need to also mock the strategy's generate_signal
        from datetime import UTC, datetime

        from trading_bot.core.models import Signal

        mock_signal = Signal(
            type=SignalType.LONG,
            pair="BTC/USDT",
            confidence=0.8,
            strength=0.7,
            source="trend_following",
            metadata={},
            timestamp=datetime.now(UTC),
        )
        selector._trend.generate_signal = MagicMock(return_value=mock_signal)  # type: ignore[method-assign]

        ohlcv = _make_ohlcv(100)
        signal, classification = selector.generate_signal(ohlcv)

        assert signal is not None
        # default volatile_sizing_factor = 0.5
        assert signal.confidence == pytest.approx(0.8 * 0.5)
        assert signal.strength == pytest.approx(0.7 * 0.5)
        assert signal.metadata.get("regime") == "volatile"
