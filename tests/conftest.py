"""Global test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ai.prediction.feature_engine import FeatureEngine
from core.models import (
    AISignals,
    CombinedSignal,
    OrderResult,
    SentimentResult,
    Signal,
    SignalType,
)


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic 100-bar OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    n = 100
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
    )


@pytest.fixture()
def sample_ohlcv_250() -> pd.DataFrame:
    """Synthetic 250-bar OHLCV DataFrame (enough for EMA_200)."""
    rng = np.random.default_rng(42)
    n = 250
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
    )


@pytest.fixture()
def sample_signal() -> Signal:
    return Signal(
        type=SignalType.LONG,
        pair="BTC/USDT",
        confidence=0.75,
        strength=0.6,
        source="trend_following",
        metadata={},
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture()
def sample_ai_signals() -> AISignals:
    return AISignals(
        sentiment_score=0.7,
        prediction_confidence=0.8,
        llm_confidence=0.6,
    )


@pytest.fixture()
def sample_combined_signal_no_ai(sample_signal: Signal) -> CombinedSignal:
    return CombinedSignal(
        type=SignalType.LONG,
        pair="BTC/USDT",
        combined_confidence=0.75,
        technical_signal=sample_signal,
        ai_signals=None,
        weights_used={"technical": 0.4},
        should_trade=True,
        recommended_sizing_pct=7.5,
        reasoning="test",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture()
def sample_combined_signal_with_ai(
    sample_signal: Signal, sample_ai_signals: AISignals
) -> CombinedSignal:
    return CombinedSignal(
        type=SignalType.LONG,
        pair="BTC/USDT",
        combined_confidence=0.72,
        technical_signal=sample_signal,
        ai_signals=sample_ai_signals,
        weights_used={"technical": 0.4},
        should_trade=True,
        recommended_sizing_pct=7.2,
        reasoning="test",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture()
def sample_sentiment_result() -> SentimentResult:
    return SentimentResult(
        score=0.65,
        confidence=0.8,
        label="positive",
        model_used="CryptoBERT",
        analyzed_text="Bitcoin breaks ATH",
    )


@pytest.fixture()
def sample_order_result() -> OrderResult:
    return OrderResult(
        order_id="test-001",
        exchange_order_id="EX-001",
        pair="BTC/USDT",
        side="buy",
        filled_price=50000.0,
        filled_quantity=0.01,
        fee=0.5,
        slippage_pct=0.01,
        status="filled",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_indicator_row(**overrides: float) -> pd.DataFrame:
    """Build a single-row DataFrame with OHLCV + indicator columns."""
    defaults: dict[str, float] = {
        "open": 50000,
        "high": 51000,
        "low": 49000,
        "close": 50500,
        "volume": 1000,
        FeatureEngine.EMA_9: 50200,
        FeatureEngine.EMA_21: 50000,
        FeatureEngine.EMA_50: 49800,
        FeatureEngine.RSI_14: 55,
        FeatureEngine.MACD_12_26_9: 100,
        FeatureEngine.MACDs_12_26_9: 80,
        FeatureEngine.ADX_14: 30,
        FeatureEngine.BBL_20: 49000,
        FeatureEngine.BBU_20: 51000,
        FeatureEngine.BBM_20: 50000,
        FeatureEngine.ATR_14: 500,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


@pytest.fixture()
def mock_feature_engine() -> MagicMock:
    """A MagicMock FeatureEngine that returns whatever DataFrame it receives."""
    fe = MagicMock(spec=FeatureEngine)
    fe.build_features = MagicMock(side_effect=lambda df: df)
    return fe


@pytest.fixture()
def bullish_ohlcv() -> pd.DataFrame:
    """Single-row DataFrame with bullish indicator values."""
    return _make_indicator_row(
        **{
            FeatureEngine.EMA_9: 50200,
            FeatureEngine.EMA_21: 50000,
            FeatureEngine.EMA_50: 49800,
            FeatureEngine.RSI_14: 55,
            FeatureEngine.MACD_12_26_9: 100,
            FeatureEngine.MACDs_12_26_9: 80,
            FeatureEngine.ADX_14: 30,
        }
    )


@pytest.fixture()
def bearish_ohlcv() -> pd.DataFrame:
    """Single-row DataFrame triggering CLOSE in TrendFollowing."""
    return _make_indicator_row(
        **{
            FeatureEngine.EMA_9: 49800,
            FeatureEngine.EMA_21: 50000,
            FeatureEngine.RSI_14: 35,
            FeatureEngine.ADX_14: 20,
        }
    )


@pytest.fixture()
def oversold_ohlcv() -> pd.DataFrame:
    """Single-row DataFrame triggering LONG in MeanReversion."""
    return _make_indicator_row(
        close=48500,
        **{
            FeatureEngine.RSI_14: 25,
            FeatureEngine.BBL_20: 49000,
            FeatureEngine.BBU_20: 51000,
        },
    )


@pytest.fixture()
def overbought_ohlcv() -> pd.DataFrame:
    """Single-row DataFrame triggering SHORT in MeanReversion."""
    return _make_indicator_row(
        close=51500,
        **{
            FeatureEngine.RSI_14: 75,
            FeatureEngine.BBL_20: 49000,
            FeatureEngine.BBU_20: 51000,
        },
    )


@pytest.fixture()
def neutral_rsi_ohlcv() -> pd.DataFrame:
    """Single-row DataFrame triggering CLOSE in MeanReversion (RSI mean zone)."""
    return _make_indicator_row(
        **{
            FeatureEngine.RSI_14: 50,
        }
    )
