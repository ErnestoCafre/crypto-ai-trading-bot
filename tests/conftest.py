"""Global test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from core.models import (
    OrderResult,
    SentimentResult,
    Signal,
    SignalType,
)


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic 100-bar OHLCV DataFrame."""
    import numpy as np

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
