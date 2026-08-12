"""Tests for StrategyOptimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.optimizer import StrategyOptimizer
from trading_bot.core.models import Signal
from trading_bot.core.strategies.base import BaseStrategy


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 50000 + np.cumsum(rng.normal(0.5, 100, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 30, n),
            "high": close + abs(rng.normal(0, 150, n)),
            "low": close - abs(rng.normal(0, 150, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 200, n)),
        },
        index=dates,
    )


class _ParamStrategy(BaseStrategy):
    """Strategy that accepts a threshold parameter."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "param_strategy"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        return None


class TestOptimizer:
    @pytest.mark.slow()
    def test_returns_best_params(self) -> None:
        def factory(threshold: float = 0.5) -> _ParamStrategy:
            return _ParamStrategy(threshold=threshold)

        optimizer = StrategyOptimizer(
            strategy_factory=factory,
            param_space={"threshold": (0.1, 0.9)},
            n_trials=3,
            walk_forward=False,
        )
        result = optimizer.optimize(_make_ohlcv(100), "BTC/USDT")
        assert "threshold" in result
        assert "_best_metric" in result

    @pytest.mark.slow()
    def test_respects_param_ranges(self) -> None:
        def factory(threshold: float = 0.5) -> _ParamStrategy:
            return _ParamStrategy(threshold=threshold)

        optimizer = StrategyOptimizer(
            strategy_factory=factory,
            param_space={"threshold": (0.2, 0.8)},
            n_trials=3,
            walk_forward=False,
        )
        result = optimizer.optimize(_make_ohlcv(100), "BTC/USDT")
        val = float(result["threshold"])  # type: ignore[arg-type]
        assert 0.2 <= val <= 0.8
