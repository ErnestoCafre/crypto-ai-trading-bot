"""Tests for StrategyOptimizer."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.optimizer import StrategyOptimizer
from trading_bot.backtest.walk_forward import WalkForwardValidator
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

    def __init__(self, threshold: float = 0.5, window: int = 5) -> None:
        self.threshold = threshold
        self.window = window

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
    def test_returns_best_params(self) -> None:
        optimizer = StrategyOptimizer(
            param_space={"threshold": (0.1, 0.9)},
            strategy_factory=_ParamStrategy,
            n_trials=3,
            seed=0,
        )
        result = optimizer.optimize(_make_ohlcv(100), "BTC/USDT")
        assert "threshold" in result
        assert "_best_metric" in result

    def test_respects_param_ranges_and_int_type(self) -> None:
        seen: list[tuple[float, int]] = []

        def factory(threshold: float, window: int) -> _ParamStrategy:
            seen.append((threshold, window))
            return _ParamStrategy(threshold, window)

        optimizer = StrategyOptimizer(
            param_space={"threshold": (0.2, 0.8), "window": (3, 9)},
            strategy_factory=factory,
            n_trials=5,
            seed=0,
        )
        result = optimizer.optimize(_make_ohlcv(100), "BTC/USDT")
        assert 0.2 <= float(result["threshold"]) <= 0.8  # type: ignore[arg-type]
        assert all(isinstance(w, int) and 3 <= w <= 9 for _, w in seen)

    def test_requires_exactly_one_factory(self) -> None:
        with pytest.raises(ValueError):
            StrategyOptimizer(param_space={"x": (0.0, 1.0)})


class TestNestedWalkForward:
    def test_optimizes_on_train_and_scores_on_test_only(self) -> None:
        """Each search sees only its fold's train bars; scoring is OOS."""
        ohlcv = _make_ohlcv(300)
        searched_on: list[pd.Timestamp] = []
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.7)

        class _SpyEngine(BacktestEngine):
            def run(self, ohlcv, pair, ai_signals_series=None, trade_from=0):  # type: ignore[no-untyped-def]
                if trade_from == 0:
                    searched_on.append(ohlcv.index[-1])
                return super().run(ohlcv, pair, ai_signals_series, trade_from)

        def engine_factory(threshold: float) -> BacktestEngine:
            return _SpyEngine(_ParamStrategy(threshold))

        optimizer = StrategyOptimizer(
            param_space={"threshold": (0.0, 1.0)},
            engine_factory=engine_factory,
            n_trials=2,
            seed=0,
        )
        result = optimizer.walk_forward(ohlcv, "BTC/USDT", validator=wf)

        folds = wf.folds(len(ohlcv))
        assert len(result.fold_results) == len(folds)
        train_ends = {ohlcv.index[f.train_end - 1] for f in folds}
        assert set(searched_on) == train_ends
        for fold, fold_result in zip(folds, result.fold_results, strict=True):
            assert fold_result.equity_curve.index[0] == ohlcv.index[fold.test_start]
            assert "threshold" in fold_result.parameters
        assert len(result.parameter_stability["threshold"]) == len(folds)

    @pytest.mark.slow()
    def test_live_pipeline_params(self) -> None:
        from trading_bot.backtest.pipeline import build_live_engine

        optimizer = StrategyOptimizer(
            param_space={"adx_min": (15.0, 35.0), "ema_fast": (5, 12)},
            engine_factory=partial(build_live_engine, "BTC/USDT"),
            n_trials=2,
            seed=0,
        )
        result = optimizer.walk_forward(
            _make_ohlcv(700),
            "BTC/USDT",
            validator=WalkForwardValidator(n_splits=2, train_ratio=0.6),
        )
        assert len(result.fold_results) == 2
        assert all(isinstance(v, float) for v in result.parameter_stability["ema_fast"])


def test_all_trials_invalid_raises_clear_error() -> None:
    def factory(threshold: float) -> _ParamStrategy:
        raise ValueError("inconsistent")

    optimizer = StrategyOptimizer(
        param_space={"threshold": (0.0, 1.0)},
        strategy_factory=factory,
        n_trials=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="no valid parameter combination"):
        optimizer.optimize(_make_ohlcv(50), "BTC/USDT")
