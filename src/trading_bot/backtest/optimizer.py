"""Strategy parameter optimizer using Optuna with walk-forward validation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Optimize strategy parameters using Optuna.

    Uses walk-forward validation by default to prevent overfitting.
    """

    def __init__(
        self,
        strategy_factory: Callable[..., Any],
        param_space: dict[str, tuple[float, float]],
        n_trials: int = 100,
        metric: str = "sharpe_ratio",
        walk_forward: bool = True,
    ) -> None:
        self._factory = strategy_factory
        self._param_space = param_space
        self._n_trials = n_trials
        self._metric = metric
        self._walk_forward = walk_forward

    def optimize(
        self,
        ohlcv: pd.DataFrame,
        pair: str,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ) -> dict[str, object]:
        """Run Optuna optimization and return the best parameters.

        Returns a dict of ``{param_name: best_value, "_best_metric": float}``.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, float] = {}
            for name, (lo, hi) in self._param_space.items():
                params[name] = trial.suggest_float(name, lo, hi)

            strategy = self._factory(**params)
            engine = BacktestEngine(
                strategy=strategy,
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                slippage_pct=slippage_pct,
            )

            if self._walk_forward:
                wf = WalkForwardValidator(n_splits=3, train_ratio=0.7)
                wf_result = wf.run(engine, ohlcv, pair)
                return wf_result.aggregate_metrics.get(self._metric, 0.0)
            else:
                result = engine.run(ohlcv, pair)
                return result.metrics.get(self._metric, 0.0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)

        best: dict[str, object] = dict(study.best_params)
        best["_best_metric"] = study.best_value
        return best
