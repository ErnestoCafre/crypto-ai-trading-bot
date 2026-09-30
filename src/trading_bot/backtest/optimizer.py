"""Strategy parameter optimizer using Optuna with nested walk-forward."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import pandas as pd

from trading_bot.backtest.data_types import BacktestResult, WalkForwardResult
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.walk_forward import (
    WalkForwardValidator,
    run_fold,
    summarize_folds,
)

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Optimize strategy parameters with Optuna.

    Build engines either from ``engine_factory(**params)`` — e.g.
    ``functools.partial(build_live_engine, "BTC/USDT")`` to tune the live
    pipeline — or from ``strategy_factory(**params)`` wrapped in a plain
    :class:`BacktestEngine`. ``param_space`` maps a name to ``(low, high)``;
    two ints are searched with ``suggest_int``, anything else as a float.

    :meth:`walk_forward` is the honest estimate: parameters are chosen on
    each fold's train window only and scored on the unseen test window.
    :meth:`optimize` searches in-sample over whatever data it is given — use
    it for the final fit after validating with :meth:`walk_forward`, never
    to report performance.
    """

    def __init__(
        self,
        param_space: dict[str, tuple[float, float]],
        strategy_factory: Callable[..., Any] | None = None,
        engine_factory: Callable[..., BacktestEngine] | None = None,
        n_trials: int = 100,
        metric: str = "sharpe_ratio",
        seed: int | None = None,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ) -> None:
        if (strategy_factory is None) == (engine_factory is None):
            raise ValueError("pass exactly one of strategy_factory / engine_factory")
        self._param_space = param_space
        self._strategy_factory = strategy_factory
        self._engine_factory = engine_factory
        self._n_trials = n_trials
        self._metric = metric
        self._seed = seed
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._slippage_pct = slippage_pct

    def build_engine(self, params: dict[str, Any]) -> BacktestEngine:
        if self._engine_factory is not None:
            return self._engine_factory(**params)
        assert self._strategy_factory is not None
        return BacktestEngine(
            strategy=self._strategy_factory(**params),
            initial_capital=self._initial_capital,
            fee_pct=self._fee_pct,
            slippage_pct=self._slippage_pct,
        )

    def optimize(self, ohlcv: pd.DataFrame, pair: str) -> dict[str, object]:
        """In-sample search over *ohlcv*.

        Returns ``{param_name: best_value, ..., "_best_metric": float}``.
        """
        best_params, best_value = self._search(ohlcv, pair)
        return {**best_params, "_best_metric": best_value}

    def walk_forward(
        self,
        ohlcv: pd.DataFrame,
        pair: str,
        validator: WalkForwardValidator | None = None,
    ) -> WalkForwardResult:
        """Nested walk-forward: optimize on each train window, score on test.

        ``parameter_stability`` holds the chosen value of every parameter per
        fold; each fold result's ``parameters`` includes its chosen values.
        """
        validator = validator or WalkForwardValidator(n_splits=3, train_ratio=0.7)
        fold_results: list[BacktestResult] = []
        chosen: dict[str, list[float]] = {name: [] for name in self._param_space}

        for fold in validator.folds(len(ohlcv)):
            train = ohlcv.iloc[fold.train_start : fold.train_end]
            params, in_sample = self._search(train, pair)
            result = run_fold(self.build_engine(params), ohlcv, fold, pair)
            result.parameters = {
                **result.parameters,
                **params,
                "in_sample_metric": in_sample,
            }
            fold_results.append(result)
            for name, value in params.items():
                chosen[name].append(float(value))
            logger.info(
                "Fold %s-%s: params=%s in-sample %s=%.3f, OOS %s=%.3f",
                fold.test_start,
                fold.test_end,
                params,
                self._metric,
                in_sample,
                self._metric,
                result.metrics.get(self._metric, 0.0),
            )

        return summarize_folds(fold_results, parameter_stability=chosen)

    def _search(self, ohlcv: pd.DataFrame, pair: str) -> tuple[dict[str, Any], float]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for name, (lo, hi) in self._param_space.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = trial.suggest_int(name, lo, hi)
                else:
                    params[name] = trial.suggest_float(name, float(lo), float(hi))
            try:
                engine = self.build_engine(params)
            except ValueError as exc:
                # Inconsistent combination (e.g. ema_fast >= ema_mid).
                raise optuna.TrialPruned(str(exc)) from exc
            value = engine.run(ohlcv, pair).metrics.get(self._metric, 0.0)
            return value if math.isfinite(value) else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._seed),
        )
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if not completed:
            raise ValueError(
                "no valid parameter combination in param_space "
                f"after {self._n_trials} trials"
            )
        return dict(study.best_params), float(study.best_value)
