"""Model trainer — training, evaluation, and Optuna hyperparameter tuning."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_bot.ai.prediction.base import BasePredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Wrapper for training, evaluating, and persisting prediction models."""

    def __init__(self, model_dir: str = "var/models/prediction") -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(
        self,
        model: BasePredictor,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Train the model and return validation metrics."""
        metrics = model.train(train_data, val_data)
        logger.info("Training complete. Metrics: %s", metrics)
        return metrics

    def train_with_optuna(
        self,
        model_class: type[BasePredictor],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        n_trials: int = 50,
    ) -> tuple[BasePredictor, dict[str, float]]:
        """Tune hyperparameters with Optuna and return the best model + metrics."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_model: BasePredictor | None = None
        best_metrics: dict[str, float] = {}
        best_score: float = -float("inf")

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_model, best_metrics, best_score

            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            }

            model = model_class()
            metrics = model.train(train_data, val_data, params)

            score = metrics.get("accuracy", 0.0)
            if score > best_score:
                best_score = score
                best_model = model
                best_metrics = metrics

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if best_model is None:
            best_model = model_class()
            best_metrics = best_model.train(train_data, val_data)

        return best_model, best_metrics

    def save_model_with_metadata(
        self,
        model: BasePredictor,
        metrics: dict[str, float],
        pair: str,
        timeframe: str,
    ) -> str:
        """Save model and metadata, return the model path."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_name = type(model).__name__.lower()
        fname = f"{model_name}_{pair.replace('/', '_')}_{timestamp}.pkl"
        model_path = str(self._model_dir / fname)

        model.save(model_path)

        # Save metadata alongside
        meta_path = model_path.replace(".pkl", "_meta.json")
        metadata = {
            "model_type": model_name,
            "pair": pair,
            "timeframe": timeframe,
            "metrics": metrics,
            "timestamp": timestamp,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to %s", model_path)
        return model_path
