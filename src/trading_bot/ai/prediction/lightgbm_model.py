"""LightGBM predictor — gradient-boosted tree model for price direction."""

from __future__ import annotations

import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.ai.prediction.base import BasePredictor
from trading_bot.core.models import Prediction

logger = logging.getLogger(__name__)


class LightGBMPredictor(BasePredictor):
    """LightGBM binary classifier for price direction prediction."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbose": -1,
    }

    def __init__(
        self,
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
        params: dict[str, Any] | None = None,
    ) -> None:
        self._pair = pair
        self._timeframe = timeframe
        self._params: dict[str, Any] = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Any = None
        self._feature_names: list[str] = []

    def predict(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> Prediction:
        """Predict price direction. Must be trained first."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        feat = features if features is not None else ohlcv
        proba = self._model.predict(feat)

        if hasattr(proba, "__len__") and len(proba) > 0:
            prob_up = float(proba[-1])
        else:
            prob_up = 0.5

        direction = "up" if prob_up > 0.5 else "down"
        predicted_return = (prob_up - 0.5) * 0.02

        return Prediction(
            pair=self._pair,
            timeframe=self._timeframe,
            predicted_return=predicted_return,
            predicted_direction=direction,
            confidence=abs(prob_up - 0.5) * 2,
            model_type="lightgbm",
            model_version="1.0",
            features_used=self._feature_names,
            timestamp=datetime.now(UTC),
        )

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Train the LightGBM model.

        Last column of *train_data* / *val_data* is the label.
        """
        import lightgbm as lgb

        params = {**self._params, **(hyperparameters or {})}

        x_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        x_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1]

        self._feature_names = list(x_train.columns)

        n_estimators = int(params.pop("n_estimators", 500))
        early_stopping = params.pop("early_stopping_rounds", 50)

        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        if early_stopping:
            callbacks.append(lgb.early_stopping(int(early_stopping)))

        model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=callbacks,
        )

        self._model = model

        from sklearn.metrics import (
            accuracy_score,
            log_loss,
            roc_auc_score,
        )

        y_pred_proba = model.predict_proba(x_val)[:, 1]
        y_pred = model.predict(x_val)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "log_loss": float(log_loss(y_val, y_pred_proba)),
        }
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_val, y_pred_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0

        return metrics

    def load(self, model_path: str) -> None:
        """Load a trained model from disk."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._model = data["model"]
        self._feature_names = data.get("feature_names", [])

    def save(self, model_path: str) -> None:
        """Save the trained model to disk."""
        if self._model is None:
            raise RuntimeError("No trained model to save.")

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "feature_names": self._feature_names,
                },
                f,
            )

    def is_loaded(self) -> bool:
        """Return True if a model is available for prediction."""
        return self._model is not None

    @property
    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        return dict(
            zip(
                self._feature_names,
                map(float, importances),
                strict=False,
            )
        )
