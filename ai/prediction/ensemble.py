"""Ensemble predictor — stacking with meta-learner over base predictors."""

from __future__ import annotations

import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.prediction.base import BasePredictor
from core.models import Prediction

logger = logging.getLogger(__name__)


class EnsemblePredictor(BasePredictor):
    """Stacking ensemble: base predictors + meta-learner.

    Uses cross-validated out-of-fold predictions from base models to train
    the meta-learner, avoiding overfitting.
    """

    def __init__(
        self,
        base_predictors: list[BasePredictor],
        meta_learner: str = "logistic_regression",
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> None:
        self._bases = base_predictors
        self._meta_learner_type = meta_learner
        self._pair = pair
        self._timeframe = timeframe
        self._meta_model: Any = None
        self._trained = False

    def predict(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> Prediction:
        """Generate ensemble prediction by combining base outputs."""
        if not self._trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        base_probas: list[float] = []
        for base in self._bases:
            try:
                pred = base.predict(ohlcv, features)
                base_probas.append(pred.confidence)
            except Exception:
                base_probas.append(0.5)

        meta_input = np.array(base_probas).reshape(1, -1)
        if self._meta_model is not None:
            prob_up = float(self._meta_model.predict_proba(meta_input)[0, 1])
        else:
            prob_up = float(np.mean(base_probas))

        direction = "up" if prob_up > 0.5 else "down"

        return Prediction(
            pair=self._pair,
            timeframe=self._timeframe,
            predicted_return=(prob_up - 0.5) * 0.02,
            predicted_direction=direction,
            confidence=abs(prob_up - 0.5) * 2,
            model_type="ensemble",
            model_version="1.0",
            features_used=[],
            timestamp=datetime.now(UTC),
        )

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Train base models and meta-learner.

        *train_data* / *val_data*: last column is label.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import KFold

        x_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        x_val = val_data.iloc[:, :-1]
        y_val = val_data.iloc[:, -1]

        # Train base models
        for base in self._bases:
            base.train(train_data, val_data, hyperparameters)

        # Out-of-fold predictions for meta-learner training
        n_bases = len(self._bases)
        oof_preds = np.zeros((len(x_train), n_bases))

        kf = KFold(n_splits=min(3, len(x_train)), shuffle=False)
        for _, (tr_idx, v_idx) in enumerate(kf.split(x_train)):
            fold_train = train_data.iloc[tr_idx]
            fold_val = train_data.iloc[v_idx]
            fold_features = x_train.iloc[v_idx]

            for base_i, base in enumerate(self._bases):
                try:
                    base.train(fold_train, fold_val)
                    pred = base.predict(fold_features, fold_features)
                    oof_preds[v_idx, base_i] = pred.confidence
                except Exception:
                    oof_preds[v_idx, base_i] = 0.5

        # Re-train base models on full training data
        for base in self._bases:
            base.train(train_data, val_data, hyperparameters)

        # Train meta-learner
        self._meta_model = LogisticRegression(max_iter=1000)
        self._meta_model.fit(oof_preds, y_train)
        self._trained = True

        # Evaluate on validation set
        val_base_preds = np.zeros((len(x_val), n_bases))
        for base_i, base in enumerate(self._bases):
            try:
                pred = base.predict(x_val, x_val)
                val_base_preds[:, base_i] = pred.confidence
            except Exception:
                val_base_preds[:, base_i] = 0.5

        y_pred = self._meta_model.predict(val_base_preds)
        accuracy = float(accuracy_score(y_val, y_pred))

        return {"accuracy": accuracy}

    def load(self, model_path: str) -> None:
        """Load ensemble state from disk."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._meta_model = data["meta_model"]
        self._trained = data.get("trained", True)

    def save(self, model_path: str) -> None:
        """Save ensemble state to disk."""
        if not self._trained:
            raise RuntimeError("Ensemble not trained.")

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {"meta_model": self._meta_model, "trained": self._trained},
                f,
            )

    def is_loaded(self) -> bool:
        """Return True if ensemble is ready for prediction."""
        return self._trained
