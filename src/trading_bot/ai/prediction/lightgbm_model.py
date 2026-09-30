"""LightGBM predictor — gradient-boosted tree model for price direction."""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.base import BasePredictor, PredictionUnavailableError
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
        """Predict the direction of the next bar from the latest feature row.

        Features are aligned to the training columns (extra columns are
        dropped, order restored). Raises PredictionUnavailableError when a
        training column is missing or the latest row has NaN.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        if not self._feature_names:
            raise PredictionUnavailableError("model has no recorded feature names")

        feat = features if features is not None else ohlcv
        if feat.empty:
            raise PredictionUnavailableError("no feature rows")
        missing = [c for c in self._feature_names if c not in feat.columns]
        if missing:
            raise PredictionUnavailableError(f"missing feature columns: {missing}")
        latest = feat[self._feature_names].iloc[[-1]]
        nan_cols = latest.columns[latest.isna().iloc[0]].tolist()
        if nan_cols:
            raise PredictionUnavailableError(f"NaN in latest row: {nan_cols}")

        # LGBMClassifier.predict returns the class (0/1); the probability of
        # the "up" class is what confidence must be derived from.
        prob_up = float(self._model.predict_proba(latest)[0, 1])

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
            eval_X=(x_val,),
            eval_y=(y_val,),
            callbacks=callbacks,
        )

        self._model = model

        from sklearn.metrics import (
            accuracy_score,
            log_loss,
            roc_auc_score,
        )

        y_pred_proba = np.asarray(model.predict_proba(x_val))[:, 1]
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
        """Load a trained model from disk.

        Trust boundary: pickle executes code on load, so only load files this
        project produced (save()/ModelTrainer) from the local model dir,
        never downloaded or user-supplied ones. When a ``.sha256`` sidecar
        exists the file must match it (guards against corrupt/partial files).
        """
        path = Path(model_path)
        payload = path.read_bytes()
        sidecar = _checksum_path(path)
        if sidecar.exists():
            expected = sidecar.read_text(encoding="utf-8").split()[0]
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {path}")
        else:
            logger.warning("No checksum sidecar for %s; loading unverified", path)
        data = pickle.loads(payload)  # noqa: S301 — trusted local artifact
        self._model = data["model"]
        self._feature_names = list(data.get("feature_names", []))

    def save(self, model_path: str) -> None:
        """Save the trained model to disk."""
        if self._model is None:
            raise RuntimeError("No trained model to save.")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(
            {
                "model": self._model,
                "feature_names": self._feature_names,
            }
        )
        path.write_bytes(payload)
        _checksum_path(path).write_text(
            f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n", encoding="utf-8"
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


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")
