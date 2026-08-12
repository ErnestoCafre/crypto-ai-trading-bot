"""GRU predictor stub."""

from __future__ import annotations

import pandas as pd

from trading_bot.ai.prediction.base import BasePredictor
from trading_bot.core.models import Prediction


class GRUPredictor(BasePredictor):
    def predict(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> Prediction:
        raise NotImplementedError

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hyperparameters: dict[str, object] | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError

    def load(self, model_path: str) -> None:
        raise NotImplementedError

    def save(self, model_path: str) -> None:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError
