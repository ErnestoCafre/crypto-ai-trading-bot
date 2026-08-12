"""Abstract base class for price prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from trading_bot.core.models import Prediction


class BasePredictor(ABC):
    @abstractmethod
    def predict(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> Prediction: ...

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, float]: ...

    @abstractmethod
    def load(self, model_path: str) -> None: ...

    @abstractmethod
    def save(self, model_path: str) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...
