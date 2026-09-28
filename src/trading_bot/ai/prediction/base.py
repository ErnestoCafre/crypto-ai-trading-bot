"""Abstract base class for price prediction models."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.core.models import Prediction


class PredictionUnavailableError(ValueError):
    """The model cannot produce a trustworthy prediction for this input.

    Raised for missing feature columns or NaN in the latest feature row;
    callers treat it as "no prediction", not as a neutral one.
    """


def model_path(model_dir: str | Path, pair: str, timeframe: str) -> Path:
    """Canonical per-pair model file: ``{dir}/BTC_USDT_1h.pkl``."""
    safe_pair = re.sub(r"[^A-Za-z0-9]+", "_", pair).strip("_")
    return Path(model_dir) / f"{safe_pair}_{timeframe}.pkl"


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
