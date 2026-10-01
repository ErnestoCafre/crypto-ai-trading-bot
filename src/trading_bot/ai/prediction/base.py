"""Abstract base class for price prediction models."""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.core.models import Prediction

logger = logging.getLogger(__name__)


class PredictionUnavailableError(ValueError):
    """The model cannot produce a trustworthy prediction for this input.

    Raised for missing feature columns or NaN in the latest feature row;
    callers treat it as "no prediction", not as a neutral one.
    """


def model_path(model_dir: str | Path, pair: str, timeframe: str) -> Path:
    """Canonical per-pair model file: ``{dir}/BTC_USDT_1h.pkl``."""
    safe_pair = re.sub(r"[^A-Za-z0-9]+", "_", pair).strip("_")
    return Path(model_dir) / f"{safe_pair}_{timeframe}.pkl"


def checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_artifact(path: str | Path, data: dict[str, Any]) -> None:
    """Pickle ``data`` to ``path`` with a ``.sha256`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(data)
    path.write_bytes(payload)
    checksum_path(path).write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n", encoding="utf-8"
    )


def read_artifact(path: str | Path) -> dict[str, Any]:
    """Load a model artifact written by :func:`write_artifact`.

    Trust boundary: pickle executes code on load, so only load files this
    project produced from the local model dir, never downloaded or
    user-supplied ones. When a ``.sha256`` sidecar exists the file must
    match it (guards against corrupt/partial files).
    """
    path = Path(path)
    payload = path.read_bytes()
    sidecar = checksum_path(path)
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {path}")
    else:
        logger.warning("No checksum sidecar for %s; loading unverified", path)
    data = pickle.loads(payload)  # noqa: S301 — trusted local artifact
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a model artifact")
    return data


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
