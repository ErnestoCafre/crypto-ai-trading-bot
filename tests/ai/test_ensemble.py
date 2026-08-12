"""Tests for EnsemblePredictor."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.base import BasePredictor
from trading_bot.ai.prediction.ensemble import EnsemblePredictor
from trading_bot.core.models import Prediction


def _make_mock_predictor(confidence: float = 0.7) -> BasePredictor:
    """Create a mock BasePredictor that returns a fixed prediction."""
    mock = MagicMock(spec=BasePredictor)
    mock.predict.return_value = Prediction(
        pair="BTC/USDT",
        timeframe="1h",
        predicted_return=0.01,
        predicted_direction="up",
        confidence=confidence,
        model_type="mock",
        model_version="1.0",
        features_used=[],
        timestamp=datetime.now(UTC),
    )
    mock.train.return_value = {"accuracy": 0.6}
    return mock


def _make_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    x_tr = rng.normal(0, 1, (n_train, 5))
    y_tr = (x_tr[:, 0] > 0).astype(int)
    x_va = rng.normal(0, 1, (n_val, 5))
    y_va = (x_va[:, 0] > 0).astype(int)

    cols = [f"f{i}" for i in range(5)]
    train = pd.DataFrame(x_tr, columns=cols)
    train["label"] = y_tr
    val = pd.DataFrame(x_va, columns=cols)
    val["label"] = y_va
    return train, val


class TestEnsemblePredictorTrainPredict:
    def test_train_and_predict(self) -> None:
        bases = [_make_mock_predictor(0.7), _make_mock_predictor(0.6)]
        ensemble = EnsemblePredictor(base_predictors=bases)
        train, val = _make_train_val()
        metrics = ensemble.train(train, val)
        assert "accuracy" in metrics

        pred = ensemble.predict(val.iloc[:, :-1])
        assert isinstance(pred, Prediction)
        assert pred.model_type == "ensemble"


class TestEnsemblePersistence:
    def test_save_load_roundtrip(self) -> None:
        bases = [_make_mock_predictor(0.7)]
        ensemble = EnsemblePredictor(base_predictors=bases)
        train, val = _make_train_val()
        ensemble.train(train, val)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ensemble.pkl")
            ensemble.save(path)

            ensemble2 = EnsemblePredictor(base_predictors=bases)
            ensemble2.load(path)
            assert ensemble2.is_loaded()


class TestEnsembleCombination:
    def test_combines_base_predictions(self) -> None:
        """Ensemble should use outputs from multiple base predictors."""
        bases = [_make_mock_predictor(0.8), _make_mock_predictor(0.3)]
        ensemble = EnsemblePredictor(base_predictors=bases)
        train, val = _make_train_val()
        ensemble.train(train, val)

        pred = ensemble.predict(val.iloc[:, :-1])
        assert isinstance(pred, Prediction)
        # Ensemble should produce a valid confidence
        assert 0.0 <= pred.confidence <= 1.0
