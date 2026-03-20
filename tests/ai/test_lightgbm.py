"""Tests for LightGBMPredictor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.prediction.lightgbm_model import LightGBMPredictor
from core.models import Prediction


def _make_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create simple train/val DataFrames (last column = label)."""
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    n_features = 5

    x_tr = rng.normal(0, 1, (n_train, n_features))
    y_tr = (x_tr[:, 0] + x_tr[:, 1] > 0).astype(int)
    x_va = rng.normal(0, 1, (n_val, n_features))
    y_va = (x_va[:, 0] + x_va[:, 1] > 0).astype(int)

    cols = [f"feat_{i}" for i in range(n_features)]
    train = pd.DataFrame(x_tr, columns=cols)
    train["label"] = y_tr
    val = pd.DataFrame(x_va, columns=cols)
    val["label"] = y_va

    return train, val


class TestLightGBMTrain:
    def test_train_returns_metrics(self) -> None:
        model = LightGBMPredictor()
        train, val = _make_train_val()
        metrics = model.train(train, val)
        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert metrics["accuracy"] > 0

    def test_predict_returns_prediction(self) -> None:
        model = LightGBMPredictor()
        train, val = _make_train_val()
        model.train(train, val)

        feat = val.iloc[:, :-1]
        pred = model.predict(feat, feat)
        assert isinstance(pred, Prediction)
        assert pred.predicted_direction in ("up", "down")
        assert 0.0 <= pred.confidence <= 1.0

    def test_predict_before_train_raises(self) -> None:
        model = LightGBMPredictor()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(df)


class TestLightGBMPersistence:
    def test_save_load_roundtrip(self) -> None:
        model = LightGBMPredictor()
        train, val = _make_train_val()
        model.train(train, val)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pkl")
            model.save(path)

            model2 = LightGBMPredictor()
            assert not model2.is_loaded()
            model2.load(path)
            assert model2.is_loaded()

            feat = val.iloc[:, :-1]
            pred = model2.predict(feat, feat)
            assert isinstance(pred, Prediction)


class TestLightGBMFeatureImportance:
    def test_feature_importance_populated(self) -> None:
        model = LightGBMPredictor()
        train, val = _make_train_val()
        model.train(train, val)
        fi = model.feature_importance
        assert isinstance(fi, dict)
        assert len(fi) > 0


class TestLightGBMIsLoaded:
    def test_is_loaded_false_initially(self) -> None:
        model = LightGBMPredictor()
        assert not model.is_loaded()

    def test_is_loaded_true_after_train(self) -> None:
        model = LightGBMPredictor()
        train, val = _make_train_val()
        model.train(train, val)
        assert model.is_loaded()
