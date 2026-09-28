"""Tests for LightGBMPredictor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor
from trading_bot.core.models import Prediction


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


# ---------------------------------------------------------------------------
# Inference correctness
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> tuple[LightGBMPredictor, pd.DataFrame]:
    model = LightGBMPredictor(params={"early_stopping_rounds": 0, "n_estimators": 50})
    train, val = _make_train_val()
    model.train(train, val)
    return model, val.iloc[:, :-1]


class TestLightGBMInference:
    def test_confidence_varies_between_zero_and_one(self, trained) -> None:
        model, feat = trained
        confidences = {
            round(model.predict(feat.iloc[: i + 1], feat.iloc[: i + 1]).confidence, 6)
            for i in range(len(feat))
        }
        # predict() used to return the class label, so confidence was 1.0.
        assert len(confidences) > 5
        assert all(0.0 <= c <= 1.0 for c in confidences)
        assert any(c < 0.99 for c in confidences)

    def test_uses_probability_of_up_class(self, trained) -> None:
        model, feat = trained
        pred = model.predict(feat, feat)
        prob_up = model._model.predict_proba(feat.iloc[[-1]])[0, 1]
        assert pred.confidence == pytest.approx(abs(prob_up - 0.5) * 2)
        assert pred.predicted_direction == ("up" if prob_up > 0.5 else "down")

    def test_columns_aligned_with_extra_and_shuffled_columns(self, trained) -> None:
        model, feat = trained
        messy = feat[list(reversed(feat.columns))].copy()
        messy["unrelated"] = 123.0
        assert model.predict(messy, messy).confidence == pytest.approx(
            model.predict(feat, feat).confidence
        )

    def test_nan_in_latest_row_means_no_prediction(self, trained) -> None:
        from trading_bot.ai.prediction.base import PredictionUnavailableError

        model, feat = trained
        bad = feat.copy()
        bad.iloc[-1, 2] = np.nan
        with pytest.raises(PredictionUnavailableError, match="NaN"):
            model.predict(bad, bad)

    def test_missing_column_means_no_prediction(self, trained) -> None:
        from trading_bot.ai.prediction.base import PredictionUnavailableError

        model, feat = trained
        with pytest.raises(PredictionUnavailableError, match="missing"):
            model.predict(feat, feat.drop(columns=["feat_0"]))

    def test_save_writes_checksum_and_load_verifies_it(
        self, trained, tmp_path: Path
    ) -> None:
        model, feat = trained
        path = tmp_path / "BTC_USDT_1h.pkl"
        model.save(str(path))
        sidecar = tmp_path / "BTC_USDT_1h.pkl.sha256"
        assert sidecar.exists()

        loaded = LightGBMPredictor()
        loaded.load(str(path))
        assert loaded.predict(feat, feat).confidence == pytest.approx(
            model.predict(feat, feat).confidence
        )

        path.write_bytes(path.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            LightGBMPredictor().load(str(path))


def test_model_path_is_sanitized() -> None:
    from trading_bot.ai.prediction.base import model_path

    assert model_path("var/m", "BTC/USDT", "1h") == Path("var/m/BTC_USDT_1h.pkl")
    assert model_path("d", "BTC/USDT:USDT", "4h").name == "BTC_USDT_USDT_4h.pkl"


def test_trainer_publishes_canonical_model(tmp_path: Path) -> None:
    from trading_bot.ai.prediction.base import model_path
    from trading_bot.ai.prediction.trainer import ModelTrainer

    model = LightGBMPredictor(params={"early_stopping_rounds": 0, "n_estimators": 10})
    train, val = _make_train_val()
    model.train(train, val)

    ModelTrainer(model_dir=str(tmp_path)).save_model_with_metadata(
        model, {"accuracy": 0.5}, "ETH/USDT", "4h"
    )

    assert model_path(tmp_path, "ETH/USDT", "4h").exists()
