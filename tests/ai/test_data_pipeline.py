"""Tests for MLDataPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.data_pipeline import MLDataPipeline


@pytest.fixture()
def ohlcv_100() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
    )


class TestPrepare:
    def test_returns_aligned_features_labels(self, ohlcv_100: pd.DataFrame) -> None:
        pipe = MLDataPipeline()
        features, labels = pipe.prepare(ohlcv_100)
        assert len(features) == len(labels)
        assert len(features) > 0
        # No NaN in features
        assert not features.isna().any().any()

    def test_labels_are_binary(self, ohlcv_100: pd.DataFrame) -> None:
        pipe = MLDataPipeline()
        _, y = pipe.prepare(ohlcv_100)
        assert set(y.unique()).issubset({0, 1})


class TestTimeSeriesSplit:
    def test_preserves_order(self, ohlcv_100: pd.DataFrame) -> None:
        pipe = MLDataPipeline()
        features, labels = pipe.prepare(ohlcv_100)
        x_tr, _, x_val, _, x_te, _ = pipe.time_series_split(features, labels)
        assert x_tr.index[-1] < x_val.index[0]
        assert x_val.index[-1] < x_te.index[0]

    def test_approximate_ratios(self, ohlcv_100: pd.DataFrame) -> None:
        pipe = MLDataPipeline()
        features, labels = pipe.prepare(ohlcv_100)
        x_tr, _, x_val, _, x_te, _ = pipe.time_series_split(
            features, labels, train_ratio=0.7, val_ratio=0.15
        )
        total = len(features)
        assert len(x_tr) == pytest.approx(total * 0.7, abs=2)
        assert len(x_val) == pytest.approx(total * 0.15, abs=2)


class TestLabelHorizon:
    def test_bars_without_forward_return_are_dropped(
        self, ohlcv_100: pd.DataFrame
    ) -> None:
        # A monotonic rise: every labelled bar must be "up" (1). The last
        # `horizon` bars have no future close and must not appear as 0.
        rising = ohlcv_100.copy()
        rising["close"] = np.linspace(100.0, 400.0, len(rising))
        horizon = 3
        _, y = MLDataPipeline().prepare(rising, label_horizon=horizon)
        assert (y == 1).all()
        assert y.index[-1] == rising.index[-1 - horizon]

    def test_feature_columns_subset(self, ohlcv_100: pd.DataFrame) -> None:
        x, y = MLDataPipeline().prepare(ohlcv_100, feature_columns=["RSI_14"])
        assert list(x.columns) == ["RSI_14"]
        assert len(x) == len(y)
