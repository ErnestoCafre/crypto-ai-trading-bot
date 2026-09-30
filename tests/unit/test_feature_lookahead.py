"""Anti look-ahead tests: features at bar i must not depend on bars after i.

The backtest engine builds features once over the whole history and slices
by index; that is only valid if every indicator is causal. These tests also
protect the LightGBM training features (``MLDataPipeline``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.feature_engine_advanced import AdvancedFeatureEngine


def _ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30_000 + np.cumsum(rng.normal(0, 80, n))
    open_ = close + rng.normal(0, 20, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + abs(rng.normal(0, 60, n)),
            "low": np.minimum(open_, close) - abs(rng.normal(0, 60, n)),
            "close": close,
            "volume": abs(rng.normal(1_000, 200, n)),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


def _perturb_after(df: pd.DataFrame, i: int) -> pd.DataFrame:
    """Copy of *df* with every bar after *i* replaced by a different path."""
    out = df.copy()
    rng = np.random.default_rng(99)
    tail = len(df) - i - 1
    shock = 1.0 + rng.uniform(0.05, 0.5, tail)
    for col in ("open", "high", "low", "close"):
        out.iloc[i + 1 :, out.columns.get_loc(col)] = df[col].iloc[i + 1 :] * shock
    out.iloc[i + 1 :, out.columns.get_loc("volume")] = df["volume"].iloc[
        i + 1 :
    ] * rng.uniform(2, 5, tail)
    return out


ENGINES = [FeatureEngine(), AdvancedFeatureEngine()]


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("i", [120, 250, 398])
def test_features_unchanged_when_future_bars_change(
    engine: FeatureEngine | AdvancedFeatureEngine, i: int
) -> None:
    df = _ohlcv()
    base = engine.build_features(df)
    moved = engine.build_features(_perturb_after(df, i))

    assert list(base.columns) == list(moved.columns)
    pd.testing.assert_frame_equal(
        base.iloc[: i + 1], moved.iloc[: i + 1], check_exact=False, rtol=1e-9
    )


@pytest.mark.parametrize("i", [150, 300])
def test_full_history_row_equals_prefix_row(i: int) -> None:
    """Row i of the full-history build equals the last row of a prefix build.

    This is what lets the backtest precompute features once and slice.
    """
    df = _ohlcv()
    fe = FeatureEngine()
    full = fe.build_features(df)
    prefix = fe.build_features(df.iloc[: i + 1])

    # Too-short prefixes omit long indicators (EMA_200...) instead of NaN.
    for col in set(full.columns) - set(prefix.columns):
        assert pd.isna(full[col].iloc[i]), col
    assert set(prefix.columns) <= set(full.columns)

    pd.testing.assert_series_equal(
        full[prefix.columns].iloc[i], prefix.iloc[-1], check_exact=False, rtol=1e-9
    )


def test_no_chikou_span() -> None:
    """Ichimoku's chikou span is close shifted back in time (a future close)."""
    cols = FeatureEngine().build_features(_ohlcv()).columns
    assert not [c for c in cols if c.startswith("ICS")]
    assert "ISA_9" in cols  # the causal Ichimoku lines are still there
