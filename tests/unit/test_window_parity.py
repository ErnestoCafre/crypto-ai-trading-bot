"""Backtest (full history) vs live (OHLCV_LOOKBACK_BARS window) decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.window_parity import (
    Decision,
    WindowParityResult,
    _record,
    compare_live_window,
)
from trading_bot.config.settings import Settings


def _synth(n: int, seed: int = 11) -> pd.DataFrame:
    """Hourly GBM with trending, flat and falling stretches (not market data)."""
    rng = np.random.default_rng(seed)
    per_year = 365 * 24
    sigma = 0.65 / np.sqrt(per_year)
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.5], size=n // 60 + 1), 60)[:n]
    close = 30000 * np.exp(np.cumsum(rng.normal(regime * 40 / per_year, sigma)))
    open_ = np.r_[30000, close[:-1]] * np.exp(rng.normal(0, sigma / 4, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sigma / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sigma / 2, n)))
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_live_window_matches_full_history(settings: Settings) -> None:
    """With the default window, the live bot decides what the backtest did.

    Bars are sampled (step) to keep the test fast; scripts/window_parity.py
    checks every bar of real candles.
    """
    ohlcv = _synth(1200)
    res = compare_live_window(ohlcv, "BTC/USDT", settings, step=17)
    assert res.window == settings.OHLCV_LOOKBACK_BARS
    assert res.checked == len(range(res.window - 1, len(ohlcv), 17))
    assert res.ok, res.as_dict()
    assert res.max_confidence_diff < 1e-3
    assert res.max_atr_rel_diff < 1e-3


def test_window_shorter_than_the_lookback_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="OHLCV_LOOKBACK_BARS"):
        compare_live_window(_synth(400), "BTC/USDT", settings, window=150)


def test_history_shorter_than_the_window_checks_nothing(settings: Settings) -> None:
    assert compare_live_window(_synth(200), "BTC/USDT", settings).checked == 0


def _d(**kw: object) -> Decision:
    base: dict[str, object] = {
        "signal": "long",
        "regime": "trending",
        "sizing_factor": None,
        "confidence": 0.7,
        "atr": 100.0,
    }
    base.update(kw)
    return Decision(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("live", "fields", "numeric_only"),
    [
        (_d(signal=None, confidence=None), ["signal"], False),
        (_d(regime="volatile", sizing_factor=0.5), ["regime", "sizing_factor"], False),
        (_d(confidence=0.71), ["confidence"], True),
        (_d(atr=100.5), ["atr"], True),
    ],
)
def test_mismatches_are_recorded(
    live: Decision, fields: list[str], numeric_only: bool
) -> None:
    res = WindowParityResult(window=250)
    _record(res, "t", _d(), live, confidence_tol=1e-3, atr_rel_tol=1e-3)
    assert not res.ok
    assert res.mismatches[0]["fields"] == fields
    assert res.numeric_only == int(numeric_only)


def test_within_tolerance_is_not_a_mismatch() -> None:
    res = WindowParityResult(window=250)
    _record(res, "t", _d(), _d(confidence=0.7000001, atr=100.00001), 1e-3, 1e-3)
    assert res.ok
    assert res.max_confidence_diff > 0
