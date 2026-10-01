"""Scale-free features for the meta-label model.

Every feature is a ratio, an oscillator or a primary-model output, so one
model can be shared across pairs and price regimes. All of them depend on at
most ~110 bars of history (EMA50 slope, 100-bar volume/ATR means, crossing
age capped at ``CROSS_AGE_CAP``), so the live bot reproduces them from its
``OHLCV_LOOKBACK_BARS`` (250) window; ``tests/ai/test_meta_label.py`` and the
training script's parity check verify it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine

CROSS_AGE_CAP = 100

PRICE_FEATURES = (
    "ret_1",
    "ret_6",
    "ret_24",
    "dist_ema9",
    "dist_ema21",
    "dist_ema50",
    "ema9_21",
    "ema21_50",
    "slope_ema21",
    "slope_ema50",
    "atr_pct",
    "atr_rel100",
    "rvol_24",
    "rsi",
    "adx",
    "di_spread",
    "macdh_pct",
    "bbb",
    "bbp",
    "stoch_k",
    "cci",
    "willr",
    "vol_rel20",
    "vol_rel100",
    "cross_age",
)
SIGNAL_FEATURES = (
    "tech_confidence",
    "strategy",
    "regime",
    "regime_confidence",
    "volatility_percentile",
    "sizing_factor",
)


def price_features(features: pd.DataFrame) -> pd.DataFrame:
    """Scale-free transforms of a ``FeatureEngine.build_features`` frame."""
    f = features
    close = f["close"]
    log_close = np.log(close)
    ema9, ema21, ema50 = f["EMA_9"], f["EMA_21"], f["EMA_50"]
    atr = f[FeatureEngine.ATR_14]
    dmp, dmn = f[FeatureEngine.DMP_14], f[FeatureEngine.DMN_14]
    above = (ema9 > ema21).astype(int)
    # Bars since EMA9 last crossed above EMA21 (0 while below), capped.
    run = above.groupby((above != above.shift()).cumsum()).cumsum()
    out = pd.DataFrame(
        {
            "ret_1": log_close.diff(1),
            "ret_6": log_close.diff(6),
            "ret_24": log_close.diff(24),
            "dist_ema9": close / ema9 - 1,
            "dist_ema21": close / ema21 - 1,
            "dist_ema50": close / ema50 - 1,
            "ema9_21": ema9 / ema21 - 1,
            "ema21_50": ema21 / ema50 - 1,
            "slope_ema21": ema21 / ema21.shift(5) - 1,
            "slope_ema50": ema50 / ema50.shift(10) - 1,
            "atr_pct": atr / close,
            "atr_rel100": atr / atr.rolling(100).mean(),
            "rvol_24": log_close.diff().rolling(24).std(),
            "rsi": f[FeatureEngine.RSI_14],
            "adx": f[FeatureEngine.ADX_14],
            "di_spread": (dmp - dmn) / (dmp + dmn),
            "macdh_pct": f[FeatureEngine.MACDh_12_26_9] / close,
            "bbb": f["BBB_20_2.0"],
            "bbp": f["BBP_20_2.0"],
            "stoch_k": f[FeatureEngine.STOCHk],
            "cci": f[FeatureEngine.CCI_20],
            "willr": f[FeatureEngine.WILLR_14],
            "vol_rel20": f["volume"] / f["VOL_SMA_20"],
            "vol_rel100": f["volume"] / f["volume"].rolling(100).mean(),
            "cross_age": run.clip(upper=CROSS_AGE_CAP).astype(float),
        },
        index=f.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def signal_features(signals: pd.DataFrame) -> pd.DataFrame:
    """Primary-model columns of ``meta_events.technical_signals``."""
    return pd.DataFrame(
        {
            "tech_confidence": signals["confidence"].astype(float),
            "strategy": signals["strategy"].astype(float),
            "regime": signals["regime"].astype(float),
            "regime_confidence": signals["regime_confidence"].astype(float),
            "volatility_percentile": signals["volatility_percentile"].astype(float),
            "sizing_factor": signals["sizing_factor"].astype(float),
        },
        index=signals.index,
    )


def meta_feature_frame(
    features: pd.DataFrame, signals: pd.DataFrame, pair_code: int
) -> pd.DataFrame:
    """Full feature matrix (one row per bar) for the model."""
    out = pd.concat(
        [price_features(features), signal_features(signals)], axis=1, join="inner"
    )
    out["pair_code"] = float(pair_code)
    return out
