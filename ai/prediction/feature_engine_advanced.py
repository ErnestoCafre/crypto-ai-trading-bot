"""Advanced feature engineering — statistical and derived features on OHLCV data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ai.prediction.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


class AdvancedFeatureEngine:
    """Extend base FeatureEngine with statistical/volatility features."""

    # --- Returns ---
    LOG_RETURN = "log_return"
    ROLLING_RETURN_5 = "rolling_return_5"
    ROLLING_RETURN_20 = "rolling_return_20"

    # --- Rolling Statistics ---
    ROLLING_SKEW_20 = "rolling_skew_20"
    ROLLING_KURT_20 = "rolling_kurt_20"
    LONGEST_STRIKE_BELOW_MEAN_20 = "longest_strike_below_mean_20"

    # --- Volatility ---
    REALIZED_VOL_20 = "realized_vol_20"
    PARKINSON_VOL_20 = "parkinson_vol_20"
    GARMAN_KLASS_VOL_20 = "garman_klass_vol_20"

    # --- Volume ---
    VOLUME_RATIO = "volume_ratio"
    VOLUME_ZSCORE_20 = "volume_zscore_20"

    # --- Price ---
    PRICE_DIST_SMA20 = "price_dist_sma20"
    HIGH_LOW_RANGE_PCT = "high_low_range_pct"
    CLOSE_POSITION_IN_RANGE = "close_position_in_range"

    def __init__(self, base_engine: FeatureEngine | None = None) -> None:
        self._base = base_engine or FeatureEngine()

    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *ohlcv* enriched with base + advanced features.

        Calls base_engine.build_features() first, then adds advanced features.
        Does **not** drop NaN rows — the caller decides.
        """
        missing = _REQUIRED_COLS - set(ohlcv.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        df = self._base.build_features(ohlcv)

        self._try(lambda: self._add_returns(df), "returns")
        self._try(lambda: self._add_rolling_stats(df), "rolling_stats")
        self._try(lambda: self._add_volatility(df), "volatility")
        self._try(lambda: self._add_volume(df), "volume")
        self._try(lambda: self._add_price(df), "price")

        return df

    def get_feature_names(self) -> list[str]:
        """Return a list of all advanced feature column names."""
        return [
            self.LOG_RETURN,
            self.ROLLING_RETURN_5,
            self.ROLLING_RETURN_20,
            self.ROLLING_SKEW_20,
            self.ROLLING_KURT_20,
            self.LONGEST_STRIKE_BELOW_MEAN_20,
            self.REALIZED_VOL_20,
            self.PARKINSON_VOL_20,
            self.GARMAN_KLASS_VOL_20,
            self.VOLUME_RATIO,
            self.VOLUME_ZSCORE_20,
            self.PRICE_DIST_SMA20,
            self.HIGH_LOW_RANGE_PCT,
            self.CLOSE_POSITION_IN_RANGE,
        ]

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> None:
        close = df["close"]
        df[AdvancedFeatureEngine.LOG_RETURN] = np.log(close / close.shift(1))
        df[AdvancedFeatureEngine.ROLLING_RETURN_5] = (
            df[AdvancedFeatureEngine.LOG_RETURN].rolling(5).sum()
        )
        df[AdvancedFeatureEngine.ROLLING_RETURN_20] = (
            df[AdvancedFeatureEngine.LOG_RETURN].rolling(20).sum()
        )

    @staticmethod
    def _add_rolling_stats(df: pd.DataFrame) -> None:
        log_ret = df.get(AdvancedFeatureEngine.LOG_RETURN)
        if log_ret is None:
            close = df["close"]
            log_ret = np.log(close / close.shift(1))

        df[AdvancedFeatureEngine.ROLLING_SKEW_20] = log_ret.rolling(20).skew()
        df[AdvancedFeatureEngine.ROLLING_KURT_20] = log_ret.rolling(20).kurt()

        # Longest strike below mean in rolling window of 20
        df[AdvancedFeatureEngine.LONGEST_STRIKE_BELOW_MEAN_20] = log_ret.rolling(
            20
        ).apply(_longest_strike_below_mean, raw=True)

    @staticmethod
    def _add_volatility(df: pd.DataFrame) -> None:
        log_ret = df.get(AdvancedFeatureEngine.LOG_RETURN)
        if log_ret is None:
            close = df["close"]
            log_ret = np.log(close / close.shift(1))

        # Realized volatility (annualized)
        df[AdvancedFeatureEngine.REALIZED_VOL_20] = log_ret.rolling(20).std() * np.sqrt(
            252
        )

        # Parkinson volatility
        high = df["high"]
        low = df["low"]
        hl_log = np.log(high / low)
        df[AdvancedFeatureEngine.PARKINSON_VOL_20] = (
            hl_log.pow(2).rolling(20).mean() / (4 * np.log(2))
        ).apply(np.sqrt)

        # Garman-Klass volatility
        close = df["close"]
        open_ = df["open"]
        hl_term = 0.5 * hl_log.pow(2)
        co_term = (2 * np.log(2) - 1) * np.log(close / open_).pow(2)
        df[AdvancedFeatureEngine.GARMAN_KLASS_VOL_20] = (
            (hl_term - co_term).rolling(20).mean()
        ).apply(np.sqrt)

    @staticmethod
    def _add_volume(df: pd.DataFrame) -> None:
        vol = df["volume"]
        vol_sma = vol.rolling(20).mean()
        df[AdvancedFeatureEngine.VOLUME_RATIO] = vol / vol_sma

        vol_std = vol.rolling(20).std()
        df[AdvancedFeatureEngine.VOLUME_ZSCORE_20] = (vol - vol_sma) / vol_std

    @staticmethod
    def _add_price(df: pd.DataFrame) -> None:
        close = df["close"]
        sma20 = close.rolling(20).mean()
        df[AdvancedFeatureEngine.PRICE_DIST_SMA20] = (close - sma20) / sma20

        high = df["high"]
        low = df["low"]
        df[AdvancedFeatureEngine.HIGH_LOW_RANGE_PCT] = (high - low) / low

        hl_range = high - low
        df[AdvancedFeatureEngine.CLOSE_POSITION_IN_RANGE] = (close - low) / hl_range

    @staticmethod
    def _try(fn: object, label: str) -> None:
        """Call *fn*; swallow exceptions so a short DataFrame doesn't crash."""
        try:
            fn()  # type: ignore[operator]
        except Exception:
            logger.warning("AdvancedFeatureEngine: could not compute %s", label)


def _longest_strike_below_mean(values: np.ndarray) -> float:
    """Count the longest consecutive run of values below the window mean."""
    if len(values) == 0:
        return 0.0
    mean = np.nanmean(values)
    below = values < mean
    max_strike = 0
    current = 0
    for b in below:
        if b:
            current += 1
            if current > max_strike:
                max_strike = current
        else:
            current = 0
    return float(max_strike)
