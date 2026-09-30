"""Feature engineering — computes technical indicators on OHLCV data."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable

import pandas as pd
import pandas_ta_classic as ta  # noqa: F401 – registers .ta accessor

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


class FeatureEngine:
    """Compute ~20 technical indicators on an OHLCV DataFrame."""

    # Column-name constants for external consumers.
    RSI_14 = "RSI_14"
    MACD_12_26_9 = "MACD_12_26_9"
    MACDh_12_26_9 = "MACDh_12_26_9"
    MACDs_12_26_9 = "MACDs_12_26_9"
    BBL_20 = "BBL_20_2.0"
    BBM_20 = "BBM_20_2.0"
    BBU_20 = "BBU_20_2.0"
    ATR_14 = "ATRr_14"
    EMA_9 = "EMA_9"
    EMA_21 = "EMA_21"
    EMA_50 = "EMA_50"
    EMA_200 = "EMA_200"
    SMA_20 = "SMA_20"
    SMA_50 = "SMA_50"
    SMA_200 = "SMA_200"
    ADX_14 = "ADX_14"
    DMP_14 = "DMP_14"
    DMN_14 = "DMN_14"
    STOCHk = "STOCHk_14_3_3"
    STOCHd = "STOCHd_14_3_3"
    OBV = "OBV"
    CCI_20 = "CCI_20_0.015"
    WILLR_14 = "WILLR_14"
    VOL_SMA_20 = "VOL_SMA_20"

    DEFAULT_EMA_PERIODS = (9, 21, 50, 200)

    def __init__(self, extra_ema_periods: Iterable[int] = ()) -> None:
        self._ema_periods = tuple(
            sorted({*self.DEFAULT_EMA_PERIODS, *(int(p) for p in extra_ema_periods)})
        )

    @staticmethod
    def ema_column(period: int) -> str:
        return f"EMA_{int(period)}"

    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *ohlcv* enriched with technical-indicator columns.

        Every indicator is causal: row *i* depends only on rows ``<= i``, so
        the frame can be built once over a whole backtest and sliced by
        index. Does **not** drop NaN rows — the caller decides.
        """
        missing = _REQUIRED_COLS - set(ohlcv.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

        df = ohlcv.copy()

        self._try(lambda: df.ta.rsi(length=14, append=True), "RSI")
        self._try(lambda: df.ta.macd(fast=12, slow=26, signal=9, append=True), "MACD")
        self._try(lambda: df.ta.bbands(length=20, std=2, append=True), "BBands")
        self._try(lambda: df.ta.atr(length=14, append=True), "ATR")

        for period in self._ema_periods:
            self._try(
                lambda p=period: df.ta.ema(length=p, append=True), f"EMA_{period}"
            )

        for period in (20, 50, 200):
            self._try(
                lambda p=period: df.ta.sma(length=p, append=True), f"SMA_{period}"
            )

        self._try(lambda: df.ta.adx(length=14, append=True), "ADX")
        self._try(lambda: df.ta.stoch(k=14, d=3, smooth_k=3, append=True), "Stochastic")
        self._try(lambda: df.ta.obv(append=True), "OBV")

        if isinstance(df.index, pd.DatetimeIndex):
            # pandas-ta anchors VWAP on index.to_period("D"), which drops the
            # timezone; UTC wall-clock days are the intended anchor.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Converting to Period")
                self._try(lambda: df.ta.vwap(append=True), "VWAP")

        self._try(lambda: df.ta.cci(length=20, append=True), "CCI")
        self._try(lambda: df.ta.willr(length=14, append=True), "Williams %R")

        if len(df) >= 100:
            # lookahead=False drops the chikou span (ICS_26 = close shifted
            # 26 bars *back*, i.e. a future close): every feature at bar i
            # must depend only on bars <= i.
            self._try(lambda: df.ta.ichimoku(lookahead=False, append=True), "Ichimoku")

        # Volume SMA
        self._try(lambda: _add_vol_sma(df, 20), "VOL_SMA_20")

        return df

    @staticmethod
    def _try(fn: object, label: str) -> None:
        """Call *fn*; swallow exceptions so a short DataFrame doesn't crash."""
        try:
            fn()  # type: ignore[operator]
        except Exception:
            logger.warning("FeatureEngine: could not compute %s", label)


class PrecomputedFeatureEngine(FeatureEngine):
    """Feature engine that computes a whole history once and slices it.

    :meth:`prime` builds features over the full backtest history; afterwards
    ``build_features(history.iloc[:i + 1])`` returns row-sliced features in
    O(1) instead of recomputing every indicator over a growing window. This
    is only equivalent because every indicator is causal (see
    ``tests/unit/test_feature_lookahead.py``). Frames that are not a prefix of
    the primed history fall back to a full computation. The returned slice is
    shared: callers must not mutate it.
    """

    def __init__(self, extra_ema_periods: Iterable[int] = ()) -> None:
        super().__init__(extra_ema_periods)
        self._full: pd.DataFrame | None = None

    def prime(self, ohlcv: pd.DataFrame) -> None:
        self._full = super().build_features(ohlcv)

    def clear(self) -> None:
        self._full = None

    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        full = self._full
        n = len(ohlcv)
        if (
            full is not None
            and 0 < n <= len(full)
            and ohlcv.index[0] == full.index[0]
            and ohlcv.index[-1] == full.index[n - 1]
            and ohlcv["close"].iloc[-1] == full["close"].iloc[n - 1]
        ):
            return full.iloc[:n]
        return super().build_features(ohlcv)


def _add_vol_sma(df: pd.DataFrame, period: int) -> None:
    df["VOL_SMA_20"] = df["volume"].rolling(window=period).mean()
