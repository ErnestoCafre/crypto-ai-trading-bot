"""Tests for TrendFollowing and MeanReversion strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.config.settings import Settings
from trading_bot.core.models import Signal, SignalType
from trading_bot.core.strategies.base import graded_confidence
from trading_bot.core.strategies.mean_reversion import MeanReversion
from trading_bot.core.strategies.trend_following import TrendFollowing


class TestTrendFollowing:
    def test_long_on_bullish_data(
        self, mock_feature_engine: MagicMock, bullish_ohlcv: pd.DataFrame
    ) -> None:
        strat = TrendFollowing(pair="BTC/USDT", feature_engine=mock_feature_engine)
        sig = strat.generate_signal(bullish_ohlcv)
        assert sig is not None
        assert sig.type == SignalType.LONG
        assert sig.pair == "BTC/USDT"
        assert 0.0 <= sig.confidence <= 1.0

    def test_close_on_bearish_data(
        self, mock_feature_engine: MagicMock, bearish_ohlcv: pd.DataFrame
    ) -> None:
        strat = TrendFollowing(pair="BTC/USDT", feature_engine=mock_feature_engine)
        sig = strat.generate_signal(bearish_ohlcv)
        assert sig is not None
        assert sig.type == SignalType.CLOSE

    def test_none_on_flat_data(self, mock_feature_engine: MagicMock) -> None:
        """When EMA alignment holds but RSI and ADX are borderline, no signal."""
        row: dict[str, float] = {
            "open": 50000,
            "high": 51000,
            "low": 49000,
            "close": 50000,
            "volume": 1000,
            FeatureEngine.EMA_9: 50100,
            FeatureEngine.EMA_21: 50050,
            FeatureEngine.EMA_50: 50000,
            FeatureEngine.RSI_14: 50,  # exactly 50 — passes >50? No.
            FeatureEngine.MACD_12_26_9: 100,
            FeatureEngine.MACDs_12_26_9: 80,
            FeatureEngine.ADX_14: 20,  # < 25 → long fails; ema9>ema21 → no close
        }
        df = pd.DataFrame([row])
        strat = TrendFollowing(pair="BTC/USDT", feature_engine=mock_feature_engine)
        sig = strat.generate_signal(df)
        assert sig is None

    def test_lookback_200(self) -> None:
        strat = TrendFollowing(pair="BTC/USDT")
        assert strat.get_required_lookback() == 200


class TestMeanReversion:
    def test_long_on_oversold(
        self, mock_feature_engine: MagicMock, oversold_ohlcv: pd.DataFrame
    ) -> None:
        strat = MeanReversion(pair="BTC/USDT", feature_engine=mock_feature_engine)
        sig = strat.generate_signal(oversold_ohlcv)
        assert sig is not None
        assert sig.type == SignalType.LONG

    def test_short_on_overbought(
        self, mock_feature_engine: MagicMock, overbought_ohlcv: pd.DataFrame
    ) -> None:
        strat = MeanReversion(pair="BTC/USDT", feature_engine=mock_feature_engine)
        sig = strat.generate_signal(overbought_ohlcv)
        assert sig is not None
        assert sig.type == SignalType.SHORT

    def test_close_on_neutral_rsi(
        self, mock_feature_engine: MagicMock, neutral_rsi_ohlcv: pd.DataFrame
    ) -> None:
        strat = MeanReversion(pair="BTC/USDT", feature_engine=mock_feature_engine)
        sig = strat.generate_signal(neutral_rsi_ohlcv)
        assert sig is not None
        assert sig.type == SignalType.CLOSE

    def test_lookback_50(self) -> None:
        strat = MeanReversion(pair="BTC/USDT")
        assert strat.get_required_lookback() == 50


class TestStrategyParameters:
    def test_invalid_thresholds_rejected(self) -> None:
        import pytest

        from trading_bot.core.strategies.mean_reversion import MeanReversion
        from trading_bot.core.strategies.trend_following import TrendFollowing

        with pytest.raises(ValueError):
            TrendFollowing("BTC/USDT", ema_fast=21, ema_mid=21)
        with pytest.raises(ValueError):
            MeanReversion("BTC/USDT", rsi_oversold=45, rsi_exit_low=40)

    def test_non_default_emas_work_with_shared_engine(self) -> None:
        """A shared engine without the EMA periods falls back to computing them."""
        import numpy as np
        import pandas as pd

        from trading_bot.ai.prediction.feature_engine import FeatureEngine
        from trading_bot.core.strategies.trend_following import TrendFollowing

        n = 400
        close = 100 * np.exp(np.linspace(0, 0.5, n))  # steady uptrend
        ohlcv = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.002,
                "low": close * 0.997,
                "close": close,
                "volume": np.full(n, 10.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )
        shared = TrendFollowing(
            "BTC/USDT", FeatureEngine(), ema_fast=7, ema_mid=17, ema_slow=40
        ).generate_signal(ohlcv)
        own = TrendFollowing(
            "BTC/USDT", ema_fast=7, ema_mid=17, ema_slow=40
        ).generate_signal(ohlcv)
        assert shared is not None and own is not None
        assert (shared.type, shared.confidence) == (own.type, own.confidence)


def _row(**cols: float) -> pd.DataFrame:
    base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}
    return pd.DataFrame([{**base, **cols}])


class TestGradedConfidence:
    """Both strategies share one scale: 0.5 = conditions just met, 1.0 = strong."""

    def test_scale(self) -> None:
        assert graded_confidence([]) == 0.5
        assert graded_confidence([0.0, 0.0]) == 0.5
        assert graded_confidence([1.0, 3.0]) == 1.0  # margins clipped to [0, 1]
        assert graded_confidence([0.5, -2.0]) == pytest.approx(0.625)
        assert graded_confidence([0.4, float("nan")]) == pytest.approx(0.7)

    def _trend_long(self, adx: float, rsi: float, ema9: float) -> float:
        df = _row(
            **{
                FeatureEngine.EMA_9: ema9,
                FeatureEngine.EMA_21: 100.0,
                FeatureEngine.EMA_50: 99.0,
                FeatureEngine.RSI_14: rsi,
                FeatureEngine.MACD_12_26_9: 1.0,
                FeatureEngine.MACDs_12_26_9: 0.5,
                FeatureEngine.ADX_14: adx,
                FeatureEngine.ATR_14: 1.0,
            }
        )
        fe = MagicMock()
        fe.build_features.side_effect = lambda d: d
        sig = TrendFollowing(pair="BTC/USDT", feature_engine=fe).generate_signal(df)
        assert sig is not None and sig.type == SignalType.LONG
        return sig.confidence

    def test_trend_long_is_graded(self) -> None:
        weak = self._trend_long(adx=26, rsi=51, ema9=100.5)
        strong = self._trend_long(adx=50, rsi=70, ema9=102.0)
        assert 0.5 <= weak < 0.6
        assert strong == pytest.approx(1.0)

    def _reversion(self, rsi: float, close: float, **kw: bool) -> Signal | None:
        df = _row(
            close=close,
            **{
                FeatureEngine.RSI_14: rsi,
                FeatureEngine.BBL_20: 95.0,
                FeatureEngine.BBU_20: 105.0,
                FeatureEngine.ATR_14: 2.0,
            },
        )
        fe = MagicMock()
        fe.build_features.side_effect = lambda d: d
        return MeanReversion(pair="BTC/USDT", feature_engine=fe, **kw).generate_signal(
            df
        )

    def test_reversion_long_clears_the_default_threshold(self) -> None:
        """Before: (30 - RSI) / 30 needed RSI <= 15 to reach 0.5."""
        sig = self._reversion(rsi=28, close=94.5)
        assert sig is not None and sig.type == SignalType.LONG
        assert sig.confidence >= Settings().MIN_CONFIDENCE_THRESHOLD
        deep = self._reversion(rsi=10, close=93.0)
        assert deep is not None and deep.confidence == pytest.approx(1.0)

    def test_reversion_entries_can_be_disabled_keeping_exits(self) -> None:
        assert self._reversion(rsi=28, close=94.5, enable_entries=False) is None
        short = self._reversion(rsi=80, close=106.0, enable_entries=False)
        assert short is not None and short.type == SignalType.SHORT
        close = self._reversion(rsi=50, close=100.0, enable_entries=False)
        assert close is not None and close.type == SignalType.CLOSE

    def test_reversion_entries_are_off_by_default_in_live_wiring(self) -> None:
        from trading_bot.ai.prediction.feature_engine import PrecomputedFeatureEngine
        from trading_bot.backtest.pipeline import build_live_selector

        assert Settings().MEAN_REVERSION_ENTRIES is False
        selector = build_live_selector("BTC/USDT", PrecomputedFeatureEngine())
        assert selector._reversion._enable_entries is False  # type: ignore[attr-defined]
        enabled = Settings().model_copy(update={"MEAN_REVERSION_ENTRIES": True})
        on = build_live_selector("BTC/USDT", PrecomputedFeatureEngine(), enabled)
        assert on._reversion._enable_entries is True  # type: ignore[attr-defined]
