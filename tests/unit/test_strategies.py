"""Tests for TrendFollowing and MeanReversion strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.core.models import SignalType
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
