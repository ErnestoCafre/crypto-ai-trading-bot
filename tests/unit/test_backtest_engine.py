"""Tests for BacktestEngine."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.data_types import BacktestResult
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.core.models import Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy


def _make_ohlcv(n: int = 300, trend: float = 0.5) -> pd.DataFrame:
    """Synthetic OHLCV with a DatetimeIndex."""
    rng = np.random.default_rng(42)
    close = 50000 + np.cumsum(rng.normal(trend, 100, n))
    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 30, n),
            "high": close + abs(rng.normal(0, 150, n)),
            "low": close - abs(rng.normal(0, 150, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 200, n)),
        },
        index=dates,
    )


def _make_signal(sig_type: SignalType, source: str = "test") -> Signal:
    return Signal(
        type=sig_type,
        pair="BTC/USDT",
        confidence=0.8,
        strength=0.7,
        source=source,
        metadata={},
        timestamp=datetime.now(UTC),
    )


class _AlwaysLongStrategy(BaseStrategy):
    """Generates LONG at every bar (for testing)."""

    @property
    def name(self) -> str:
        return "always_long"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        return _make_signal(SignalType.LONG, "always_long")


class _LongThenCloseStrategy(BaseStrategy):
    """LONG on first valid bar, CLOSE on second."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def name(self) -> str:
        return "long_then_close"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        self._call_count += 1
        if self._call_count == 1:
            return _make_signal(SignalType.LONG, "long_then_close")
        if self._call_count == 2:
            return Signal(
                type=SignalType.CLOSE,
                pair="BTC/USDT",
                confidence=0.6,
                strength=0.5,
                source="long_then_close",
                metadata={},
                timestamp=datetime.now(UTC),
            )
        return None


class _NeverTradeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "never_trade"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        return None


class TestBacktestEngineRun:
    def test_returns_backtest_result(self) -> None:
        engine = BacktestEngine(strategy=_LongThenCloseStrategy())
        ohlcv = _make_ohlcv(50)
        result = engine.run(ohlcv, "BTC/USDT")
        assert isinstance(result, BacktestResult)
        assert result.pair == "BTC/USDT"
        assert result.strategy_name == "long_then_close"

    def test_no_look_ahead_bias(self) -> None:
        """Signal at bar i should only see ohlcv[:i+1]."""
        seen_lengths: list[int] = []

        class _SpyStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "spy"

            def get_required_lookback(self) -> int:
                return 1

            def generate_signal(
                self,
                ohlcv: pd.DataFrame,
                ai_signals: object = None,
            ) -> Signal | None:
                seen_lengths.append(len(ohlcv))
                return None

        engine = BacktestEngine(strategy=_SpyStrategy())
        ohlcv = _make_ohlcv(20)
        engine.run(ohlcv, "BTC/USDT")

        for i in range(1, len(seen_lengths)):
            assert seen_lengths[i] >= seen_lengths[i - 1]

    def test_slippage_applied(self) -> None:
        engine = BacktestEngine(strategy=_LongThenCloseStrategy(), slippage_pct=0.1)
        result = engine.run(_make_ohlcv(50), "BTC/USDT")
        if result.trades:
            trade = result.trades[0]
            assert trade.slippage > 0

    def test_fees_deducted(self) -> None:
        engine = BacktestEngine(strategy=_LongThenCloseStrategy(), fee_pct=0.1)
        result = engine.run(_make_ohlcv(50), "BTC/USDT")
        if result.trades:
            trade = result.trades[0]
            assert trade.fees > 0

    def test_equity_curve_length(self) -> None:
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        ohlcv = _make_ohlcv(50)
        result = engine.run(ohlcv, "BTC/USDT")
        assert len(result.equity_curve) == len(ohlcv)

    def test_empty_ohlcv_graceful(self) -> None:
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        ohlcv = pd.DataFrame(
            {
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )
        result = engine.run(ohlcv, "BTC/USDT")
        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0

    def test_single_bar_graceful(self) -> None:
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        ohlcv = pd.DataFrame(
            {
                "open": [100],
                "high": [110],
                "low": [90],
                "close": [105],
                "volume": [1000],
            }
        )
        result = engine.run(ohlcv, "BTC/USDT")
        assert isinstance(result, BacktestResult)

    def test_no_trades_when_never_signal(self) -> None:
        engine = BacktestEngine(strategy=_NeverTradeStrategy())
        result = engine.run(_make_ohlcv(100), "BTC/USDT")
        assert len(result.trades) == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000, rel=1e-6)

    def test_stop_loss_triggers(self) -> None:
        """Stop-loss should trigger on price drop."""

        class _LongOnceStrategy(BaseStrategy):
            _done = False

            @property
            def name(self) -> str:
                return "long_once"

            def get_required_lookback(self) -> int:
                return 1

            def generate_signal(
                self,
                ohlcv: pd.DataFrame,
                ai_signals: object = None,
            ) -> Signal | None:
                if not self._done:
                    self._done = True
                    return _make_signal(SignalType.LONG, "long_once")
                return None

        engine = BacktestEngine(strategy=_LongOnceStrategy())
        n = 50
        close = np.concatenate(
            [
                np.full(5, 50000.0),
                np.linspace(50000, 45000, n - 5),
            ]
        )
        ohlcv = pd.DataFrame(
            {
                "open": close + 10,
                "high": close + 100,
                "low": close - 500,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        )
        result = engine.run(ohlcv, "BTC/USDT")
        assert len(result.trades) >= 1

    def test_trade_pnl_consistency(self) -> None:
        """Trade PnL consistent with entry/exit prices."""
        engine = BacktestEngine(
            strategy=_LongThenCloseStrategy(),
            fee_pct=0.0,
            slippage_pct=0.0,
        )
        result = engine.run(_make_ohlcv(50), "BTC/USDT")
        for trade in result.trades:
            if trade.side == "long":
                expected = (trade.exit_price - trade.entry_price) * trade.quantity
                assert trade.pnl == pytest.approx(expected, rel=1e-6)


class _ScriptedStrategy(BaseStrategy):
    """Emits a scripted signal type at given window lengths."""

    def __init__(self, script: dict[int, SignalType]) -> None:
        self._script = script

    @property
    def name(self) -> str:
        return "scripted"

    def get_required_lookback(self) -> int:
        return 1

    def generate_signal(
        self, ohlcv: pd.DataFrame, ai_signals: object = None
    ) -> Signal | None:
        sig = self._script.get(len(ohlcv))
        return _make_signal(sig) if sig is not None else None


def _flat_ohlcv(n: int = 20, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price + 1),
            "low": np.full(n, price - 1),
            "close": np.full(n, price),
            "volume": np.full(n, 1_000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


class TestSpotRules:
    def test_short_without_position_is_ignored(self) -> None:
        engine = BacktestEngine(_ScriptedStrategy({3: SignalType.SHORT}))
        assert engine.run(_flat_ohlcv(), "BTC/USDT").trades == []

    def test_short_closes_long(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.LONG, 6: SignalType.SHORT})
        )
        trades = engine.run(_flat_ohlcv(), "BTC/USDT").trades
        assert [(t.side, t.exit_reason) for t in trades] == [("long", "signal")]
        # Executed at the open of the bar after each signal bar.
        ohlcv = _flat_ohlcv()
        assert trades[0].entry_time == ohlcv.index[3]
        assert trades[0].exit_time == ohlcv.index[6]

    def test_long_while_holding_does_not_pyramid(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy(
                {3: SignalType.LONG, 4: SignalType.LONG, 8: SignalType.CLOSE}
            )
        )
        trades = engine.run(_flat_ohlcv(), "BTC/USDT").trades
        assert len(trades) == 1

    def test_allow_short_opens_short(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.SHORT, 6: SignalType.LONG}),
            allow_short=True,
        )
        trades = engine.run(_flat_ohlcv(), "BTC/USDT").trades
        assert [t.side for t in trades] == ["short"]

    def test_close_closes_short(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.SHORT, 6: SignalType.CLOSE}),
            allow_short=True,
        )
        trades = engine.run(_flat_ohlcv(), "BTC/USDT").trades
        assert [(t.side, t.exit_reason) for t in trades] == [("short", "signal")]


class TestRegimeSizing:
    """VOLATILE shrinks the position through metadata["sizing_factor"]."""

    class _Sizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def calculate(self, **kwargs: object) -> object:
            from trading_bot.core.models import PositionSize

            self.calls.append(kwargs)
            return PositionSize(
                pair="BTC/USDT",
                quantity=1.0,
                value_usd=100.0,
                risk_pct=1.0,
                sizing_method="stub",
                ai_adjustment=1.0,
                regime_adjustment=1.0,
                stop_loss_price=50.0,
                stop_loss_pct=50.0,
            )

    def test_sizing_factor_reaches_sizer(self) -> None:
        class _Volatile(_ScriptedStrategy):
            def generate_signal(
                self, ohlcv: pd.DataFrame, ai_signals: object = None
            ) -> Signal | None:
                sig = super().generate_signal(ohlcv)
                if sig is not None:
                    sig.metadata["sizing_factor"] = 0.5
                return sig

        sizer = self._Sizer()
        engine = BacktestEngine(_Volatile({3: SignalType.LONG}), position_sizer=sizer)
        engine.run(_flat_ohlcv(), "BTC/USDT")
        assert sizer.calls[0]["regime_size_multiplier"] == 0.5
        assert sizer.calls[0]["ai_size_multiplier"] is None

    def test_default_sizing_applies_factor(self) -> None:
        class _Volatile(_ScriptedStrategy):
            def generate_signal(
                self, ohlcv: pd.DataFrame, ai_signals: object = None
            ) -> Signal | None:
                sig = super().generate_signal(ohlcv)
                if sig is not None:
                    sig.metadata["sizing_factor"] = 0.5
                return sig

        full = BacktestEngine(_ScriptedStrategy({3: SignalType.LONG}), fee_pct=0.0)
        half = BacktestEngine(_Volatile({3: SignalType.LONG}), fee_pct=0.0)
        q_full = full.run(_flat_ohlcv(), "BTC/USDT").trades[0].quantity
        q_half = half.run(_flat_ohlcv(), "BTC/USDT").trades[0].quantity
        assert q_half == pytest.approx(q_full / 2)


class TestExecutionCosts:
    def test_pnl_includes_entry_and_exit_fees(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.LONG, 6: SignalType.CLOSE}),
            fee_pct=0.1,
            slippage_pct=0.0,
        )
        result = engine.run(_flat_ohlcv(), "BTC/USDT")
        (trade,) = result.trades
        # Flat price: the round trip loses exactly both fees.
        assert trade.pnl == pytest.approx(-trade.fees)
        assert trade.fees == pytest.approx(2 * 100.0 * trade.quantity * 0.001)
        assert result.equity_curve.iloc[-1] == pytest.approx(100_000 + trade.pnl)


class TestStopGap:
    def _gap_ohlcv(self) -> pd.DataFrame:
        df = _flat_ohlcv()
        # Bar 8 gaps down through the 2% stop (98) and opens at 90.
        df.iloc[8, df.columns.get_loc("open")] = 90.0
        df.iloc[8, df.columns.get_loc("low")] = 89.0
        df.iloc[8, df.columns.get_loc("close")] = 95.0
        return df

    def test_gap_through_stop_fills_at_open(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.LONG}), fee_pct=0.0, slippage_pct=0.0
        )
        (trade,) = engine.run(self._gap_ohlcv(), "BTC/USDT").trades
        assert trade.exit_reason == "stop_loss"
        assert trade.exit_price == pytest.approx(90.0)

    def test_touch_fills_at_stop(self) -> None:
        df = _flat_ohlcv()
        df.iloc[8, df.columns.get_loc("low")] = 97.0
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.LONG}), fee_pct=0.0, slippage_pct=0.0
        )
        (trade,) = engine.run(df, "BTC/USDT").trades
        assert trade.exit_price == pytest.approx(98.0)


class TestTradeFrom:
    def test_warmup_bars_are_not_traded_or_reported(self) -> None:
        engine = BacktestEngine(
            _ScriptedStrategy({3: SignalType.LONG, 12: SignalType.LONG})
        )
        ohlcv = _flat_ohlcv()
        result = engine.run(ohlcv, "BTC/USDT", trade_from=10)
        assert result.equity_curve.index[0] == ohlcv.index[10]
        assert len(result.equity_curve) == 10
        assert [t.entry_time for t in result.trades] == [ohlcv.index[12]]
        assert result.start_date == ohlcv.index[10]

    def test_invalid_trade_from(self) -> None:
        engine = BacktestEngine(_NeverTradeStrategy())
        with pytest.raises(ValueError):
            engine.run(_flat_ohlcv(), "BTC/USDT", trade_from=20)


def _gbm_ohlcv(n: int = 600, seed: int = 1) -> pd.DataFrame:
    """Geometric random walk with consistent OHLC (opens at the prior close)."""
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.003,
            "low": np.minimum(open_, close) * 0.997,
            "close": close,
            "volume": rng.uniform(100, 200, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
    )


class TestLivePipeline:
    def test_precomputed_features_match_per_bar_recompute(self) -> None:
        """Slicing precomputed features gives the same trades as recomputing."""
        from trading_bot.ai.prediction.feature_engine import FeatureEngine
        from trading_bot.backtest.pipeline import build_live_engine
        from trading_bot.core.regime import RegimeDetector
        from trading_bot.core.signal_combiner import SignalCombiner
        from trading_bot.core.strategies.mean_reversion import MeanReversion
        from trading_bot.core.strategies.trend_following import TrendFollowing
        from trading_bot.core.strategy_selector import StrategySelector
        from trading_bot.risk.ai_confidence import AIConfidenceFilter
        from trading_bot.risk.position_sizer import PositionSizer

        # 340 bars, seed 4: exercises a stop exit and a signal exit.
        ohlcv = _gbm_ohlcv(340, seed=4)
        fast = build_live_engine("BTC/USDT").run(ohlcv, "BTC/USDT")

        fe = FeatureEngine()
        slow_engine = BacktestEngine(
            StrategySelector(
                TrendFollowing("BTC/USDT", fe),
                MeanReversion("BTC/USDT", fe),
                RegimeDetector(feature_engine=fe),
                0.5,
            ),
            position_sizer=PositionSizer(),
            signal_combiner=SignalCombiner(),
            ai_confidence_filter=AIConfidenceFilter(),
        )
        slow = slow_engine.run(ohlcv, "BTC/USDT")

        assert {t.exit_reason for t in fast.trades} >= {"stop_loss", "signal"}
        assert [(t.entry_time, t.exit_time, t.exit_reason) for t in fast.trades] == [
            (t.entry_time, t.exit_time, t.exit_reason) for t in slow.trades
        ]
        pd.testing.assert_series_equal(fast.equity_curve, slow.equity_curve)

    def test_live_engine_is_long_only_and_tags_regime(self) -> None:
        from trading_bot.backtest.pipeline import build_live_engine

        result = build_live_engine("BTC/USDT").run(_gbm_ohlcv(), "BTC/USDT")
        assert all(t.side == "long" for t in result.trades)
        assert all(
            t.regime in {"trending", "ranging", "volatile"} for t in result.trades
        )

    def test_live_settings_reach_the_selector(self) -> None:
        """USE_REGIME_DETECTION=false routes every bar to trend following."""
        from trading_bot.ai.prediction.feature_engine import PrecomputedFeatureEngine
        from trading_bot.backtest.pipeline import build_live_selector
        from trading_bot.config.settings import Settings

        ohlcv = _gbm_ohlcv(340, seed=4)
        off = build_live_selector(
            "BTC/USDT",
            PrecomputedFeatureEngine(),
            Settings(USE_REGIME_DETECTION=False),
        )
        on = build_live_selector("BTC/USDT", PrecomputedFeatureEngine(), Settings())
        routed_off = {
            off.select(ohlcv.iloc[: i + 1])[0].name for i in range(250, 340, 5)
        }
        routed_on = {on.select(ohlcv.iloc[: i + 1])[0].name for i in range(250, 340, 5)}
        assert routed_off == {"trend_following"}
        assert "mean_reversion" in routed_on

    def test_unknown_strategy_param_rejected(self) -> None:
        from trading_bot.backtest.pipeline import build_live_engine

        with pytest.raises(TypeError):
            build_live_engine("BTC/USDT", rsi_typo=10)
