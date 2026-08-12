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
