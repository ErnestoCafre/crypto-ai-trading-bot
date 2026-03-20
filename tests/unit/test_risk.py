"""Tests for risk module: PositionSizer, StopLoss, Drawdown, AIConfidence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.models import AISignals, CombinedSignal, SignalType
from risk.ai_confidence import AIConfidenceFilter
from risk.drawdown import DrawdownMonitor
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossManager

# ---------- StopLossManager ----------


class TestStopLossManager:
    def test_long_stop_below_entry(self) -> None:
        sl = StopLossManager()
        stop = sl.calculate_stop("BTC/USDT", "buy", 50000.0, atr_value=500.0)
        assert stop < 50000.0

    def test_short_stop_above_entry(self) -> None:
        sl = StopLossManager()
        stop = sl.calculate_stop("BTC/USDT", "sell", 50000.0, atr_value=500.0)
        assert stop > 50000.0

    def test_multiplier_applied(self) -> None:
        sl = StopLossManager(atr_multiplier=3.0)
        stop = sl.calculate_stop("BTC/USDT", "buy", 50000.0, atr_value=500.0)
        # stop = 50000 - 500*3 = 48500
        assert stop == pytest.approx(48500.0)

    def test_fallback_without_atr(self) -> None:
        sl = StopLossManager(default_pct=0.02)
        stop = sl.calculate_stop("BTC/USDT", "buy", 50000.0)
        # fallback atr = 50000*0.02 = 1000, stop = 50000 - 1000*2 = 48000
        assert stop == pytest.approx(48000.0)

    def test_stop_never_negative(self) -> None:
        sl = StopLossManager(atr_multiplier=100.0)
        stop = sl.calculate_stop("BTC/USDT", "buy", 100.0, atr_value=50.0)
        assert stop >= 0.0


# ---------- PositionSizer ----------


class TestPositionSizer:
    def test_respects_max_risk(self) -> None:
        sl = StopLossManager()
        ps = PositionSizer(stop_loss_mgr=sl)
        result = ps.calculate(
            pair="BTC/USDT",
            signal_confidence=0.8,
            portfolio_value=100_000,
            entry_price=50000.0,
            side="buy",
        )
        assert result.quantity > 0
        assert result.value_usd <= 100_000 * 0.10  # MAX_POSITION_PCT default=10%

    def test_max_position_cap(self) -> None:
        # Tiny ATR multiplier → tiny stop distance → huge quantity → gets capped
        sl = StopLossManager(atr_multiplier=0.001)
        ps = PositionSizer(stop_loss_mgr=sl)
        result = ps.calculate(
            pair="BTC/USDT",
            signal_confidence=1.0,
            portfolio_value=100_000,
            entry_price=50000.0,
            side="buy",
        )
        assert result.value_usd <= 100_000 * 0.10 + 1  # small tolerance

    def test_ai_confidence_reduces_risk(self) -> None:
        sl = StopLossManager()
        ps = PositionSizer(stop_loss_mgr=sl)
        full = ps.calculate(
            pair="BTC/USDT",
            signal_confidence=0.8,
            portfolio_value=100_000,
            entry_price=50000.0,
        )
        reduced = ps.calculate(
            pair="BTC/USDT",
            signal_confidence=0.8,
            portfolio_value=100_000,
            entry_price=50000.0,
            ai_confidence=0.5,
        )
        assert reduced.quantity <= full.quantity

    def test_quantity_non_negative(self) -> None:
        sl = StopLossManager()
        ps = PositionSizer(stop_loss_mgr=sl)
        result = ps.calculate(
            pair="BTC/USDT",
            signal_confidence=0.1,
            portfolio_value=100,
            entry_price=50000.0,
        )
        assert result.quantity >= 0.0


# ---------- DrawdownMonitor ----------


class TestDrawdownMonitor:
    def test_kill_switch_triggers_on_high_drawdown(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        dm.check(100_000)  # set peak
        result = dm.check(80_000)  # 20% drawdown, MAX_DRAWDOWN_PCT=15%
        assert result is True

    def test_no_trigger_in_normal_range(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        dm.check(100_000)
        result = dm.check(98_000)  # 2% drawdown — below both thresholds
        assert result is False

    def test_peak_tracking(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        dm.check(110_000)  # new peak
        dm.check(105_000)
        assert dm.current_drawdown_pct == pytest.approx(
            (110_000 - 105_000) / 110_000 * 100, rel=0.01
        )

    def test_drawdown_properties(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        dm.check(100_000)
        dm.check(97_000)
        assert dm.current_drawdown_pct > 0
        # daily_pnl_pct should be negative
        assert dm.daily_pnl_pct < 0


# ---------- AIConfidenceFilter ----------


class TestAIConfidenceFilter:
    def _make_combined(
        self, confidence: float, ai: AISignals | None = None
    ) -> CombinedSignal:
        return CombinedSignal(
            type=SignalType.LONG,
            pair="BTC/USDT",
            combined_confidence=confidence,
            technical_signal=None,
            ai_signals=ai,
            weights_used={"technical": 0.4},
            should_trade=True,
            recommended_sizing_pct=5.0,
            reasoning="test",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def test_pass_through_without_ai(self) -> None:
        f = AIConfidenceFilter()
        sig = self._make_combined(0.3, ai=None)
        assert f.should_trade(sig) is True

    def test_blocks_low_confidence_with_ai(self) -> None:
        f = AIConfidenceFilter()
        ai = AISignals(sentiment_score=0.2)
        sig = self._make_combined(0.3, ai=ai)
        assert f.should_trade(sig) is False

    def test_passes_high_confidence_with_ai(self) -> None:
        f = AIConfidenceFilter()
        ai = AISignals(sentiment_score=0.9)
        sig = self._make_combined(0.8, ai=ai)
        assert f.should_trade(sig) is True
