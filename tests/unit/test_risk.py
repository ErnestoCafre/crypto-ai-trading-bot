"""Tests for risk module: PositionSizer, StopLoss, Drawdown, AIConfidence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import AISignals, CombinedSignal, SignalType
from trading_bot.risk.ai_confidence import AIConfidenceFilter
from trading_bot.risk.drawdown import DrawdownMonitor, TradingState
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.stop_loss import StopLossManager

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
        assert result.value_usd <= 100_000 * 0.25  # MAX_POSITION_PCT default=25%

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
        assert result.value_usd <= 100_000 * 0.25 + 1  # small tolerance

    def test_ai_confidence_reduces_risk(self) -> None:
        sl = StopLossManager()
        # Wide position cap so the risk budget, not the cap, sets the size.
        settings = Settings(_env_file=None, MAX_POSITION_PCT=100.0)  # type: ignore[call-arg]
        ps = PositionSizer(stop_loss_mgr=sl, settings=settings)
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
            ai_size_multiplier=0.5,
        )
        assert reduced.quantity == pytest.approx(full.quantity * 0.5)
        assert reduced.ai_adjustment == 0.5

    def test_regime_multiplier_stacks_with_ai(self) -> None:
        settings = Settings(_env_file=None, MAX_POSITION_PCT=100.0)  # type: ignore[call-arg]
        ps = PositionSizer(stop_loss_mgr=StopLossManager(), settings=settings)
        kwargs = dict(
            pair="BTC/USDT",
            signal_confidence=0.8,
            portfolio_value=100_000,
            entry_price=50000.0,
        )
        full = ps.calculate(**kwargs)
        reduced = ps.calculate(
            **kwargs, ai_size_multiplier=0.5, regime_size_multiplier=0.5
        )
        assert reduced.quantity == pytest.approx(full.quantity * 0.25)
        assert reduced.regime_adjustment == 0.5

    def test_ai_multiplier_never_raises_risk(self) -> None:
        settings = Settings(_env_file=None, MAX_POSITION_PCT=100.0)  # type: ignore[call-arg]
        ps = PositionSizer(stop_loss_mgr=StopLossManager(), settings=settings)
        kwargs = dict(
            pair="BTC/USDT",
            signal_confidence=0.8,
            portfolio_value=100_000,
            entry_price=50000.0,
        )
        full = ps.calculate(**kwargs)
        boosted = ps.calculate(**kwargs, ai_size_multiplier=3.0)
        assert boosted.quantity == pytest.approx(full.quantity)

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


class TestPositionCapCalibration:
    """The risk budget should size entries; MAX_POSITION_PCT is a backstop."""

    def test_capped_flag(self) -> None:
        tight = PositionSizer(stop_loss_mgr=StopLossManager(atr_multiplier=0.001))
        wide = PositionSizer(
            stop_loss_mgr=StopLossManager(),
            settings=Settings(_env_file=None, MAX_POSITION_PCT=100.0),  # type: ignore[call-arg]
        )
        kwargs = dict(
            pair="BTC/USDT",
            signal_confidence=0.8,
            portfolio_value=100_000,
            entry_price=50_000.0,
        )
        assert tight.calculate(**kwargs).capped is True
        assert wide.calculate(**kwargs).capped is False

    @pytest.mark.parametrize("stop_pct", [1.0, 2.0, 3.0, 4.0, 6.0])
    def test_default_cap_rarely_binds_for_typical_1h_stops(
        self, stop_pct: float
    ) -> None:
        """Defaults: a 2xATR stop at >=1% of price is sized by risk, not the cap.

        If this fails after changing MAX_RISK_PER_TRADE_PCT / MAX_POSITION_PCT
        the cap would set most entries (validate with scripts/run_backtest.py:
        "Cap binding" should stay well under 50%).
        """
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        ps = PositionSizer(stop_loss_mgr=StopLossManager(), settings=settings)
        entry = 50_000.0
        atr = entry * stop_pct / 100 / 2
        result = ps._sl.calculate_stop("BTC/USDT", "buy", entry, atr_value=atr)
        assert result == pytest.approx(entry * (1 - stop_pct / 100))
        risk_qty = 100_000 * settings.MAX_RISK_PER_TRADE_PCT / 100 / (entry - result)
        cap_qty = 100_000 * settings.MAX_POSITION_PCT / 100 / entry
        assert risk_qty <= cap_qty

    def test_warns_when_cap_sets_most_entries(self) -> None:
        from loguru import logger

        ps = PositionSizer(stop_loss_mgr=StopLossManager(atr_multiplier=0.001))
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING")
        try:
            for _ in range(12):
                ps.calculate(
                    pair="BTC/USDT",
                    signal_confidence=0.8,
                    portfolio_value=100_000,
                    entry_price=50_000.0,
                )
        finally:
            logger.remove(sink)
        assert ps.cap_hit_rate == 1.0
        assert sum("MAX_POSITION_PCT" in m for m in messages) == 1  # once


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


# ---------- Injected settings ----------


class TestInjectedSettings:
    """Risk components use the Settings they are given, not a fresh .env read."""

    @staticmethod
    def _settings(**overrides: object) -> Settings:
        return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]

    def test_ai_filter_uses_injected_threshold(self) -> None:
        f = AIConfidenceFilter(settings=self._settings(MIN_CONFIDENCE_THRESHOLD=0.9))
        sig = TestAIConfidenceFilter()._make_combined(0.8, ai=AISignals(0.5))
        assert f.should_trade(sig) is False

    def test_position_sizer_uses_injected_limits(self) -> None:
        ps = PositionSizer(
            stop_loss_mgr=StopLossManager(),
            settings=self._settings(MAX_RISK_PER_TRADE_PCT=50.0, MAX_POSITION_PCT=2.0),
        )
        result = ps.calculate(
            pair="BTC/USDT",
            signal_confidence=1.0,
            portfolio_value=100_000.0,
            entry_price=50_000.0,
        )
        assert result.value_usd == pytest.approx(2_000.0)
        assert result.risk_pct == 50.0

    def test_drawdown_monitor_uses_injected_limits(self) -> None:
        settings = self._settings(MAX_DRAWDOWN_PCT=5.0, MAX_DAILY_LOSS_PCT=50.0)
        dm = DrawdownMonitor(initial_value=100_000, settings=settings)
        assert dm.update(94_000) == TradingState.HALTED

        restored = DrawdownMonitor.from_dict(dm.to_dict(), settings=settings)
        assert restored.state == TradingState.HALTED

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()


# ---------- AI sizing multiplier ----------


class TestAISizeMultiplier:
    @staticmethod
    def _sizer(**overrides: object) -> PositionSizer:
        values: dict[str, object] = {
            "MIN_CONFIDENCE_THRESHOLD": 0.5,
            "AI_SIZING_FLOOR": 0.25,
        }
        values.update(overrides)
        settings = Settings(_env_file=None, **values)  # type: ignore[arg-type]
        return PositionSizer(stop_loss_mgr=StopLossManager(), settings=settings)

    @staticmethod
    def _combined(confidence: float, weights: dict[str, float]) -> CombinedSignal:
        return CombinedSignal(
            type=SignalType.LONG,
            pair="BTC/USDT",
            combined_confidence=confidence,
            technical_signal=None,
            ai_signals=AISignals(sentiment_score=0.1),
            weights_used=weights,
            should_trade=True,
            reasoning="test",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def test_technical_only_is_not_scaled(self) -> None:
        sig = self._combined(0.6, {"technical": 1.0})
        assert self._sizer().ai_size_multiplier(sig) is None

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [(0.5, 0.25), (0.75, 0.625), (1.0, 1.0), (0.3, 0.25)],
    )
    def test_linear_from_floor_at_threshold(
        self, confidence: float, expected: float
    ) -> None:
        sig = self._combined(confidence, {"technical": 0.6, "prediction": 0.4})
        assert self._sizer().ai_size_multiplier(sig) == pytest.approx(expected)

    def test_opposing_prediction_shrinks_position(self) -> None:
        """Regression: a strong opposing prediction used to size near full."""
        from trading_bot.core.models import Signal
        from trading_bot.core.signal_combiner import SignalCombiner

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        combiner = SignalCombiner(settings=settings)
        sizer = PositionSizer(stop_loss_mgr=StopLossManager(), settings=settings)
        tech = Signal(
            type=SignalType.LONG,
            pair="BTC/USDT",
            confidence=0.9,
            strength=0.8,
            source="test",
            metadata={},
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        against = combiner.combine(
            tech, AISignals(prediction_confidence=0.8, prediction_direction="down")
        )
        along = combiner.combine(
            tech, AISignals(prediction_confidence=0.8, prediction_direction="up")
        )
        m_against = sizer.ai_size_multiplier(against)
        m_along = sizer.ai_size_multiplier(along)
        assert m_against is not None and m_along is not None
        assert m_against < m_along
