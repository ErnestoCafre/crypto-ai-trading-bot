"""Tests for core.signal_combiner.SignalCombiner."""

from __future__ import annotations

from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.signal_combiner import SignalCombiner


class TestSignalCombiner:
    def test_technical_only_passthrough(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner()
        result = combiner.combine(technical=sample_signal, ai=None)
        assert result.combined_confidence == sample_signal.confidence
        assert result.type == SignalType.LONG
        assert result.should_trade is True  # 0.75 >= 0.5

    def test_technical_plus_ai_weighted(
        self, sample_signal: Signal, sample_ai_signals: AISignals
    ) -> None:
        combiner = SignalCombiner()
        result = combiner.combine(
            technical=sample_signal,
            ai=sample_ai_signals,
            weights={"technical": 0.4},
        )
        # tech=0.75*0.4=0.30, ai_avg=(0.7+0.8+0.6)/3=0.7, ai_weight=0.6 → 0.42
        # combined = 0.30 + 0.42 = 0.72
        assert abs(result.combined_confidence - 0.72) < 0.01

    def test_low_confidence_should_not_trade(self, sample_signal: Signal) -> None:
        low_signal = Signal(
            type=SignalType.LONG,
            pair="BTC/USDT",
            confidence=0.3,
            strength=0.2,
            source="test",
            metadata={},
            timestamp=sample_signal.timestamp,
        )
        combiner = SignalCombiner()
        result = combiner.combine(technical=low_signal, ai=None)
        assert result.should_trade is False

    def test_no_signals_neutral(self) -> None:
        combiner = SignalCombiner()
        result = combiner.combine(technical=None, ai=None)
        assert result.type == SignalType.NEUTRAL
        assert result.should_trade is False
        assert result.combined_confidence == 0.0

    def test_reasoning_contains_info(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner()
        result = combiner.combine(technical=sample_signal, ai=None)
        assert "Technical" in result.reasoning
        assert "Combined" in result.reasoning
