"""Tests for core.signal_combiner.SignalCombiner."""

from __future__ import annotations

import pytest

from trading_bot.config.settings import Settings
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
        # All sources agree with LONG (weights 0.4/0.2/0.2/0.2):
        # 0.4*0.75 + 0.2*0.7 (sentiment) + 0.2*0.8 (prediction) + 0.2*0.6 (LLM)
        # = 0.72
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


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "TECHNICAL_WEIGHT": 0.4,
        "SENTIMENT_WEIGHT": 0.2,
        "DL_WEIGHT": 0.2,
        "LLM_WEIGHT": 0.2,
        "MIN_CONFIDENCE_THRESHOLD": 0.5,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


class TestDirectionalAgreement:
    """AI sources vote on the technical direction, not on raw confidence."""

    def test_bearish_sentiment_lowers_long_below_technical_only(
        self, sample_signal: Signal
    ) -> None:
        combiner = SignalCombiner(settings=_settings())
        tech_only = combiner.combine(sample_signal, None).combined_confidence
        result = combiner.combine(sample_signal, AISignals(sentiment_score=-0.5))
        # (0.4*0.75 + 0.2*-0.5) / 0.6
        assert result.combined_confidence == pytest.approx(0.2 / 0.6)
        assert result.combined_confidence < tech_only
        assert result.should_trade is False

    def test_bullish_sentiment_raises_long(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(sample_signal, AISignals(sentiment_score=0.9))
        assert result.combined_confidence > sample_signal.confidence

    def test_prediction_down_opposes_long(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner(settings=_settings())
        down = combiner.combine(
            sample_signal,
            AISignals(prediction_confidence=0.8, prediction_direction="down"),
        )
        up = combiner.combine(
            sample_signal,
            AISignals(prediction_confidence=0.8, prediction_direction="up"),
        )
        assert down.combined_confidence == pytest.approx((0.3 - 0.16) / 0.6)
        assert up.combined_confidence == pytest.approx((0.3 + 0.16) / 0.6)

    def test_prediction_direction_falls_back_to_return_sign(
        self, sample_signal: Signal
    ) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(
            sample_signal,
            AISignals(prediction_confidence=0.8, prediction_return=-0.004),
        )
        assert result.combined_confidence < sample_signal.confidence

    def test_prediction_without_direction_is_ignored(
        self, sample_signal: Signal
    ) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(sample_signal, AISignals(prediction_confidence=0.9))
        assert result.combined_confidence == sample_signal.confidence
        assert result.weights_used == {"technical": 1.0}

    def test_llm_short_without_consensus_lowers_confidence(
        self, sample_signal: Signal
    ) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(
            sample_signal,
            AISignals(
                llm_recommendation="SHORT", llm_confidence=0.9, llm_consensus=False
            ),
        )
        assert result.combined_confidence == pytest.approx((0.3 - 0.18) / 0.6)

    def test_llm_neutral_dilutes_but_does_not_oppose(
        self, sample_signal: Signal
    ) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(
            sample_signal,
            AISignals(llm_recommendation="NEUTRAL", llm_confidence=0.0),
        )
        assert result.combined_confidence == pytest.approx(0.3 / 0.6)

    def test_result_is_clipped_to_unit_interval(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner(settings=_settings(TECHNICAL_WEIGHT=0.01))
        result = combiner.combine(sample_signal, AISignals(sentiment_score=-1.0))
        assert result.combined_confidence == 0.0


class TestLLMVeto:
    def test_llm_consensus_short_vetoes_long(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner(settings=_settings(LLM_WEIGHT=0.01))
        result = combiner.combine(
            sample_signal,
            AISignals(
                sentiment_score=1.0,
                llm_recommendation="SHORT",
                llm_confidence=0.6,
                llm_consensus=True,
            ),
        )
        # Confidence alone would allow the trade; the veto overrides it.
        assert result.combined_confidence >= 0.5
        assert result.should_trade is False
        assert "VETO" in result.reasoning

    def test_llm_consensus_long_does_not_veto(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(
            sample_signal,
            AISignals(
                llm_recommendation="LONG", llm_confidence=0.8, llm_consensus=True
            ),
        )
        assert result.should_trade is True
        assert "VETO" not in result.reasoning

    def test_no_veto_on_exit_signals(self, sample_signal: Signal) -> None:
        close = Signal(
            type=SignalType.CLOSE,
            pair="BTC/USDT",
            confidence=0.75,
            strength=0.6,
            source="test",
            metadata={},
            timestamp=sample_signal.timestamp,
        )
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(
            close,
            AISignals(
                llm_recommendation="SHORT", llm_confidence=0.9, llm_consensus=True
            ),
        )
        assert "VETO" not in result.reasoning


class TestWeights:
    def test_per_source_weights_from_settings(self, sample_signal: Signal) -> None:
        ai = AISignals(
            sentiment_score=1.0, llm_recommendation="SHORT", llm_confidence=1.0
        )
        heavy_sentiment = SignalCombiner(
            settings=_settings(SENTIMENT_WEIGHT=0.5, LLM_WEIGHT=0.1)
        ).combine(sample_signal, ai)
        heavy_llm = SignalCombiner(
            settings=_settings(SENTIMENT_WEIGHT=0.1, LLM_WEIGHT=0.3)
        ).combine(sample_signal, ai)
        assert heavy_sentiment.combined_confidence == pytest.approx(
            (0.4 * 0.75 + 0.5 - 0.1) / 1.0
        )
        assert heavy_llm.combined_confidence == pytest.approx(
            (0.4 * 0.75 + 0.1 - 0.3) / 0.8
        )

    def test_weights_renormalised_over_present_sources(
        self, sample_signal: Signal
    ) -> None:
        combiner = SignalCombiner(settings=_settings())
        result = combiner.combine(sample_signal, AISignals(sentiment_score=0.5))
        assert result.weights_used == pytest.approx(
            {"technical": 0.4 / 0.6, "sentiment": 0.2 / 0.6}
        )

    def test_technical_only_unchanged_by_settings(self, sample_signal: Signal) -> None:
        combiner = SignalCombiner(settings=_settings(TECHNICAL_WEIGHT=0.1))
        result = combiner.combine(sample_signal, None)
        assert result.combined_confidence == sample_signal.confidence
        assert result.should_trade is True

    def test_threshold_comes_from_injected_settings(
        self, sample_signal: Signal
    ) -> None:
        strict = SignalCombiner(settings=_settings(MIN_CONFIDENCE_THRESHOLD=0.9))
        assert strict.combine(sample_signal, None).should_trade is False
