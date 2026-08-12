"""Signal combiner — merges technical and AI signals into a single decision."""

from __future__ import annotations

from datetime import UTC, datetime

from trading_bot.config.settings import get_settings
from trading_bot.core.models import AISignals, CombinedSignal, Signal, SignalType


class SignalCombiner:
    def combine(
        self,
        technical: Signal | None,
        ai: AISignals | None,
        weights: dict[str, float] | None = None,
    ) -> CombinedSignal:
        settings = get_settings()

        if weights is None:
            weights = {"technical": settings.TECHNICAL_WEIGHT}

        # No signal at all -> neutral
        if technical is None and ai is None:
            return CombinedSignal(
                type=SignalType.NEUTRAL,
                pair="UNKNOWN",
                combined_confidence=0.0,
                technical_signal=None,
                ai_signals=None,
                weights_used=weights,
                should_trade=False,
                recommended_sizing_pct=0.0,
                reasoning="No signals available.",
                timestamp=datetime.now(UTC),
            )

        pair = technical.pair if technical else "UNKNOWN"
        signal_type = technical.type if technical else SignalType.NEUTRAL

        # Technical-only (Phase 2 default)
        if ai is None:
            combined_confidence = technical.confidence if technical else 0.0
        else:
            tech_weight = weights.get("technical", settings.TECHNICAL_WEIGHT)
            ai_weight = 1.0 - tech_weight
            tech_conf = technical.confidence if technical else 0.0
            ai_scores = [
                s
                for s in [
                    ai.prediction_confidence,
                    ai.llm_confidence,
                    ai.sentiment_score,
                ]
                if s is not None
            ]
            ai_avg = sum(ai_scores) / len(ai_scores) if ai_scores else 0.0
            combined_confidence = tech_conf * tech_weight + ai_avg * ai_weight

        should_trade = combined_confidence >= settings.MIN_CONFIDENCE_THRESHOLD
        recommended_sizing_pct = combined_confidence * settings.MAX_POSITION_PCT

        # Reasoning
        parts: list[str] = []
        if technical:
            parts.append(
                f"Technical: {technical.type.value} conf={technical.confidence:.2f}"
            )
        if ai:
            parts.append(
                f"AI: sent={ai.sentiment_score} pred_conf={ai.prediction_confidence}"
                f" llm_conf={ai.llm_confidence}"
            )
        parts.append(
            f"Combined conf={combined_confidence:.2f}, should_trade={should_trade}"
        )
        reasoning = " | ".join(parts)

        return CombinedSignal(
            type=signal_type,
            pair=pair,
            combined_confidence=combined_confidence,
            technical_signal=technical,
            ai_signals=ai,
            weights_used=weights,
            should_trade=should_trade,
            recommended_sizing_pct=recommended_sizing_pct,
            reasoning=reasoning,
            timestamp=datetime.now(UTC),
        )
