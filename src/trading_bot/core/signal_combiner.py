"""Signal combiner — merges technical and AI signals into a single decision.

The technical signal sets the direction. Each AI source then contributes its
*agreement* with that direction, in [-1, 1]:

- sentiment: ``score * direction`` (score is already in [-1, 1]);
- ML prediction: ``+confidence`` if it predicts the same direction, else
  ``-confidence``;
- LLM agents: ``+confidence`` / ``-confidence`` for LONG / SHORT, 0 when
  NEUTRAL.

``combined = (w_tech * tech_conf + sum(w_i * agreement_i)) / sum(w)`` over
the sources present, clipped to [0, 1]. A source that disagrees therefore
lowers confidence instead of raising it.

Veto (ADR-006): when the LLM agents reach consensus against a LONG entry,
the entry is not taken whatever the combined confidence. Exits never depend
on this class (see TradingOrchestrator._process_pair).
"""

from __future__ import annotations

from datetime import UTC, datetime

from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import AISignals, CombinedSignal, Signal, SignalType

_DIRECTION: dict[SignalType, int] = {
    SignalType.LONG: 1,
    SignalType.SHORT: -1,
    SignalType.CLOSE: -1,
    SignalType.NEUTRAL: 0,
}
_LLM_DIRECTION: dict[str, int] = {"LONG": 1, "SHORT": -1, "NEUTRAL": 0}


class SignalCombiner:
    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._default_weights: dict[str, float] = {
            "technical": s.TECHNICAL_WEIGHT,
            "sentiment": s.SENTIMENT_WEIGHT,
            "prediction": s.DL_WEIGHT,
            "llm": s.LLM_WEIGHT,
        }
        self._min_confidence = s.MIN_CONFIDENCE_THRESHOLD

    def combine(
        self,
        technical: Signal | None,
        ai: AISignals | None,
        weights: dict[str, float] | None = None,
    ) -> CombinedSignal:
        """Combine signals; ``weights`` overrides the per-source defaults."""
        base_weights = {**self._default_weights, **(weights or {})}

        # No signal at all -> neutral
        if technical is None and ai is None:
            return CombinedSignal(
                type=SignalType.NEUTRAL,
                pair="UNKNOWN",
                combined_confidence=0.0,
                technical_signal=None,
                ai_signals=None,
                weights_used={},
                should_trade=False,
                reasoning="No signals available.",
                timestamp=datetime.now(UTC),
            )

        pair = technical.pair if technical else "UNKNOWN"
        signal_type = technical.type if technical else SignalType.NEUTRAL
        direction = _DIRECTION[signal_type]
        tech_conf = technical.confidence if technical else 0.0

        agreements = self._agreements(ai, direction) if ai is not None else {}

        # Technical-only: the technical confidence, unchanged.
        if not agreements:
            combined_confidence = tech_conf
            weights_used = {"technical": 1.0}
        else:
            raw = {"technical": base_weights["technical"]}
            raw.update({k: base_weights[k] for k in agreements})
            total = sum(raw.values())
            if total <= 0:
                raise ValueError(f"Signal weights must sum to > 0, got {raw}")
            weights_used = {k: w / total for k, w in raw.items()}
            weighted = weights_used["technical"] * tech_conf + sum(
                weights_used[k] * a for k, a in agreements.items()
            )
            combined_confidence = min(1.0, max(0.0, weighted))

        should_trade = combined_confidence >= self._min_confidence
        veto = self._llm_veto(ai, signal_type)
        if veto is not None:
            should_trade = False

        # Reasoning
        parts: list[str] = []
        if technical:
            parts.append(
                f"Technical: {technical.type.value} conf={technical.confidence:.2f}"
            )
        if agreements:
            parts.append(
                "AI agreement: "
                + ", ".join(f"{k}={a:+.2f}" for k, a in agreements.items())
            )
        if veto is not None:
            parts.append(veto)
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
            weights_used=weights_used,
            should_trade=should_trade,
            reasoning=reasoning,
            timestamp=datetime.now(UTC),
        )

    @staticmethod
    def _agreements(ai: AISignals, direction: int) -> dict[str, float]:
        """Agreement in [-1, 1] of each available AI source with ``direction``."""
        out: dict[str, float] = {}

        if ai.sentiment_score is not None:
            score = min(1.0, max(-1.0, ai.sentiment_score))
            out["sentiment"] = score * direction

        pred_dir = _prediction_direction(ai)
        if ai.prediction_confidence is not None and pred_dir is not None:
            conf = min(1.0, max(0.0, ai.prediction_confidence))
            out["prediction"] = conf * pred_dir * direction

        if ai.llm_recommendation is not None and ai.llm_confidence is not None:
            llm_dir = _LLM_DIRECTION.get(ai.llm_recommendation.upper(), 0)
            conf = min(1.0, max(0.0, ai.llm_confidence))
            out["llm"] = conf * llm_dir * direction

        return out

    @staticmethod
    def _llm_veto(ai: AISignals | None, signal_type: SignalType) -> str | None:
        """ADR-006: LLM consensus against a LONG entry blocks it."""
        if (
            ai is None
            or signal_type != SignalType.LONG
            or ai.llm_consensus is not True
            or ai.llm_recommendation is None
        ):
            return None
        if _LLM_DIRECTION.get(ai.llm_recommendation.upper(), 0) >= 0:
            return None
        return f"VETO: LLM consensus is {ai.llm_recommendation.upper()} against LONG"


def _prediction_direction(ai: AISignals) -> int | None:
    """+1 up, -1 down, None when the prediction carries no direction."""
    if ai.prediction_direction is not None:
        return {"up": 1, "down": -1}.get(ai.prediction_direction.lower())
    if ai.prediction_return is not None and ai.prediction_return != 0:
        return 1 if ai.prediction_return > 0 else -1
    return None
