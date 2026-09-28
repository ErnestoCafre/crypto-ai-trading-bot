"""Transformer sentiment analyzer for crypto text (CryptoBERT family)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from trading_bot.ai.sentiment.base import BaseSentimentAnalyzer
from trading_bot.core.models import SentimentResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ElKulako/cryptobert"


@dataclass(frozen=True)
class ModelProfile:
    # Lower-cased pipeline label -> polarity in [-1, 1].
    labels: dict[str, float]
    # Max token length the model was trained on.
    max_length: int


# Labels verified against each model's config.json / model card on
# Hugging Face. kk08/CryptoBERT ships no id2label, so the pipeline emits
# generic LABEL_0 (negative) / LABEL_1 (positive) per its model card.
MODEL_PROFILES: dict[str, ModelProfile] = {
    "elkulako/cryptobert": ModelProfile(
        {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0},
        max_length=128,  # trained at 128; the card advises against more
    ),
    "kk08/cryptobert": ModelProfile(
        {"label_0": -1.0, "label_1": 1.0},
        max_length=512,
    ),
    "prosusai/finbert": ModelProfile(
        {"negative": -1.0, "neutral": 0.0, "positive": 1.0},
        max_length=512,
    ),
    "burakutf/finetuned-finbert-crypto": ModelProfile(
        {"negative": -1.0, "neutral": 0.0, "positive": 1.0},
        max_length=512,
    ),
}

# Fallback for models without a profile, matched against config.id2label.
_GENERIC_POLARITY = {
    "positive": 1.0,
    "bullish": 1.0,
    "negative": -1.0,
    "bearish": -1.0,
    "neutral": 0.0,
}


class CryptoBERTAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analysis with a Hugging Face text-classification model.

    Lazy-loads the transformers pipeline on first use to avoid downloading
    ~500 MB on import. The score is the probability-weighted polarity over
    all classes (P(bullish) - P(bearish) for three-class models), so it is
    continuous in [-1, 1] instead of the winning label's raw probability.
    Inference is synchronous: async callers should run it in a thread.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int | None = None,
        max_length: int | None = None,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._profile = MODEL_PROFILES.get(model_name.lower())
        self._max_length = max_length or (
            self._profile.max_length if self._profile else 512
        )
        self._batch_size = batch_size
        self._pipeline: Any = None
        self._labels: dict[str, float] = (
            dict(self._profile.labels) if self._profile else {}
        )
        self._warned_labels: set[str] = set()

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single text for sentiment."""
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze several texts in one pipeline call."""
        results: list[SentimentResult | None] = [None] * len(texts)
        pending = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = self._neutral(text)

        if pending:
            pipe = self._get_pipeline()
            outputs = pipe(
                [t for _, t in pending],
                top_k=None,
                truncation=True,
                max_length=self._max_length,
                batch_size=self._batch_size,
            )
            for (i, text), scores in zip(pending, outputs, strict=True):
                results[i] = self._to_result(text, scores)

        return [r for r in results if r is not None]

    def is_loaded(self) -> bool:
        """Return True if the pipeline has been loaded."""
        return self._pipeline is not None

    def _to_result(self, text: str, scores: Any) -> SentimentResult:
        # top_k=None yields every class; tolerate a bare dict just in case.
        entries = [scores] if isinstance(scores, dict) else list(scores)
        score = 0.0
        known_mass = 0.0
        best_label, best_prob = "unknown", 0.0
        for entry in entries:
            label = str(entry["label"]).lower()
            prob = float(entry["score"])
            polarity = self._labels.get(label)
            if polarity is None:
                self._warn_unknown(label)
                continue
            score += polarity * prob
            known_mass += prob
            if prob > best_prob:
                best_label, best_prob = label, prob

        if known_mass <= 0:
            # Nothing mappable: report zero confidence rather than a
            # confident-looking neutral.
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label="unknown",
                model_used=self._model_name,
                analyzed_text=text[:200],
            )

        return SentimentResult(
            score=max(-1.0, min(1.0, score / known_mass)),
            confidence=max(0.0, min(1.0, best_prob)),
            label=best_label,
            model_used=self._model_name,
            analyzed_text=text[:200],
        )

    def _warn_unknown(self, label: str) -> None:
        if label not in self._warned_labels:
            self._warned_labels.add(label)
            logger.warning(
                "Sentiment model %s emitted unmapped label %r; it is ignored",
                self._model_name,
                label,
            )

    def _neutral(self, text: str) -> SentimentResult:
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            label="neutral",
            model_used=self._model_name,
            analyzed_text=text,
        )

    def _get_pipeline(self) -> Any:
        """Lazy-load the transformers pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            kwargs: dict[str, Any] = {"model": self._model_name}
            if self._device is not None:
                kwargs["device"] = self._device

            self._pipeline = pipeline("text-classification", **kwargs)  # type: ignore[call-overload]
            if self._profile is None:
                self._labels = self._labels_from_config(self._pipeline)
            logger.info(
                "Sentiment pipeline loaded: %s (labels=%s, max_length=%d)",
                self._model_name,
                self._labels,
                self._max_length,
            )

        return self._pipeline

    def _labels_from_config(self, pipe: Any) -> dict[str, float]:
        """Derive polarities from config.id2label for unprofiled models."""
        id2label = getattr(getattr(pipe.model, "config", None), "id2label", {}) or {}
        labels: dict[str, float] = {}
        for name in id2label.values():
            label = str(name).lower()
            if label in _GENERIC_POLARITY:
                labels[label] = _GENERIC_POLARITY[label]
            else:
                logger.warning(
                    "Sentiment model %s: cannot infer polarity of label %r",
                    self._model_name,
                    label,
                )
        return labels
