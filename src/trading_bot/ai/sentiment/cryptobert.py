"""CryptoBERT sentiment analyzer — crypto-specific BERT model."""

from __future__ import annotations

import logging
from typing import Any

from trading_bot.ai.sentiment.base import BaseSentimentAnalyzer
from trading_bot.core.models import SentimentResult

logger = logging.getLogger(__name__)


class CryptoBERTAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analysis using the kk08/CryptoBERT model.

    Lazy-loads the transformers pipeline on first use to avoid
    downloading ~500 MB on import.
    """

    def __init__(
        self,
        model_name: str = "kk08/CryptoBERT",
        device: int | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._pipeline: Any = None

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single text for sentiment."""
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label="neutral",
                model_used=self._model_name,
                analyzed_text=text,
            )

        pipe = self._get_pipeline()
        result = pipe(text[: self._max_length])[0]  # type: ignore[operator]

        label = str(result["label"]).lower()
        raw_score = float(result["score"])

        # Map to [-1, 1] score
        if "positive" in label or "bullish" in label:
            score = raw_score
        elif "negative" in label or "bearish" in label:
            score = -raw_score
        else:
            score = 0.0

        return SentimentResult(
            score=max(-1.0, min(1.0, score)),
            confidence=max(0.0, min(1.0, raw_score)),
            label=label,
            model_used=self._model_name,
            analyzed_text=text[:200],
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]

    def is_loaded(self) -> bool:
        """Return True if the pipeline has been loaded."""
        return self._pipeline is not None

    def _get_pipeline(self) -> Any:
        """Lazy-load the transformers pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            kwargs: dict[str, Any] = {
                "model": self._model_name,
                "truncation": True,
                "max_length": self._max_length,
            }
            if self._device is not None:
                kwargs["device"] = self._device

            self._pipeline = pipeline("text-classification", **kwargs)  # type: ignore[call-overload]
            logger.info("CryptoBERT pipeline loaded: %s", self._model_name)

        return self._pipeline
