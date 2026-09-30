"""Tests for CryptoBERTAnalyzer (mocked pipeline — never downloads model in CI)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_bot.ai.sentiment.cryptobert import (
    DEFAULT_MODEL,
    MODEL_PROFILES,
    CryptoBERTAnalyzer,
)
from trading_bot.core.models import SentimentResult


def _pipe(*per_text: dict[str, float]) -> MagicMock:
    """Mock pipeline returning, per input text, all class scores (top_k=None)."""
    pipe = MagicMock()
    outputs = [
        [{"label": k, "score": v} for k, v in scores.items()] for scores in per_text
    ]
    pipe.side_effect = lambda texts, **kw: outputs[: len(texts)]
    return pipe


def _analyzer(model: str, pipe: MagicMock) -> CryptoBERTAnalyzer:
    a = CryptoBERTAnalyzer(model_name=model)
    a._pipeline = pipe
    return a


# Real label sets as emitted by each model's pipeline (config.json id2label;
# kk08 has none, so transformers emits LABEL_0/LABEL_1).
LABEL_TABLE: list[tuple[str, dict[str, float], float]] = [
    ("kk08/CryptoBERT", {"LABEL_1": 0.9, "LABEL_0": 0.1}, 0.8),
    ("kk08/CryptoBERT", {"LABEL_0": 0.9, "LABEL_1": 0.1}, -0.8),
    ("ElKulako/cryptobert", {"Bullish": 0.7, "Neutral": 0.2, "Bearish": 0.1}, 0.6),
    ("ElKulako/cryptobert", {"Bearish": 0.8, "Neutral": 0.15, "Bullish": 0.05}, -0.75),
    ("ElKulako/cryptobert", {"Neutral": 0.9, "Bullish": 0.05, "Bearish": 0.05}, 0.0),
    ("ProsusAI/finbert", {"positive": 0.6, "negative": 0.1, "neutral": 0.3}, 0.5),
    ("ProsusAI/finbert", {"negative": 0.7, "positive": 0.1, "neutral": 0.2}, -0.6),
    (
        "burakutf/finetuned-finbert-crypto",
        {"Positive": 0.5, "Neutral": 0.3, "Negative": 0.2},
        0.3,
    ),
]


class TestLabelMapping:
    @pytest.mark.parametrize(("model", "scores", "expected"), LABEL_TABLE)
    def test_real_labels_map_to_polarity(
        self, model: str, scores: dict[str, float], expected: float
    ) -> None:
        result = _analyzer(model, _pipe(scores)).analyze("some crypto headline")
        assert result.score == pytest.approx(expected)
        assert result.confidence == pytest.approx(max(scores.values()))
        assert result.label == max(scores, key=scores.__getitem__).lower()

    def test_default_model_is_elkulako(self) -> None:
        assert DEFAULT_MODEL == "ElKulako/cryptobert"
        assert CryptoBERTAnalyzer()._max_length == 128

    def test_every_profile_is_symmetric(self) -> None:
        for profile in MODEL_PROFILES.values():
            assert min(profile.labels.values()) == -1.0
            assert max(profile.labels.values()) == 1.0

    def test_unknown_label_warns_and_is_not_silent_neutral(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        a = _analyzer("ElKulako/cryptobert", _pipe({"LABEL_7": 0.99}))
        with caplog.at_level(logging.WARNING):
            result = a.analyze("text")
        assert "unmapped label" in caplog.text
        assert result.label == "unknown"
        assert result.confidence == 0.0

    def test_unknown_label_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        a = _analyzer("ElKulako/cryptobert", _pipe({"weird": 0.5, "Bullish": 0.5}))
        with caplog.at_level(logging.WARNING):
            a.analyze("one")
            a.analyze("two")
        assert caplog.text.count("unmapped label") == 1

    def test_unprofiled_model_uses_config_id2label(self) -> None:
        a = CryptoBERTAnalyzer(model_name="someone/crypto-sentiment")
        pipe = _pipe({"BULLISH": 0.8, "BEARISH": 0.2})
        pipe.model.config.id2label = {0: "BEARISH", 1: "BULLISH"}
        a._labels = a._labels_from_config(pipe)
        a._pipeline = pipe
        assert a.analyze("text").score == pytest.approx(0.6)


class TestAnalyze:
    def test_returns_sentiment_result(self) -> None:
        a = _analyzer("ElKulako/cryptobert", _pipe({"Bullish": 0.9, "Bearish": 0.1}))
        result = a.analyze("Bitcoin is going to the moon!")
        assert isinstance(result, SentimentResult)
        assert -1.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_text_handled_without_pipeline(self) -> None:
        pipe = _pipe()
        result = _analyzer("ElKulako/cryptobert", pipe).analyze("")
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.label == "neutral"
        pipe.assert_not_called()


class TestBatch:
    def test_batch_uses_one_pipeline_call(self) -> None:
        pipe = _pipe(
            {"Bullish": 0.9, "Bearish": 0.1},
            {"Bearish": 0.9, "Bullish": 0.1},
            {"Neutral": 1.0},
        )
        a = _analyzer("ElKulako/cryptobert", pipe)

        results = a.analyze_batch(["up", "", "down", "flat"])

        assert pipe.call_count == 1
        texts: Any = pipe.call_args.args[0]
        assert texts == ["up", "down", "flat"]
        kwargs = pipe.call_args.kwargs
        assert kwargs["top_k"] is None
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 128
        assert [r.label for r in results] == [
            "bullish",
            "neutral",
            "bearish",
            "neutral",
        ]
        assert results[0].score > 0 > results[2].score


class TestIsLoaded:
    def test_not_loaded_initially(self) -> None:
        a = CryptoBERTAnalyzer()
        assert not a.is_loaded()

    def test_loaded_after_pipeline_set(self) -> None:
        assert _analyzer("ElKulako/cryptobert", _pipe()).is_loaded()


def test_missing_optional_backend_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trading_bot.ai.sentiment.cryptobert as mod

    monkeypatch.undo()  # drop the conftest stub
    monkeypatch.setattr(mod, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="--extra sentiment"):
        mod.CryptoBERTAnalyzer()
