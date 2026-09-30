"""AI test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _sentiment_backend_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests mock the transformers pipeline, so the optional extra is not needed."""
    monkeypatch.setattr(
        "trading_bot.ai.sentiment.cryptobert._require_backend", lambda: None
    )
