"""Per-model LLM pricing (USD per million tokens, input and output apart).

Anthropic first-party API list prices as of 2026-09 (Claude API model
table). Dated IDs (``claude-haiku-4-5-20251001``) resolve to their alias.
Models not listed are priced at ``UNKNOWN_MODEL_PRICE`` — the most expensive
listed rate — so an unknown model can only overstate spend, never understate
it. Add a row here when switching LLM_MODEL.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


_PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00),
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-fable-5-1": ModelPrice(10.00, 50.00),
}

UNKNOWN_MODEL_PRICE = ModelPrice(
    input_per_mtok=max(p.input_per_mtok for p in _PRICES.values()),
    output_per_mtok=max(p.output_per_mtok for p in _PRICES.values()),
)

_DATE_SUFFIX = re.compile(r"-\d{8}$")
_warned: set[str] = set()


def get_price(model: str) -> ModelPrice:
    """Price for ``model``; conservative fallback (and one warning) if unknown."""
    key = _DATE_SUFFIX.sub("", model.strip().lower())
    price = _PRICES.get(key)
    if price is not None:
        return price
    if model not in _warned:
        _warned.add(model)
        logger.warning(
            "No price for LLM model %r — budgeting at the conservative "
            "$%.2f/$%.2f per MTok (input/output). Add it to ai/pricing.py.",
            model,
            UNKNOWN_MODEL_PRICE.input_per_mtok,
            UNKNOWN_MODEL_PRICE.output_per_mtok,
        )
    return UNKNOWN_MODEL_PRICE


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return get_price(model).cost(input_tokens, output_tokens)
