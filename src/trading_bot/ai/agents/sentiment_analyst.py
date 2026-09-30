"""Sentiment analyst agent — LLM-powered sentiment analysis."""

from __future__ import annotations

import re

from trading_bot.ai.agents.base import BaseAgent
from trading_bot.ai.agents.prompts.sentiment import SENTIMENT_ANALYST_PROMPT

# Headlines come from external news feeds: bound their size so they cannot
# crowd the instructions out of the prompt.
MAX_HEADLINES = 10
MAX_HEADLINE_CHARS = 200
_TAG = re.compile(r"</?\s*headlines?\s*>", re.IGNORECASE)


def _sanitize_headline(text: str) -> str:
    """One line, no delimiter tags, bounded length."""
    flat = " ".join(str(text).split())
    flat = _TAG.sub("", flat)
    return flat[:MAX_HEADLINE_CHARS]


class SentimentAnalystAgent(BaseAgent):
    """Analyses market sentiment via an LLM and returns a recommendation."""

    @property
    def name(self) -> str:
        return "sentiment_analyst"

    def _build_prompt(self, context: dict[str, object]) -> str:
        headlines_raw = context.get("headlines", [])
        headlines: list[str] = []
        if isinstance(headlines_raw, list):
            headlines = [
                h for h in (_sanitize_headline(x) for x in headlines_raw) if h
            ][:MAX_HEADLINES]
        if headlines:
            headlines_str = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
        else:
            headlines_str = "No recent headlines available."

        return SENTIMENT_ANALYST_PROMPT.format(
            pair=context.get("pair", "UNKNOWN"),
            sentiment_score=context.get("sentiment_score", 0.0),
            fear_greed=context.get("fear_greed", "N/A"),
            headlines=headlines_str,
        )
