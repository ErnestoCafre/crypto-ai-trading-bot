"""Sentiment analyst prompt template."""

from __future__ import annotations

SENTIMENT_ANALYST_PROMPT = (
    "You are a cryptocurrency market sentiment analyst. "
    "Analyze the following sentiment data for {pair}.\n\n"
    "Aggregated Sentiment Score: {sentiment_score} "
    "(-1=bearish, +1=bullish)\n"
    "Fear & Greed Index: {fear_greed}\n"
    "Recent Headlines:\n{headlines}\n\n"
    "Return ONLY valid JSON:\n"
    '{{"recommendation": "LONG|SHORT|NEUTRAL", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "2-3 sentences", '
    '"key_factors": ["factor1", "factor2"]}}'
)
