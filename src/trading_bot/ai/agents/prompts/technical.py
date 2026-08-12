"""Technical analyst prompt template."""

from __future__ import annotations

TECHNICAL_ANALYST_PROMPT = (
    "You are a senior cryptocurrency technical analyst. "
    "Analyze the following technical indicators for {pair} "
    "on the {timeframe} timeframe.\n\n"
    "Current Indicators:\n{indicators}\n\n"
    "Market Regime: {regime}\n\n"
    "Return ONLY valid JSON:\n"
    '{{"recommendation": "LONG|SHORT|NEUTRAL", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "2-3 sentences", '
    '"key_factors": ["factor1", "factor2"]}}'
)
