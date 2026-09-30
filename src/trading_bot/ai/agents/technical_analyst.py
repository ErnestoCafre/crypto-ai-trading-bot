"""Technical analyst agent — LLM-powered technical analysis."""

from __future__ import annotations

from trading_bot.ai.agents.base import BaseAgent
from trading_bot.ai.agents.prompts.technical import TECHNICAL_ANALYST_PROMPT


class TechnicalAnalystAgent(BaseAgent):
    """Analyses technical indicators via an LLM and returns a recommendation."""

    @property
    def name(self) -> str:
        return "technical_analyst"

    def _build_prompt(self, context: dict[str, object]) -> str:
        indicators = context.get("indicators", {})
        if isinstance(indicators, dict):
            indicators_str = "\n".join(f"  {k}: {v}" for k, v in indicators.items())
        else:
            indicators_str = str(indicators)

        return TECHNICAL_ANALYST_PROMPT.format(
            pair=context.get("pair", "UNKNOWN"),
            timeframe=context.get("timeframe", "1h"),
            indicators=indicators_str,
            regime=context.get("regime", "UNKNOWN"),
        )
