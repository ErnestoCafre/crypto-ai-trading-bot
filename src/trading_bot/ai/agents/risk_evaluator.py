"""Risk evaluator agent — LLM-powered risk assessment."""

from __future__ import annotations

from trading_bot.ai.agents.base import BaseAgent
from trading_bot.ai.agents.prompts.risk import RISK_EVALUATOR_PROMPT


class RiskEvaluatorAgent(BaseAgent):
    """Evaluates portfolio risk via an LLM and returns a recommendation."""

    @property
    def name(self) -> str:
        return "risk_evaluator"

    def _build_prompt(self, context: dict[str, object]) -> str:
        pv = context.get("portfolio_value", 0.0)
        portfolio_val = float(pv) if isinstance(pv, (int, float, str)) else 0.0
        return RISK_EVALUATOR_PROMPT.format(
            pair=context.get("pair", "UNKNOWN"),
            portfolio_value=portfolio_val,
            drawdown_pct=context.get("drawdown_pct", 0.0),
            open_positions=context.get("open_positions", 0),
            max_risk_pct=context.get("max_risk_pct", 1.0),
            max_position_pct=context.get("max_position_pct", 10.0),
            max_drawdown_pct=context.get("max_drawdown_pct", 15.0),
        )
