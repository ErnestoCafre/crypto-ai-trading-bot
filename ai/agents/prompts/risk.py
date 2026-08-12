"""Risk evaluator prompt template."""

from __future__ import annotations

RISK_EVALUATOR_PROMPT = (
    "You are a cryptocurrency risk manager. Evaluate "
    "whether it is safe to trade {pair}.\n\n"
    "Portfolio Value: ${portfolio_value:,.2f}\n"
    "Current Drawdown: {drawdown_pct}%\n"
    "Open Positions: {open_positions}\n"
    "Max Risk/Trade: {max_risk_pct}%\n"
    "Max Position: {max_position_pct}%\n"
    "Max Drawdown Limit: {max_drawdown_pct}%\n\n"
    "Return ONLY valid JSON:\n"
    '{{"recommendation": "LONG|SHORT|NEUTRAL", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "2-3 sentences", '
    '"key_factors": ["factor1", "factor2"]}}'
)
