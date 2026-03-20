"""Sentiment cost tracker — daily budget enforcement for LLM API calls."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class SentimentCostTracker:
    """Track daily API spending and enforce a budget cap."""

    def __init__(self, max_daily_cost_usd: float = 5.0) -> None:
        self._max_daily = max_daily_cost_usd
        self._costs: list[tuple[datetime, float]] = []

    def record_cost(self, cost_usd: float) -> None:
        """Record an API call cost."""
        self._costs.append((datetime.now(UTC), cost_usd))

    def can_spend(self, estimated_cost_usd: float) -> bool:
        """Return True if spending *estimated_cost_usd* would stay within budget."""
        return self.daily_spent_usd + estimated_cost_usd <= self._max_daily

    @property
    def daily_spent_usd(self) -> float:
        """Return total spending for the current UTC day."""
        today = datetime.now(UTC).date()
        return sum(cost for ts, cost in self._costs if ts.date() == today)
