"""Sentiment cost tracker — daily budget enforcement for LLM API calls."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from trading_bot.state_file import atomic_write_json, read_json_dict

logger = logging.getLogger(__name__)

# Fractions of the daily budget that raise an alert, once per UTC day.
# 1.0 is raised by mark_exhausted() when a run is refused: the pre-check
# blocks on the worst case, so recorded spend may never reach 100 %.
ALERT_THRESHOLDS = (0.5, 0.8)
EXHAUSTED = 1.0


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SentimentCostTracker:
    """Track daily API spending (UTC day) and enforce a budget cap.

    With ``state_path`` the day's spend (and which alerts already fired) is
    persisted after every change and restored on construction, so a restart
    can neither reset the budget nor repeat an alert.
    """

    def __init__(
        self,
        max_daily_cost_usd: float = 5.0,
        state_path: Path | str | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._max_daily = max_daily_cost_usd
        self._clock = clock
        self._path = Path(state_path) if state_path is not None else None
        self._day: date = self._today()
        self._spent: float = 0.0
        self._alerted: set[float] = set()
        self._pending: list[float] = []
        self._load()

    def record_cost(self, cost_usd: float) -> None:
        """Record an API call cost."""
        self._roll_day()
        self._spent += cost_usd
        for threshold in ALERT_THRESHOLDS:
            if self._spent >= threshold * self._max_daily:
                self._raise_alert(threshold)
        self._save()

    def mark_exhausted(self) -> None:
        """Record that a run was refused for lack of budget (alerts once/day)."""
        self._roll_day()
        self._raise_alert(EXHAUSTED)
        self._save()

    def pop_alerts(self) -> list[float]:
        """Thresholds (fractions of the budget) newly reached since last call."""
        alerts, self._pending = self._pending, []
        return alerts

    @property
    def max_daily_usd(self) -> float:
        return self._max_daily

    def can_spend(self, estimated_cost_usd: float) -> bool:
        """Return True if spending *estimated_cost_usd* would stay within budget."""
        return self.daily_spent_usd + estimated_cost_usd <= self._max_daily

    @property
    def daily_spent_usd(self) -> float:
        """Return total spending for the current UTC day."""
        self._roll_day()
        return self._spent

    def _today(self) -> date:
        return self._clock().date()

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._spent = 0.0
            self._alerted = set()

    def _raise_alert(self, threshold: float) -> None:
        if threshold not in self._alerted:
            self._alerted.add(threshold)
            self._pending.append(threshold)

    def _load(self) -> None:
        if self._path is None:
            return
        data = read_json_dict(self._path)
        if data is None:
            return
        try:
            day = date.fromisoformat(str(data["day"]))
            spent = float(data["spent_usd"])
            alerted = {float(t) for t in data.get("alerted", [])}
        except (KeyError, TypeError, ValueError):
            logger.warning("LLM cost state at %s is invalid — ignoring", self._path)
            return
        if day == self._day:
            self._spent = spent
            self._alerted = alerted
            logger.info(
                "LLM spend restored from %s: $%.4f of $%.2f today",
                self._path,
                spent,
                self._max_daily,
            )

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            atomic_write_json(
                self._path,
                {
                    "day": self._day.isoformat(),
                    "spent_usd": self._spent,
                    "alerted": sorted(self._alerted),
                },
            )
        except OSError:
            logger.exception("Failed to persist LLM cost state to %s", self._path)
