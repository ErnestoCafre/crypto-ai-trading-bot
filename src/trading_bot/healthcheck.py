"""Container health check: is the trading cycle still completing?

The bot rewrites its risk state file (``RISK_STATE_PATH``) with a
``saved_at`` timestamp at the end of every trading cycle that could value the
portfolio. The process is healthy while that heartbeat is younger than
``HEALTH_MAX_CYCLES`` trading intervals (default 3): TRADING_INTERVAL_MINUTES,
or one hour in donchian_ensemble mode (its cycle runs hourly).

Usage (exit code 0 = healthy, 1 = unhealthy):
    python -m trading_bot.healthcheck
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trading_bot.config.settings import Settings, get_settings


def risk_state_path(settings: Settings | None = None) -> Path:
    """Resolved risk state file for the configured BOT_MODE (as core.bot)."""
    s = settings or get_settings()
    return Path(str(s.RISK_STATE_PATH).replace("{mode}", str(s.BOT_MODE)))


def cycle_interval(settings: Settings | None = None) -> timedelta:
    """Time between trading cycles (as core.bot schedules them)."""
    s = settings or get_settings()
    if s.STRATEGY_MODE == "donchian_ensemble":
        return timedelta(hours=1)
    return timedelta(minutes=s.TRADING_INTERVAL_MINUTES)


def load_risk_state(settings: Settings | None = None) -> dict[str, Any] | None:
    """Read-only view of the bot's risk state; None if missing or unreadable."""
    try:
        data = json.loads(risk_state_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def heartbeat_age(
    state: dict[str, Any] | None, now: datetime | None = None
) -> timedelta | None:
    """Time since the state was saved; None if missing or malformed."""
    if not state or not isinstance(state.get("saved_at"), str):
        return None
    try:
        saved = datetime.fromisoformat(state["saved_at"])
    except ValueError:
        return None
    if saved.tzinfo is None:
        saved = saved.replace(tzinfo=UTC)
    return (now or datetime.now(UTC)) - saved


def check(settings: Settings | None = None, now: datetime | None = None) -> str | None:
    """Return None when healthy, else the reason."""
    s = settings or get_settings()
    age = heartbeat_age(load_risk_state(s), now)
    if age is None:
        return f"no heartbeat in {risk_state_path(s)}"
    cycles = int(os.environ.get("HEALTH_MAX_CYCLES", "3"))
    limit = cycle_interval(s) * cycles
    if age > limit:
        return f"last completed cycle {age} ago (limit {limit})"
    return None


def main() -> None:
    reason = check()
    if reason is not None:
        print(f"unhealthy: {reason}", file=sys.stderr)
        sys.exit(1)
    print("healthy")


if __name__ == "__main__":
    main()
