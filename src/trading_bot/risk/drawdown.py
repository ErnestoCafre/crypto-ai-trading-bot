"""Drawdown monitor — latching kill-switch on excessive losses."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from trading_bot.config.settings import Settings, get_settings


class TradingState(StrEnum):
    ACTIVE = "active"
    # Daily loss limit hit: no new entries until the next UTC day.
    DAILY_PAUSED = "daily_paused"
    # Max drawdown hit: no new entries until a manual reset.
    HALTED = "halted"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DrawdownMonitor:
    def __init__(
        self,
        initial_value: float = 0.0,
        clock: Callable[[], datetime] = _utc_now,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._max_dd_pct = settings.MAX_DRAWDOWN_PCT
        self._max_daily_loss_pct = settings.MAX_DAILY_LOSS_PCT
        self._clock = clock

        self._peak_value: float = initial_value
        self._day_start_value: float = initial_value
        self._current_day: str = self._today()
        self._last_value: float = initial_value

        self._state = TradingState.ACTIVE
        self._halt_reason: str | None = None
        self._halted_at: str | None = None

    def update(self, portfolio_value: float) -> TradingState | None:
        """Evaluate the limits against a new portfolio value.

        Returns the new state only on a transition into DAILY_PAUSED or
        HALTED, so callers liquidate and notify exactly once. Once HALTED
        the monitor stays latched until :meth:`reset`; DAILY_PAUSED lifts by
        itself on the next UTC day.
        """
        self._last_value = portfolio_value

        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        today = self._today()
        if today != self._current_day:
            self._current_day = today
            self._day_start_value = portfolio_value
            if self._state == TradingState.DAILY_PAUSED:
                self._set_state(TradingState.ACTIVE, None)

        if self._state == TradingState.HALTED:
            return None

        if self._peak_value > 0 and self.current_drawdown_pct >= self._max_dd_pct:
            self._set_state(
                TradingState.HALTED,
                f"max drawdown {self.current_drawdown_pct:.2f}% "
                f">= {self._max_dd_pct:.2f}%",
            )
            return TradingState.HALTED

        if self._state == TradingState.DAILY_PAUSED:
            return None

        if (
            self._day_start_value > 0
            and self.daily_pnl_pct < 0
            and abs(self.daily_pnl_pct) >= self._max_daily_loss_pct
        ):
            self._set_state(
                TradingState.DAILY_PAUSED,
                f"daily loss {abs(self.daily_pnl_pct):.2f}% "
                f">= {self._max_daily_loss_pct:.2f}%",
            )
            return TradingState.DAILY_PAUSED

        return None

    def check(self, portfolio_value: float) -> bool:
        """Return True if trading should STOP (kill-switch engaged)."""
        self.update(portfolio_value)
        return self._state != TradingState.ACTIVE

    def reset(self, portfolio_value: float) -> None:
        """Manual reset: clear any halt and re-baseline at ``portfolio_value``."""
        self._set_state(TradingState.ACTIVE, None)
        self._peak_value = portfolio_value
        self._day_start_value = portfolio_value
        self._last_value = portfolio_value
        self._current_day = self._today()

    def rebaseline(self, portfolio_value: float) -> None:
        """Re-anchor peak and day start at ``portfolio_value``, keeping the state.

        For accounts that restart from scratch (paper): the latch survives,
        but the old peak no longer relates to the new balance.
        """
        self._peak_value = portfolio_value
        self._day_start_value = portfolio_value
        self._last_value = portfolio_value
        self._current_day = self._today()

    @property
    def state(self) -> TradingState:
        return self._state

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    @property
    def entries_allowed(self) -> bool:
        return self._state == TradingState.ACTIVE

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._last_value) / self._peak_value * 100

    @property
    def daily_pnl_pct(self) -> float:
        if self._day_start_value <= 0:
            return 0.0
        return (self._last_value - self._day_start_value) / self._day_start_value * 100

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_value": self._peak_value,
            "day_start_value": self._day_start_value,
            "current_day": self._current_day,
            "last_value": self._last_value,
            "state": self._state.value,
            "halt_reason": self._halt_reason,
            "halted_at": self._halted_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        clock: Callable[[], datetime] = _utc_now,
        settings: Settings | None = None,
    ) -> DrawdownMonitor:
        """Rebuild a monitor from :meth:`to_dict` output.

        Raises KeyError/TypeError/ValueError on malformed data.
        """
        monitor = cls(
            initial_value=float(data["peak_value"]), clock=clock, settings=settings
        )
        monitor._day_start_value = float(data["day_start_value"])
        monitor._current_day = str(data["current_day"])
        monitor._last_value = float(data["last_value"])
        monitor._state = TradingState(data["state"])
        reason = data.get("halt_reason")
        halted_at = data.get("halted_at")
        monitor._halt_reason = str(reason) if reason is not None else None
        monitor._halted_at = str(halted_at) if halted_at is not None else None
        return monitor

    def _set_state(self, state: TradingState, reason: str | None) -> None:
        self._state = state
        self._halt_reason = reason
        self._halted_at = None if reason is None else self._clock().isoformat()

    def _today(self) -> str:
        return self._clock().strftime("%Y-%m-%d")
