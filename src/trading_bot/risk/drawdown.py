"""Drawdown monitor — triggers kill-switch on excessive losses."""

from __future__ import annotations

from datetime import UTC, datetime

from trading_bot.config.settings import get_settings


class DrawdownMonitor:
    def __init__(self, initial_value: float = 0.0) -> None:
        settings = get_settings()
        self._max_dd_pct = settings.MAX_DRAWDOWN_PCT
        self._max_daily_loss_pct = settings.MAX_DAILY_LOSS_PCT

        self._peak_value: float = initial_value
        self._day_start_value: float = initial_value
        self._current_day: str = datetime.now(UTC).strftime("%Y-%m-%d")
        self._last_value: float = initial_value

    def check(self, portfolio_value: float) -> bool:
        """Return True if trading should STOP (kill-switch triggered)."""
        self._last_value = portfolio_value

        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        # Day rollover
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_day:
            self._current_day = today
            self._day_start_value = portfolio_value

        # Drawdown from peak
        if self._peak_value > 0 and self.current_drawdown_pct >= self._max_dd_pct:
            return True

        # Daily loss
        return (
            self._day_start_value > 0
            and self.daily_pnl_pct < 0
            and abs(self.daily_pnl_pct) >= self._max_daily_loss_pct
        )

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
