"""Telegram notifier stub."""

from __future__ import annotations

from trading_bot.notifications.base import BaseNotifier


class TelegramNotifier(BaseNotifier):
    async def send_alert(self, message: str, level: str = "info") -> bool:
        raise NotImplementedError

    async def send_trade_notification(self, trade: dict[str, object]) -> bool:
        raise NotImplementedError

    async def send_ai_report(self, analysis: dict[str, object]) -> bool:
        raise NotImplementedError
