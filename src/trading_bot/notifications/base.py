"""Abstract base class for notification systems."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseNotifier(ABC):
    @abstractmethod
    async def send_alert(self, message: str, level: str = "info") -> bool: ...

    @abstractmethod
    async def send_trade_notification(self, trade: dict[str, object]) -> bool: ...

    @abstractmethod
    async def send_ai_report(self, analysis: dict[str, object]) -> bool: ...
