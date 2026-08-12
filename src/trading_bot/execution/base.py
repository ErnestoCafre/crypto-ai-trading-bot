"""Abstract base class for order executors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trading_bot.core.models import OrderResult


class BaseExecutor(ABC):
    @abstractmethod
    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_open_orders(
        self, pair: str | None = None
    ) -> list[dict[str, object]]: ...

    @abstractmethod
    async def get_balance(self) -> dict[str, float]: ...

    @abstractmethod
    async def close_all_positions(self) -> list[OrderResult]: ...
