"""Abstract base class for order executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
    async def cancel_order(self, order_id: str, pair: str | None = None) -> bool: ...

    @abstractmethod
    async def get_open_orders(
        self, pair: str | None = None
    ) -> list[dict[str, object]]: ...

    @abstractmethod
    async def get_balance(self) -> dict[str, float]: ...

    @abstractmethod
    async def get_position(self, pair: str) -> float:
        """Base-asset quantity currently held (and sellable) for ``pair``."""

    @abstractmethod
    async def close_all_positions(self) -> list[OrderResult]: ...

    def update_prices(self, prices: dict[str, float]) -> None:  # noqa: B027
        """Receive the latest market price per pair.

        No-op by default: executors backed by an exchange price positions
        themselves. The paper executor uses it for mark-to-market valuation.
        """

    async def place_stop_loss(
        self, pair: str, quantity: float, stop_price: float
    ) -> str | None:
        """Place a native sell stop; return its order id.

        None means no native stop was placed and the caller must watch the
        stop itself. Default: no native stops.
        """
        return None

    async def get_order(self, order_id: str, pair: str) -> dict[str, Any]:
        """Fetch a native order (ccxt order structure)."""
        raise NotImplementedError(f"{type(self).__name__} has no native orders")

    async def close(self) -> None:  # noqa: B027
        """Release exchange connections. No-op by default."""
