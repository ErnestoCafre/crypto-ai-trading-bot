"""Paper trading executor stub."""

from __future__ import annotations

from core.models import OrderResult
from execution.base import BaseExecutor


class PaperExecutor(BaseExecutor):
    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    async def get_open_orders(self, pair: str | None = None) -> list[dict[str, object]]:
        raise NotImplementedError

    async def get_balance(self) -> dict[str, float]:
        raise NotImplementedError

    async def close_all_positions(self) -> list[OrderResult]:
        raise NotImplementedError
