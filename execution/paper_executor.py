"""Paper trading executor — simulates order execution in-memory."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from core.models import OrderResult
from execution.base import BaseExecutor


class PaperExecutor(BaseExecutor):
    def __init__(
        self,
        initial_balance: float = 100_000.0,
        slippage_pct: float = 0.05,
        fee_pct: float = 0.1,
    ) -> None:
        self._cash: float = initial_balance
        self._positions: dict[str, dict[str, float]] = {}
        self._open_orders: dict[str, dict[str, object]] = {}
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct

    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult:
        base_price = price if price is not None else 0.0

        # Apply slippage
        if side == "buy":
            fill_price = base_price * (1 + self._slippage_pct / 100)
        else:
            fill_price = base_price * (1 - self._slippage_pct / 100)

        cost = fill_price * quantity
        fee = cost * (self._fee_pct / 100)
        slippage_actual = (
            abs(fill_price - base_price) / base_price * 100 if base_price > 0 else 0.0
        )

        order_id = str(uuid.uuid4())

        if side == "buy":
            self._cash -= cost + fee
            pos = self._positions.get(pair, {"qty": 0.0, "avg_price": 0.0})
            total_qty = pos["qty"] + quantity
            if total_qty > 0:
                pos["avg_price"] = (
                    pos["avg_price"] * pos["qty"] + fill_price * quantity
                ) / total_qty
            pos["qty"] = total_qty
            self._positions[pair] = pos
        else:  # sell
            self._cash += cost - fee
            pos = self._positions.get(pair, {"qty": 0.0, "avg_price": 0.0})
            pos["qty"] -= quantity
            if pos["qty"] <= 0:
                self._positions.pop(pair, None)
            else:
                self._positions[pair] = pos

        return OrderResult(
            order_id=order_id,
            exchange_order_id=None,
            pair=pair,
            side=side,
            filled_price=fill_price,
            filled_quantity=quantity,
            fee=fee,
            slippage_pct=slippage_actual,
            status="filled",
            timestamp=datetime.now(UTC),
        )

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._open_orders:
            del self._open_orders[order_id]
            return True
        return False

    async def get_open_orders(self, pair: str | None = None) -> list[dict[str, object]]:
        if pair is None:
            return list(self._open_orders.values())
        return [o for o in self._open_orders.values() if o.get("pair") == pair]

    async def get_balance(self) -> dict[str, float]:
        positions_value = sum(
            pos["qty"] * pos["avg_price"] for pos in self._positions.values()
        )
        return {
            "cash": self._cash,
            "positions_value": positions_value,
            "total": self._cash + positions_value,
        }

    async def close_all_positions(self) -> list[OrderResult]:
        results: list[OrderResult] = []
        for pair, pos in list(self._positions.items()):
            if pos["qty"] > 0:
                result = await self.place_order(
                    pair=pair,
                    side="sell",
                    order_type="market",
                    quantity=pos["qty"],
                    price=pos["avg_price"],
                )
                results.append(result)
        return results
