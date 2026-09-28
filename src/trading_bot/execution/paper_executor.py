"""Paper trading executor — simulates order execution in-memory."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from trading_bot.core.models import OrderResult
from trading_bot.execution.base import BaseExecutor

logger = logging.getLogger(__name__)

# Quantities at or below this are float residue, not a real position.
_QTY_EPSILON = 1e-12


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
        self._last_prices: dict[str, float] = {}
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct

    def update_prices(self, prices: dict[str, float]) -> None:
        """Record the latest market price per pair for mark-to-market."""
        for pair, price in prices.items():
            if price > 0:
                self._last_prices[pair] = price

    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult:
        base_price = price if price is not None else self._last_prices.get(pair)
        if base_price is None or base_price <= 0:
            logger.warning("Paper %s %s rejected: no market price", side, pair)
            return self._rejected(pair, side, base_price or 0.0)
        # Market prices come from update_prices; a fill price (e.g. a stop
        # level) only seeds the mark when none is known yet.
        self._last_prices.setdefault(pair, base_price)

        # Apply slippage
        if side == "buy":
            fill_price = base_price * (1 + self._slippage_pct / 100)
        else:
            fill_price = base_price * (1 - self._slippage_pct / 100)
        fee_rate = self._fee_pct / 100

        if side == "buy":
            # Spot account: never spend more cash than available, fees included.
            affordable = max(self._cash, 0.0) / (fill_price * (1 + fee_rate))
            if quantity > affordable:
                logger.warning(
                    "Paper buy %s reduced from %.8f to %.8f: insufficient cash",
                    pair,
                    quantity,
                    affordable,
                )
                quantity = affordable
        else:
            # Spot account: can only sell what is held.
            held = self._positions.get(pair, {}).get("qty", 0.0)
            if quantity > held:
                logger.warning(
                    "Paper sell %s reduced from %.8f to held %.8f",
                    pair,
                    quantity,
                    held,
                )
                quantity = held

        if quantity <= _QTY_EPSILON:
            logger.warning("Paper %s %s rejected: zero executable quantity", side, pair)
            return self._rejected(pair, side, base_price)

        cost = fill_price * quantity
        fee = cost * fee_rate
        slippage_actual = abs(fill_price - base_price) / base_price * 100

        if side == "buy":
            self._cash -= cost + fee
            pos = self._positions.get(pair, {"qty": 0.0, "avg_price": 0.0})
            total_qty = pos["qty"] + quantity
            pos["avg_price"] = (
                pos["avg_price"] * pos["qty"] + fill_price * quantity
            ) / total_qty
            pos["qty"] = total_qty
            self._positions[pair] = pos
        else:  # sell
            self._cash += cost - fee
            pos = self._positions[pair]
            pos["qty"] -= quantity
            if pos["qty"] <= _QTY_EPSILON:
                self._positions.pop(pair, None)

        return OrderResult(
            order_id=str(uuid.uuid4()),
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

    async def cancel_order(self, order_id: str, pair: str | None = None) -> bool:
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
            pos["qty"] * self._mark_price(pair, pos)
            for pair, pos in self._positions.items()
        )
        return {
            "cash": self._cash,
            "positions_value": positions_value,
            "total": self._cash + positions_value,
        }

    async def get_position(self, pair: str) -> float:
        return self._positions.get(pair, {}).get("qty", 0.0)

    async def close_all_positions(self) -> list[OrderResult]:
        results: list[OrderResult] = []
        for pair, pos in list(self._positions.items()):
            if pos["qty"] > 0:
                result = await self.place_order(
                    pair=pair,
                    side="sell",
                    order_type="market",
                    quantity=pos["qty"],
                    price=self._mark_price(pair, pos),
                )
                results.append(result)
        return results

    def _mark_price(self, pair: str, pos: dict[str, float]) -> float:
        """Last market price for ``pair``; entry price if none seen yet."""
        return self._last_prices.get(pair, pos["avg_price"])

    @staticmethod
    def _rejected(pair: str, side: str, price: float) -> OrderResult:
        return OrderResult(
            order_id=str(uuid.uuid4()),
            exchange_order_id=None,
            pair=pair,
            side=side,
            filled_price=price,
            filled_quantity=0.0,
            fee=0.0,
            slippage_pct=0.0,
            status="rejected",
            timestamp=datetime.now(UTC),
        )
