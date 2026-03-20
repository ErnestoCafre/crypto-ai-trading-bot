"""Live trading executor — wraps ccxt async exchange."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from core.models import OrderResult
from execution.base import BaseExecutor

logger = logging.getLogger(__name__)


class LiveExecutor(BaseExecutor):
    def __init__(self, exchange: Any) -> None:
        self._exchange = exchange

    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult:
        try:
            result = await self._exchange.create_order(
                symbol=pair,
                type=order_type,
                side=side,
                amount=quantity,
                price=price,
            )
            return OrderResult(
                order_id=str(uuid.uuid4()),
                exchange_order_id=str(result.get("id", "")),
                pair=pair,
                side=side,
                filled_price=float(result.get("average", result.get("price", 0))),
                filled_quantity=float(result.get("filled", quantity)),
                fee=float(result.get("fee", {}).get("cost", 0))
                if result.get("fee")
                else 0.0,
                slippage_pct=0.0,
                status=str(result.get("status", "unknown")),
                timestamp=datetime.now(UTC),
            )
        except Exception:
            logger.exception("LiveExecutor.place_order failed for %s", pair)
            raise

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id)
            return True
        except Exception:
            logger.exception("LiveExecutor.cancel_order failed for %s", order_id)
            return False

    async def get_open_orders(self, pair: str | None = None) -> list[dict[str, object]]:
        try:
            orders = await self._exchange.fetch_open_orders(symbol=pair)
            return [dict(o) for o in orders]  # type: ignore[arg-type]
        except Exception:
            logger.exception("LiveExecutor.get_open_orders failed")
            return []

    async def get_balance(self) -> dict[str, float]:
        try:
            balance = await self._exchange.fetch_balance()
            return {
                "total": float(balance.get("total", {}).get("USDT", 0)),
                "free": float(balance.get("free", {}).get("USDT", 0)),
                "used": float(balance.get("used", {}).get("USDT", 0)),
            }
        except Exception:
            logger.exception("LiveExecutor.get_balance failed")
            return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def close_all_positions(self) -> list[OrderResult]:
        logger.warning("LiveExecutor.close_all_positions: not fully implemented")
        return []
