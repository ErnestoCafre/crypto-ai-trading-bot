"""Live trading executor — wraps ccxt async exchange (spot only)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from trading_bot.core.models import OrderResult
from trading_bot.execution.base import BaseExecutor

logger = logging.getLogger(__name__)


class LiveExecutor(BaseExecutor):
    def __init__(
        self,
        exchange: Any,
        pairs: Sequence[str] = (),
        quote_currency: str = "USDT",
    ) -> None:
        self._exchange = exchange
        # Only these pairs are managed by the bot: valuation and liquidation
        # never touch other assets held in the account.
        self._pairs = list(pairs)
        self._quote = quote_currency

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
            # ccxt returns None for average/filled/fee on many market orders.
            filled = result.get("filled")
            fee = result.get("fee") or {}
            return OrderResult(
                order_id=str(uuid.uuid4()),
                exchange_order_id=str(result.get("id", "")),
                pair=pair,
                side=side,
                filled_price=float(
                    result.get("average") or result.get("price") or price or 0.0
                ),
                filled_quantity=float(filled) if filled is not None else quantity,
                fee=float(fee.get("cost") or 0.0),
                slippage_pct=0.0,
                status=str(result.get("status") or "unknown"),
                timestamp=datetime.now(UTC),
            )
        except Exception:
            logger.exception("LiveExecutor.place_order failed for %s", pair)
            raise

    async def cancel_order(self, order_id: str, pair: str | None = None) -> bool:
        try:
            # Binance (and most ccxt exchanges) require the symbol to cancel.
            await self._exchange.cancel_order(order_id, pair)
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
        """Quote cash plus managed base assets valued at last market price.

        Raises on any exchange error instead of reporting a zero balance: a
        fabricated 0.0 total would read as a 100% drawdown and fire the
        kill-switch.
        """
        balance = await self._exchange.fetch_balance()
        totals = balance.get("total") or {}
        free = float((balance.get("free") or {}).get(self._quote) or 0.0)
        used = float((balance.get("used") or {}).get(self._quote) or 0.0)
        cash_total = float(totals.get(self._quote) or 0.0)

        held = {
            pair: float(totals.get(self._base(pair)) or 0.0) for pair in self._pairs
        }
        held = {pair: qty for pair, qty in held.items() if qty > 0}

        positions_value = 0.0
        if held:
            tickers = await self._exchange.fetch_tickers(list(held))
            for pair, qty in held.items():
                price = self._ticker_price(tickers.get(pair))
                if price is None:
                    raise RuntimeError(f"No market price available to value {pair}")
                positions_value += qty * price

        return {
            "total": cash_total + positions_value,
            "free": free,
            "used": used,
            "cash": free,
            "positions_value": positions_value,
        }

    async def get_position(self, pair: str) -> float:
        """Held base-asset amount for ``pair`` (dust below market minimums is 0).

        Uses the total balance: an open stop order locks the position, so the
        free balance alone would report it as flat.
        """
        balance = await self._exchange.fetch_balance()
        qty = float((balance.get("total") or {}).get(self._base(pair)) or 0.0)
        if qty <= 0:
            return 0.0
        ticker = await self._exchange.fetch_ticker(pair)
        return await self._sellable_amount(pair, qty, self._ticker_price(ticker))

    async def close_all_positions(self) -> list[OrderResult]:
        """Cancel the bot's open orders and sell its base assets to quote."""
        logger.warning(
            "LiveExecutor.close_all_positions initiated — emergency liquidation"
        )
        results: list[OrderResult] = []

        # 1. Cancel open orders (incl. native stops, which lock the balance) on
        # managed pairs only. Queried per symbol: Binance rejects symbol-less
        # fetchOpenOrders by default.
        for pair in self._pairs:
            for order in await self.get_open_orders(pair):
                oid = str(order.get("id", ""))
                if oid and order.get("symbol", pair) == pair:
                    await self.cancel_order(oid, pair)

        # 2. Sell managed base assets at market
        try:
            balance = await self._exchange.fetch_balance()
            free = balance.get("free") or {}
            to_sell = {
                pair: float(free.get(self._base(pair)) or 0.0) for pair in self._pairs
            }
            to_sell = {pair: qty for pair, qty in to_sell.items() if qty > 0}
            tickers = (
                await self._exchange.fetch_tickers(list(to_sell)) if to_sell else {}
            )
        except Exception:
            logger.exception("Failed to fetch balances for liquidation")
            return results

        for pair, qty in to_sell.items():
            try:
                amount = await self._sellable_amount(
                    pair, qty, self._ticker_price(tickers.get(pair))
                )
                if amount <= 0:
                    logger.warning(
                        "Skipping %s liquidation: %.8f below market minimum",
                        pair,
                        qty,
                    )
                    continue
                res = await self.place_order(
                    pair=pair,
                    side="sell",
                    order_type="market",
                    quantity=amount,
                )
                results.append(res)
            except Exception:
                logger.exception("Failed to sell spot balance for %s", pair)

        return results

    async def place_stop_loss(
        self, pair: str, quantity: float, stop_price: float
    ) -> str | None:
        """Place a native stop-market sell (Binance spot ``STOP_LOSS``).

        Uses the ccxt unified ``stopLossPrice`` param. Returns None, so the bot
        watches the stop itself, when the market lacks ``STOP_LOSS`` or the
        order is refused (e.g. the stop is already above the market).
        """
        try:
            await self._exchange.load_markets()
            market = self._exchange.market(pair)
            order_types = (market.get("info") or {}).get("orderTypes")
            if order_types is not None and "STOP_LOSS" not in order_types:
                logger.warning(
                    "%s has no STOP_LOSS order type; bot will watch the stop", pair
                )
                return None
            amount = float(self._exchange.amount_to_precision(pair, quantity))
            trigger = float(self._exchange.price_to_precision(pair, stop_price))
            order = await self._exchange.create_order(
                symbol=pair,
                type="market",
                side="sell",
                amount=amount,
                price=None,
                params={"stopLossPrice": trigger},
            )
            order_id = order.get("id")
            if not order_id:
                raise ValueError("exchange returned no order id")
            logger.info(
                "Stop-loss placed for %s: %s @ %s (order %s)",
                pair,
                amount,
                trigger,
                order_id,
            )
            return str(order_id)
        except Exception:
            logger.warning(
                "Could not place exchange stop for %s at %.8f; bot will watch it",
                pair,
                stop_price,
                exc_info=True,
            )
            return None

    async def close(self) -> None:
        try:
            await self._exchange.close()
        except Exception:
            logger.exception("LiveExecutor.close failed")

    async def get_order(self, order_id: str, pair: str) -> dict[str, Any]:
        return dict(await self._exchange.fetch_order(order_id, pair))

    async def tradable_amount(self, pair: str, quantity: float, price: float) -> float:
        """Market precision and minimum amount/notional (buys and sells)."""
        if quantity <= 0:
            return 0.0
        return await self._sellable_amount(pair, quantity, price)

    async def _sellable_amount(
        self, pair: str, qty: float, price: float | None
    ) -> float:
        """Round ``qty`` to the market's precision; 0.0 if under its minimums."""
        await self._exchange.load_markets()
        try:
            amount = float(self._exchange.amount_to_precision(pair, qty))
        except Exception:
            # ccxt raises when the rounded amount would be zero.
            return 0.0

        limits = self._exchange.market(pair).get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min") or 0.0
        min_cost = (limits.get("cost") or {}).get("min") or 0.0
        if amount <= 0 or amount < min_amount:
            return 0.0
        if price is not None and amount * price < min_cost:
            return 0.0
        return amount

    @staticmethod
    def _base(pair: str) -> str:
        return pair.split("/")[0]

    @staticmethod
    def _ticker_price(ticker: dict[str, Any] | None) -> float | None:
        if not ticker:
            return None
        price = ticker.get("last") or ticker.get("close") or ticker.get("bid")
        return float(price) if price else None
