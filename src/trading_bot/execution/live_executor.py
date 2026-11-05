"""Live trading executor — wraps ccxt async exchange (spot only).

Position ledger: the account may hold base assets the bot did not buy (the
owner's own coins, airdrops, a manual trade). The executor keeps the
quantity *it* bought per pair and treats ``min(ledger, exchange balance)`` as
the bot's position for sizing, valuation, exits and liquidation, so it never
sells what it does not own. The ledger is persisted by the bot with the risk
state (:meth:`to_dict` / :meth:`restore`); taking it below the exchange
balance only ever ignores coins, so the bot errs on the safe side.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Mapping, Sequence
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
        owned: Mapping[str, float] | None = None,
    ) -> None:
        self._exchange = exchange
        # Only these pairs are managed by the bot: valuation and liquidation
        # never touch other assets held in the account.
        self._pairs = list(pairs)
        self._quote = quote_currency
        # Base quantity the bot bought and still holds, per pair (see module
        # docstring). Empty = the bot owns nothing yet.
        self._owned: dict[str, float] = {}
        # Native stop orders the bot placed: liquidation cancels only these.
        self._own_orders: dict[str, str] = {}  # order id -> pair
        if owned:
            self.restore({"owned": dict(owned)})

    # -- position ledger --

    def to_dict(self) -> dict[str, Any]:
        """Ledger snapshot for :meth:`restore` (persisted with the risk state)."""
        return {"owned": dict(self._owned), "orders": dict(self._own_orders)}

    def restore(self, data: Mapping[str, Any]) -> None:
        """Replace the ledger with a :meth:`to_dict` snapshot.

        Raises ValueError/TypeError/KeyError on malformed data and leaves the
        ledger unchanged.
        """
        owned: dict[str, float] = {}
        for pair, qty in dict(data["owned"]).items():
            value = float(qty)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid owned quantity for {pair}: {qty}")
            if value > 0:
                owned[str(pair)] = value
        orders = {str(k): str(v) for k, v in dict(data.get("orders") or {}).items()}
        self._owned, self._own_orders = owned, orders

    def owned(self, pair: str) -> float:
        """Quantity the ledger attributes to the bot (before reconciliation)."""
        return self._owned.get(pair, 0.0)

    def record_external_fill(self, pair: str, side: str, quantity: float) -> None:
        """Apply a fill that did not go through :meth:`place_order` (a native
        stop executed by the exchange)."""
        self._apply_fill(pair, side, quantity)

    async def exchange_holdings(self) -> dict[str, float]:
        """Total base balance on the exchange per managed pair (bot or not)."""
        totals = (await self._exchange.fetch_balance()).get("total") or {}
        return {p: float(totals.get(self._base(p)) or 0.0) for p in self._pairs}

    async def reconcile(self) -> dict[str, float]:
        """Clamp the ledger to the exchange balance; return foreign holdings.

        A ledger above the balance means coins left outside the bot (manual
        sell, a stop filled while it was down): the bot cannot own them any
        more. Holdings above the ledger are the owner's and are left alone.
        """
        holdings = await self.exchange_holdings()
        foreign: dict[str, float] = {}
        for pair, total in holdings.items():
            owned = self.owned(pair)
            if owned > total:
                logger.warning(
                    "%s: ledger %.8f above exchange balance %.8f — clamped",
                    pair,
                    owned,
                    total,
                )
                self._set_owned(pair, total)
            elif total > owned:
                foreign[pair] = total - owned
        return foreign

    def _apply_fill(self, pair: str, side: str, quantity: float) -> None:
        if quantity <= 0:
            return
        owned = self.owned(pair)
        self._set_owned(pair, owned + quantity if side == "buy" else owned - quantity)

    def _set_owned(self, pair: str, quantity: float) -> None:
        if quantity > 0:
            self._owned[pair] = quantity
        else:
            self._owned.pop(pair, None)

    async def _bot_holdings(self, totals: Mapping[str, Any]) -> dict[str, float]:
        """min(ledger, exchange total) per managed pair, positive only."""
        held = {
            pair: min(self.owned(pair), float(totals.get(self._base(pair)) or 0.0))
            for pair in self._pairs
        }
        return {pair: qty for pair, qty in held.items() if qty > 0}

    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> OrderResult:
        """Send an order and report what actually filled.

        For market orders ``price`` is only the reference used when the
        exchange reports no fill price; it is not sent (ccxt could turn it
        into a limit on some exchanges). When the response lacks ``filled``
        the order is fetched once; if it is still unknown the fill is
        measured as the change of the base balance and the status is
        ``"unconfirmed"``. The ledger moves by the base actually received or
        given (a buy fee charged in the base asset is deducted).
        """
        before = await self._base_total(pair)
        try:
            order = await self._exchange.create_order(
                symbol=pair,
                type=order_type,
                side=side,
                amount=quantity,
                price=None if order_type == "market" else price,
            )
        except Exception:
            logger.exception("LiveExecutor.place_order failed for %s", pair)
            raise
        order = dict(order or {})
        if order.get("filled") is None and order.get("id"):
            order = await self._refresh_order(order, pair)

        filled = order.get("filled")
        status = str(order.get("status") or "unknown")
        if filled is None:
            filled = await self._measured_fill(pair, side, before, quantity)
            logger.warning(
                "%s %s order %s: fill not reported by the exchange — %.8f "
                "measured from the balance",
                pair,
                side,
                order.get("id"),
                filled,
            )
            status = "unconfirmed"
            net = float(filled)  # a balance change is already net of fees
        else:
            net = self._net_base(order, pair, side, float(filled))
        filled_qty = float(filled)
        self._apply_fill(pair, side, net)
        fee = order.get("fee") or {}
        return OrderResult(
            order_id=str(uuid.uuid4()),
            exchange_order_id=str(order.get("id", "")),
            pair=pair,
            side=side,
            filled_price=self._fill_price(order, filled_qty, price),
            filled_quantity=filled_qty,
            fee=float(fee.get("cost") or 0.0) if isinstance(fee, dict) else 0.0,
            slippage_pct=0.0,
            status=status,
            timestamp=datetime.now(UTC),
        )

    async def _base_total(self, pair: str) -> float | None:
        """Total base balance of ``pair``; None if it cannot be read."""
        try:
            totals = (await self._exchange.fetch_balance()).get("total") or {}
            return float(totals.get(self._base(pair)) or 0.0)
        except Exception:
            logger.warning("Could not read the %s balance", pair, exc_info=True)
            return None

    async def _measured_fill(
        self, pair: str, side: str, before: float | None, requested: float
    ) -> float:
        """Fill size from the base balance change around the order.

        Falls back to ``requested`` (logged) when a balance cannot be read:
        the ledger is then clamped to the balance on the next reconcile.
        """
        after = await self._base_total(pair)
        if before is None or after is None:
            logger.error(
                "%s: cannot measure an unreported fill — assuming %.8f",
                pair,
                requested,
            )
            return requested
        change = after - before if side == "buy" else before - after
        return max(0.0, change)

    def _net_base(
        self, order: dict[str, Any], pair: str, side: str, filled: float
    ) -> float:
        """Base received on a buy: ``filled`` minus fees charged in the base
        asset (Binance without BNB). Sells give exactly ``filled``."""
        if side != "buy":
            return filled
        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        base = self._base(pair)
        in_base = sum(
            float(f.get("cost") or 0.0)
            for f in fees
            if isinstance(f, dict) and f.get("currency") == base
        )
        return max(0.0, filled - in_base)

    async def _refresh_order(self, order: dict[str, Any], pair: str) -> dict[str, Any]:
        """Merge the exchange's current view of ``order`` (one fetch_order)."""
        try:
            fetched = dict(await self._exchange.fetch_order(str(order["id"]), pair))
        except Exception:
            logger.warning(
                "Could not fetch order %s on %s", order["id"], pair, exc_info=True
            )
            return order
        return {**order, **{k: v for k, v in fetched.items() if v is not None}}

    @staticmethod
    def _fill_price(
        order: dict[str, Any], filled: float, reference: float | None
    ) -> float:
        """Average fill price; cost / filled; order price; else the reference."""
        if order.get("average"):
            return float(order["average"])
        cost = order.get("cost")
        if cost and filled > 0:
            return float(cost) / filled
        return float(order.get("price") or reference or 0.0)

    async def cancel_order(self, order_id: str, pair: str | None = None) -> bool:
        try:
            # Binance (and most ccxt exchanges) require the symbol to cancel.
            await self._exchange.cancel_order(order_id, pair)
            self._own_orders.pop(order_id, None)
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

        held = await self._bot_holdings(totals)

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
        """The bot's sellable amount for ``pair``: min(ledger, total balance).

        Dust below market minimums is 0. Uses the total balance: an open stop
        order locks the position, so the free balance alone would read flat.
        """
        balance = await self._exchange.fetch_balance()
        qty = (await self._bot_holdings(balance.get("total") or {})).get(pair, 0.0)
        if qty <= 0:
            return 0.0
        ticker = await self._exchange.fetch_ticker(pair)
        return await self._sellable_amount(pair, qty, self._ticker_price(ticker))

    async def close_all_positions(self) -> list[OrderResult]:
        """Cancel the bot's native stops and sell the bot's holdings to quote.

        Only what the ledger attributes to the bot is sold, and only orders the
        bot placed are cancelled: the owner's coins and orders are left alone.
        """
        logger.warning(
            "LiveExecutor.close_all_positions initiated — emergency liquidation"
        )
        results: list[OrderResult] = []

        # 1. Cancel the bot's open stops (they lock the balance). Queried per
        # symbol: Binance rejects symbol-less fetchOpenOrders by default.
        for pair in self._pairs:
            for order in await self.get_open_orders(pair):
                oid = str(order.get("id", ""))
                if oid in self._own_orders and order.get("symbol", pair) == pair:
                    await self.cancel_order(oid, pair)

        # 2. Sell the bot's holdings at market
        try:
            balance = await self._exchange.fetch_balance()
            to_sell = await self._bot_holdings(balance.get("free") or {})
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
            self._own_orders[str(order_id)] = pair
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
        order = dict(await self._exchange.fetch_order(order_id, pair))
        if order.get("status") not in (None, "open"):
            self._own_orders.pop(order_id, None)  # no longer live: forget it
        return order

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
