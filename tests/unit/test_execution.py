"""Tests for execution.paper_executor.PaperExecutor and LiveExecutor."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_bot.execution.live_executor import LiveExecutor
from trading_bot.execution.paper_executor import PaperExecutor


@pytest.fixture()
def executor() -> PaperExecutor:
    return PaperExecutor(initial_balance=100_000.0, slippage_pct=0.0, fee_pct=0.0)


class TestPaperExecutor:
    @pytest.mark.asyncio()
    async def test_snapshot_round_trip(self, executor: PaperExecutor) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 0.5, price=40_000.0)
        executor.update_prices({"BTC/USDT": 42_000.0})

        restored = PaperExecutor(initial_balance=1.0)
        restored.restore(executor.to_dict())

        assert await restored.get_balance() == await executor.get_balance()
        assert await restored.get_position("BTC/USDT") == 0.5

    @pytest.mark.parametrize(
        "snapshot",
        [
            {},
            {"cash": "inf", "positions": {}},
            {"cash": 1.0, "positions": {"BTC/USDT": {"qty": 1.0}}},
            {"cash": 1.0, "positions": {"BTC/USDT": {"qty": 1.0, "avg_price": 0}}},
            {"cash": 1.0, "positions": []},
        ],
    )
    @pytest.mark.asyncio()
    async def test_restore_rejects_malformed_snapshot(
        self, executor: PaperExecutor, snapshot: dict[str, Any]
    ) -> None:
        with pytest.raises((KeyError, TypeError, ValueError, AttributeError)):
            executor.restore(snapshot)
        assert (await executor.get_balance())["total"] == 100_000.0

    @pytest.mark.asyncio()
    async def test_fill_price_does_not_override_market_mark(
        self, executor: PaperExecutor
    ) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 1.0, price=50000.0)
        await executor.place_order("ETH/USDT", "buy", "market", 1.0, price=3000.0)
        executor.update_prices({"BTC/USDT": 50000.0, "ETH/USDT": 3000.0})
        # A stop-style sale at 2500 must not re-mark ETH at the fill price.
        await executor.place_order("ETH/USDT", "sell", "market", 0.5, price=2500.0)
        balance = await executor.get_balance()
        assert balance["positions_value"] == pytest.approx(50000.0 + 0.5 * 3000.0)

    @pytest.mark.asyncio()
    async def test_buy_reduces_balance(self, executor: PaperExecutor) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 0.1, price=50000.0)
        balance = await executor.get_balance()
        assert balance["cash"] < 100_000.0

    @pytest.mark.asyncio()
    async def test_sell_increases_balance(self, executor: PaperExecutor) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 0.1, price=50000.0)
        await executor.place_order("BTC/USDT", "sell", "market", 0.1, price=51000.0)
        balance = await executor.get_balance()
        # Sold at higher price → profit
        assert balance["cash"] > 100_000.0 - 50000 * 0.1

    @pytest.mark.asyncio()
    async def test_cancel_order(self, executor: PaperExecutor) -> None:
        # Non-existent order returns False
        assert await executor.cancel_order("nonexistent") is False

    @pytest.mark.asyncio()
    async def test_get_balance_structure(self, executor: PaperExecutor) -> None:
        balance = await executor.get_balance()
        assert "cash" in balance
        assert "positions_value" in balance
        assert "total" in balance
        assert balance["total"] == pytest.approx(100_000.0)

    @pytest.mark.asyncio()
    async def test_close_all_positions(self, executor: PaperExecutor) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 0.1, price=50000.0)
        await executor.place_order("ETH/USDT", "buy", "market", 1.0, price=3000.0)
        results = await executor.close_all_positions()
        assert len(results) == 2
        balance = await executor.get_balance()
        assert balance["positions_value"] == pytest.approx(0.0)

    @pytest.mark.asyncio()
    async def test_slippage_and_fees(self) -> None:
        ex = PaperExecutor(initial_balance=100_000.0, slippage_pct=1.0, fee_pct=1.0)
        result = await ex.place_order("BTC/USDT", "buy", "market", 1.0, price=10000.0)
        # fill_price should be > 10000 (slippage on buy)
        assert result.filled_price > 10000.0
        assert result.fee > 0.0

    @pytest.mark.asyncio()
    async def test_sell_without_position_leaves_cash_unchanged(
        self, executor: PaperExecutor
    ) -> None:
        result = await executor.place_order(
            "BTC/USDT", "sell", "market", 0.5, price=50000.0
        )
        balance = await executor.get_balance()
        assert result.status == "rejected"
        assert result.filled_quantity == 0.0
        assert balance["cash"] == pytest.approx(100_000.0)
        assert await executor.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio()
    async def test_sell_more_than_held_is_capped(self, executor: PaperExecutor) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 0.1, price=50000.0)
        result = await executor.place_order(
            "BTC/USDT", "sell", "market", 1.0, price=50000.0
        )
        balance = await executor.get_balance()
        assert result.filled_quantity == pytest.approx(0.1)
        assert balance["cash"] == pytest.approx(100_000.0)
        assert await executor.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio()
    async def test_buy_reduced_to_available_cash_including_fees(self) -> None:
        ex = PaperExecutor(initial_balance=1_000.0, slippage_pct=0.0, fee_pct=1.0)
        result = await ex.place_order("BTC/USDT", "buy", "market", 1.0, price=10000.0)
        balance = await ex.get_balance()
        assert result.status == "filled"
        assert result.filled_quantity == pytest.approx(1_000.0 / (10000.0 * 1.01))
        assert balance["cash"] == pytest.approx(0.0, abs=1e-9)
        assert balance["cash"] >= -1e-9

    @pytest.mark.asyncio()
    async def test_buy_without_cash_is_rejected(self) -> None:
        ex = PaperExecutor(initial_balance=0.0, slippage_pct=0.0, fee_pct=0.1)
        result = await ex.place_order("BTC/USDT", "buy", "market", 1.0, price=10000.0)
        assert result.status == "rejected"
        assert await ex.get_position("BTC/USDT") == 0.0
        assert (await ex.get_balance())["cash"] == 0.0

    @pytest.mark.asyncio()
    async def test_get_position(self, executor: PaperExecutor) -> None:
        assert await executor.get_position("BTC/USDT") == 0.0
        await executor.place_order("BTC/USDT", "buy", "market", 0.25, price=50000.0)
        assert await executor.get_position("BTC/USDT") == pytest.approx(0.25)

    @pytest.mark.asyncio()
    async def test_price_drop_reduces_total(self, executor: PaperExecutor) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 1.0, price=50000.0)
        executor.update_prices({"BTC/USDT": 40000.0})
        balance = await executor.get_balance()
        assert balance["positions_value"] == pytest.approx(40000.0)
        assert balance["total"] == pytest.approx(90_000.0)

    @pytest.mark.asyncio()
    async def test_close_all_positions_uses_market_price(
        self, executor: PaperExecutor
    ) -> None:
        await executor.place_order("BTC/USDT", "buy", "market", 1.0, price=50000.0)
        executor.update_prices({"BTC/USDT": 40000.0})
        results = await executor.close_all_positions()
        balance = await executor.get_balance()
        assert results[0].filled_price == pytest.approx(40000.0)
        assert balance["cash"] == pytest.approx(90_000.0)

    @pytest.mark.asyncio()
    async def test_market_order_without_price_uses_last_price(
        self, executor: PaperExecutor
    ) -> None:
        executor.update_prices({"BTC/USDT": 30000.0})
        result = await executor.place_order("BTC/USDT", "buy", "market", 0.1)
        assert result.filled_price == pytest.approx(30000.0)

    @pytest.mark.asyncio()
    async def test_order_without_any_price_is_rejected(
        self, executor: PaperExecutor
    ) -> None:
        result = await executor.place_order("BTC/USDT", "buy", "market", 0.1)
        assert result.status == "rejected"
        assert (await executor.get_balance())["cash"] == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# LiveExecutor
# ---------------------------------------------------------------------------


def _mock_exchange(
    balance: dict[str, Any],
    tickers: dict[str, dict[str, Any]] | None = None,
    limits: dict[str, dict[str, Any]] | None = None,
) -> MagicMock:
    """ccxt-like exchange mock with 5-decimal amount precision."""
    tickers = tickers or {}
    limits = limits or {}
    ex = MagicMock()
    ex.has = {"fetchPositions": False}
    ex.fetch_balance = AsyncMock(return_value=balance)
    ex.fetch_tickers = AsyncMock(
        side_effect=lambda symbols: {s: tickers[s] for s in symbols if s in tickers}
    )
    ex.fetch_ticker = AsyncMock(side_effect=lambda symbol: tickers.get(symbol))
    ex.fetch_open_orders = AsyncMock(return_value=[])
    ex.cancel_order = AsyncMock(return_value=True)
    ex.load_markets = AsyncMock(return_value={})
    ex.amount_to_precision = MagicMock(
        side_effect=lambda symbol, amount: f"{math.floor(amount * 1e5) / 1e5:.5f}"
    )
    ex.price_to_precision = MagicMock(
        side_effect=lambda symbol, price: f"{math.floor(price * 1e2) / 1e2:.2f}"
    )
    ex.fetch_order = AsyncMock(return_value={"id": "stop-1", "status": "open"})
    ex.market = MagicMock(side_effect=lambda symbol: {"limits": limits.get(symbol, {})})
    ex.create_order = AsyncMock(
        return_value={
            "id": "ord-1",
            "status": "closed",
            "average": None,
            "price": None,
            "filled": None,
            "fee": None,
        }
    )
    return ex


class TestLiveExecutor:
    @pytest.mark.asyncio()
    async def test_get_balance(self) -> None:
        mock_exchange = _mock_exchange(
            {
                "total": {"USDT": 5000.0},
                "free": {"USDT": 4500.0},
                "used": {"USDT": 500.0},
            }
        )
        executor = LiveExecutor(exchange=mock_exchange)
        balance = await executor.get_balance()
        assert balance["total"] == 5000.0
        assert balance["free"] == 4500.0
        assert balance["cash"] == 4500.0

    @pytest.mark.asyncio()
    async def test_get_balance_values_pair_assets_at_market(self) -> None:
        mock_exchange = _mock_exchange(
            {
                "total": {"USDT": 1000.0, "BTC": 0.5, "ETH": 2.0, "BNB": 10.0},
                "free": {"USDT": 1000.0, "BTC": 0.5, "ETH": 2.0, "BNB": 10.0},
                "used": {},
            },
            tickers={
                "BTC/USDT": {"last": 60000.0},
                "ETH/USDT": {"last": 3000.0},
                "BNB/USDT": {"last": 500.0},
            },
        )
        executor = LiveExecutor(
            exchange=mock_exchange, pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        )
        balance = await executor.get_balance()

        # BNB is not a managed pair and SOL is not held: neither is valued.
        mock_exchange.fetch_tickers.assert_awaited_once_with(["BTC/USDT", "ETH/USDT"])
        assert balance["positions_value"] == pytest.approx(36_000.0)
        assert balance["total"] == pytest.approx(37_000.0)
        assert balance["cash"] == pytest.approx(1000.0)

    @pytest.mark.asyncio()
    async def test_get_balance_raises_instead_of_reporting_zero(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.fetch_balance = AsyncMock(side_effect=ConnectionError("down"))
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        with pytest.raises(ConnectionError):
            await executor.get_balance()

    @pytest.mark.asyncio()
    async def test_get_balance_raises_without_price_for_held_asset(self) -> None:
        mock_exchange = _mock_exchange(
            {"total": {"USDT": 1000.0, "BTC": 0.5}, "free": {}, "used": {}}
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        with pytest.raises(RuntimeError, match="BTC/USDT"):
            await executor.get_balance()

    @pytest.mark.asyncio()
    async def test_place_order_handles_none_average(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.fetch_order = AsyncMock(return_value={"id": "ord-1"})
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])

        result = await executor.place_order(
            "BTC/USDT", "buy", "market", 0.01, price=50000.0
        )
        assert result.filled_price == 50000.0
        assert result.filled_quantity == 0.01
        assert result.fee == 0.0
        # Fill neither in the response nor in fetch_order: flagged (audit M-2).
        assert result.status == "unconfirmed"
        mock_exchange.fetch_order.assert_awaited_with("ord-1", "BTC/USDT")

        no_price = await executor.place_order("BTC/USDT", "sell", "market", 0.01)
        assert no_price.filled_price == 0.0

    @pytest.mark.asyncio()
    async def test_place_order_fetches_the_fill_when_not_reported(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.fetch_order = AsyncMock(
            return_value={
                "id": "ord-1",
                "status": "expired",
                "filled": 0.0,
                "average": None,
                "cost": 0.0,
            }
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])

        result = await executor.place_order(
            "BTC/USDT", "buy", "market", 0.01, price=50000.0
        )

        assert result.filled_quantity == 0.0  # nothing filled: not assumed
        assert result.status == "expired"

    @pytest.mark.asyncio()
    async def test_fill_price_from_cost_when_average_missing(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.create_order = AsyncMock(
            return_value={"id": "1", "status": "closed", "filled": 0.02, "cost": 990.0}
        )
        executor = LiveExecutor(exchange=mock_exchange)
        result = await executor.place_order(
            "BTC/USDT", "sell", "market", 0.02, price=50000.0
        )
        assert result.filled_price == pytest.approx(49_500.0)
        mock_exchange.fetch_order.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_market_order_does_not_send_a_price(self) -> None:
        mock_exchange = _mock_exchange({})
        executor = LiveExecutor(exchange=mock_exchange)
        await executor.place_order("BTC/USDT", "buy", "market", 0.01, price=50000.0)
        await executor.place_order("BTC/USDT", "buy", "limit", 0.01, price=49000.0)
        calls = mock_exchange.create_order.await_args_list
        assert calls[0].kwargs["price"] is None
        assert calls[1].kwargs["price"] == 49000.0

    @pytest.mark.asyncio()
    async def test_place_order_prefers_average_then_price(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.create_order = AsyncMock(
            return_value={
                "id": "1",
                "status": "closed",
                "average": None,
                "price": 49900.0,
                "filled": 0.009,
                "fee": {"cost": None, "currency": "BNB"},
            }
        )
        executor = LiveExecutor(exchange=mock_exchange)
        result = await executor.place_order(
            "BTC/USDT", "buy", "market", 0.01, price=50000.0
        )
        assert result.filled_price == 49900.0
        assert result.filled_quantity == 0.009
        assert result.fee == 0.0

    @pytest.mark.asyncio()
    async def test_get_position_returns_sellable_base_amount(self) -> None:
        mock_exchange = _mock_exchange(
            {"free": {}, "total": {"BTC": 0.123456789, "USDT": 10.0}, "used": {}},
            tickers={"BTC/USDT": {"last": 60000.0}},
            limits={"BTC/USDT": {"amount": {"min": 0.00001}}},
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        assert await executor.get_position("BTC/USDT") == pytest.approx(0.12345)
        assert await executor.get_position("ETH/USDT") == 0.0

    @pytest.mark.asyncio()
    async def test_get_position_treats_dust_as_flat(self) -> None:
        mock_exchange = _mock_exchange(
            {"free": {}, "total": {"BTC": 0.00005}, "used": {}},
            tickers={"BTC/USDT": {"last": 60000.0}},
            limits={"BTC/USDT": {"amount": {"min": 0.00001}, "cost": {"min": 5.0}}},
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        # 0.00005 BTC * 60000 = 3 USDT < 5 USDT min notional
        assert await executor.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio()
    async def test_tradable_amount_applies_precision_and_minimums(self) -> None:
        mock_exchange = _mock_exchange(
            {"free": {}, "total": {}, "used": {}},
            limits={"BTC/USDT": {"amount": {"min": 0.00001}, "cost": {"min": 5.0}}},
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        assert await executor.tradable_amount(
            "BTC/USDT", 0.123456789, 60000.0
        ) == pytest.approx(0.12345)
        # 0.00005 BTC x 60000 = 3 USDT < 5 USDT min notional
        assert await executor.tradable_amount("BTC/USDT", 0.00005, 60000.0) == 0.0
        assert await executor.tradable_amount("BTC/USDT", 0.0, 60000.0) == 0.0

    @pytest.mark.asyncio()
    async def test_paper_tradable_amount_is_unchanged(self) -> None:
        paper = PaperExecutor()
        assert await paper.tradable_amount("BTC/USDT", 0.123456789, 1.0) == (
            0.123456789
        )

    @pytest.mark.asyncio()
    async def test_close_all_positions_spot(self) -> None:
        mock_exchange = _mock_exchange(
            {
                "free": {"BTC": 0.5, "USDT": 1000.0},
                "total": {"BTC": 0.5, "USDT": 1000.0},
                "used": {},
            },
            tickers={"BTC/USDT": {"last": 65000.0}},
        )
        mock_exchange.fetch_open_orders = AsyncMock(
            return_value=[{"id": "ord-1", "symbol": "BTC/USDT"}]
        )
        mock_exchange.create_order = AsyncMock(
            return_value={
                "id": "close-1",
                "status": "closed",
                "average": 65000.0,
                "filled": 0.5,
            }
        )

        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        results = await executor.close_all_positions()

        mock_exchange.cancel_order.assert_called_once_with("ord-1", "BTC/USDT")
        mock_exchange.create_order.assert_called_once_with(
            symbol="BTC/USDT",
            type="market",
            side="sell",
            amount=0.5,
            price=None,
        )
        assert len(results) == 1
        assert results[0].pair == "BTC/USDT"

    @pytest.mark.asyncio()
    async def test_close_all_positions_limited_to_trading_pairs(self) -> None:
        mock_exchange = _mock_exchange(
            {
                "free": {
                    "BTC": 0.123456789,
                    "ETH": 0.0001,
                    "BNB": 3.0,
                    "DOGE": 1000.0,
                    "USDT": 50.0,
                },
                "total": {},
                "used": {},
            },
            tickers={
                "BTC/USDT": {"last": 60000.0},
                "ETH/USDT": {"last": 3000.0},
            },
            limits={"ETH/USDT": {"amount": {"min": 0.001}}},
        )
        open_orders = {
            "BTC/USDT": [{"id": "bot-1", "symbol": "BTC/USDT"}],
            "BNB/USDT": [{"id": "manual-1", "symbol": "BNB/USDT"}],
        }
        mock_exchange.fetch_open_orders = AsyncMock(
            side_effect=lambda symbol=None: open_orders.get(symbol, [])
        )

        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT", "ETH/USDT"])
        results = await executor.close_all_positions()

        # Only the bot's pair order is cancelled; BNB/DOGE are untouched;
        # ETH dust below the market minimum is skipped; BTC is rounded.
        mock_exchange.cancel_order.assert_called_once_with("bot-1", "BTC/USDT")
        mock_exchange.create_order.assert_called_once_with(
            symbol="BTC/USDT",
            type="market",
            side="sell",
            amount=0.12345,
            price=None,
        )
        assert [r.pair for r in results] == ["BTC/USDT"]

    @pytest.mark.asyncio()
    async def test_get_position_counts_balance_locked_by_stop(self) -> None:
        mock_exchange = _mock_exchange(
            {"free": {"BTC": 0.0}, "used": {"BTC": 0.5}, "total": {"BTC": 0.5}},
            tickers={"BTC/USDT": {"last": 60000.0}},
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        assert await executor.get_position("BTC/USDT") == pytest.approx(0.5)

    @pytest.mark.asyncio()
    async def test_place_stop_loss_applies_precision(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.create_order = AsyncMock(return_value={"id": "stop-1"})
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])

        order_id = await executor.place_stop_loss("BTC/USDT", 0.123456789, 58123.4567)

        assert order_id == "stop-1"
        mock_exchange.create_order.assert_awaited_once_with(
            symbol="BTC/USDT",
            type="market",
            side="sell",
            amount=0.12345,
            price=None,
            params={"stopLossPrice": 58123.45},
        )

    @pytest.mark.asyncio()
    async def test_place_stop_loss_failure_returns_none(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.create_order = AsyncMock(
            side_effect=Exception("Stop price would trigger immediately.")
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        assert await executor.place_stop_loss("BTC/USDT", 0.1, 58000.0) is None

    @pytest.mark.asyncio()
    async def test_place_stop_loss_unsupported_market_returns_none(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.market = MagicMock(
            return_value={"limits": {}, "info": {"orderTypes": ["LIMIT", "MARKET"]}}
        )
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        assert await executor.place_stop_loss("BTC/USDT", 0.1, 58000.0) is None
        mock_exchange.create_order.assert_not_called()

    @pytest.mark.asyncio()
    async def test_cancel_order_passes_symbol(self) -> None:
        mock_exchange = _mock_exchange({})
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])
        assert await executor.cancel_order("stop-1", "BTC/USDT") is True
        mock_exchange.cancel_order.assert_awaited_once_with("stop-1", "BTC/USDT")

    @pytest.mark.asyncio()
    async def test_close_all_queries_open_orders_per_symbol(self) -> None:
        """Binance rejects symbol-less fetchOpenOrders by default."""
        mock_exchange = _mock_exchange(
            {"free": {"BTC": 0.5}, "total": {"BTC": 0.5}, "used": {}},
            tickers={"BTC/USDT": {"last": 60000.0}},
        )

        async def fetch_open_orders(symbol=None):
            if symbol is None:
                raise Exception("fetchOpenOrders() WARNING: ... without a symbol")
            return [{"id": "stop-1", "symbol": symbol}]

        mock_exchange.fetch_open_orders = AsyncMock(side_effect=fetch_open_orders)
        executor = LiveExecutor(exchange=mock_exchange, pairs=["BTC/USDT"])

        await executor.close_all_positions()

        mock_exchange.cancel_order.assert_awaited_once_with("stop-1", "BTC/USDT")
        assert mock_exchange.create_order.await_count == 1

    @pytest.mark.asyncio()
    async def test_close_releases_exchange(self) -> None:
        mock_exchange = _mock_exchange({})
        mock_exchange.close = AsyncMock()
        executor = LiveExecutor(exchange=mock_exchange)
        await executor.close()
        mock_exchange.close.assert_awaited_once()
