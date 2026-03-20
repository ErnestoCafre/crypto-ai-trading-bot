"""Tests for execution.paper_executor.PaperExecutor."""

from __future__ import annotations

import pytest

from execution.paper_executor import PaperExecutor


@pytest.fixture()
def executor() -> PaperExecutor:
    return PaperExecutor(initial_balance=100_000.0, slippage_pct=0.0, fee_pct=0.0)


class TestPaperExecutor:
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
