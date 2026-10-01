"""Best-effort persistence of the bot's fills, trades and portfolio snapshots.

Wraps :class:`TradingRepository` for :class:`TradingOrchestrator`. A database
problem must never affect trading, so every method is bounded by a timeout,
logs failures and returns instead of raising. Each operation uses its own
short-lived session: a failed commit cannot poison later writes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from trading_bot.core.models import OrderResult
from trading_bot.db.base import build_url
from trading_bot.db.models import Trade
from trading_bot.db.repository import TradingRepository

T = TypeVar("T")


@dataclass
class _OpenTrade:
    trade_id: uuid.UUID
    entry_price: float
    quantity: float


class TradeJournal:
    """Records orders, trade lifecycles and snapshots; never raises."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        exchange: str,
        timeout_s: float = 5.0,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._factory = session_factory
        self._exchange = exchange
        self._timeout_s = timeout_s
        self._engine = engine
        self._open: dict[str, _OpenTrade] = {}
        self._max_drawdown_pct = 0.0

    @classmethod
    async def connect(
        cls, url: str, exchange: str, timeout_s: float = 5.0
    ) -> TradeJournal | None:
        """Journal on ``url``, or None (with a warning) if it is unreachable."""
        engine: AsyncEngine | None = None
        try:
            engine = create_async_engine(build_url(url), pool_pre_ping=True)
            async with asyncio.timeout(timeout_s), engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            logger.warning(
                "Database unavailable ({}: {}) — running without persistence "
                "(trades and snapshots will not reach the dashboard)",
                type(exc).__name__,
                exc,
            )
            if engine is not None:
                await engine.dispose()
            return None
        journal = cls(
            async_sessionmaker(engine, expire_on_commit=False),
            exchange,
            timeout_s=timeout_s,
            engine=engine,
        )
        await journal.restore()
        logger.info("Trade journal connected")
        return journal

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    # ------------------------------------------------------------------

    async def _run(
        self, what: str, op: Callable[[TradingRepository], Awaitable[T]]
    ) -> T | None:
        try:
            async with asyncio.timeout(self._timeout_s), self._factory() as session:
                return await op(TradingRepository(session))
        except Exception:
            logger.exception("DB: failed to {} — trading continues", what)
            return None

    async def restore(self) -> None:
        """Reload open trades (to close them later) and the max drawdown."""

        async def op(repo: TradingRepository) -> None:
            for trade in await repo.open_trades():
                self._open[trade.pair] = _OpenTrade(
                    trade.id, float(trade.entry_price), float(trade.quantity)
                )
            snap = await repo.latest_snapshot()
            if snap is not None:
                self._max_drawdown_pct = float(snap.max_drawdown_pct)

        await self._run("restore open trades", op)
        if self._open:
            logger.info("Trade journal: {} open trades restored", len(self._open))

    async def reconcile(self, held: dict[str, float]) -> None:
        """Abandon journaled open trades whose pair holds no position.

        Without it a stale open trade (paper account reset on restart, or an
        exit whose write failed) would stay open on the dashboard and absorb
        the next exit of that pair.
        """
        stale = {p: t for p, t in self._open.items() if held.get(p, 0.0) <= 0}
        if not stale:
            return
        logger.warning(
            "Trade journal: {} open trade(s) with no position ({}) marked "
            "abandoned — exit not recorded",
            len(stale),
            ", ".join(sorted(stale)),
        )
        for pair in stale:
            del self._open[pair]

        async def op(repo: TradingRepository) -> None:
            for trade in stale.values():
                await repo.abandon_trade(trade.trade_id, "no_position")

        await self._run("abandon stale trades", op)

    def open_trade_id(self, pair: str) -> uuid.UUID | None:
        trade = self._open.get(pair)
        return trade.trade_id if trade else None

    async def record_entry(
        self, result: OrderResult, strategy: str, confidence: float | None = None
    ) -> None:
        """A filled buy: open a trade and record its order."""
        if result.pair in self._open:
            # Entries only happen when flat: a journaled open trade here is
            # stale (its exit was never recorded).
            await self.reconcile({result.pair: 0.0})

        async def op(repo: TradingRepository) -> uuid.UUID:
            trade = await repo.open_trade(
                exchange=self._exchange,
                pair=result.pair,
                strategy=strategy,
                entry_price=result.filled_price,
                quantity=result.filled_quantity,
                fees=result.fee,
                entry_time=result.timestamp,
                ai_combined_confidence=confidence,
            )
            await self._add_order(repo, result, strategy, trade.id)
            return trade.id

        trade_id = await self._run(f"record entry on {result.pair}", op)
        if trade_id is not None:
            self._open[result.pair] = _OpenTrade(
                trade_id, result.filled_price, result.filled_quantity
            )

    async def record_exit(
        self, result: OrderResult, strategy: str, reason: str
    ) -> None:
        """A filled sell (signal, stop or liquidation): record and close."""
        open_trade = self._open.pop(result.pair, None)

        async def op(repo: TradingRepository) -> None:
            trade_id = open_trade.trade_id if open_trade else None
            await self._add_order(repo, result, strategy, trade_id)
            if trade_id is not None:
                await repo.close_trade(
                    trade_id,
                    exit_price=result.filled_price,
                    exit_reason=reason,
                    exit_fees=result.fee,
                    exit_time=result.timestamp,
                )

        if open_trade is None:
            logger.debug("{}: exit with no journaled trade", result.pair)
        await self._run(f"record exit on {result.pair}", op)

    async def record_rebalance(
        self,
        result: OrderResult,
        strategy: str,
        position_after: float,
        confidence: float | None = None,
    ) -> None:
        """A target-exposure fill: open, scale in, partially or fully close.

        A buy from flat opens a trade and a buy while holding averages into
        it. A sell that leaves the pair flat closes the trade; a partial sell
        is closed as its own trade (pro-rata entry fees) so realised P&L is
        recorded on every reduction.
        """
        open_trade = self._open.get(result.pair)
        if result.side == "buy" and open_trade is None:
            await self.record_entry(result, strategy=strategy, confidence=confidence)
            return
        if result.side == "sell" and (open_trade is None or position_after <= 0):
            await self.record_exit(result, strategy=strategy, reason="rebalance_exit")
            return
        assert open_trade is not None
        trade_id = open_trade.trade_id

        if result.side == "buy":

            async def op(repo: TradingRepository) -> Trade:
                await self._add_order(repo, result, strategy, trade_id)
                return await repo.scale_in_trade(
                    trade_id,
                    price=result.filled_price,
                    quantity=result.filled_quantity,
                    fees=result.fee,
                )

        else:

            async def op(repo: TradingRepository) -> Trade:
                await self._add_order(repo, result, strategy, trade_id)
                await repo.close_partial_trade(
                    trade_id,
                    quantity=result.filled_quantity,
                    exit_price=result.filled_price,
                    exit_reason="rebalance",
                    exit_fees=result.fee,
                    exit_time=result.timestamp,
                )
                remaining = await repo.get_trade(trade_id)
                assert remaining is not None
                return remaining

        # In-memory view follows the database only after a successful write.
        trade = await self._run(f"record rebalance on {result.pair}", op)
        if trade is not None:
            open_trade.entry_price = float(trade.entry_price)
            open_trade.quantity = float(trade.quantity)

    async def _add_order(
        self,
        repo: TradingRepository,
        result: OrderResult,
        strategy: str,
        trade_id: uuid.UUID | None,
    ) -> None:
        await repo.record_order(
            exchange=self._exchange,
            pair=result.pair,
            side=result.side,
            quantity=result.filled_quantity,
            status=result.status,
            strategy=strategy,
            filled_price=result.filled_price,
            filled_quantity=result.filled_quantity,
            fee=result.fee,
            exchange_order_id=result.exchange_order_id,
            trade_id=trade_id,
        )

    async def record_snapshot(
        self,
        *,
        total_value: float,
        positions_value: float,
        prices: dict[str, float],
        drawdown_pct: float,
        open_positions: int,
        trigger: str = "cycle",
    ) -> None:
        self._max_drawdown_pct = max(self._max_drawdown_pct, drawdown_pct)
        unrealized = sum(
            (prices[pair] - t.entry_price) * t.quantity
            for pair, t in self._open.items()
            if pair in prices
        )
        day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        async def op(repo: TradingRepository) -> None:
            await repo.record_snapshot(
                total_value_usd=total_value,
                cash_balance_usd=total_value - positions_value,
                positions_value_usd=positions_value,
                unrealized_pnl=unrealized,
                realized_pnl_day=await repo.realized_pnl_since(day_start),
                drawdown_pct=drawdown_pct,
                max_drawdown_pct=self._max_drawdown_pct,
                open_positions_count=open_positions,
                snapshot_trigger=trigger,
            )

        await self._run("record portfolio snapshot", op)
