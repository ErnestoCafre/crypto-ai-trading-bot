"""Runtime persistence: orders, trades and portfolio snapshots.

Written by ``core.trade_journal.TradeJournal`` from the trading loop: every
fill (:meth:`record_order`), entries (:meth:`open_trade`), exits, stops and
kill-switch liquidations (:meth:`close_trade`) and one snapshot per trading
cycle (:meth:`record_snapshot`). The dashboard reads trades and snapshots
through this class.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from trading_bot.db.models import Order, PortfolioSnapshot, Trade


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


class TradingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Writes ---------------------------------------------------------

    async def record_order(
        self,
        *,
        exchange: str,
        pair: str,
        side: str,
        quantity: float,
        status: str,
        strategy: str,
        order_type: str = "market",
        price: float | None = None,
        filled_price: float | None = None,
        filled_quantity: float | None = None,
        fee: float | None = None,
        fee_currency: str | None = None,
        exchange_order_id: str | None = None,
        trade_id: uuid.UUID | None = None,
        signal_source: str | None = None,
    ) -> Order:
        now = datetime.now(UTC)
        order = Order(
            exchange=exchange,
            pair=pair,
            side=side,
            order_type=order_type,
            price=_dec(price),
            quantity=_dec(quantity),
            filled_price=_dec(filled_price),
            filled_quantity=_dec(filled_quantity),
            fee=_dec(fee),
            fee_currency=fee_currency,
            exchange_order_id=exchange_order_id,
            trade_id=trade_id,
            status=status,
            strategy=strategy,
            signal_source=signal_source,
            created_at=now,
            updated_at=now,
        )
        self._session.add(order)
        await self._session.commit()
        return order

    async def open_trade(
        self,
        *,
        exchange: str,
        pair: str,
        strategy: str,
        entry_price: float,
        quantity: float,
        fees: float = 0.0,
        side: str = "long",
        entry_time: datetime | None = None,
        ai_combined_confidence: float | None = None,
    ) -> Trade:
        now = datetime.now(UTC)
        trade = Trade(
            exchange=exchange,
            pair=pair,
            side=side,
            strategy=strategy,
            entry_price=_dec(entry_price),
            entry_time=entry_time or now,
            quantity=_dec(quantity),
            total_fees=_dec(fees),
            ai_combined_confidence=_dec(ai_combined_confidence),
            status="open",
            created_at=now,
        )
        self._session.add(trade)
        await self._session.commit()
        return trade

    async def close_trade(
        self,
        trade_id: uuid.UUID,
        *,
        exit_price: float,
        exit_reason: str,
        exit_fees: float = 0.0,
        exit_time: datetime | None = None,
    ) -> Trade:
        """Close an open trade; PnL is net of entry and exit fees."""
        trade = await self._session.get(Trade, trade_id)
        if trade is None:
            raise LookupError(f"trade {trade_id} not found")
        exit_time = exit_time or datetime.now(UTC)
        entry = float(trade.entry_price)
        qty = float(trade.quantity)
        fees = float(trade.total_fees or 0) + exit_fees
        direction = 1.0 if trade.side == "long" else -1.0
        pnl = direction * (exit_price - entry) * qty - fees

        trade.exit_price = _dec(exit_price)
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason
        trade.total_fees = _dec(fees)
        trade.pnl = _dec(pnl)
        trade.pnl_pct = (
            _dec(round(pnl / (entry * qty) * 100, 6)) if entry * qty else None
        )
        entry_time = trade.entry_time
        if entry_time.tzinfo is None:  # SQLite drops tzinfo
            entry_time = entry_time.replace(tzinfo=UTC)
        hours = (exit_time - entry_time).total_seconds() / 3600
        trade.holding_time_hours = _dec(round(hours, 2))
        trade.status = "closed"
        await self._session.commit()
        return trade

    async def abandon_trade(self, trade_id: uuid.UUID, reason: str) -> None:
        """Mark an open trade whose position no longer exists (no exit fill
        known, e.g. the paper account was reset); it leaves open/closed views
        without a fabricated exit price or PnL."""
        trade = await self._session.get(Trade, trade_id)
        if trade is None or trade.status != "open":
            return
        trade.status = "abandoned"
        trade.exit_reason = reason
        await self._session.commit()

    async def record_snapshot(
        self,
        *,
        total_value_usd: float,
        cash_balance_usd: float,
        positions_value_usd: float,
        unrealized_pnl: float,
        realized_pnl_day: float,
        drawdown_pct: float,
        max_drawdown_pct: float,
        open_positions_count: int,
        snapshot_trigger: str = "cycle",
        timestamp: datetime | None = None,
    ) -> PortfolioSnapshot:
        snapshot = PortfolioSnapshot(
            timestamp=timestamp or datetime.now(UTC),
            total_value_usd=_dec(total_value_usd),
            cash_balance_usd=_dec(cash_balance_usd),
            positions_value_usd=_dec(positions_value_usd),
            unrealized_pnl=_dec(unrealized_pnl),
            realized_pnl_day=_dec(realized_pnl_day),
            drawdown_pct=_dec(drawdown_pct),
            max_drawdown_pct=_dec(max_drawdown_pct),
            open_positions_count=open_positions_count,
            snapshot_trigger=snapshot_trigger,
        )
        self._session.add(snapshot)
        await self._session.commit()
        return snapshot

    # -- Reads ----------------------------------------------------------

    async def open_trades(self) -> list[Trade]:
        result = await self._session.execute(
            select(Trade).where(Trade.status == "open").order_by(Trade.entry_time)
        )
        return list(result.scalars())

    async def recent_trades(self, limit: int = 50) -> list[Trade]:
        result = await self._session.execute(
            select(Trade)
            .where(Trade.status == "closed")
            .order_by(Trade.exit_time.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def realized_pnl_since(self, since: datetime) -> float:
        """Net PnL of trades closed at or after ``since``."""
        result = await self._session.execute(
            select(func.coalesce(func.sum(Trade.pnl), 0)).where(
                Trade.status == "closed", Trade.exit_time >= since
            )
        )
        return float(result.scalar_one() or 0)

    async def latest_snapshot(self) -> PortfolioSnapshot | None:
        result = await self._session.execute(
            select(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def snapshots(self, limit: int = 1000) -> list[PortfolioSnapshot]:
        """Most recent snapshots, oldest first."""
        result = await self._session.execute(
            select(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(limit)
        )
        return list(reversed(list(result.scalars())))
