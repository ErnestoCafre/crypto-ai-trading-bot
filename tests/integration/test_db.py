"""Integration tests for database ORM models and tables creation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trading_bot.db.base import Base
from trading_bot.db.models import Order, PortfolioSnapshot


@pytest.fixture
async def async_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_order_and_trade_roundtrip(async_db: AsyncSession) -> None:
    now = datetime.now(UTC)
    order_id = uuid.uuid4()

    order = Order(
        id=order_id,
        exchange="binance",
        pair="BTC/USDT",
        side="buy",
        order_type="market",
        price=Decimal("65000.0"),
        quantity=Decimal("0.1"),
        status="filled",
        strategy="trend_following",
        created_at=now,
        updated_at=now,
    )
    async_db.add(order)
    await async_db.commit()

    stmt = select(Order).where(Order.id == order_id)
    result = await async_db.execute(stmt)
    fetched_order = result.scalar_one_or_none()

    assert fetched_order is not None
    assert fetched_order.pair == "BTC/USDT"
    assert fetched_order.quantity == Decimal("0.1")


@pytest.mark.asyncio
async def test_portfolio_snapshot_creation(async_db: AsyncSession) -> None:
    now = datetime.now(UTC)
    snapshot = PortfolioSnapshot(
        timestamp=now,
        total_value_usd=Decimal("100000.00"),
        cash_balance_usd=Decimal("90000.00"),
        positions_value_usd=Decimal("10000.00"),
        unrealized_pnl=Decimal("250.00"),
        realized_pnl_day=Decimal("50.00"),
        drawdown_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.05"),
        open_positions_count=1,
        snapshot_trigger="timer",
    )
    async_db.add(snapshot)
    await async_db.commit()

    stmt = select(PortfolioSnapshot)
    result = await async_db.execute(stmt)
    fetched = result.scalars().all()
    assert len(fetched) == 1
    assert fetched[0].total_value_usd == Decimal("100000.00")
