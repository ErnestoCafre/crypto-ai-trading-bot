"""TradingRepository and timezone-aware model columns."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import DateTime
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from trading_bot.db.base import Base
from trading_bot.db.repository import TradingRepository


def test_every_datetime_column_is_timezone_aware() -> None:
    """Models must match the migrations' DateTime(timezone=True) columns."""
    import trading_bot.db.models  # noqa: F401 — registers the tables

    naive = [
        f"{table.name}.{col.name}"
        for table in Base.metadata.tables.values()
        for col in table.columns
        if isinstance(col.type, DateTime) and not col.type.timezone
    ]
    assert naive == []


def test_importing_db_does_not_create_engine() -> None:
    import trading_bot.db.base as base

    base.get_engine.cache_clear()
    import trading_bot.db.models  # noqa: F401

    assert base.get_engine.cache_info().currsize == 0


@pytest.fixture
async def repo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield TradingRepository(session)
    await engine.dispose()


async def test_trade_lifecycle(repo: TradingRepository) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trade = await repo.open_trade(
        exchange="binance",
        pair="BTC/USDT",
        strategy="trend_following",
        entry_price=100.0,
        quantity=2.0,
        fees=0.2,
        entry_time=t0,
    )
    assert [t.id for t in await repo.open_trades()] == [trade.id]

    closed = await repo.close_trade(
        trade.id,
        exit_price=110.0,
        exit_reason="signal",
        exit_fees=0.22,
        exit_time=t0 + timedelta(hours=5),
    )
    assert float(closed.pnl) == pytest.approx(20.0 - 0.42)
    assert float(closed.holding_time_hours) == pytest.approx(5.0)
    assert await repo.open_trades() == []
    assert [t.id for t in await repo.recent_trades()] == [trade.id]


async def test_scale_in_and_partial_close(repo: TradingRepository) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    trade = await repo.open_trade(
        exchange="binance",
        pair="BTC/USDT",
        strategy="donchian_ensemble",
        entry_price=100.0,
        quantity=1.0,
        fees=0.1,
        entry_time=t0,
    )
    await repo.scale_in_trade(trade.id, price=200.0, quantity=1.0, fees=0.2)
    piece = await repo.close_partial_trade(
        trade.id,
        quantity=0.5,
        exit_price=300.0,
        exit_reason="rebalance",
        exit_fees=0.15,
        exit_time=t0 + timedelta(days=2),
    )
    # Slice: avg entry 150, a quarter of the 0.3 entry fees.
    assert float(piece.entry_price) == pytest.approx(150.0)
    assert float(piece.pnl) == pytest.approx(150 * 0.5 - 0.075 - 0.15)
    assert piece.entry_time.replace(tzinfo=UTC) == t0
    (open_,) = await repo.open_trades()
    assert open_.id == trade.id
    assert float(open_.quantity) == pytest.approx(1.5)
    assert float(open_.total_fees) == pytest.approx(0.225)


async def test_snapshots_oldest_first(repo: TradingRepository) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    for h in range(3):
        await repo.record_snapshot(
            total_value_usd=1000 + h,
            cash_balance_usd=1000,
            positions_value_usd=h,
            unrealized_pnl=0,
            realized_pnl_day=0,
            drawdown_pct=0,
            max_drawdown_pct=0,
            open_positions_count=0,
            timestamp=t0 + timedelta(hours=h),
        )
    snaps = await repo.snapshots(limit=2)
    assert [float(s.total_value_usd) for s in snaps] == [1001, 1002]


async def test_record_order(
    repo: TradingRepository,
) -> None:
    order = await repo.record_order(
        exchange="binance",
        pair="BTC/USDT",
        side="buy",
        quantity=0.5,
        status="filled",
        strategy="trend_following",
        filled_price=100.0,
    )
    assert order.id is not None


async def test_get_trade(repo: TradingRepository) -> None:
    trade = await repo.open_trade(
        exchange="binance",
        pair="ETH/USDT",
        strategy="donchian_ensemble",
        entry_price=10.0,
        quantity=1.0,
    )
    assert (await repo.get_trade(trade.id)) is trade
