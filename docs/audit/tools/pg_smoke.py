"""TradeJournal smoke test against a real PostgreSQL (after alembic upgrade head).

    DATABASE_URL=postgresql://postgres:x@127.0.0.1:55432/trading_bot \
        uv run python -I docs/audit/tools/pg_smoke.py
"""

import asyncio
import os
import uuid
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from trading_bot.core.models import OrderResult
from trading_bot.core.trade_journal import TradeJournal
from trading_bot.db.base import build_url

URL = os.environ["DATABASE_URL"]


def fill(side: str, qty: float, px: float) -> OrderResult:
    return OrderResult(
        order_id=str(uuid.uuid4()),
        exchange_order_id=None,
        pair="BTC/USDT",
        side=side,
        filled_price=px,
        filled_quantity=qty,
        fee=px * qty * 0.001,
        slippage_pct=0.05,
        status="filled",
        timestamp=datetime.now(UTC),
    )


async def main() -> None:
    journal = await TradeJournal.connect(URL, "binance")
    assert journal is not None, "database unreachable"
    for side, qty, px, after in [
        ("buy", 0.1, 60000, 0.1),
        ("buy", 0.05, 62000, 0.15),
        ("sell", 0.05, 63000, 0.10),
        ("sell", 0.10, 64000, 0.0),
    ]:
        await journal.record_rebalance(fill(side, qty, px), "donchian_ensemble", after)
    await journal.record_snapshot(
        total_value=100500,
        positions_value=0,
        prices={"BTC/USDT": 64000},
        drawdown_pct=0.5,
        open_positions=0,
        trigger="daily_paused",
    )
    await journal.close()
    engine = create_async_engine(build_url(URL))
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "select status, quantity, entry_price, exit_price, pnl, exit_reason "
                "from trades order by created_at"
            )
        )
        for row in rows:
            print("trade", tuple(row))
        print(
            "orders", (await conn.execute(text("select count(*) from orders"))).scalar()
        )
    await engine.dispose()


asyncio.run(main())
