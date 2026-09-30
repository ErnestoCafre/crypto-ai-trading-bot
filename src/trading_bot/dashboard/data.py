"""Data sources for the dashboard — real data or ``None``, never placeholders.

* Risk state: the JSON the bot persists every cycle (``RISK_STATE_PATH``).
* Trades / portfolio snapshots: PostgreSQL via :class:`TradingRepository`
  (empty until the bot is wired to persist them — see ``db/repository.py``).
* Market data: public OHLCV through :class:`CCXTFetcher`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from trading_bot.config.settings import Settings
from trading_bot.healthcheck import heartbeat_age, load_risk_state


@dataclass
class StopView:
    pair: str
    stop_price: float
    quantity: float
    exchange_order: bool


@dataclass
class RiskView:
    state: str
    halt_reason: str | None
    peak_value: float
    last_value: float
    day_start_value: float
    saved_at: str | None
    heartbeat_age: timedelta | None
    stops: list[StopView] = field(default_factory=list)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.last_value) / self.peak_value * 100)

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_value <= 0:
            return 0.0
        return (self.last_value - self.day_start_value) / self.day_start_value * 100


def parse_risk_state(
    data: dict[str, Any] | None, now: datetime | None = None
) -> RiskView | None:
    """Build a :class:`RiskView` from the persisted JSON; None if unusable."""
    if not data:
        return None
    try:
        dd = data["drawdown"]
        stops = [
            StopView(
                pair=str(raw["pair"]),
                stop_price=float(raw["stop_price"]),
                quantity=float(raw["quantity"]),
                exchange_order=raw.get("order_id") is not None,
            )
            for raw in (data.get("stops") or {}).values()
        ]
        return RiskView(
            state=str(dd["state"]),
            halt_reason=dd.get("halt_reason"),
            peak_value=float(dd["peak_value"]),
            last_value=float(dd["last_value"]),
            day_start_value=float(dd["day_start_value"]),
            saved_at=data.get("saved_at"),
            heartbeat_age=heartbeat_age(data, now),
            stops=stops,
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("Risk state has an unexpected format")
        return None


def load_risk_view(settings: Settings) -> RiskView | None:
    return parse_risk_state(load_risk_state(settings))


@dataclass
class DbView:
    trades: pd.DataFrame  # closed trades, newest first
    open_trades: pd.DataFrame
    equity: pd.Series  # total portfolio value by snapshot time

    @property
    def win_rate(self) -> float | None:
        if self.trades.empty:
            return None
        return float((self.trades["PnL"] > 0).mean())


async def _read_db() -> DbView:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from trading_bot.db.base import build_url
    from trading_bot.db.repository import TradingRepository

    # Each Streamlit rerun calls asyncio.run() with a fresh event loop, so a
    # pooled (loop-bound) engine cannot be shared between reads.
    engine = create_async_engine(build_url(), poolclass=NullPool)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            repo = TradingRepository(session)
            closed = await repo.recent_trades(limit=100)
            opened = await repo.open_trades()
            snaps = await repo.snapshots(limit=2000)
    finally:
        await engine.dispose()

    def rows(trades: list[Any]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Par": t.pair,
                    "Lado": t.side,
                    "Estrategia": t.strategy,
                    "Entrada": t.entry_time,
                    "Precio entrada": float(t.entry_price),
                    "Cantidad": float(t.quantity),
                    "Salida": t.exit_time,
                    "Precio salida": (
                        float(t.exit_price) if t.exit_price is not None else None
                    ),
                    "Motivo": t.exit_reason,
                    "PnL": float(t.pnl) if t.pnl is not None else None,
                    "Fees": float(t.total_fees) if t.total_fees is not None else None,
                }
                for t in trades
            ]
        )

    equity = pd.Series(
        [float(s.total_value_usd) for s in snaps],
        index=pd.DatetimeIndex([s.timestamp for s in snaps]),
        name="equity",
        dtype=float,
    )
    return DbView(trades=rows(closed), open_trades=rows(opened), equity=equity)


def load_db_view() -> DbView | None:
    """Trades and snapshots from the database, or None if it is unreachable."""
    try:
        return asyncio.run(_read_db())
    except Exception as exc:
        logger.warning("Dashboard: database unavailable ({})", exc)
        return None


async def _fetch_ohlcv(
    exchange_id: str, pair: str, timeframe: str, bars: int
) -> pd.DataFrame:
    from trading_bot.data.market.candles import timeframe_to_timedelta
    from trading_bot.data.market.ccxt_fetcher import CCXTFetcher

    since = datetime.now(UTC) - (bars + 1) * timeframe_to_timedelta(timeframe)
    # Public market data: no keys, production endpoints (testnet prices are
    # not representative).
    async with CCXTFetcher(
        exchange_id=exchange_id, sandbox=False, max_retries=2, timeout_s=20
    ) as fetcher:
        return await fetcher.fetch_ohlcv(pair, timeframe, since=since, limit=bars)


def load_ohlcv(
    settings: Settings, pair: str, timeframe: str, bars: int = 200
) -> pd.DataFrame | None:
    """Recent OHLCV from the exchange, or None on any failure."""
    try:
        df = asyncio.run(_fetch_ohlcv(settings.EXCHANGE_ID, pair, timeframe, bars))
    except Exception as exc:
        logger.warning("Dashboard: OHLCV unavailable for {} ({})", pair, exc)
        return None
    return df if not df.empty else None
