"""Live position ledger: the bot only trades the coins it bought (audit H-1).

The account may hold the owner's own coins of a traded pair. Before the
ledger the bot treated the whole balance as its position: a target of 0 or a
kill-switch sold the owner's coins too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.test_bot import _make_settings
from trading_bot.core.bot import TradingOrchestrator
from trading_bot.execution.live_executor import LiveExecutor
from trading_bot.risk.drawdown import DrawdownMonitor
from trading_bot.risk.stop_tracker import StopTracker

PAIR = "BTC/USDT"


def _exchange(total_btc: float, free_btc: float | None = None) -> MagicMock:
    free_btc = total_btc if free_btc is None else free_btc
    ex = MagicMock()
    ex.fetch_balance = AsyncMock(
        return_value={
            "total": {"USDT": 1_000.0, "BTC": total_btc},
            "free": {"USDT": 1_000.0, "BTC": free_btc},
            "used": {"BTC": total_btc - free_btc},
        }
    )
    ex.fetch_ticker = AsyncMock(return_value={"last": 50_000.0})
    ex.fetch_tickers = AsyncMock(return_value={PAIR: {"last": 50_000.0}})
    ex.fetch_open_orders = AsyncMock(return_value=[])
    ex.load_markets = AsyncMock(return_value={})
    ex.market = MagicMock(return_value={"limits": {}})
    ex.amount_to_precision = MagicMock(
        side_effect=lambda s, a: f"{math.floor(a * 1e5) / 1e5:.5f}"
    )
    ex.cancel_order = AsyncMock(return_value={})
    ex.create_order = AsyncMock(
        side_effect=lambda **kw: {"id": "o", "status": "closed", "filled": kw["amount"]}
    )
    return ex


class TestExecutorLedger:
    async def test_position_ignores_the_owners_coins(self) -> None:
        executor = LiveExecutor(_exchange(1.5), pairs=[PAIR], owned={PAIR: 0.5})
        assert await executor.get_position(PAIR) == pytest.approx(0.5)

    async def test_position_never_exceeds_the_exchange_balance(self) -> None:
        executor = LiveExecutor(_exchange(0.2), pairs=[PAIR], owned={PAIR: 0.5})
        assert await executor.get_position(PAIR) == pytest.approx(0.2)

    async def test_empty_ledger_owns_nothing(self) -> None:
        executor = LiveExecutor(_exchange(1.5), pairs=[PAIR])
        assert await executor.get_position(PAIR) == 0.0
        assert (await executor.get_balance())["positions_value"] == 0.0

    async def test_fills_move_the_ledger(self) -> None:
        executor = LiveExecutor(_exchange(10.0), pairs=[PAIR])
        await executor.place_order(PAIR, "buy", "market", 0.3, price=50_000.0)
        await executor.place_order(PAIR, "buy", "market", 0.2, price=50_000.0)
        assert executor.owned(PAIR) == pytest.approx(0.5)
        await executor.place_order(PAIR, "sell", "market", 0.1, price=50_000.0)
        assert executor.owned(PAIR) == pytest.approx(0.4)
        executor.record_external_fill(PAIR, "sell", 1.0)  # native stop, oversized
        assert executor.owned(PAIR) == 0.0

    async def test_unfilled_order_does_not_move_the_ledger(self) -> None:
        ex = _exchange(1.0)
        ex.create_order = AsyncMock(
            return_value={"id": "o", "status": "expired", "filled": 0.0}
        )
        executor = LiveExecutor(ex, pairs=[PAIR])
        await executor.place_order(PAIR, "buy", "market", 0.3, price=50_000.0)
        assert executor.owned(PAIR) == 0.0

    async def test_balance_values_only_the_bots_coins(self) -> None:
        executor = LiveExecutor(_exchange(1.5), pairs=[PAIR], owned={PAIR: 0.5})
        balance = await executor.get_balance()
        assert balance["positions_value"] == pytest.approx(25_000.0)
        assert balance["total"] == pytest.approx(26_000.0)

    async def test_liquidation_sells_and_cancels_only_the_bots(self) -> None:
        ex = _exchange(1.5)
        ex.fetch_open_orders = AsyncMock(
            return_value=[
                {"id": "bot-stop", "symbol": PAIR},
                {"id": "owner-order", "symbol": PAIR},
            ]
        )
        executor = LiveExecutor(ex, pairs=[PAIR])
        executor.restore({"owned": {PAIR: 0.5}, "orders": {"bot-stop": PAIR}})

        await executor.close_all_positions()

        ex.cancel_order.assert_awaited_once_with("bot-stop", PAIR)
        assert ex.create_order.await_args.kwargs["amount"] == pytest.approx(0.5)
        assert executor.owned(PAIR) == 0.0

    async def test_reconcile_clamps_and_reports_foreign_coins(self) -> None:
        executor = LiveExecutor(_exchange(0.3), pairs=[PAIR], owned={PAIR: 0.5})
        assert await executor.reconcile() == {}
        assert executor.owned(PAIR) == pytest.approx(0.3)

        executor = LiveExecutor(_exchange(1.5), pairs=[PAIR], owned={PAIR: 0.5})
        assert await executor.reconcile() == {PAIR: pytest.approx(1.0)}
        assert executor.owned(PAIR) == pytest.approx(0.5)

    async def test_snapshot_round_trip_and_validation(self) -> None:
        executor = LiveExecutor(_exchange(1.0), pairs=[PAIR], owned={PAIR: 0.5})
        executor.restore({"owned": {PAIR: 0.5}, "orders": {"s1": PAIR}})
        copy = LiveExecutor(_exchange(1.0), pairs=[PAIR])
        copy.restore(json.loads(json.dumps(executor.to_dict())))
        assert copy.to_dict() == executor.to_dict()
        for bad in ({PAIR: -1.0}, {PAIR: float("nan")}, {PAIR: "x"}):
            with pytest.raises((ValueError, TypeError)):
                copy.restore({"owned": bad})
        assert copy.to_dict() == executor.to_dict()  # unchanged on error

    async def test_finished_native_order_is_forgotten(self) -> None:
        ex = _exchange(1.0)
        ex.fetch_order = AsyncMock(return_value={"id": "s1", "status": "closed"})
        executor = LiveExecutor(ex, pairs=[PAIR])
        executor.restore({"owned": {PAIR: 0.5}, "orders": {"s1": PAIR}})
        await executor.get_order("s1", PAIR)
        assert executor.to_dict()["orders"] == {}


def _live_orch(tmp_path: Path, exchange: MagicMock, **overrides) -> TradingOrchestrator:
    settings = _make_settings(
        BOT_MODE="live",
        TRADING_PAIRS=[PAIR],
        RISK_STATE_PATH=str(tmp_path / "risk_state_{mode}.json"),
        **overrides,
    )
    orch = TradingOrchestrator(settings=settings)
    orch._executor = LiveExecutor(exchange, pairs=[PAIR])
    return orch


def _old_state(tmp_path: Path, stops: dict | None = None) -> Path:
    """A live state file written before the ledger existed."""
    path = tmp_path / "risk_state_live.json"
    monitor = DrawdownMonitor(initial_value=76_000.0)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "live",
                "drawdown": monitor.to_dict(),
                "stops": stops or {},
                "rebalanced": {},
            }
        ),
        encoding="utf-8",
    )
    return path


class TestBotLedgerStartup:
    async def test_first_start_ignores_existing_coins(self, tmp_path: Path) -> None:
        orch = _live_orch(tmp_path, _exchange(1.5))
        await orch._load_risk_state(76_000.0)

        assert await orch._executor.get_position(PAIR) == 0.0
        saved = json.loads((tmp_path / "risk_state_live.json").read_text())
        assert saved["live_ledger"]["owned"] == {}

    async def test_adopt_flag_takes_the_whole_balance(self, tmp_path: Path) -> None:
        orch = _live_orch(tmp_path, _exchange(1.5), LIVE_ADOPT_EXISTING_BALANCE=True)
        await orch._load_risk_state(76_000.0)
        assert await orch._executor.get_position(PAIR) == pytest.approx(1.5)

    async def test_upgrade_adopts_what_saved_stops_cover(self, tmp_path: Path) -> None:
        tracker = StopTracker()
        tracker.register(PAIR, 45_000.0, 0.5, "stop-1", None)
        _old_state(tmp_path, stops=tracker.to_dict())
        orch = _live_orch(tmp_path, _exchange(1.5))

        await orch._load_risk_state(76_000.0)

        assert await orch._executor.get_position(PAIR) == pytest.approx(0.5)
        assert orch._executor.to_dict()["orders"] == {"stop-1": PAIR}
        assert PAIR in orch._stop_tracker  # still protected

    async def test_upgrade_rebaselines_instead_of_halting(self, tmp_path: Path) -> None:
        # The old peak valued the owner's 1.5 BTC; the ledger values none.
        _old_state(tmp_path)
        orch = _live_orch(tmp_path, _exchange(1.5))
        await orch._load_risk_state(76_000.0)
        assert orch._drawdown_monitor is not None
        assert orch._drawdown_monitor.update(1_000.0) is None  # no false HALT

    async def test_ledger_survives_a_restart(self, tmp_path: Path) -> None:
        ex = _exchange(1.5)
        first = _live_orch(tmp_path, ex)
        await first._load_risk_state(76_000.0)
        await first._execute_order(PAIR, "buy", 0.2, 50_000.0)  # saved at once

        second = _live_orch(tmp_path, ex)
        await second._load_risk_state(76_000.0)
        assert await second._executor.get_position(PAIR) == pytest.approx(0.2)
