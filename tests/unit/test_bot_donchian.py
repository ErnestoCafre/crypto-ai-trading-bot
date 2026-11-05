"""TradingOrchestrator in STRATEGY_MODE=donchian_ensemble (target exposure)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_bot import _db_rows, _make_settings, journal_db  # noqa: F401
from trading_bot.core import bot as bot_module
from trading_bot.core.bot import TradingOrchestrator
from trading_bot.core.models import AISignals
from trading_bot.core.strategies.donchian_ensemble import (
    HISTORY_BARS,
    target_exposure,
)
from trading_bot.execution.paper_executor import PaperExecutor
from trading_bot.risk.drawdown import DrawdownMonitor, TradingState
from trading_bot.risk.stop_tracker import StopTracker

PAIR = "BTC/USDT"
LOOKBACKS = [5, 10, 20, 30, 60, 90, 150, 250, 360]


def _settings(tmp_path, **overrides):
    values = {
        "STRATEGY_MODE": "donchian_ensemble",
        "TRADING_PAIRS": [PAIR],
        "MAX_POSITION_PCT": 25.0,
        "DONCHIAN_LOOKBACKS": LOOKBACKS,
        "DONCHIAN_VOL_TARGET": 0.25,
        "DONCHIAN_VOL_WINDOW": 90,
        "REBALANCE_THRESHOLD": 0.05,
        "RISK_STATE_PATH": str(tmp_path / "risk_state_{mode}.json"),
        "DONCHIAN_DECISION_LOG_PATH": str(tmp_path / "decisions_{mode}.jsonl"),
    }
    values.update(overrides)
    return _make_settings(**values)


def _daily(n: int = HISTORY_BARS + 1, seed: int = 3, drift: float = 0.002):
    """Daily candles ending with today's forming candle (open 00:00 UTC)."""
    rng = np.random.default_rng(seed)
    close = 20_000 * np.exp(np.cumsum(rng.normal(drift, 0.03, n)))
    today = pd.Timestamp(datetime.now(UTC).date(), tz="UTC")
    index = pd.date_range(end=today, periods=n, freq="1D")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def _paper(cash: float = 100_000.0, fee_pct: float = 0.0) -> PaperExecutor:
    return PaperExecutor(initial_balance=cash, slippage_pct=0.0, fee_pct=fee_pct)


def _orchestrator(tmp_path, ohlcv, executor=None, **overrides):
    orch = TradingOrchestrator(settings=_settings(tmp_path, **overrides))
    orch._executor = executor or _paper()
    orch._fetcher = AsyncMock()
    orch._fetcher.fetch_ohlcv = AsyncMock(return_value=ohlcv)
    orch._drawdown_monitor = DrawdownMonitor(initial_value=100_000.0)
    return orch


def _expected_w(ohlcv: pd.DataFrame) -> float:
    return target_exposure(ohlcv.iloc[:-1], LOOKBACKS).exposure


async def _position_value(executor: PaperExecutor, price: float) -> float:
    return await executor.get_position(PAIR) * price


class TestTargetRebalancing:
    async def test_buys_allocation_times_target(self, tmp_path):
        ohlcv = _daily()
        w = _expected_w(ohlcv)
        assert 0.1 < w <= 1.0
        orch = _orchestrator(tmp_path, ohlcv)

        await orch.run_trading_cycle()

        price = float(ohlcv["close"].iloc[-1])
        value = await _position_value(orch._executor, price)
        assert value == pytest.approx(100_000 * 0.25 * w, rel=1e-9)

    async def test_fetches_daily_candles_with_full_history(self, tmp_path):
        orch = _orchestrator(tmp_path, _daily(), TRADING_TIMEFRAME="1h")
        await orch.run_trading_cycle()
        kwargs = orch._fetcher.fetch_ohlcv.call_args.kwargs
        assert kwargs["timeframe"] == "1d"
        assert kwargs["limit"] == HISTORY_BARS + 1

    async def test_same_candle_is_traded_once(self, tmp_path):
        executor = _paper()
        executor.place_order = AsyncMock(wraps=executor.place_order)
        orch = _orchestrator(tmp_path, _daily(), executor)

        for _ in range(3):
            await orch.run_trading_cycle()

        assert executor.place_order.await_count == 1

    async def test_forming_daily_candle_is_ignored(self, tmp_path):
        ohlcv = _daily()
        crashed = ohlcv.copy()
        crashed.iloc[-1, crashed.columns.get_loc("close")] *= 0.5
        orch = _orchestrator(tmp_path, crashed)
        seen = []

        def spy(ohlcv, *args):
            seen.append(target_exposure(ohlcv, *args))
            return seen[-1]

        with patch.object(bot_module, "target_exposure", side_effect=spy) as mock:
            await orch.run_trading_cycle()
        assert mock.call_args.args[0].index[-1] == ohlcv.index[-2]
        assert seen[0].exposure == pytest.approx(_expected_w(ohlcv))

    async def test_restart_mid_trend_keeps_exposure(self, tmp_path):
        ohlcv = _daily()
        executor = _paper()
        first = _orchestrator(tmp_path, ohlcv, executor)
        await first._load_risk_state(100_000.0)
        await first.run_trading_cycle()
        held = await executor.get_position(PAIR)
        assert held > 0

        # Restart with the persisted risk state: the candle is already done.
        executor.place_order = AsyncMock(wraps=executor.place_order)
        second = _orchestrator(tmp_path, ohlcv, executor)
        await second._load_risk_state(100_000.0)
        await second.run_trading_cycle()
        executor.place_order.assert_not_awaited()

        # Even without the persisted mark the state is rebuilt from OHLCV:
        # same target, nothing to trade.
        (tmp_path / "risk_state_paper.json").unlink()
        third = _orchestrator(tmp_path, ohlcv, executor)
        await third._load_risk_state(100_000.0)
        await third.run_trading_cycle()
        executor.place_order.assert_not_awaited()
        assert await executor.get_position(PAIR) == held

    async def test_insufficient_history_holds_position_and_alerts(self, tmp_path):
        # Too little history is a data problem, not a sell signal (audit M-1).
        short = _daily(n=400)
        executor = _paper()
        await executor.place_order(PAIR, "buy", "market", 0.5, 20_000.0)
        orch = _orchestrator(tmp_path, short, executor)
        orch._notifier = AsyncMock()

        await orch.run_trading_cycle()
        await orch.run_trading_cycle()

        assert await executor.get_position(PAIR) == 0.5
        assert PAIR not in orch._rebalanced  # retried next cycle
        orch._notifier.send_alert.assert_awaited_once()  # once per candle
        records = [
            json.loads(line)
            for line in (tmp_path / "decisions_paper.jsonl").read_text().splitlines()
        ]
        assert records[-1]["held_not_ready"] is True

    async def test_unfilled_order_is_retried_next_cycle(self, tmp_path):
        # A rejected/expired order must not mark the candle done (audit H-2).
        executor = _paper()
        real_place = executor.place_order
        rejected = PaperExecutor._rejected(PAIR, "buy", 0.0)
        executor.place_order = AsyncMock(return_value=rejected)
        orch = _orchestrator(tmp_path, _daily(), executor)

        await orch.run_trading_cycle()
        assert PAIR not in orch._rebalanced
        records = [
            json.loads(line)
            for line in (tmp_path / "decisions_paper.jsonl").read_text().splitlines()
        ]
        assert records[-1]["unfilled"] is True

        executor.place_order = AsyncMock(wraps=real_place)
        await orch.run_trading_cycle()
        executor.place_order.assert_awaited_once()
        assert PAIR in orch._rebalanced
        assert await executor.get_position(PAIR) > 0

    async def test_no_history_and_flat_places_nothing(self, tmp_path):
        executor = _paper()
        executor.place_order = AsyncMock(wraps=executor.place_order)
        orch = _orchestrator(tmp_path, _daily(n=400), executor)
        await orch.run_trading_cycle()
        executor.place_order.assert_not_awaited()

    async def test_gap_below_threshold_is_not_traded(self, tmp_path):
        ohlcv = _daily()
        price = float(ohlcv["close"].iloc[-1])
        w = _expected_w(ohlcv)
        executor = _paper()
        # Hold 3 % of the sleeve less than the target (threshold 5 %).
        qty = 100_000 * 0.25 * (w - 0.03) / price
        await executor.place_order(PAIR, "buy", "market", qty, price)
        executor.place_order = AsyncMock(wraps=executor.place_order)
        orch = _orchestrator(tmp_path, ohlcv, executor)

        await orch.run_trading_cycle()

        executor.place_order.assert_not_awaited()


class TestPartialRebalanceOnPaper:
    async def test_partial_sell_keeps_average_price_and_credits_cash(self, tmp_path):
        ohlcv = _daily()
        price = float(ohlcv["close"].iloc[-1])
        w = _expected_w(ohlcv)
        executor = _paper(fee_pct=0.1)
        # Twice the target value, bought in two fills at different prices.
        target_value = 100_000 * 0.25 * w
        await executor.place_order(PAIR, "buy", "market", target_value / price, price)
        await executor.place_order(
            PAIR, "buy", "market", target_value / (0.9 * price), 0.9 * price
        )
        avg_before = executor._positions[PAIR]["avg_price"]
        qty_before = await executor.get_position(PAIR)
        cash_before = (await executor.get_balance())["cash"]
        orch = _orchestrator(tmp_path, ohlcv, executor)

        await orch.run_trading_cycle()

        sold = qty_before - await executor.get_position(PAIR)
        assert 0 < sold < qty_before
        assert executor._positions[PAIR]["avg_price"] == pytest.approx(avg_before)
        cash_after = (await executor.get_balance())["cash"]
        assert cash_after - cash_before == pytest.approx(sold * price * (1 - 0.001))
        balance = await executor.get_balance()
        # Sold back to the target, measured on the equity before the sale.
        assert balance["positions_value"] == pytest.approx(
            orch._settings.MAX_POSITION_PCT
            / 100
            * w
            * (cash_before + qty_before * price),
            rel=1e-9,
        )

    async def test_buys_what_the_cash_affords(self, tmp_path):
        ohlcv = _daily()
        price = float(ohlcv["close"].iloc[-1])
        executor = _paper()
        # Cash tied up in another (unmanaged here) asset.
        await executor.place_order("ETH/USDT", "buy", "market", 99.0, 1_000.0)
        orch = _orchestrator(tmp_path, ohlcv, executor)

        await orch.run_trading_cycle()

        bought = await executor.get_position(PAIR)
        assert bought * price == pytest.approx(1_000.0 * 0.99)
        assert (await executor.get_balance())["cash"] == pytest.approx(10.0)


class TestRiskAndAI:
    async def test_kill_switch_blocks_increase(self, tmp_path):
        executor = _paper()
        executor.place_order = AsyncMock(wraps=executor.place_order)
        orch = _orchestrator(tmp_path, _daily(), executor)
        orch._drawdown_monitor._state = TradingState.HALTED

        await orch.run_trading_cycle()

        executor.place_order.assert_not_awaited()
        assert orch._rebalanced == {}  # retried once trading is allowed

    async def test_kill_switch_still_allows_reduction(self, tmp_path):
        ohlcv = _daily()
        price = float(ohlcv["close"].iloc[-1])
        executor = _paper()
        await executor.place_order(PAIR, "buy", "market", 60_000 / price, price)
        orch = _orchestrator(tmp_path, ohlcv, executor)
        orch._drawdown_monitor._state = TradingState.DAILY_PAUSED

        await orch.run_trading_cycle()

        value = await _position_value(executor, price)
        assert value < 60_000
        assert value == pytest.approx(100_000 * 0.25 * _expected_w(ohlcv), rel=1e-9)

    async def test_llm_consensus_short_blocks_increase(self, tmp_path):
        executor = _paper()
        executor.place_order = AsyncMock(wraps=executor.place_order)
        orch = _orchestrator(tmp_path, _daily(), executor)
        orch._build_ai_signals = MagicMock(
            return_value=AISignals(
                llm_recommendation="SHORT", llm_confidence=0.0, llm_consensus=True
            )
        )

        await orch.run_trading_cycle()

        executor.place_order.assert_not_awaited()

    async def test_bearish_sentiment_shrinks_target(self, tmp_path):
        ohlcv = _daily()
        orch = _orchestrator(tmp_path, ohlcv, AI_SIZING_FLOOR=0.25)
        orch._build_ai_signals = MagicMock(return_value=AISignals(sentiment_score=-1.0))

        await orch.run_trading_cycle()

        price = float(ohlcv["close"].iloc[-1])
        value = await _position_value(orch._executor, price)
        assert value == pytest.approx(100_000 * 0.25 * _expected_w(ohlcv) * 0.25)

    async def test_bullish_sentiment_never_raises_target(self, tmp_path):
        ohlcv = _daily()
        orch = _orchestrator(tmp_path, ohlcv)
        orch._build_ai_signals = MagicMock(return_value=AISignals(sentiment_score=1.0))

        await orch.run_trading_cycle()

        price = float(ohlcv["close"].iloc[-1])
        value = await _position_value(orch._executor, price)
        assert value == pytest.approx(100_000 * 0.25 * _expected_w(ohlcv))

    async def test_kill_switch_trigger_liquidates_without_rebalancing(self, tmp_path):
        ohlcv = _daily()
        price = float(ohlcv["close"].iloc[-1])
        executor = _paper()
        await executor.place_order(PAIR, "buy", "market", 10_000 / price, price)
        orch = _orchestrator(tmp_path, ohlcv, executor)
        # Peak far above the current value: max drawdown fires this cycle.
        orch._drawdown_monitor = DrawdownMonitor(initial_value=200_000.0)

        await orch.run_trading_cycle()

        assert orch._drawdown_monitor.state == TradingState.HALTED
        assert await executor.get_position(PAIR) == 0.0
        await orch.run_trading_cycle()  # HALTED: no re-entry
        assert await executor.get_position(PAIR) == 0.0

    async def test_trade_stops_are_dropped_in_target_mode(self, tmp_path):
        executor = _paper()
        executor.cancel_order = AsyncMock(return_value=True)
        orch = _orchestrator(tmp_path, _daily(), executor)
        orch._stop_tracker = StopTracker()
        orch._stop_tracker.register(PAIR, 15_000.0, 0.5, "native-1", None)

        await orch._drop_trade_stops()

        executor.cancel_order.assert_awaited_once_with("native-1", PAIR)
        assert len(orch._stop_tracker) == 0

    async def test_live_amounts_use_market_minimums(self, tmp_path):
        executor = AsyncMock()
        executor.get_balance = AsyncMock(
            return_value={"total": 100_000.0, "free": 100_000.0}
        )
        executor.get_position = AsyncMock(return_value=0.0)
        executor.tradable_amount = AsyncMock(return_value=0.0)  # below minimum
        executor.update_prices = MagicMock()
        orch = _orchestrator(tmp_path, _daily(), executor)

        await orch.run_trading_cycle()

        executor.tradable_amount.assert_awaited()
        executor.place_order.assert_not_awaited()


class TestRebalanceJournal:
    async def test_open_scale_in_partial_and_full_exit(
        self,
        tmp_path,
        journal_db,  # noqa: F811
    ):
        journal, factory = journal_db
        price = 100.0
        executor = _paper()
        orch = _orchestrator(tmp_path, _daily(), executor)
        orch._journal = journal

        buy = await executor.place_order(PAIR, "buy", "market", 1.0, price)
        await journal.record_rebalance(buy, "donchian_ensemble", 1.0)
        add = await executor.place_order(PAIR, "buy", "market", 1.0, 2 * price)
        await journal.record_rebalance(add, "donchian_ensemble", 2.0)
        part = await executor.place_order(PAIR, "sell", "market", 0.5, 3 * price)
        await journal.record_rebalance(part, "donchian_ensemble", 1.5)
        rest = await executor.place_order(PAIR, "sell", "market", 1.5, 3 * price)
        await journal.record_rebalance(rest, "donchian_ensemble", 0.0)

        trades, orders, _ = await _db_rows(factory)
        assert [o.side for o in orders] == ["buy", "buy", "sell", "sell"]
        by_status = sorted(
            (t.exit_reason, float(t.quantity), float(t.entry_price), float(t.pnl))
            for t in trades
        )
        assert by_status == [
            ("rebalance", 0.5, pytest.approx(1.5 * price), pytest.approx(0.75 * price)),
            (
                "rebalance_exit",
                1.5,
                pytest.approx(1.5 * price),
                pytest.approx(2.25 * price),
            ),
        ]
        assert all(t.status == "closed" for t in trades)


class TestDecisionLog:
    async def test_cycle_logs_decision_matching_the_backtest(self, tmp_path):
        from trading_bot.backtest.parity import check_parity
        from trading_bot.core.decision_log import read_decisions

        ohlcv = _daily()
        orch = _orchestrator(tmp_path, ohlcv)
        await orch.run_trading_cycle()
        await orch.run_trading_cycle()  # same candle: no new decision

        records = read_decisions(tmp_path / "decisions_paper.jsonl")
        assert len(records) == 1
        record = records[0]
        assert record["candle"] == ohlcv.index[-2].isoformat()
        assert record["target"] == pytest.approx(_expected_w(ohlcv))
        assert record["fill"]["side"] == "buy"
        assert record["fill"]["quantity"] > 0

        report = check_parity(
            records, {PAIR: ohlcv}, LOOKBACKS, 0.25, 90, last_closed=ohlcv.index[-2]
        )
        assert report.rows[0].target_gap == pytest.approx(0.0, abs=1e-12)
        assert report.missing == {}

    async def test_blocked_increase_is_logged_without_fill(self, tmp_path):
        from trading_bot.core.decision_log import read_decisions

        orch = _orchestrator(tmp_path, _daily())
        orch._drawdown_monitor.update(80_000.0)  # HALTED: no increases
        await orch.run_trading_cycle()

        (record,) = read_decisions(tmp_path / "decisions_paper.jsonl")
        assert record["blocked"] is True and record["fill"] is None

    async def test_log_failure_does_not_stop_trading(self, tmp_path):
        orch = _orchestrator(tmp_path, _daily())
        with patch.object(
            bot_module.DecisionLog, "append", side_effect=OSError("disk full")
        ):
            await orch.run_trading_cycle()
        assert await orch._executor.get_position(PAIR) > 0
        assert orch._rebalanced


class TestMarketDataSource:
    @pytest.mark.parametrize(
        ("mode", "sandbox", "expected_sandbox", "keyed"),
        [
            ("paper", True, False, False),
            ("paper", False, False, False),
            ("live", True, True, True),
            ("live", False, False, True),
        ],
    )
    def test_paper_reads_the_real_market(
        self, tmp_path, mode, sandbox, expected_sandbox, keyed
    ):
        orch = _orchestrator(
            tmp_path,
            _daily(),
            BOT_MODE=mode,
            EXCHANGE_SANDBOX=sandbox,
            EXCHANGE_API_KEY="key",
            EXCHANGE_API_SECRET="secret",
        )
        with patch.object(bot_module, "CCXTFetcher") as fetcher_cls:
            orch._build_fetcher()
        kwargs = fetcher_cls.call_args.kwargs
        assert kwargs["sandbox"] is expected_sandbox
        assert (kwargs["api_key"], kwargs["api_secret"]) == (
            ("key", "secret") if keyed else ("", "")
        )


class TestSchedule:
    def test_cycle_runs_hourly_just_after_the_daily_close(self, tmp_path):
        from apscheduler.triggers.cron import CronTrigger

        trigger = _orchestrator(tmp_path, _daily())._trading_trigger()
        assert isinstance(trigger, CronTrigger)
        after_close = datetime(2026, 10, 1, 0, 0, 30, tzinfo=UTC)
        fire = trigger.get_next_fire_time(None, after_close)
        assert fire == datetime(2026, 10, 1, 0, 2, tzinfo=UTC)
        fire = trigger.get_next_fire_time(fire, fire.replace(minute=3))
        assert fire == datetime(2026, 10, 1, 1, 2, tzinfo=UTC)

    def test_regime_selector_keeps_the_interval(self, tmp_path):
        from apscheduler.triggers.interval import IntervalTrigger

        orch = _orchestrator(tmp_path, _daily(), STRATEGY_MODE="regime_selector")
        trigger = orch._trading_trigger()
        assert isinstance(trigger, IntervalTrigger)
        assert trigger.interval.total_seconds() == 60 * 60


class TestRebalanceMarkOnRestart:
    async def test_live_restart_skips_the_traded_candle(self, tmp_path):
        ohlcv = _daily()
        executor = _paper()
        first = _orchestrator(tmp_path, ohlcv, executor, BOT_MODE="live")
        await first._load_risk_state(100_000.0)
        await first.run_trading_cycle()

        second = _orchestrator(tmp_path, ohlcv, executor, BOT_MODE="live")
        await second._load_risk_state(100_000.0)
        assert second._rebalanced == first._rebalanced != {}
        with patch.object(bot_module, "target_exposure") as target:
            await second.run_trading_cycle()
        target.assert_not_called()

    async def test_paper_restart_restores_account_and_mark(self, tmp_path):
        ohlcv = _daily()
        first = _orchestrator(tmp_path, ohlcv)
        await first._load_risk_state(100_000.0)
        await first.run_trading_cycle()
        held = await first._executor.get_position(PAIR)
        assert held > 0 and first._rebalanced

        # New process, new executor at INITIAL_BALANCE: the saved account wins.
        second = _orchestrator(tmp_path, ohlcv)
        await second._load_risk_state(100_000.0)
        assert await second._executor.get_position(PAIR) == pytest.approx(held)
        assert second._rebalanced == first._rebalanced
        with patch.object(bot_module, "target_exposure") as target:
            await second.run_trading_cycle()
        target.assert_not_called()

    async def test_paper_state_without_account_forgets_the_mark(self, tmp_path):
        ohlcv = _daily()
        first = _orchestrator(tmp_path, ohlcv)
        await first._load_risk_state(100_000.0)
        await first.run_trading_cycle()
        assert first._rebalanced
        path = first._risk_state_path()
        state = json.loads(path.read_text(encoding="utf-8"))
        del state["paper_account"]  # file written before the account was saved
        path.write_text(json.dumps(state), encoding="utf-8")

        # The account starts from cash again: rebuy the same candle.
        second = _orchestrator(tmp_path, ohlcv)
        await second._load_risk_state(100_000.0)
        assert second._rebalanced == {}
        await second.run_trading_cycle()
        assert await second._executor.get_position(PAIR) > 0

    def test_history_grows_with_longer_channels(self, tmp_path):
        orch = _orchestrator(
            tmp_path, _daily(), DONCHIAN_LOOKBACKS=[20, 1000], DONCHIAN_VOL_WINDOW=90
        )
        assert orch._lookback_bars == 1000 + 90 + 1000
        default = _orchestrator(tmp_path, _daily())
        assert default._lookback_bars == HISTORY_BARS


async def test_journal_keeps_memory_in_sync_when_db_write_fails(journal_db):  # noqa: F811
    journal, _ = journal_db
    buy = await _paper().place_order(PAIR, "buy", "market", 1.0, 100.0)
    await journal.record_rebalance(buy, "donchian_ensemble", 1.0)
    before = journal._open[PAIR]
    snapshot = (before.entry_price, before.quantity)

    def broken():
        raise RuntimeError("db down")

    journal._factory = broken
    add = await _paper().place_order(PAIR, "buy", "market", 1.0, 200.0)
    await journal.record_rebalance(add, "donchian_ensemble", 2.0)

    after = journal._open[PAIR]
    assert (after.entry_price, after.quantity) == snapshot


class TestDailyLossPolicy:
    """DONCHIAN_DAILY_LOSS_ACTION (audit M-4): the daily limit must not force
    an intraday exit the strategy was never validated with, unless asked."""

    async def _paused_cycle(self, tmp_path, action: str):
        ohlcv = _daily()
        price = float(ohlcv["close"].iloc[-1])
        executor = _paper()
        await executor.place_order(PAIR, "buy", "market", 20_000 / price, price)
        orch = _orchestrator(
            tmp_path, ohlcv, executor, DONCHIAN_DAILY_LOSS_ACTION=action
        )
        orch._notifier = AsyncMock()
        # Day started 3.4 % higher; peak equal, so only the daily limit fires.
        orch._drawdown_monitor = DrawdownMonitor(initial_value=103_500.0)
        held = await executor.get_position(PAIR)
        await orch.run_trading_cycle()
        return orch, executor, held

    async def test_block_increases_sells_nothing(self, tmp_path):
        orch, executor, held = await self._paused_cycle(tmp_path, "block_increases")
        assert orch._drawdown_monitor.state == TradingState.DAILY_PAUSED
        assert await executor.get_position(PAIR) == pytest.approx(held)
        assert "Not liquidating" in orch._notifier.send_alert.call_args.args[0]

    async def test_liquidate_sells_everything(self, tmp_path):
        orch, executor, _ = await self._paused_cycle(tmp_path, "liquidate")
        assert orch._drawdown_monitor.state == TradingState.DAILY_PAUSED
        assert await executor.get_position(PAIR) == 0.0

    async def test_halve_sells_half(self, tmp_path):
        orch, executor, held = await self._paused_cycle(tmp_path, "halve")
        assert orch._drawdown_monitor.state == TradingState.DAILY_PAUSED
        assert await executor.get_position(PAIR) == pytest.approx(held / 2)

    async def test_paused_day_never_raises_exposure(self, tmp_path):
        orch, executor, held = await self._paused_cycle(tmp_path, "block_increases")
        orch._rebalanced = {}  # force a new rebalance the same day
        await orch.run_trading_cycle()
        assert await executor.get_position(PAIR) <= held + 1e-12

    async def test_off_disables_the_daily_limit(self, tmp_path):
        orch = _orchestrator(tmp_path, _daily(), DONCHIAN_DAILY_LOSS_ACTION="off")
        await orch._load_risk_state(100_000.0)
        assert orch._drawdown_monitor.update(96_000.0) is None  # -4 % in a day
        assert orch._drawdown_monitor.state == TradingState.ACTIVE
        # The max drawdown HALT still applies.
        assert orch._drawdown_monitor.update(80_000.0) == TradingState.HALTED

    def test_regime_selector_always_liquidates(self, tmp_path):
        orch = _orchestrator(
            tmp_path,
            _daily(),
            STRATEGY_MODE="regime_selector",
            DONCHIAN_DAILY_LOSS_ACTION="block_increases",
        )
        assert orch._daily_loss_action == "liquidate"
