"""Tests for the latching DrawdownMonitor, StopTracker and RiskStateStore."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from trading_bot.risk.drawdown import DrawdownMonitor, TradingState
from trading_bot.risk.risk_state import RiskStateStore
from trading_bot.risk.stop_tracker import StopTracker


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _bars(rows: list[tuple[float, float]]) -> pd.DataFrame:
    """Hourly closed-candle OHLCV from (open, low) tuples."""
    index = pd.date_range("2026-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [o for o, _ in rows],
            "high": [o + 100 for o, _ in rows],
            "low": [low for _, low in rows],
            "close": [o for o, _ in rows],
            "volume": [1.0] * len(rows),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# DrawdownMonitor latch
# ---------------------------------------------------------------------------


class TestDrawdownLatch:
    def test_max_drawdown_halts_once_and_stays_halted(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        assert dm.update(100_000) is None
        assert dm.update(80_000) == TradingState.HALTED
        # Transition is reported once; recovery does not unlatch it.
        assert dm.update(79_000) is None
        assert dm.update(120_000) is None
        assert dm.state == TradingState.HALTED
        assert not dm.entries_allowed
        assert dm.halt_reason is not None and "drawdown" in dm.halt_reason

    def test_daily_loss_pauses_until_next_utc_day(self) -> None:
        clock = _Clock(datetime(2026, 3, 1, 10, tzinfo=UTC))
        dm = DrawdownMonitor(initial_value=100_000, clock=clock)

        assert dm.update(96_000) == TradingState.DAILY_PAUSED  # -4% > 3%
        assert dm.update(95_500) is None
        assert not dm.entries_allowed

        clock.now = datetime(2026, 3, 1, 23, 59, tzinfo=UTC)
        dm.update(95_500)
        assert dm.state == TradingState.DAILY_PAUSED

        clock.now = datetime(2026, 3, 2, 0, 1, tzinfo=UTC)
        assert dm.update(95_500) is None
        assert dm.state == TradingState.ACTIVE
        assert dm.entries_allowed

    def test_daily_pause_escalates_to_halt(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        assert dm.update(96_000) == TradingState.DAILY_PAUSED
        assert dm.update(84_000) == TradingState.HALTED

    def test_reset_clears_halt_and_rebaselines(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        dm.update(80_000)
        dm.reset(80_000)
        assert dm.state == TradingState.ACTIVE
        assert dm.halt_reason is None
        assert dm.update(80_000) is None
        assert dm.current_drawdown_pct == pytest.approx(0.0)

    def test_check_keeps_bool_contract(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        assert dm.check(99_000) is False
        assert dm.check(80_000) is True
        assert dm.check(100_000) is True  # latched

    def test_round_trip_dict(self) -> None:
        dm = DrawdownMonitor(initial_value=100_000)
        dm.update(110_000)
        dm.update(90_000)  # 18% from peak -> HALTED
        restored = DrawdownMonitor.from_dict(json.loads(json.dumps(dm.to_dict())))
        assert restored.state == TradingState.HALTED
        assert restored.to_dict() == dm.to_dict()

    def test_from_dict_rejects_malformed(self) -> None:
        with pytest.raises((KeyError, ValueError, TypeError)):
            DrawdownMonitor.from_dict({"peak_value": "abc"})
        with pytest.raises(ValueError):
            DrawdownMonitor.from_dict(
                {
                    "peak_value": 1.0,
                    "day_start_value": 1.0,
                    "current_day": "2026-01-01",
                    "last_value": 1.0,
                    "state": "exploded",
                }
            )


# ---------------------------------------------------------------------------
# StopTracker
# ---------------------------------------------------------------------------


class TestStopTracker:
    def _tracker(self, bars: pd.DataFrame, stop: float = 48_000.0) -> StopTracker:
        tracker = StopTracker()
        tracker.register("BTC/USDT", stop, 1.0, None, entry_candle=bars.index[0])
        return tracker

    def test_low_touching_stop_fills_at_stop(self) -> None:
        bars = _bars([(50_000, 49_900), (49_500, 47_900), (49_000, 48_500)])
        assert self._tracker(bars).check_hit("BTC/USDT", bars) == 48_000.0

    def test_gap_below_stop_fills_at_open(self) -> None:
        bars = _bars([(50_000, 49_900), (46_000, 45_000), (46_000, 45_500)])
        assert self._tracker(bars).check_hit("BTC/USDT", bars) == 46_000.0

    def test_untouched_stop_does_nothing(self) -> None:
        bars = _bars([(50_000, 49_900), (49_500, 48_100), (49_000, 48_500)])
        assert self._tracker(bars).check_hit("BTC/USDT", bars) is None

    def test_entry_candle_is_ignored(self) -> None:
        # The entry candle's low may predate the fill.
        bars = _bars([(50_000, 40_000), (49_500, 48_100), (48_200, 48_150)])
        assert self._tracker(bars).check_hit("BTC/USDT", bars) is None

    def test_last_row_is_evaluated_as_closed(self) -> None:
        # Callers pass closed candles only, so the last row counts.
        bars = _bars([(50_000, 49_900), (49_500, 48_100), (48_200, 47_000)])
        assert self._tracker(bars).check_hit("BTC/USDT", bars) == 48_000.0

    def test_candles_are_checked_only_once(self) -> None:
        bars = _bars([(50_000, 49_900), (49_500, 48_100), (49_000, 48_500)])
        tracker = self._tracker(bars)
        assert tracker.check_hit("BTC/USDT", bars) is None
        stop = tracker.get("BTC/USDT")
        assert stop is not None
        assert stop.last_checked == bars.index[2].isoformat()
        # Re-checking the same candles does nothing.
        assert tracker.check_hit("BTC/USDT", bars) is None

        # Next cycle: a new closed candle breaches the stop.
        later = _bars(
            [(50_000, 49_900), (49_500, 48_100), (49_000, 48_500), (47_500, 47_400)]
        )
        assert tracker.check_hit("BTC/USDT", later) == 47_500.0  # gap open

    def test_round_trip_dict(self) -> None:
        bars = _bars([(50_000, 49_900), (49_500, 48_100)])
        tracker = self._tracker(bars)
        tracker.get("BTC/USDT").order_id = "stop-1"  # type: ignore[union-attr]
        restored = StopTracker.from_dict(json.loads(json.dumps(tracker.to_dict())))
        assert restored.to_dict() == tracker.to_dict()


# ---------------------------------------------------------------------------
# RiskStateStore
# ---------------------------------------------------------------------------


class TestRiskStateStore:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert RiskStateStore(tmp_path / "state.json").load() is None

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        store = RiskStateStore(tmp_path / "nested" / "state.json")
        store.save({"a": 1})
        assert store.load() == {"a": 1}
        # No temp files left behind by the atomic write.
        assert [p.name for p in store.path.parent.iterdir()] == ["state.json"]

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")
        assert RiskStateStore(path).load() is None

    def test_non_object_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert RiskStateStore(path).load() is None

    def test_failed_write_keeps_previous_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RiskStateStore(tmp_path / "state.json")
        store.save({"version": 1})

        def boom(*_: object, **__: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("trading_bot.risk.risk_state.os.replace", boom)
        with pytest.raises(OSError):
            store.save({"version": 2})
        assert store.load() == {"version": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
