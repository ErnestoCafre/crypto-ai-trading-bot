"""Paper/live vs backtest parity (backtest/parity.py) and the decision log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.parity import check_parity, latest_decisions
from trading_bot.backtest.target_engine import TargetExposureEngine
from trading_bot.core.decision_log import DecisionLog, read_decisions

PAIR = "BTC/USDT"
LOOKBACKS = [5, 10, 20, 60]
VOL_WINDOW = 30
LIMITS = {
    "tolerance": 1e-6,
    "max_slippage_pct": 0.05,
    "max_fee_pct": 0.1,
    "max_lag_min": 60.0,
}


def _daily(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20_000 * np.exp(np.cumsum(rng.normal(0.003, 0.03, n)))
    index = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": open_, "close": close}, index=index)


def _targets(ohlcv: pd.DataFrame) -> pd.Series:
    engine = TargetExposureEngine(
        lookbacks=LOOKBACKS, vol_target=0.25, vol_window=VOL_WINDOW
    )
    return engine.targets(ohlcv)


def _records(ohlcv: pd.DataFrame, days: int = 5) -> list[dict]:
    """Decisions as the bot logs them for the last ``days`` closed candles."""
    targets = _targets(ohlcv)
    out = []
    for candle in ohlcv.index[-days - 1 : -1]:
        decided = candle + pd.Timedelta(24 * 60 + 2, unit="min")
        out.append(
            {
                "pair": PAIR,
                "candle": candle.isoformat(),
                "decided_at": decided.isoformat(),
                "target": float(targets.loc[candle]),
                "wanted": float(targets.loc[candle]),
                "held_w": 0.0,
                "fill": None,
            }
        )
    return out


def _check(records, ohlcv):
    return check_parity(
        records,
        {PAIR: ohlcv},
        LOOKBACKS,
        0.25,
        VOL_WINDOW,
        last_closed=ohlcv.index[-2],
    )


def test_matching_run_has_no_failures():
    ohlcv = _daily()
    report = _check(_records(ohlcv), ohlcv)
    assert len(report.rows) == 5
    assert report.failures(**LIMITS) == []


def test_target_mismatch_is_reported():
    ohlcv = _daily()
    records = _records(ohlcv)
    records[2]["target"] += 0.01
    failures = _check(records, ohlcv).failures(**LIMITS)
    assert len(failures) == 1 and "!= backtest" in failures[0]


def test_missing_day_and_late_decision_are_reported():
    ohlcv = _daily()
    records = _records(ohlcv)
    del records[1]
    records[-1]["decided_at"] = (
        pd.Timestamp(records[-1]["candle"]) + pd.Timedelta(27, unit="h")
    ).isoformat()
    failures = _check(records, ohlcv).failures(**LIMITS)
    assert any("no decision for" in f for f in failures)
    assert any("min after close" in f for f in failures)


def test_fill_measured_against_next_open():
    ohlcv = _daily()
    records = _records(ohlcv)
    candle = pd.Timestamp(records[0]["candle"])
    next_open = float(ohlcv.loc[candle + pd.Timedelta(1, unit="D"), "open"])
    price = next_open * 1.002  # 0.2 % worse than the open
    records[0]["fill"] = {
        "side": "buy",
        "status": "filled",
        "quantity": 1.0,
        "price": price,
        "fee": price * 0.001,
    }
    report = _check(records, ohlcv)
    row = report.fills[0]
    assert row.slippage_pct == pytest.approx(0.2)
    assert row.fee_pct == pytest.approx(0.1)
    failures = report.failures(**LIMITS)
    assert failures == ["mean adverse slippage 0.200% > backtest 0.050%"]


def test_sell_below_open_is_adverse():
    ohlcv = _daily()
    records = _records(ohlcv)
    candle = pd.Timestamp(records[0]["candle"])
    next_open = float(ohlcv.loc[candle + pd.Timedelta(1, unit="D"), "open"])
    records[0]["fill"] = {"side": "sell", "quantity": 1.0, "price": next_open * 0.999}
    assert _check(records, ohlcv).fills[0].slippage_pct == pytest.approx(0.1)


def test_lag_counts_from_the_first_decision_of_a_candle():
    ohlcv = _daily()
    records = _records(ohlcv)
    retry = {
        **records[-1],
        "decided_at": (
            pd.Timestamp(records[-1]["candle"]) + pd.Timedelta(29, unit="h")
        ).isoformat(),
    }
    report = _check([*records, retry], ohlcv)
    assert report.rows[-1].lag_min == pytest.approx(2)
    assert report.failures(**LIMITS) == []


def test_latest_record_per_candle_wins():
    first = {"pair": PAIR, "candle": "2026-03-01T00:00:00+00:00", "target": 0.1}
    retry = {**first, "target": 0.2}
    latest = latest_decisions([first, retry])
    assert list(latest.values()) == [retry]


def test_decision_log_round_trip_skips_malformed_lines(tmp_path):
    path = tmp_path / "state" / "decisions.jsonl"
    log = DecisionLog(path)
    log.append({"pair": PAIR, "target": 0.5})
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{truncated\n[1, 2]\n\n")
    log.append({"pair": PAIR, "target": 0.6})
    assert [r["target"] for r in read_decisions(path)] == [0.5, 0.6]
    assert read_decisions(tmp_path / "missing.jsonl") == []


def test_held_not_ready_record_is_reported_not_crashing():
    # The bot logs no target when it holds a pair with too little history
    # (audit M-1 fix); parity must report it instead of raising KeyError.
    ohlcv = _daily()
    records = _records(ohlcv)
    held = dict(records[-1])
    held.pop("target")
    held["held_not_ready"] = True
    records[-1] = held
    report = _check(records, ohlcv)
    assert report.rows[-1].bot_target is None
    assert report.rows[-1].target_gap is None
    assert any("bot computed no target" in f for f in report.failures(**LIMITS))
