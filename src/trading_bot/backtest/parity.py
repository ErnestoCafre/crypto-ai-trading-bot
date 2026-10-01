"""Paper/live vs backtest parity for STRATEGY_MODE=donchian_ensemble.

Compares the bot's decision log (``core/decision_log.py``) with the
:class:`TargetExposureEngine` replayed on the full daily history:

- **Target**: the strategy exposure the bot computed for candle t (before AI)
  must equal the engine's target for t. A gap means the bot saw different
  candles, too short a history (see ``unsynced``) or different settings.
- **Execution**: the engine fills at the open of t+1 with
  ``BACKTEST_SLIPPAGE_PCT`` and ``BACKTEST_FEE_PCT``; each real fill is
  measured against that open (adverse slippage) and its notional (fee).
- **Continuity**: every closed candle since the first decision must have one.
- **Lag**: minutes from the close of t to the decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from trading_bot.backtest.target_engine import TargetExposureEngine

DAY = pd.Timedelta(days=1)
# Float residue allowed over a cost budget (a 0.1 % fee is not "> 0.1 %").
_COST_EPSILON = 1e-9


@dataclass(frozen=True)
class ParityRow:
    pair: str
    candle: pd.Timestamp
    bot_target: float
    ref_target: float | None  # None: candle missing from the reference data
    held_w: float
    wanted: float
    lag_min: float
    fill_side: str | None = None
    fill_price: float | None = None
    next_open: float | None = None
    slippage_pct: float | None = None  # adverse vs next open; >0 = worse
    fee_pct: float | None = None

    @property
    def target_gap(self) -> float | None:
        if self.ref_target is None:
            return None
        return abs(self.bot_target - self.ref_target)


@dataclass(frozen=True)
class ParityReport:
    rows: list[ParityRow]
    missing: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    @property
    def fills(self) -> list[ParityRow]:
        return [r for r in self.rows if r.fill_side is not None]

    def mean_slippage_pct(self) -> float | None:
        values = [r.slippage_pct for r in self.fills if r.slippage_pct is not None]
        return sum(values) / len(values) if values else None

    def mean_fee_pct(self) -> float | None:
        values = [r.fee_pct for r in self.fills if r.fee_pct is not None]
        return sum(values) / len(values) if values else None

    def failures(
        self,
        tolerance: float,
        max_slippage_pct: float,
        max_fee_pct: float,
        max_lag_min: float,
    ) -> list[str]:
        """Human-readable reasons the run does not match the backtest."""
        out: list[str] = []
        for r in self.rows:
            day = r.candle.date().isoformat()
            if r.ref_target is None:
                out.append(f"{r.pair} {day}: candle not in the reference data")
            elif r.target_gap is not None and r.target_gap > tolerance:
                out.append(
                    f"{r.pair} {day}: target {r.bot_target:.6f} != backtest "
                    f"{r.ref_target:.6f}"
                )
            if r.lag_min > max_lag_min:
                out.append(f"{r.pair} {day}: decided {r.lag_min:.0f} min after close")
        for pair, days in self.missing.items():
            listed = ", ".join(d.date().isoformat() for d in days)
            out.append(f"{pair}: no decision for {listed}")
        slip = self.mean_slippage_pct()
        if slip is not None and slip > max_slippage_pct + _COST_EPSILON:
            out.append(
                f"mean adverse slippage {slip:.3f}% > backtest {max_slippage_pct:.3f}%"
            )
        fee = self.mean_fee_pct()
        if fee is not None and fee > max_fee_pct + _COST_EPSILON:
            out.append(f"mean fee {fee:.3f}% > backtest {max_fee_pct:.3f}%")
        return out


def latest_decisions(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, pd.Timestamp], Mapping[str, Any]]:
    """Last record per (pair, candle): retries of a blocked or failed
    rebalance supersede earlier ones."""
    latest: dict[tuple[str, pd.Timestamp], Mapping[str, Any]] = {}
    for record in records:
        key = (str(record["pair"]), _utc(record["candle"]))
        latest[key] = record
    return latest


def check_parity(
    records: Iterable[Mapping[str, Any]],
    data: Mapping[str, pd.DataFrame],
    lookbacks: Sequence[int],
    vol_target: float,
    vol_window: int,
    last_closed: pd.Timestamp,
) -> ParityReport:
    """Compare decisions with the engine on ``data`` ({pair: daily OHLCV}).

    ``last_closed`` is the open time of the last closed daily candle: the
    continuity check expects one decision per pair per day up to it.
    """
    engine = TargetExposureEngine(
        lookbacks=lookbacks, vol_target=vol_target, vol_window=vol_window
    )
    targets = {pair: engine.targets(df) for pair, df in data.items()}
    decisions = latest_decisions(records)

    rows: list[ParityRow] = []
    seen: dict[str, set[pd.Timestamp]] = {}
    for (pair, candle), record in sorted(decisions.items()):
        seen.setdefault(pair, set()).add(candle)
        ref = targets.get(pair)
        ref_target = (
            float(ref.loc[candle]) if ref is not None and candle in ref.index else None
        )
        decided = _utc(record["decided_at"])
        row = ParityRow(
            pair=pair,
            candle=candle,
            bot_target=float(record["target"]),
            ref_target=ref_target,
            held_w=float(record.get("held_w") or 0.0),
            wanted=float(record.get("wanted") or 0.0),
            lag_min=(decided - (candle + DAY)).total_seconds() / 60,
        )
        rows.append(_with_fill(row, record.get("fill"), data.get(pair)))

    missing: dict[str, list[pd.Timestamp]] = {}
    for pair, days in seen.items():
        expected = pd.date_range(min(days), last_closed, freq="1D")
        gaps = [d for d in expected if d not in days]
        if gaps:
            missing[pair] = gaps
    return ParityReport(rows=rows, missing=missing)


def _with_fill(
    row: ParityRow, fill: Mapping[str, Any] | None, ohlcv: pd.DataFrame | None
) -> ParityRow:
    if not fill or float(fill.get("quantity") or 0.0) <= 0:
        return row
    side = str(fill["side"])
    price = float(fill["price"])
    fee = float(fill.get("fee") or 0.0)
    notional = price * float(fill["quantity"])
    next_day = row.candle + DAY
    next_open = (
        float(ohlcv.loc[next_day, "open"])
        if ohlcv is not None and next_day in ohlcv.index
        else None
    )
    slippage = None
    if next_open:
        move = price / next_open - 1
        slippage = (move if side == "buy" else -move) * 100
    return replace(
        row,
        fill_side=side,
        fill_price=price,
        next_open=next_open,
        slippage_pct=slippage,
        fee_pct=fee / notional * 100 if notional > 0 else None,
    )


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
