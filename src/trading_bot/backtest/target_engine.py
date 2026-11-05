"""Target-exposure backtest — daily rebalancing towards a strategy's weight.

Simulates the ``STRATEGY_MODE=donchian_ensemble`` bot (``core/bot.py``): for
every pair the target value is ``equity x allocation x w``, where ``w`` is
the Donchian ensemble exposure decided at the close of day t. The difference
to the held value is traded only when it exceeds
``rebalance_threshold x allocation x equity`` (a target of 0 always closes the
position). All pairs share one cash balance, as on the exchange: sells run
before buys and buys are capped by the cash available.

Execution (``execution``):

- ``"next_open"`` (default, as the live bot and :class:`BacktestEngine`):
  the target of day t is traded at the open of day t+1, valued at that open;
- ``"close"``: traded at the close of day t. With a zero threshold, zero
  costs and ``allocation=1`` this reproduces the research script exactly.

Costs: slippage moves the fill price against the order and the fee is a
percentage of the filled notional, per side, on the traded amount only.

Kill-switch (optional, ``max_drawdown_pct`` / ``max_daily_loss_pct``): the
live bot values the portfolio every hour and reacts when it falls
``max_daily_loss_pct`` below the day's first valuation (daily limit, action
``daily_loss_action``) or ``max_drawdown_pct`` below its peak (HALT: liquidate
and never raise exposure again). With daily bars the intraday path is
approximated: every held pair moves linearly from its open to its low, and
a limit fires at the first point of that path where the portfolio crosses
it; sells fill there. The day starts at the open valuation and the peak is
tracked on open and close valuations. This is pessimistic (all lows at the
same moment, continuous rather than hourly checks) on purpose.

Not simulated: AI sources and exchange minimums.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from trading_bot.backtest.data_types import BacktestTrade
from trading_bot.backtest.metrics import MetricsCalculator
from trading_bot.core.strategies.donchian_ensemble import (
    DEFAULT_LOOKBACKS,
    DEFAULT_VOL_TARGET,
    DEFAULT_VOL_WINDOW,
    exposure_series,
    rebalance_delta,
    required_history,
)

STRATEGY_NAME = "donchian_ensemble"
Execution = Literal["next_open", "close"]
DailyLossAction = Literal["liquidate", "block_increases", "halve", "off"]
DAILY_LOSS_ACTIONS: tuple[str, ...] = ("liquidate", "block_increases", "halve", "off")
# Share of each position a daily-loss action sells.
_DAILY_SELL_SHARE = {"liquidate": 1.0, "halve": 0.5}

# Share of a lot below which a sell is treated as closing it (float residue).
_EPSILON = 1e-12


@dataclass
class TargetExposureResult:
    equity_curve: pd.Series  # portfolio value at each daily close
    # Held exposure per pair at each close: position value / (equity x alloc).
    exposure: pd.DataFrame
    # Target decided at each close, per pair.
    target: pd.DataFrame
    trades: list[BacktestTrade]  # one per sell fill (realised P&L)
    traded_notional: pd.Series  # per day, all pairs
    fees: float
    parameters: dict[str, object] = field(default_factory=dict)
    # Kill-switch events: (day, "daily_<action>" | "halted").
    risk_events: list[tuple[pd.Timestamp, str]] = field(default_factory=list)

    def metrics(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        risk_free_rate: float = 0.0,
    ) -> dict[str, float]:
        """Metrics over ``[start, end]`` (daily closes, inclusive).

        The equity curve is anchored at the close before ``start`` so the
        first day's return counts, as with daily returns. ``risk_free_rate``
        defaults to 0 (the research study's Sharpe); pass 0.04 to match
        :meth:`MetricsCalculator.calculate`'s default.
        """
        eq = self.equity_curve
        lo = 0 if start is None else max(0, int(eq.index.searchsorted(_ts(start))) - 1)
        hi = len(eq) if end is None else int(eq.index.searchsorted(_ts(end), "right"))
        part = eq.iloc[lo:hi]
        out = MetricsCalculator.equity_metrics(part, risk_free_rate=risk_free_rate)
        if len(part) < 2:
            return out
        days = part.index[1:]
        alloc = float(self.parameters.get("allocation", 1.0))  # type: ignore[arg-type]
        out["avg_exposure_pct"] = float(
            self.exposure.loc[days].sum(axis=1).mean() * alloc * 100
        )
        years = len(days) / 365
        notional = float(self.traded_notional.loc[days].sum())
        out["turnover_per_year"] = (
            notional / float(part.mean()) / years if years > 0 else 0.0
        )
        sells = [t for t in self.trades if days[0] <= t.exit_time <= days[-1]]
        out["sell_fills"] = float(len(sells))
        out["fills"] = float((self.traded_notional.loc[days] > 0).sum())
        events = [kind for ts, kind in self.risk_events if days[0] <= ts <= days[-1]]
        out["daily_loss_events"] = float(sum(k.startswith("daily") for k in events))
        out["halted"] = float("halted" in events)
        return out


class TargetExposureEngine:
    """Portfolio backtest of the Donchian ensemble with threshold rebalancing."""

    def __init__(
        self,
        lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
        vol_target: float = DEFAULT_VOL_TARGET,
        vol_window: int = DEFAULT_VOL_WINDOW,
        allocation: float = 1.0,
        rebalance_threshold: float = 0.05,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
        execution: Execution = "next_open",
        min_history: int | None = None,
        max_drawdown_pct: float | None = None,
        max_daily_loss_pct: float | None = None,
        daily_loss_action: DailyLossAction = "liquidate",
    ) -> None:
        if not 0 < allocation <= 1:
            raise ValueError("allocation must be in (0, 1]")
        if rebalance_threshold < 0:
            raise ValueError("rebalance_threshold must be >= 0")
        if execution not in ("next_open", "close"):
            raise ValueError(f"unknown execution {execution!r}")
        if daily_loss_action not in DAILY_LOSS_ACTIONS:
            raise ValueError(f"unknown daily_loss_action {daily_loss_action!r}")
        for name, pct in (
            ("max_drawdown_pct", max_drawdown_pct),
            ("max_daily_loss_pct", max_daily_loss_pct),
        ):
            if pct is not None and not 0 < pct <= 100:
                raise ValueError(f"{name} must be in (0, 100]")
        self._max_dd = None if max_drawdown_pct is None else max_drawdown_pct / 100
        self._max_daily = (
            None
            if max_daily_loss_pct is None or daily_loss_action == "off"
            else max_daily_loss_pct / 100
        )
        self._daily_action: DailyLossAction = daily_loss_action
        self._lookbacks = tuple(lookbacks)
        self._vol_target = vol_target
        self._vol_window = vol_window
        self._allocation = allocation
        self._threshold = rebalance_threshold
        self._capital = initial_capital
        self._fee = fee_pct / 100
        self._slip = slippage_pct / 100
        self._execution: Execution = execution
        # Live rule by default: no exposure before the longest channel and the
        # volatility window are fully warmed up. 0 = research script warm-up.
        self._min_history = (
            required_history(self._lookbacks, vol_window)
            if min_history is None
            else min_history
        )

    @property
    def strategy_name(self) -> str:
        return STRATEGY_NAME

    def targets(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Target exposure decided at each daily close."""
        return exposure_series(
            ohlcv["close"].astype(float),
            self._lookbacks,
            self._vol_target,
            self._vol_window,
            self._min_history,
        )["exposure"]

    def run(self, data: Mapping[str, pd.DataFrame]) -> TargetExposureResult:
        """Backtest ``{pair: daily OHLCV}`` as one portfolio."""
        if not data:
            raise ValueError("no data")
        pairs = list(data)
        index = pd.DatetimeIndex(
            sorted(set().union(*(df.index for df in data.values())))
        )

        def aligned(col: str) -> pd.DataFrame:
            return pd.DataFrame(
                {p: data[p][col].astype(float) for p in pairs}, index=index
            )

        opens, closes, lows = aligned("open"), aligned("close"), aligned("low")
        decided = pd.DataFrame({p: self.targets(data[p]) for p in pairs}, index=index)
        # A day without a bar for a pair decides nothing for it: the live bot
        # has no new closed candle and leaves the pair alone. Filling it with
        # 0 would liquidate the pair at the next open.
        known = decided.notna()
        targets = decided.fillna(0.0)
        # Last known close values held positions on days a pair has no bar.
        marks = closes.ffill()

        book = _Book(pairs, self._capital, self._fee, self._slip)
        risk = _RiskState(peak=self._capital) if self._risk_enabled else None
        equity = np.empty(len(index))
        held_w = np.zeros((len(index), len(pairs)))
        notional = np.zeros(len(index))

        for i, ts in enumerate(index):
            # The day opens at the open (or the last mark without a bar).
            prev_marks = marks.iloc[i - 1] if i > 0 else closes.iloc[i]
            day_open = opens.iloc[i].where(opens.iloc[i].notna(), prev_marks)
            day_low = lows.iloc[i].where(lows.iloc[i].notna(), day_open)
            if risk is not None:
                risk.start_day(book.equity(day_open))
            if self._execution == "next_open" and i > 0:
                tradable = opens.iloc[i].notna() & known.iloc[i - 1]
                notional[i] += self._rebalance(
                    book,
                    targets.iloc[i - 1],
                    day_open,
                    tradable,
                    ts,
                    allow_increase=risk is None or not risk.halted,
                )
            if risk is not None:
                notional[i] += self._intraday_risk(book, risk, day_open, day_low, ts)
            if self._execution == "close":
                notional[i] += self._rebalance(
                    book,
                    targets.iloc[i],
                    marks.iloc[i],
                    closes.iloc[i].notna(),
                    ts,
                    allow_increase=risk is None or not (risk.halted or risk.paused),
                )
            value = book.equity(marks.iloc[i])
            if risk is not None:
                risk.peak = max(risk.peak, value)
            equity[i] = value
            for j, p in enumerate(pairs):
                px = marks.iloc[i][p]
                if value > 0 and book.qty[p] > 0 and pd.notna(px):
                    held_w[i, j] = book.qty[p] * px / (value * self._allocation)

        return TargetExposureResult(
            equity_curve=pd.Series(equity, index=index, name="equity"),
            exposure=pd.DataFrame(held_w, index=index, columns=pairs),
            target=targets,
            trades=book.trades,
            traded_notional=pd.Series(notional, index=index),
            fees=book.fees,
            risk_events=list(risk.events) if risk is not None else [],
            parameters={
                "strategy": STRATEGY_NAME,
                "pairs": pairs,
                "lookbacks": list(self._lookbacks),
                "vol_target": self._vol_target,
                "vol_window": self._vol_window,
                "allocation": self._allocation,
                "rebalance_threshold": self._threshold,
                "fee_pct": self._fee * 100,
                "slippage_pct": self._slip * 100,
                "execution": self._execution,
                "min_history": self._min_history,
                "max_drawdown_pct": None
                if self._max_dd is None
                else self._max_dd * 100,
                "max_daily_loss_pct": (
                    None if self._max_daily is None else self._max_daily * 100
                ),
                "daily_loss_action": self._daily_action,
            },
        )

    @property
    def _risk_enabled(self) -> bool:
        return self._max_dd is not None or self._max_daily is not None

    def _intraday_risk(
        self,
        book: _Book,
        risk: _RiskState,
        day_open: pd.Series,
        day_low: pd.Series,
        ts: pd.Timestamp,
    ) -> float:
        """Fire the limits crossed on the open→low path; returns notional sold."""
        limits: list[tuple[str, float]] = []
        if self._max_daily is not None and not risk.paused:
            limits.append(("daily", risk.day_start * (1 - self._max_daily)))
        if self._max_dd is not None and not risk.halted:
            limits.append(("drawdown", risk.peak * (1 - self._max_dd)))
        traded = 0.0
        when = ts.to_pydatetime()
        reached = 0.0  # position on the path already walked, in [0, 1]
        while limits and not risk.halted:
            hits = [
                (f, kind)
                for kind, level in limits
                if (f := _crossing(book, day_open, day_low, level, reached)) is not None
            ]
            if not hits:
                break
            # Earliest first; on a tie the HALT wins (it liquidates anyway).
            reached, kind = min(hits, key=lambda h: (h[0], h[1] != "drawdown"))
            limits = [(k, lv) for k, lv in limits if k != kind]
            prices = day_open + reached * (day_low - day_open)
            if kind == "drawdown":
                traded += book.sell_share(1.0, prices, when, "kill_switch")
                risk.halted = True
                risk.events.append((ts, "halted"))
            else:
                risk.paused = True
                risk.events.append((ts, f"daily_{self._daily_action}"))
                share = _DAILY_SELL_SHARE.get(self._daily_action, 0.0)
                if share:
                    traded += book.sell_share(share, prices, when, "daily_loss_limit")
        return traded

    def _rebalance(
        self,
        book: _Book,
        target: pd.Series,
        prices: pd.Series,
        tradable: pd.Series,
        ts: pd.Timestamp,
        allow_increase: bool = True,
    ) -> float:
        """Trade every pair towards its target; returns the notional traded.

        With ``allow_increase=False`` (kill-switch engaged) only sells run.
        """
        equity = book.equity(prices)
        if equity <= 0:
            return 0.0
        orders: dict[str, float] = {}  # pair -> signed value to trade
        for pair in book.pairs:
            px = prices[pair]
            if not tradable[pair] or pd.isna(px) or px <= 0:
                continue
            delta = rebalance_delta(
                target_exposure=float(target[pair]),
                held_value=book.qty[pair] * px,
                equity=equity,
                allocation=self._allocation,
                threshold=self._threshold,
            )
            if delta < 0 or (delta > 0 and allow_increase):
                orders[pair] = delta
        traded = 0.0
        when = ts.to_pydatetime()
        # Sells first: they fund the buys.
        for pair, delta in sorted(orders.items(), key=lambda kv: kv[1]):
            px = float(prices[pair])
            if delta < 0:
                qty = book.qty[pair] if -delta >= book.qty[pair] * px else -delta / px
                traded += book.sell(pair, qty, px, when)
            else:
                traded += book.buy(pair, delta / px, px, when)
        return traded


@dataclass
class _RiskState:
    """Kill-switch state across the simulated days."""

    peak: float
    day_start: float = 0.0
    paused: bool = False  # daily limit hit today
    halted: bool = False  # max drawdown hit: latched
    events: list[tuple[pd.Timestamp, str]] = field(default_factory=list)

    def start_day(self, value: float) -> None:
        self.day_start = value
        self.paused = False
        self.peak = max(self.peak, value)


def _crossing(
    book: _Book,
    day_open: pd.Series,
    day_low: pd.Series,
    level: float,
    start: float,
) -> float | None:
    """First f in [start, 1] where the portfolio on the open→low path is at
    or below ``level``; None if it never gets there.

    Value(f) = cash + sum(qty x (open + f x (low - open))): linear in f.
    """
    at_open = book.cash
    drop = 0.0
    for pair, qty in book.qty.items():
        if qty <= 0 or pd.isna(day_open[pair]):
            continue
        at_open += qty * float(day_open[pair])
        drop += qty * float(day_open[pair] - day_low[pair])
    if at_open - start * drop <= level:
        return start
    if drop <= 0:
        return None
    f = (at_open - level) / drop
    return f if f <= 1.0 else None


@dataclass
class _Lot:
    qty: float = 0.0
    avg_price: float = 0.0  # fill price incl. slippage
    fees: float = 0.0  # entry fees not yet attributed to a sell
    opened: datetime | None = None


class _Book:
    """Spot cash + per-pair positions with average cost."""

    def __init__(
        self, pairs: list[str], cash: float, fee_rate: float, slip_rate: float
    ) -> None:
        self.pairs = pairs
        self.cash = cash
        self._fee = fee_rate
        self._slip = slip_rate
        self._lots = {p: _Lot() for p in pairs}
        self.trades: list[BacktestTrade] = []
        self.fees = 0.0

    @property
    def qty(self) -> dict[str, float]:
        return {p: lot.qty for p, lot in self._lots.items()}

    def equity(self, prices: pd.Series) -> float:
        value = self.cash
        for pair, lot in self._lots.items():
            if lot.qty > 0 and pd.notna(prices[pair]):
                value += lot.qty * float(prices[pair])
        return value

    def buy(self, pair: str, qty: float, price: float, when: datetime) -> float:
        fill = price * (1 + self._slip)
        affordable = max(self.cash, 0.0) / (fill * (1 + self._fee))
        qty = min(qty, affordable)
        if qty <= 0:
            return 0.0
        cost = fill * qty
        fee = cost * self._fee
        lot = self._lots[pair]
        if lot.qty <= 0:
            lot.opened = when
        lot.avg_price = (lot.avg_price * lot.qty + cost) / (lot.qty + qty)
        lot.qty += qty
        lot.fees += fee
        self.cash -= cost + fee
        self.fees += fee
        return cost

    def sell_share(
        self, share: float, prices: pd.Series, when: datetime, reason: str
    ) -> float:
        """Sell ``share`` of every position at ``prices``; returns proceeds."""
        traded = 0.0
        for pair, lot in self._lots.items():
            px = prices[pair]
            if lot.qty <= 0 or pd.isna(px) or px <= 0:
                continue
            qty = lot.qty if share >= 1.0 else lot.qty * share
            traded += self.sell(pair, qty, float(px), when, reason)
        return traded

    def sell(
        self,
        pair: str,
        qty: float,
        price: float,
        when: datetime,
        reason: str | None = None,
    ) -> float:
        lot = self._lots[pair]
        qty = min(qty, lot.qty)
        if qty <= 0:
            return 0.0
        fill = price * (1 - self._slip)
        proceeds = fill * qty
        fee = proceeds * self._fee
        share = qty / lot.qty
        entry_fee = lot.fees * share
        pnl = (fill - lot.avg_price) * qty - fee - entry_fee
        closing = share >= 1.0 - _EPSILON
        self.trades.append(
            BacktestTrade(
                entry_time=lot.opened or when,
                exit_time=when,
                pair=pair,
                side="long",
                strategy=STRATEGY_NAME,
                regime=None,
                entry_price=lot.avg_price,
                exit_price=fill,
                quantity=qty,
                pnl=pnl,
                pnl_pct=pnl / (lot.avg_price * qty) * 100 if lot.avg_price else 0.0,
                fees=fee + entry_fee,
                slippage=price * self._slip * qty,
                exit_reason=reason or ("exit" if closing else "rebalance"),
            )
        )
        self.cash += proceeds - fee
        self.fees += fee
        if closing:
            self._lots[pair] = _Lot()
        else:
            lot.qty -= qty
            lot.fees -= entry_fee
        return proceeds


def _ts(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts
