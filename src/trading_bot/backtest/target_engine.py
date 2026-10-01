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
Not simulated: AI sources, the drawdown kill-switch and exchange minimums.
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
    ) -> None:
        if not 0 < allocation <= 1:
            raise ValueError("allocation must be in (0, 1]")
        if rebalance_threshold < 0:
            raise ValueError("rebalance_threshold must be >= 0")
        if execution not in ("next_open", "close"):
            raise ValueError(f"unknown execution {execution!r}")
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

        opens, closes = aligned("open"), aligned("close")
        targets = pd.DataFrame(
            {p: self.targets(data[p]) for p in pairs}, index=index
        ).fillna(0.0)
        # Last known close values held positions on days a pair has no bar.
        marks = closes.ffill()

        book = _Book(pairs, self._capital, self._fee, self._slip)
        equity = np.empty(len(index))
        held_w = np.zeros((len(index), len(pairs)))
        notional = np.zeros(len(index))

        for i, ts in enumerate(index):
            if self._execution == "next_open" and i > 0:
                prices = opens.iloc[i].where(opens.iloc[i].notna(), marks.iloc[i - 1])
                tradable = opens.iloc[i].notna()
                notional[i] += self._rebalance(
                    book, targets.iloc[i - 1], prices, tradable, ts
                )
            if self._execution == "close":
                notional[i] += self._rebalance(
                    book, targets.iloc[i], marks.iloc[i], closes.iloc[i].notna(), ts
                )
            value = book.equity(marks.iloc[i])
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
            },
        )

    def _rebalance(
        self,
        book: _Book,
        target: pd.Series,
        prices: pd.Series,
        tradable: pd.Series,
        ts: pd.Timestamp,
    ) -> float:
        """Trade every pair towards its target; returns the notional traded."""
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
            if delta != 0.0:
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

    def sell(self, pair: str, qty: float, price: float, when: datetime) -> float:
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
                exit_reason="exit" if closing else "rebalance",
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
