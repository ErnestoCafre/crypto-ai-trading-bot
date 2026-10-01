"""Backtest data types — pure dataclasses for results and trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    pair: str
    side: str  # "long" / "short"
    strategy: str
    regime: str | None
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float
    exit_reason: str  # "signal", "stop_loss", "take_profit", "timeout"
    # Sizing diagnostics: loss at the initial stop as % of equity at entry,
    # and whether MAX_POSITION_PCT (not the risk budget) set the quantity.
    entry_risk_pct: float = 0.0
    size_capped: bool = False


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    equity_curve: pd.Series  # datetime → portfolio value
    metrics: dict[str, float]
    parameters: dict[str, object]
    start_date: datetime
    end_date: datetime
    pair: str
    strategy_name: str


@dataclass
class WalkForwardResult:
    fold_results: list[BacktestResult]
    aggregate_metrics: dict[str, float]
    oos_equity_curve: pd.Series
    parameter_stability: dict[str, list[float]] = field(default_factory=dict)
