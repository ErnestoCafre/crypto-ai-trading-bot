"""Backtesting engine — bar-by-bar simulation with anti look-ahead guarantee."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from trading_bot.backtest.data_types import BacktestResult, BacktestTrade
from trading_bot.backtest.metrics import MetricsCalculator
from trading_bot.core.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Event-driven backtest engine.

    Anti look-ahead guarantee: signal at bar *i* is generated using
    ``ohlcv[:i+1]``; execution happens at the **open** of bar ``i+1``.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
        position_sizer: Any = None,
        stop_loss_manager: Any = None,
        regime_detector: Any = None,
    ) -> None:
        self._strategy = strategy
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._slippage_pct = slippage_pct
        self._position_sizer = position_sizer
        self._stop_loss_manager = stop_loss_manager
        self._regime_detector = regime_detector

    def run(
        self,
        ohlcv: pd.DataFrame,
        pair: str,
        ai_signals_series: pd.Series | None = None,
    ) -> BacktestResult:
        """Run a full backtest on *ohlcv* data.

        Returns a :class:`BacktestResult` with trades, equity curve, and metrics.
        """
        if len(ohlcv) < 2:
            return self._empty_result(ohlcv, pair)

        cash = self._initial_capital
        position_qty = 0.0
        position_side: str | None = None
        entry_price = 0.0
        entry_time: datetime | None = None
        stop_price: float | None = None

        trades: list[BacktestTrade] = []
        equity_values: list[float] = []
        equity_index: list[object] = []

        lookback = self._strategy.get_required_lookback()

        for i in range(len(ohlcv)):
            row = ohlcv.iloc[i]
            current_close = float(row["close"])
            current_low = float(row["low"])
            current_high = float(row["high"])

            # -- Record equity --
            portfolio_value = cash
            if position_qty > 0 and position_side == "long":
                portfolio_value += position_qty * current_close
            elif position_qty > 0 and position_side == "short":
                portfolio_value += position_qty * (2 * entry_price - current_close)

            if hasattr(ohlcv.index, "to_pydatetime"):
                try:
                    equity_index.append(ohlcv.index[i])
                except Exception:
                    equity_index.append(i)
            else:
                equity_index.append(i)
            equity_values.append(portfolio_value)

            # -- Check stop-loss on current bar --
            if position_qty > 0 and stop_price is not None:
                triggered = False
                sl_long = position_side == "long" and current_low <= stop_price
                sl_short = position_side == "short" and current_high >= stop_price
                if sl_long or sl_short:
                    triggered = True
                    fill_price = stop_price

                if triggered:
                    # Exit at stop
                    slippage = fill_price * (self._slippage_pct / 100)
                    if position_side == "long":
                        actual_fill = fill_price - slippage
                        pnl = (actual_fill - entry_price) * position_qty
                    else:
                        actual_fill = fill_price + slippage
                        pnl = (entry_price - actual_fill) * position_qty

                    fee = actual_fill * position_qty * (self._fee_pct / 100)
                    pnl -= fee

                    pnl_pct = (
                        pnl / (entry_price * position_qty) * 100
                        if entry_price > 0
                        else 0.0
                    )

                    bar_time = _bar_time(ohlcv, i)
                    trades.append(
                        BacktestTrade(
                            entry_time=entry_time or bar_time,
                            exit_time=bar_time,
                            pair=pair,
                            side=position_side or "long",
                            strategy=self._strategy.name,
                            regime=None,
                            entry_price=entry_price,
                            exit_price=actual_fill,
                            quantity=position_qty,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            fees=fee,
                            slippage=slippage * position_qty,
                            exit_reason="stop_loss",
                        )
                    )
                    cash += entry_price * position_qty + pnl
                    position_qty = 0.0
                    position_side = None
                    stop_price = None

            # -- Generate signal (only use data up to and including bar i) --
            if i < lookback:
                continue

            window = ohlcv.iloc[: i + 1]
            try:
                signal = self._strategy.generate_signal(window)
            except Exception:
                logger.debug("Strategy signal error at bar %d", i)
                continue

            if signal is None:
                continue

            # Execution happens at open of NEXT bar
            if i + 1 >= len(ohlcv):
                continue

            next_open = float(ohlcv.iloc[i + 1]["open"])

            # -- Process signal --
            from trading_bot.core.models import SignalType

            if signal.type == SignalType.CLOSE and position_qty > 0:
                # Close position at next bar open
                slippage = next_open * (self._slippage_pct / 100)
                if position_side == "long":
                    fill_price = next_open - slippage
                    pnl = (fill_price - entry_price) * position_qty
                else:
                    fill_price = next_open + slippage
                    pnl = (entry_price - fill_price) * position_qty

                fee = fill_price * position_qty * (self._fee_pct / 100)
                pnl -= fee

                pnl_pct = (
                    pnl / (entry_price * position_qty) * 100 if entry_price > 0 else 0.0
                )

                bar_time = _bar_time(ohlcv, i + 1)
                trades.append(
                    BacktestTrade(
                        entry_time=entry_time or bar_time,
                        exit_time=bar_time,
                        pair=pair,
                        side=position_side or "long",
                        strategy=self._strategy.name,
                        regime=None,
                        entry_price=entry_price,
                        exit_price=fill_price,
                        quantity=position_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        fees=fee,
                        slippage=slippage * position_qty,
                        exit_reason="signal",
                    )
                )
                cash += entry_price * position_qty + pnl
                position_qty = 0.0
                position_side = None
                stop_price = None

            elif (
                signal.type in (SignalType.LONG, SignalType.SHORT) and position_qty == 0
            ):
                # Open position at next bar open
                side = "long" if signal.type == SignalType.LONG else "short"
                slippage = next_open * (self._slippage_pct / 100)
                if side == "long":
                    fill_price = next_open + slippage
                else:
                    fill_price = next_open - slippage

                # Position sizing
                qty = self._calculate_position_size(cash, fill_price, signal.confidence)
                if qty <= 0:
                    continue

                fee = fill_price * qty * (self._fee_pct / 100)
                cost = fill_price * qty + fee

                if cost > cash:
                    qty = (cash - fee) / fill_price
                    fee = fill_price * qty * (self._fee_pct / 100)
                    cost = fill_price * qty + fee

                if qty <= 0 or cost > cash:
                    continue

                cash -= cost
                position_qty = qty
                position_side = side
                entry_price = fill_price
                entry_time = _bar_time(ohlcv, i + 1)

                # Set stop-loss
                stop_price = self._calculate_stop(fill_price, side, window)

        # -- Close any remaining position at last bar's close --
        if position_qty > 0:
            last_close = float(ohlcv.iloc[-1]["close"])
            slippage = last_close * (self._slippage_pct / 100)
            if position_side == "long":
                fill_price = last_close - slippage
                pnl = (fill_price - entry_price) * position_qty
            else:
                fill_price = last_close + slippage
                pnl = (entry_price - fill_price) * position_qty

            fee = fill_price * position_qty * (self._fee_pct / 100)
            pnl -= fee

            pnl_pct = (
                pnl / (entry_price * position_qty) * 100 if entry_price > 0 else 0.0
            )

            bar_time = _bar_time(ohlcv, len(ohlcv) - 1)
            trades.append(
                BacktestTrade(
                    entry_time=entry_time or bar_time,
                    exit_time=bar_time,
                    pair=pair,
                    side=position_side or "long",
                    strategy=self._strategy.name,
                    regime=None,
                    entry_price=entry_price,
                    exit_price=fill_price,
                    quantity=position_qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    fees=fee,
                    slippage=slippage * position_qty,
                    exit_reason="end_of_data",
                )
            )

        equity_curve = pd.Series(equity_values, index=equity_index, name="equity")
        metrics = MetricsCalculator.calculate(trades, equity_curve)

        start = _bar_time(ohlcv, 0)
        end = _bar_time(ohlcv, len(ohlcv) - 1)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            parameters={
                "initial_capital": self._initial_capital,
                "fee_pct": self._fee_pct,
                "slippage_pct": self._slippage_pct,
            },
            start_date=start,
            end_date=end,
            pair=pair,
            strategy_name=self._strategy.name,
        )

    def _calculate_position_size(
        self, cash: float, price: float, confidence: float
    ) -> float:
        """Calculate position size (external sizer or 10% of cash)."""
        if self._position_sizer is not None:
            try:
                ps = self._position_sizer.calculate(
                    pair="",
                    signal_confidence=confidence,
                    portfolio_value=cash,
                    entry_price=price,
                )
                return float(ps.quantity)
            except Exception:
                pass

        # Default: allocate 10% of available cash
        allocation = cash * 0.1
        return allocation / price if price > 0 else 0.0

    def _calculate_stop(
        self, entry_price: float, side: str, ohlcv: pd.DataFrame
    ) -> float | None:
        """Calculate stop-loss price. Uses external manager if available."""
        if self._stop_loss_manager is not None:
            try:
                return float(
                    self._stop_loss_manager.calculate_stop(
                        pair="",
                        side="buy" if side == "long" else "sell",
                        entry_price=entry_price,
                        ohlcv=ohlcv,
                    )
                )
            except Exception:
                pass

        # Default: 2% stop-loss
        distance = entry_price * 0.02
        if side == "long":
            return entry_price - distance
        return entry_price + distance

    def _empty_result(self, ohlcv: pd.DataFrame, pair: str) -> BacktestResult:
        start = _bar_time(ohlcv, 0) if len(ohlcv) > 0 else datetime.now(UTC)
        end = _bar_time(ohlcv, len(ohlcv) - 1) if len(ohlcv) > 0 else datetime.now(UTC)
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            metrics={"total_trades": 0.0},
            parameters={},
            start_date=start,
            end_date=end,
            pair=pair,
            strategy_name=self._strategy.name,
        )


def _bar_time(ohlcv: pd.DataFrame, idx: int) -> datetime:
    """Extract a datetime from the DataFrame index or create one."""
    if hasattr(ohlcv.index, "to_pydatetime"):
        try:
            ts = ohlcv.index[idx]
            if hasattr(ts, "to_pydatetime"):
                return ts.to_pydatetime()  # type: ignore[no-any-return]
            return ts  # type: ignore[no-any-return]
        except Exception:
            pass
    return datetime(2020, 1, 1, tzinfo=UTC)
