"""Backtest performance metrics calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.backtest.data_types import BacktestTrade

_SECONDS_PER_YEAR = 365 * 86_400
_DEFAULT_PERIODS_PER_YEAR = 365.0


class MetricsCalculator:
    """Compute performance metrics from trades and equity curve."""

    @staticmethod
    def calculate(
        trades: list[BacktestTrade],
        equity_curve: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: float | None = None,
    ) -> dict[str, float]:
        """Return a dictionary of performance metrics.

        ``risk_free_rate`` is annual. ``periods_per_year`` defaults to the
        bar frequency of a DatetimeIndex (8760 for 1h, 2190 for 4h, 365 for
        1d — crypto trades every day); without timestamps it falls back to
        daily bars (365).
        """
        metrics: dict[str, float] = {}

        # --- Trade-based metrics ---
        n_trades = len(trades)
        metrics["total_trades"] = float(n_trades)

        if n_trades == 0:
            return _empty_metrics(metrics)

        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        metrics["win_rate"] = float(len(wins) / n_trades) if n_trades > 0 else 0.0
        metrics["avg_win"] = float(np.mean(wins)) if len(wins) > 0 else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if len(losses) > 0 else 0.0
        metrics["largest_win"] = float(np.max(pnls)) if n_trades > 0 else 0.0
        metrics["largest_loss"] = float(np.min(pnls)) if n_trades > 0 else 0.0

        gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_losses = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        metrics["profit_factor"] = (
            gross_wins / gross_losses if gross_losses > 0 else float("inf")
        )
        metrics["expectancy"] = float(np.mean(pnls))

        # Trade durations
        durations_h = [
            (t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades
        ]
        metrics["avg_trade_duration_hours"] = float(np.mean(durations_h))

        # Sizing: realised risk at entry and how often the position cap bound
        metrics["avg_entry_risk_pct"] = float(
            np.mean([t.entry_risk_pct for t in trades])
        )
        metrics["cap_hit_pct"] = 100.0 * sum(t.size_capped for t in trades) / n_trades

        # Trades per month over the backtest span (trade span as fallback)
        span_years = _span_years(equity_curve.index)
        if span_years is not None:
            span_days = span_years * 365
        else:
            first_entry = min(t.entry_time for t in trades)
            last_exit = max(t.exit_time for t in trades)
            span_days = (last_exit - first_entry).total_seconds() / 86400
        months = span_days / (365 / 12)
        metrics["trades_per_month"] = n_trades / months if months > 0 else 0.0

        # --- Equity-curve metrics ---
        if len(equity_curve) < 2:
            metrics["total_return_pct"] = 0.0
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0
            metrics["calmar_ratio"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["max_drawdown_duration_days"] = 0.0
            metrics["cagr"] = 0.0
            metrics["recovery_factor"] = 0.0
            return metrics

        equity = equity_curve.to_numpy(dtype=float)
        initial = equity[0]
        final = equity[-1]
        ppy = periods_per_year or infer_periods_per_year(equity_curve.index)

        # Total return
        total_ret = (final - initial) / initial * 100 if initial > 0 else 0.0
        metrics["total_return_pct"] = total_ret

        # CAGR over the calendar span (bar count as fallback)
        years = _span_years(equity_curve.index)
        if years is None:
            years = (len(equity) - 1) / ppy
        if years > 0 and initial > 0 and final > 0:
            metrics["cagr"] = ((final / initial) ** (1 / years) - 1) * 100
        else:
            metrics["cagr"] = 0.0

        # Per-bar returns; crypto trades 24/7, so annualize with the bar
        # frequency (8760 for 1h), not 252 trading days.
        returns = np.diff(equity) / equity[:-1]
        rf_per_bar = (1 + risk_free_rate) ** (1 / ppy) - 1
        excess = returns - rf_per_bar

        if len(returns) > 1 and np.std(excess, ddof=1) > 0:
            metrics["sharpe_ratio"] = float(
                np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(ppy)
            )
        else:
            metrics["sharpe_ratio"] = 0.0

        # Sortino: downside deviation over *all* bars (target = rf per bar)
        downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
        if len(returns) > 1 and downside_dev > 0:
            metrics["sortino_ratio"] = float(
                np.mean(excess) / downside_dev * np.sqrt(ppy)
            )
        else:
            metrics["sortino_ratio"] = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        metrics["max_drawdown_pct"] = float(np.max(drawdowns))
        metrics["max_drawdown_duration_days"] = _max_drawdown_days(
            equity_curve.index, equity < running_max, ppy
        )

        # Calmar ratio
        max_dd = metrics["max_drawdown_pct"]
        if max_dd > 0:
            metrics["calmar_ratio"] = metrics["cagr"] / max_dd
        else:
            metrics["calmar_ratio"] = 0.0

        # Recovery factor
        if max_dd > 0:
            metrics["recovery_factor"] = total_ret / max_dd
        else:
            metrics["recovery_factor"] = 0.0

        return metrics


def _empty_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Fill all metrics with zeros for an empty trade list."""
    for key in (
        "win_rate",
        "avg_win",
        "avg_loss",
        "largest_win",
        "largest_loss",
        "profit_factor",
        "expectancy",
        "avg_trade_duration_hours",
        "avg_entry_risk_pct",
        "cap_hit_pct",
        "trades_per_month",
        "total_return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown_pct",
        "max_drawdown_duration_days",
        "cagr",
        "recovery_factor",
    ):
        metrics.setdefault(key, 0.0)
    return metrics


def infer_periods_per_year(index: pd.Index) -> float:
    """Bars per year from the median spacing of a DatetimeIndex (24/7 market)."""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        step = pd.Series(index).diff().median()
        seconds = step.total_seconds() if pd.notna(step) else 0.0
        if seconds > 0:
            return _SECONDS_PER_YEAR / seconds
    return _DEFAULT_PERIODS_PER_YEAR


def _span_years(index: pd.Index) -> float | None:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        return float((index[-1] - index[0]).total_seconds()) / _SECONDS_PER_YEAR
    return None


def _max_drawdown_days(
    index: pd.Index, in_drawdown: np.ndarray, periods_per_year: float
) -> float:
    """Longest time underwater, from the last peak to recovery (or the end)."""
    longest = 0.0
    start: int | None = None
    timestamps = isinstance(index, pd.DatetimeIndex)
    bars_per_day = periods_per_year / 365.0
    for i, under in enumerate([*in_drawdown, False]):
        if under and start is None:
            start = i - 1  # the peak bar
        elif not under and start is not None:
            end = min(i, len(in_drawdown) - 1)
            if timestamps:
                days = (index[end] - index[start]).total_seconds() / 86_400
            else:
                days = (end - start) / bars_per_day
            longest = max(longest, float(days))
            start = None
    return longest
