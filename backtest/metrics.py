"""Backtest performance metrics calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data_types import BacktestTrade


class MetricsCalculator:
    """Compute performance metrics from trades and equity curve."""

    @staticmethod
    def calculate(
        trades: list[BacktestTrade],
        equity_curve: pd.Series,
        risk_free_rate: float = 0.04,
    ) -> dict[str, float]:
        """Return a dictionary of performance metrics."""
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

        # Trades per month
        first_entry = min(t.entry_time for t in trades)
        last_exit = max(t.exit_time for t in trades)
        span_days = (last_exit - first_entry).total_seconds() / 86400
        months = span_days / 30.0
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

        equity = equity_curve.values.astype(float)
        initial = equity[0]
        final = equity[-1]

        # Total return
        total_ret = (final - initial) / initial * 100 if initial > 0 else 0.0
        metrics["total_return_pct"] = total_ret

        # CAGR
        if hasattr(equity_curve.index, "to_pydatetime"):
            try:
                dates = equity_curve.index.to_pydatetime()
                years = (dates[-1] - dates[0]).total_seconds() / (365.25 * 86400)
            except Exception:
                years = len(equity) / 252
        else:
            years = len(equity) / 252

        if years > 0 and initial > 0 and final > 0:
            metrics["cagr"] = ((final / initial) ** (1 / years) - 1) * 100
        else:
            metrics["cagr"] = 0.0

        # Daily returns
        returns = np.diff(equity) / equity[:-1]

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            daily_rf = risk_free_rate / 252
            excess = returns - daily_rf
            metrics["sharpe_ratio"] = float(
                np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)
            )
        else:
            metrics["sharpe_ratio"] = 0.0

        # Sortino ratio (annualized)
        if len(returns) > 1:
            daily_rf = risk_free_rate / 252
            excess = returns - daily_rf
            downside = excess[excess < 0]
            if len(downside) > 0 and np.std(downside, ddof=1) > 0:
                metrics["sortino_ratio"] = float(
                    np.mean(excess) / np.std(downside, ddof=1) * np.sqrt(252)
                )
            else:
                metrics["sortino_ratio"] = 0.0
        else:
            metrics["sortino_ratio"] = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        metrics["max_drawdown_pct"] = float(np.max(drawdowns))

        # Max drawdown duration
        in_drawdown = equity < running_max
        if np.any(in_drawdown):
            max_dd_dur = _max_consecutive_true(in_drawdown)
            metrics["max_drawdown_duration_days"] = float(max_dd_dur)
        else:
            metrics["max_drawdown_duration_days"] = 0.0

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


def _max_consecutive_true(arr: np.ndarray) -> int:
    """Return the longest streak of True values in a boolean array."""
    max_run = 0
    current = 0
    for val in arr:
        if val:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run
