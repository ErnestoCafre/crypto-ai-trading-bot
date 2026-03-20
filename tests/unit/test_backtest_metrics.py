"""Tests for MetricsCalculator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from backtest.data_types import BacktestTrade
from backtest.metrics import MetricsCalculator


def _make_trade(pnl: float, pnl_pct: float = 0.0, hours: float = 24.0) -> BacktestTrade:
    now = datetime.now(UTC)
    return BacktestTrade(
        entry_time=now,
        exit_time=now + timedelta(hours=hours),
        pair="BTC/USDT",
        side="long",
        strategy="test",
        regime=None,
        entry_price=50000.0,
        exit_price=50000.0 + pnl / 0.01,
        quantity=0.01,
        pnl=pnl,
        pnl_pct=pnl_pct,
        fees=1.0,
        slippage=0.5,
        exit_reason="signal",
    )


def _make_equity(values: list[float]) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=dates, name="equity")


class TestMetricsWithProfitableTrades:
    def test_sharpe_positive(self) -> None:
        """Consistently profitable trades → positive Sharpe."""
        trades = [_make_trade(100, 1.0) for _ in range(20)]
        equity = _make_equity([100000 + i * 100 for i in range(100)])
        metrics = MetricsCalculator.calculate(trades, equity)
        assert metrics["sharpe_ratio"] > 0

    def test_win_rate_correct(self) -> None:
        trades = [_make_trade(100, 1.0)] * 3 + [_make_trade(-50, -0.5)] * 2
        equity = _make_equity([100000, 100100, 100200, 100300, 100250, 100200])
        metrics = MetricsCalculator.calculate(trades, equity)
        assert metrics["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_profit_factor_correct(self) -> None:
        trades = [_make_trade(100)] * 3 + [_make_trade(-50)] * 2
        equity = _make_equity([100000, 100050, 100100])
        metrics = MetricsCalculator.calculate(trades, equity)
        # profit_factor = gross_wins / gross_losses = 300 / 100 = 3.0
        assert metrics["profit_factor"] == pytest.approx(3.0)


class TestMetricsFlatEquity:
    def test_sharpe_zero_for_flat(self) -> None:
        """Flat equity → Sharpe should be ~0."""
        equity = _make_equity([100000.0] * 50)
        metrics = MetricsCalculator.calculate([], equity)
        assert metrics["sharpe_ratio"] == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self) -> None:
        """Equity: 100 → 110 → 88 → max DD = (110-88)/110 = 20%."""
        equity = _make_equity([100.0, 110.0, 88.0])
        trades = [_make_trade(-22, -20.0)]
        metrics = MetricsCalculator.calculate(trades, equity)
        assert metrics["max_drawdown_pct"] == pytest.approx(20.0, abs=0.1)


class TestAllMetricsPresent:
    def test_all_expected_keys(self) -> None:
        trades = [_make_trade(50)]
        equity = _make_equity([100000, 100050])
        metrics = MetricsCalculator.calculate(trades, equity)
        expected = {
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown_pct",
            "max_drawdown_duration_days",
            "total_return_pct",
            "cagr",
            "win_rate",
            "profit_factor",
            "expectancy",
            "avg_win",
            "avg_loss",
            "largest_win",
            "largest_loss",
            "total_trades",
            "trades_per_month",
            "avg_trade_duration_hours",
            "recovery_factor",
        }
        for key in expected:
            assert key in metrics, f"Missing metric: {key}"


class TestNoTrades:
    def test_empty_trades_zeroed(self) -> None:
        equity = _make_equity([100000, 100050])
        metrics = MetricsCalculator.calculate([], equity)
        assert metrics["total_trades"] == 0.0
        assert metrics["win_rate"] == 0.0
