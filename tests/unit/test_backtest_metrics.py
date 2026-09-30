"""Tests for MetricsCalculator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.data_types import BacktestTrade
from trading_bot.backtest.metrics import MetricsCalculator, infer_periods_per_year


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


def _curve(returns: np.ndarray, freq: str) -> pd.Series:
    equity = 100_000 * np.cumprod(np.r_[1.0, 1.0 + returns])
    idx = pd.date_range("2024-01-01", periods=len(equity), freq=freq, tz="UTC")
    return pd.Series(equity, index=idx)


class TestAnnualization:
    @pytest.mark.parametrize(
        ("freq", "expected"), [("h", 8760), ("4h", 2190), ("D", 365)]
    )
    def test_periods_per_year_from_index(self, freq: str, expected: int) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq=freq, tz="UTC")
        assert infer_periods_per_year(idx) == pytest.approx(expected)

    def test_no_datetime_index_defaults_to_daily(self) -> None:
        assert infer_periods_per_year(pd.RangeIndex(10)) == 365

    @pytest.mark.parametrize(("freq", "ppy"), [("h", 8760), ("4h", 2190), ("D", 365)])
    def test_sharpe_uses_bar_frequency(self, freq: str, ppy: int) -> None:
        rng = np.random.default_rng(3)
        rets = rng.normal(0.0002, 0.004, 2_000)
        metrics = MetricsCalculator.calculate(
            [_make_trade(1.0)], _curve(rets, freq), risk_free_rate=0.0
        )
        expected = rets.mean() / rets.std(ddof=1) * np.sqrt(ppy)
        assert metrics["sharpe_ratio"] == pytest.approx(expected, rel=1e-6)

    def test_risk_free_is_per_bar(self) -> None:
        """A curve growing exactly at rf has zero excess return."""
        rf = 0.04
        per_bar = (1 + rf) ** (1 / 8760) - 1
        rng = np.random.default_rng(4)
        rets = per_bar + rng.normal(0, 1e-4, 5_000)
        rets -= rets.mean() - per_bar  # exact mean
        metrics = MetricsCalculator.calculate(
            [_make_trade(1.0)], _curve(rets, "h"), risk_free_rate=rf
        )
        assert metrics["sharpe_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_sortino_uses_downside_deviation(self) -> None:
        rets = np.array([0.01, -0.01, 0.02, -0.005] * 50)
        metrics = MetricsCalculator.calculate(
            [_make_trade(1.0)], _curve(rets, "D"), risk_free_rate=0.0
        )
        dd = np.sqrt(np.mean(np.minimum(rets, 0) ** 2))
        assert metrics["sortino_ratio"] == pytest.approx(
            rets.mean() / dd * np.sqrt(365), rel=1e-6
        )

    def test_cagr_uses_calendar_span_for_hourly_bars(self) -> None:
        # One year of hourly bars doubling the equity → CAGR 100%.
        n = 8760
        rets = np.full(n, 2 ** (1 / n) - 1)
        metrics = MetricsCalculator.calculate([_make_trade(1.0)], _curve(rets, "h"))
        assert metrics["cagr"] == pytest.approx(100.0, rel=1e-6)


class TestDrawdownDuration:
    def test_duration_in_days_for_hourly_bars(self) -> None:
        # Peak at bar 10, underwater for 48 hourly bars, recovers at bar 58.
        values = [100.0 + i for i in range(11)]
        values += [100.0] * 47 + [111.0, 112.0]
        idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
        metrics = MetricsCalculator.calculate(
            [_make_trade(1.0)], pd.Series(values, index=idx)
        )
        assert metrics["max_drawdown_duration_days"] == pytest.approx(2.0)

    def test_unrecovered_drawdown_runs_to_the_end(self) -> None:
        values = [100.0, 110.0] + [90.0] * 3
        metrics = MetricsCalculator.calculate([_make_trade(1.0)], _make_equity(values))
        assert metrics["max_drawdown_duration_days"] == pytest.approx(3.0)
