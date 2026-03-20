"""Backtest report generation — console summary and optional HTML via quantstats."""

from __future__ import annotations

import logging

from backtest.data_types import BacktestResult

logger = logging.getLogger(__name__)


class BacktestReporter:
    """Generate human-readable reports from backtest results."""

    @staticmethod
    def print_summary(result: BacktestResult) -> None:
        """Print a text summary of backtest results to stdout."""
        m = result.metrics
        print(f"\n{'=' * 60}")
        print(f"  Backtest Report: {result.strategy_name} on {result.pair}")
        print(f"  Period: {result.start_date} → {result.end_date}")
        print(f"{'=' * 60}")
        print(f"  Total Return:    {m.get('total_return_pct', 0):.2f}%")
        print(f"  CAGR:            {m.get('cagr', 0):.2f}%")
        print(f"  Sharpe Ratio:    {m.get('sharpe_ratio', 0):.2f}")
        print(f"  Sortino Ratio:   {m.get('sortino_ratio', 0):.2f}")
        print(f"  Calmar Ratio:    {m.get('calmar_ratio', 0):.4f}")
        print(f"  Max Drawdown:    {m.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Win Rate:        {m.get('win_rate', 0):.1%}")
        print(f"  Profit Factor:   {m.get('profit_factor', 0):.2f}")
        print(f"  Total Trades:    {int(m.get('total_trades', 0))}")
        print(f"  Expectancy:      {m.get('expectancy', 0):.2f}")
        print(f"{'=' * 60}\n")

    @staticmethod
    def generate_html(
        result: BacktestResult,
        output_path: str = "reports/backtest_report.html",
        benchmark: object | None = None,
    ) -> str:
        """Generate an HTML report using quantstats.

        Returns the output path on success, or an empty string on failure.
        """
        try:
            import quantstats as qs

            returns = result.equity_curve.pct_change().dropna()

            qs.reports.html(
                returns,
                benchmark=benchmark,
                output=output_path,
                title=f"{result.strategy_name} on {result.pair}",
            )
            logger.info("HTML report saved to %s", output_path)
            return output_path
        except ImportError:
            logger.warning("quantstats not available — skipping HTML report")
            return ""
        except Exception:
            logger.exception("Failed to generate HTML report")
            return ""
