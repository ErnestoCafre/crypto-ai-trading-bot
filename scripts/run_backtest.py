"""Backtest the live technical pipeline on cached OHLCV.

Data comes from the parquet cache written by ``scripts/download_historical.py``
(``var/cache/<exchange>/<PAIR>/<timeframe>.parquet``) or from ``--csv``
(columns: timestamp, open, high, low, close, volume).

Usage:
    uv run python scripts/run_backtest.py --pair BTC/USDT --timeframe 1h
    uv run python scripts/run_backtest.py --pair ETH/USDT --splits 5 --html
    uv run python scripts/run_backtest.py --max-risk-pct 0.25 --max-position-pct 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trading_bot.backtest.metrics import infer_periods_per_year
from trading_bot.backtest.pipeline import build_live_engine
from trading_bot.backtest.report import BacktestReporter
from trading_bot.backtest.walk_forward import WalkForwardValidator
from trading_bot.config import get_settings
from trading_bot.data.storage import OHLCVRepository


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    else:
        exchange = args.exchange or get_settings().EXCHANGE_ID
        cached = OHLCVRepository.load_parquet(exchange, args.pair, args.timeframe)
        if cached is None:
            sys.exit(
                f"No cache for {exchange} {args.pair} {args.timeframe}; run "
                "scripts/download_historical.py first or pass --csv."
            )
        df = cached
    if not isinstance(df.index, pd.DatetimeIndex):
        sys.exit("OHLCV index must be timestamps")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def main() -> None:
    # Defaults come from the BACKTEST_* settings so .env drives both.
    base = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--capital", type=float, default=base.BACKTEST_INITIAL_CAPITAL)
    parser.add_argument("--fee-pct", type=float, default=base.BACKTEST_FEE_PCT)
    parser.add_argument(
        "--slippage-pct", type=float, default=base.BACKTEST_SLIPPAGE_PCT
    )
    parser.add_argument("--splits", type=int, default=base.BACKTEST_WALK_FORWARD_SPLITS)
    parser.add_argument(
        "--train-ratio", type=float, default=base.BACKTEST_WALK_FORWARD_TRAIN_RATIO
    )
    parser.add_argument("--html", action="store_true", help="quantstats report")
    parser.add_argument(
        "--max-risk-pct", type=float, default=None, help="MAX_RISK_PER_TRADE_PCT"
    )
    parser.add_argument(
        "--max-position-pct", type=float, default=None, help="MAX_POSITION_PCT"
    )
    args = parser.parse_args()

    overrides: dict[str, float] = {}
    if args.max_risk_pct is not None:
        overrides["MAX_RISK_PER_TRADE_PCT"] = args.max_risk_pct
    if args.max_position_pct is not None:
        overrides["MAX_POSITION_PCT"] = args.max_position_pct
    settings = base.model_copy(update=overrides)

    ohlcv = _load(args)
    print(
        f"{args.pair} {args.timeframe}: {len(ohlcv)} bars "
        f"{ohlcv.index[0]} → {ohlcv.index[-1]} "
        f"({infer_periods_per_year(ohlcv.index):.0f} bars/year)"
    )

    engine = build_live_engine(
        args.pair,
        initial_capital=args.capital,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        settings=settings,
    )
    print(
        f"Sizing: MAX_RISK_PER_TRADE_PCT={settings.MAX_RISK_PER_TRADE_PCT}% "
        f"MAX_POSITION_PCT={settings.MAX_POSITION_PCT}%"
    )

    full = engine.run(ohlcv, args.pair)
    print("\nFull period (default parameters — not tuned on this data):")
    BacktestReporter.print_summary(full)
    cap_hit = full.metrics.get("cap_hit_pct", 0.0)
    if cap_hit > 50:
        print(
            f"WARNING: MAX_POSITION_PCT set the size of {cap_hit:.0f}% of the "
            "trades; sizing is not following the risk budget / ATR stop."
        )

    wf = WalkForwardValidator(n_splits=args.splits, train_ratio=args.train_ratio)
    result = wf.run(engine, ohlcv, args.pair)
    print(f"Walk-forward, {len(result.fold_results)} out-of-sample folds:")
    for fold in result.fold_results:
        m = fold.metrics
        print(
            f"  {fold.start_date:%Y-%m-%d} → {fold.end_date:%Y-%m-%d}  "
            f"return {m.get('total_return_pct', 0):7.2f}%  "
            f"sharpe {m.get('sharpe_ratio', 0):6.2f}  "
            f"maxDD {m.get('max_drawdown_pct', 0):6.2f}%  "
            f"trades {int(m.get('total_trades', 0))}  "
            f"cap {m.get('cap_hit_pct', 0):5.1f}%"
        )
    agg = result.aggregate_metrics
    print(
        f"  chained OOS: return {agg.get('total_return_pct', 0):.2f}%  "
        f"sharpe {agg.get('sharpe_ratio', 0):.2f}  "
        f"maxDD {agg.get('max_drawdown_pct', 0):.2f}%"
    )

    if args.html:
        out = f"reports/backtest_{args.pair.replace('/', '')}_{args.timeframe}.html"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        BacktestReporter.generate_html(full, output_path=out)


if __name__ == "__main__":
    main()
