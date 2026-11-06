"""Backtest the live technical pipeline on cached OHLCV.

Data comes from the parquet cache written by ``scripts/download_historical.py``
(``var/cache/<exchange>/<PAIR>/<timeframe>.parquet``) or from ``--csv``
(columns: timestamp, open, high, low, close, volume).

With STRATEGY_MODE=donchian_ensemble (or ``--strategy-mode``) it backtests
the daily target-exposure portfolio over ``--pairs`` instead: each pair as a
full sleeve and the portfolio with MAX_POSITION_PCT per pair.

Usage:
    uv run python scripts/run_backtest.py --pair BTC/USDT --timeframe 1h
    uv run python scripts/run_backtest.py --pair ETH/USDT --splits 5 --html
    uv run python scripts/run_backtest.py --max-risk-pct 0.25 --max-position-pct 25
    uv run python scripts/run_backtest.py --strategy-mode donchian_ensemble \
        --pairs BTC/USDT,ETH/USDT,SOL/USDT --holdout-start 2024-01-01
    # compare the daily-loss policies (audit M-4) on the portfolio
    uv run python scripts/run_backtest.py --strategy-mode donchian_ensemble \
        --compare-daily-loss --holdout-start 2024-01-01

donchian_ensemble simulates the kill-switch like the bot (MAX_DRAWDOWN_PCT,
MAX_DAILY_LOSS_PCT, DONCHIAN_DAILY_LOSS_ACTION) unless --no-kill-switch.

Robustness (both modes, ``--robustness``): random starts (``--samples``
windows of ``--window`` bars launched flat on seeded random bars) and
slippage stress (``--stress``, multiples of the modelled slippage, judged on
the walk-forward OOS in regime_selector and on the holdout, or the full
period, in donchian_ensemble). It ends with a pass/fail verdict; exit code
stays 0 (it is a report). ``--json PATH`` writes every number plus the
provenance block (validation version, commit, data hashes, settings):

    uv run python scripts/run_backtest.py --pair BTC/USDT --robustness \
        --json reports/bt_btc_1h.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pandas as pd

from trading_bot.backtest.metrics import infer_periods_per_year
from trading_bot.backtest.pipeline import build_live_engine, build_target_engine
from trading_bot.backtest.provenance import VALIDATION_VERSION, provenance
from trading_bot.backtest.report import BacktestReporter
from trading_bot.backtest.robustness import (
    DEFAULT_STRESS_MULTIPLIERS,
    RandomStartResult,
    RobustnessCriteria,
    RobustnessReport,
    StressPoint,
    random_starts,
    random_starts_target,
    slippage_stress,
)
from trading_bot.backtest.target_engine import (
    DAILY_LOSS_ACTIONS,
    DailyLossAction,
    TargetExposureEngine,
)
from trading_bot.backtest.walk_forward import WalkForwardValidator
from trading_bot.config import get_settings
from trading_bot.config.settings import Settings
from trading_bot.data.storage import OHLCVRepository


def _load(
    args: argparse.Namespace, pair: str | None = None, timeframe: str | None = None
) -> pd.DataFrame:
    pair = pair or args.pair
    timeframe = timeframe or args.timeframe
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    else:
        exchange = args.exchange or get_settings().EXCHANGE_ID
        cached = OHLCVRepository.load_parquet(exchange, pair, timeframe)
        if cached is None:
            sys.exit(
                f"No cache for {exchange} {pair} {timeframe}; run "
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
    parser.add_argument(
        "--strategy-mode",
        choices=["regime_selector", "donchian_ensemble"],
        default=None,
        help="STRATEGY_MODE",
    )
    parser.add_argument(
        "--pairs", default=None, help="donchian_ensemble: comma-separated pairs"
    )
    parser.add_argument(
        "--rebalance-threshold", type=float, default=None, help="REBALANCE_THRESHOLD"
    )
    parser.add_argument(
        "--execution",
        choices=["next_open", "close"],
        default="next_open",
        help="donchian_ensemble: trade at the next open (live) or the close",
    )
    parser.add_argument(
        "--holdout-start",
        default=None,
        help="donchian_ensemble: also report before/after this date",
    )
    parser.add_argument(
        "--daily-loss-action",
        choices=list(DAILY_LOSS_ACTIONS),
        default=None,
        help="donchian_ensemble: DONCHIAN_DAILY_LOSS_ACTION",
    )
    parser.add_argument(
        "--no-kill-switch",
        action="store_true",
        help="donchian_ensemble: do not simulate the drawdown/daily limits",
    )
    parser.add_argument(
        "--compare-daily-loss",
        action="store_true",
        help="donchian_ensemble: portfolio under every daily-loss policy",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="random starts + slippage stress, with a pass/fail verdict",
    )
    parser.add_argument(
        "--samples", type=int, default=20, help="robustness: random windows"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="robustness: bars per random window (default: 1/4 of the history)",
    )
    parser.add_argument("--seed", type=int, default=42, help="robustness: RNG seed")
    parser.add_argument(
        "--stress",
        default=",".join(f"{k:g}" for k in DEFAULT_STRESS_MULTIPLIERS),
        help="robustness: slippage multipliers, comma-separated",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write results + provenance here"
    )
    args = parser.parse_args()
    print(f"Validation {VALIDATION_VERSION}")

    overrides: dict[str, float] = {}
    if args.max_risk_pct is not None:
        overrides["MAX_RISK_PER_TRADE_PCT"] = args.max_risk_pct
    if args.max_position_pct is not None:
        overrides["MAX_POSITION_PCT"] = args.max_position_pct
    settings = base.model_copy(update=overrides)
    if (args.strategy_mode or settings.STRATEGY_MODE) == "donchian_ensemble":
        _run_donchian(args, settings)
        return

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

    robustness = None
    if args.robustness:
        starts = random_starts(
            engine, ohlcv, args.pair, args.samples, args.window, args.seed
        )

        def oos(slippage: float) -> dict[str, float]:
            stressed = build_live_engine(
                args.pair,
                initial_capital=args.capital,
                fee_pct=args.fee_pct,
                slippage_pct=slippage,
                settings=settings,
            )
            return wf.run(stressed, ohlcv, args.pair).aggregate_metrics

        stress = slippage_stress(oos, args.slippage_pct, _multipliers(args.stress))
        robustness = RobustnessReport(starts, stress, RobustnessCriteria())
        _print_robustness(robustness, "walk-forward OOS")

    if args.html:
        out = f"reports/backtest_{args.pair.replace('/', '')}_{args.timeframe}.html"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        BacktestReporter.generate_html(full, output_path=out)

    if args.json:
        _write_json(
            args.json,
            {
                "provenance": provenance(
                    settings,
                    {args.pair: ohlcv},
                    strategy_mode="regime_selector",
                    timeframe=args.timeframe,
                    fee_pct=args.fee_pct,
                    slippage_pct=args.slippage_pct,
                    walk_forward={
                        "splits": args.splits,
                        "train_ratio": args.train_ratio,
                    },
                ),
                "full": full.metrics,
                "walk_forward": {
                    "folds": [
                        {
                            "start": str(f.start_date),
                            "end": str(f.end_date),
                            "metrics": f.metrics,
                        }
                        for f in result.fold_results
                    ],
                    "aggregate": agg,
                },
                "robustness": None if robustness is None else robustness.as_dict(),
            },
        )


def _run_donchian(args: argparse.Namespace, settings: Settings) -> None:
    """Daily target-exposure backtest: per-pair sleeves and the portfolio."""
    if args.csv:
        if args.pairs:
            sys.exit("--csv holds one pair: use --pair, not --pairs")
        pairs = [args.pair]  # daily candles expected in the CSV
    elif args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    else:
        pairs = list(settings.TRADING_PAIRS)
    data = {pair: _load(args, pair, "1d") for pair in pairs}
    threshold = (
        settings.REBALANCE_THRESHOLD
        if args.rebalance_threshold is None
        else args.rebalance_threshold
    )
    print(
        f"donchian_ensemble 1d: lookbacks {settings.DONCHIAN_LOOKBACKS}, vol "
        f"target {settings.DONCHIAN_VOL_TARGET:.0%} over "
        f"{settings.DONCHIAN_VOL_WINDOW}d, threshold {threshold}, "
        f"execution {args.execution}, costs {args.fee_pct}% fee + "
        f"{args.slippage_pct}% slippage per side (Sharpe with rf = 0)"
    )
    periods: list[tuple[str, str | None, str | None]] = [("full", None, None)]
    if args.holdout_start:
        cut = date.fromisoformat(args.holdout_start)
        periods += [
            ("before", None, str(cut - timedelta(days=1))),
            ("holdout", str(cut), None),
        ]

    action = args.daily_loss_action or settings.DONCHIAN_DAILY_LOSS_ACTION
    kill_switch = not args.no_kill_switch
    print(
        "kill-switch: "
        + (
            f"max DD {settings.MAX_DRAWDOWN_PCT}% (HALT), daily loss "
            f"{settings.MAX_DAILY_LOSS_PCT}% -> {action}"
            if kill_switch
            else "not simulated"
        )
    )

    def build(
        allocation: float,
        policy: str = action,
        simulate: bool = kill_switch,
        slippage: float = args.slippage_pct,
    ) -> TargetExposureEngine:
        return build_target_engine(
            initial_capital=args.capital,
            fee_pct=args.fee_pct,
            slippage_pct=slippage,
            settings=settings,
            allocation=allocation,
            rebalance_threshold=threshold,
            execution=args.execution,
            kill_switch=simulate,
            daily_loss_action=cast(DailyLossAction, policy),
        )

    alloc = settings.MAX_POSITION_PCT / 100
    if args.compare_daily_loss:
        runs = [("no kill-switch", build(alloc, simulate=False).run(data))]
        runs += [(p, build(alloc, policy=p).run(data)) for p in DAILY_LOSS_ACTIONS]
        print(f"portfolio ({alloc:.0%}/pair), daily-loss policies:")
    else:
        runs = [(pair, build(1.0).run({pair: df})) for pair, df in data.items()]
        runs.append((f"portfolio ({alloc:.0%}/pair)", build(alloc).run(data)))
    table: dict[str, dict[str, dict[str, float]]] = {}
    for name, result in runs:
        for label, start, end in periods:
            m = result.metrics(start, end)
            table.setdefault(name, {})[label] = m
            print(
                f"  {name:24} {label:8} sharpe {m['sharpe_ratio']:5.2f}  "
                f"CAGR {m['cagr']:6.2f}%  maxDD {m['max_drawdown_pct']:5.2f}%  "
                f"exposure {m.get('avg_exposure_pct', 0):5.1f}%  "
                f"turnover {m.get('turnover_per_year', 0):5.2f}x/y  "
                f"fills {int(m.get('fills', 0))}  "
                f"daily-loss {int(m.get('daily_loss_events', 0))}"
                + ("  HALTED" if m.get("halted") else "")
            )

    robustness = None
    if args.robustness:
        # The portfolio the bot runs: MAX_POSITION_PCT per pair.
        starts = random_starts_target(
            build(alloc), data, args.samples, args.window, args.seed
        )
        label, start, end = periods[-1]  # holdout when given, else full

        def judged(slippage: float) -> dict[str, float]:
            return build(alloc, slippage=slippage).run(data).metrics(start, end)

        stress = slippage_stress(judged, args.slippage_pct, _multipliers(args.stress))
        robustness = RobustnessReport(starts, stress, RobustnessCriteria())
        _print_robustness(robustness, f"portfolio, {label}")

    if args.json:
        _write_json(
            args.json,
            {
                "provenance": provenance(
                    settings,
                    data,
                    strategy_mode="donchian_ensemble",
                    timeframe="1d",
                    fee_pct=args.fee_pct,
                    slippage_pct=args.slippage_pct,
                    rebalance_threshold=threshold,
                    execution=args.execution,
                    kill_switch=kill_switch,
                    daily_loss_action=action,
                    holdout_start=args.holdout_start,
                ),
                "runs": table,
                "robustness": None if robustness is None else robustness.as_dict(),
            },
        )


def _multipliers(text: str) -> list[float]:
    values = [float(v) for v in text.split(",") if v.strip()]
    if not values or any(v <= 0 for v in values):
        sys.exit("--stress takes positive multipliers, e.g. 1,2,3")
    return values


def _print_robustness(report: RobustnessReport, judged_on: str) -> None:
    rs: RandomStartResult | None = report.random_starts
    if rs is not None:
        s = rs.summary()
        print(
            f"\nRandom starts: {int(s['n_samples'])} windows of "
            f"{int(s['window_bars'])} bars (seed {rs.seed})\n"
            f"  profitable {s['profitable_share']:.0%}  return p10/median/p90 "
            f"{s['return_p10_pct']:.2f} / {s['return_median_pct']:.2f} / "
            f"{s['return_p90_pct']:.2f}%  Sharpe median {s['sharpe_median']:.2f} "
            f"(min {s['sharpe_min']:.2f})  worst maxDD "
            f"{s['max_drawdown_worst_pct']:.2f}%  trades median "
            f"{s['trades_median']:.0f}"
        )
    print(f"Slippage stress ({judged_on}):")
    point: StressPoint
    for point in report.stress:
        m = point.metrics
        print(
            f"  x{point.multiplier:<4g} slippage {point.slippage_pct:.3f}%  "
            f"return {m.get('total_return_pct', 0):7.2f}%  "
            f"sharpe {m.get('sharpe_ratio', 0):6.2f}  "
            f"maxDD {m.get('max_drawdown_pct', 0):6.2f}%"
        )
    passed, reasons = report.verdict()
    print(f"Robustness verdict: {'PASS' if passed else 'FAIL'}")
    for reason in reasons:
        print(f"  - {reason}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nJSON report: {path}")


if __name__ == "__main__":
    main()
