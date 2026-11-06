"""Check that regime_selector decides the same on the live window as in the backtest.

The backtest computes indicators over the whole history; the live bot only
fetches the last OHLCV_LOOKBACK_BARS closed candles. This replays cached
candles both ways (``trading_bot.backtest.window_parity``) and exits with
code 1 if any decision differs: signal, regime, sizing factor, confidence
beyond ``--confidence-tol`` or ATR-14 beyond ``--atr-tol`` (relative).

Usage:
    uv run python scripts/window_parity.py --pair BTC/USDT --timeframe 1h
    uv run python scripts/window_parity.py --pair ETH/USDT --window 200 --step 5 \
        --json reports/window_parity_eth.json

Each checked bar recomputes every indicator on the live window (~60 ms), so
a year of 1h candles takes ~10 minutes with ``--step 1``; ``--last N`` or
``--step`` shorten it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trading_bot.backtest.provenance import VALIDATION_VERSION, provenance
from trading_bot.backtest.window_parity import compare_live_window
from trading_bot.config import get_settings
from trading_bot.data.storage import OHLCVRepository


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", default="BTC/USDT")
    p.add_argument("--timeframe", default=s.TRADING_TIMEFRAME)
    p.add_argument("--exchange", default=s.EXCHANGE_ID)
    p.add_argument(
        "--window", type=int, default=None, help="default OHLCV_LOOKBACK_BARS"
    )
    p.add_argument("--step", type=int, default=1, help="check every N bars")
    p.add_argument("--last", type=int, default=None, help="only the last N bars")
    p.add_argument("--confidence-tol", type=float, default=1e-3)
    p.add_argument("--atr-tol", type=float, default=1e-3)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    ohlcv = OHLCVRepository.load_parquet(args.exchange, args.pair, args.timeframe)
    if ohlcv is None:
        sys.exit(
            f"No cache for {args.exchange} {args.pair} {args.timeframe}; run "
            "scripts/download_historical.py first."
        )
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()
    start = None if args.last is None else max(0, len(ohlcv) - args.last)
    result = compare_live_window(
        ohlcv,
        args.pair,
        s,
        window=args.window,
        step=args.step,
        start=start,
        confidence_tol=args.confidence_tol,
        atr_rel_tol=args.atr_tol,
    )
    summary = result.as_dict()
    print(
        f"Validation {VALIDATION_VERSION} — {args.pair} {args.timeframe}: "
        f"live window {result.window} bars, {result.checked} bars checked\n"
        f"  decision mismatches {summary['decision_mismatches']}  "
        f"numeric-only {result.numeric_only}  "
        f"max |Δconfidence| {result.max_confidence_diff:.2e}  "
        f"max ATR rel. diff {result.max_atr_rel_diff:.2e}"
    )
    for ex in result.mismatches[:10]:
        print(
            f"  {ex['timestamp']}: {ex['fields']} full={ex['full']} live={ex['live']}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": provenance(
                s, {args.pair: ohlcv}, timeframe=args.timeframe, step=args.step
            ),
            "result": summary,
        }
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"JSON report: {args.json}")
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
