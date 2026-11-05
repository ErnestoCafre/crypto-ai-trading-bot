"""Check a donchian_ensemble run (paper or live) against the backtest engine.

Reads the bot's decision log (DONCHIAN_DECISION_LOG_PATH), downloads the
full daily history of each pair from the exchange's public endpoints (or the
parquet cache with ``--cache``) and reports, per pair and day: the bot's
target vs the engine's, the fill vs the next open, and missing days. Exits 1
when something does not match, so it can run from cron after 00:02 UTC.

Usage:
    uv run python scripts/paper_parity.py
    uv run python scripts/paper_parity.py --mode live --cache
    uv run python scripts/paper_parity.py --since 2026-10-02  # skip start-up day
    uv run python scripts/paper_parity.py --log var/state/donchian_decisions_paper.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from trading_bot.backtest.parity import ParityReport, check_parity
from trading_bot.config import get_settings
from trading_bot.core.decision_log import read_decisions
from trading_bot.data.market.ccxt_fetcher import CCXTFetcher
from trading_bot.data.storage import OHLCVRepository

HISTORY_START = datetime(2017, 1, 1, tzinfo=UTC)
TIMEFRAME = "1d"


async def _download(exchange: str, pairs: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    async with CCXTFetcher(exchange_id=exchange, sandbox=False) as fetcher:
        for pair in pairs:
            # The forming candle stays: its open is the t+1 open of the last
            # decision, and targets are only read at closed candles.
            data[pair] = await fetcher.fetch_ohlcv(pair, TIMEFRAME, since=HISTORY_START)
    return data


def _from_cache(exchange: str, pairs: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = OHLCVRepository.load_parquet(exchange, pair, TIMEFRAME)
        if df is None:
            sys.exit(f"No cache for {exchange} {pair} {TIMEFRAME}")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[pair] = df[~df.index.duplicated(keep="last")].sort_index()
    return data


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return format("-", spec.split(".")[0])  # same alignment and width
    return format(value, spec)


def _print(report: ParityReport) -> None:
    header = (
        f"{'pair':<10} {'candle':<10} {'bot w':>7} {'bt w':>7} {'gap':>9} "
        f"{'lag min':>7} {'side':>4} {'fill':>12} {'open t+1':>12} "
        f"{'slip %':>7} {'fee %':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in report.rows:
        print(
            f"{r.pair:<10} {r.candle.date().isoformat():<10} "
            f"{_fmt(r.bot_target, '>7.3f')} {_fmt(r.ref_target, '>7.3f')} "
            f"{_fmt(r.target_gap, '>9.2e')} {r.lag_min:>7.0f} "
            f"{r.fill_side or '-':>4} {_fmt(r.fill_price, '>12.4f')} "
            f"{_fmt(r.next_open, '>12.4f')} {_fmt(r.slippage_pct, '>7.3f')} "
            f"{_fmt(r.fee_pct, '>6.3f')}"
        )
    print()
    print(
        f"decisions: {len(report.rows)}  fills: {len(report.fills)}  "
        f"mean slippage: {_fmt(report.mean_slippage_pct(), '.3f')}%  "
        f"mean fee: {_fmt(report.mean_fee_pct(), '.3f')}%"
    )


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default=s.BOT_MODE, help="paper | live")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--exchange", default=s.EXCHANGE_ID)
    parser.add_argument("--cache", action="store_true", help="use the parquet cache")
    parser.add_argument(
        "--since",
        type=pd.Timestamp,
        default=None,
        help="ignore decisions for candles before this date (e.g. the start-up day)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--max-slippage-pct", type=float, default=s.BACKTEST_SLIPPAGE_PCT
    )
    parser.add_argument("--max-fee-pct", type=float, default=s.BACKTEST_FEE_PCT)
    parser.add_argument("--max-lag-min", type=float, default=60.0)
    args = parser.parse_args()

    log = args.log or Path(s.DONCHIAN_DECISION_LOG_PATH.replace("{mode}", args.mode))
    records = read_decisions(log)
    if args.since is not None:
        since = (
            args.since.tz_localize("UTC") if args.since.tzinfo is None else args.since
        )
        records = [r for r in records if pd.Timestamp(r["candle"]) >= since]
    if not records:
        sys.exit(f"No decisions in {log}")
    pairs = sorted({str(r["pair"]) for r in records})

    if args.cache:
        data = _from_cache(args.exchange, pairs)
    else:
        data = asyncio.run(_download(args.exchange, pairs))
    report = check_parity(
        records,
        data,
        s.DONCHIAN_LOOKBACKS,
        s.DONCHIAN_VOL_TARGET,
        s.DONCHIAN_VOL_WINDOW,
        last_closed=pd.Timestamp(
            datetime.now(UTC).date() - timedelta(days=1), tz="UTC"
        ),
    )
    _print(report)

    failures = report.failures(
        tolerance=args.tolerance,
        max_slippage_pct=args.max_slippage_pct,
        max_fee_pct=args.max_fee_pct,
        max_lag_min=args.max_lag_min,
    )
    if failures:
        print("\nMISMATCH:")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print("\nOK: the run matches the backtest.")


if __name__ == "__main__":
    main()
