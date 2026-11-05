"""Offline donchian_ensemble decision log for paper_parity.py, WITHOUT running
the bot: target_exposure() on the trailing WINDOW closed daily bars, as the
live bot computes it (HISTORY_BARS = 1000). Fills are simulated at the next
open with the backtest costs (0.05 % slippage, 0.1 % fee). The last candle
has no t+1 open in the CSV, so it gets no fill.

Also writes the CSV to the parquet cache (var/cache/binance/SYNUSDT/1d.parquet,
relative to the CURRENT directory): run it from a scratch directory.

    cd SCRATCH
    uv run --project REPO python -I \
        REPO/docs/audit/tools/make_decisions.py synth_1d.csv . 200 1000
    uv run --project REPO python REPO/scripts/paper_parity.py \
        --log ./decisions.jsonl --cache
"""

import json
import sys
from pathlib import Path

import pandas as pd

from trading_bot.core.strategies.donchian_ensemble import target_exposure
from trading_bot.data.storage import OHLCVRepository

csv, outdir = sys.argv[1], Path(sys.argv[2])
days, window = int(sys.argv[3]), int(sys.argv[4])
df = pd.read_csv(csv, parse_dates=["timestamp"], index_col="timestamp")
pair = "SYN/USDT"
OHLCVRepository.save_parquet(df, "binance", pair, "1d")
log = outdir / "decisions.jsonl"
with log.open("w") as fh:
    for i in range(len(df) - days, len(df)):
        t = target_exposure(df.iloc[max(0, i + 1 - window) : i + 1])
        candle = df.index[i]
        fill = None
        if i + 1 < len(df):
            px = float(df["open"].iloc[i + 1]) * 1.0005
            fill = {"side": "buy", "quantity": 0.01, "price": px, "fee": px * 1e-5}
        rec = {
            "mode": "paper",
            "pair": pair,
            "candle": candle.isoformat(),
            "decided_at": (
                candle + pd.Timedelta("1D") + pd.Timedelta("2min")
            ).isoformat(),
            "target": t.exposure,
            "raw": t.raw_exposure,
            "held_w": 0.0,
            "wanted": t.exposure,
            "ready": t.ready,
            "unsynced": [c.lookback for c in t.channels if not c.synced],
            "fill": fill,
        }
        fh.write(json.dumps(rec) + "\n")
print(log)
