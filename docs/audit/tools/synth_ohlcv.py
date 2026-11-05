"""Synthetic OHLCV (GBM with regime-switching drift). NOT market data.

For offline backtests when exchanges are unreachable. The series ends at the
last closed candle before now (UTC), so paper_parity's continuity check
applies to it.

    uv run python -I docs/audit/tools/synth_ohlcv.py OUT.csv 1D 2200 7
    uv run python -I docs/audit/tools/synth_ohlcv.py OUT.csv h 6000 11
"""

import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

out, freq, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(seed)
step = pd.Timedelta("1D") if freq.upper() == "1D" else pd.Timedelta("1h")
per_year = 365 if step == pd.Timedelta("1D") else 365 * 24
sigma = 0.65 / np.sqrt(per_year)  # ~65 % annualised volatility
regime = np.repeat(rng.choice([-1.0, 0.0, 1.5], size=n // 60 + 1), 60)[:n]
r = rng.normal(regime * 0.8 / per_year, sigma)
close = 30000 * np.exp(np.cumsum(r))
open_ = np.r_[30000, close[:-1]] * np.exp(rng.normal(0, sigma / 4, n))
high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sigma / 2, n)))
low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sigma / 2, n)))
last_open = pd.Timestamp(datetime.now(UTC)).floor(step) - step  # last closed bar
index = pd.date_range(end=last_open, periods=n, freq=step)
pd.DataFrame(
    {
        "timestamp": index,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    }
).to_csv(out, index=False)
print(out, n, index[0], "->", index[-1])
