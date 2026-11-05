"""Shows the effect of one missing daily bar on TargetExposureEngine.

Before commit "fix(backtest): a missing daily bar no longer liquidates the
pair" the held exposure of B on ``day`` dropped to 0; now it is kept.

    uv run python -I docs/audit/tools/gap_check.py synth_1d.csv
"""

import sys

import pandas as pd

from trading_bot.backtest.target_engine import TargetExposureEngine

df = pd.read_csv(sys.argv[1], parse_dates=["timestamp"], index_col="timestamp")
eng = TargetExposureEngine(
    allocation=0.5, rebalance_threshold=0.0, fee_pct=0, slippage_pct=0
)
full = eng.run({"A": df, "B": df})
held = full.exposure["B"]
day = held[held > 0.2].index[min(300, int((held > 0.2).sum()) - 1)]
gap = day - pd.Timedelta("1D")
res = eng.run({"A": df, "B": df.drop(index=gap)})
print(
    "day",
    day.date(),
    "| held B without gap",
    round(float(full.exposure.loc[day, "B"]), 4),
    "| with the bar of",
    gap.date(),
    "missing",
    round(float(res.exposure.loc[day, "B"]), 4),
)
