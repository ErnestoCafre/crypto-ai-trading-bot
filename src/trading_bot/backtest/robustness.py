"""Robustness checks on top of walk-forward: random starts and cost stress.

Walk-forward answers "did the fixed rules make money out of sample"; these
two checks answer "was that luck of the window or of the cost model":

- **Random starts** (:func:`random_starts`, :func:`random_starts_target`):
  the same rules launched on ``n_samples`` random days, each traded for a
  fixed ``window`` of bars from a flat book. A robust edge is profitable in
  most windows, not only in the one the full period happens to start on.
  Starts are drawn with a seeded RNG, so a report is reproducible.
- **Slippage stress** (:func:`slippage_stress`): the same evaluation with the
  slippage multiplied (x1, x2, x3 by default). Thin edges that only exist
  at the modelled cost die at x2.

Both are the "randomized start" and "perturbation" stages of public
strategy-screening pipelines (e.g. CacheCarti/Crypto-Strategies), adapted
to this repo's two engines: :class:`BacktestEngine` (``regime_selector``)
and :class:`TargetExposureEngine` (``donchian_ensemble``). Neither changes
how the engines simulate a trade; they only choose what to run.

:class:`RobustnessCriteria` turns the numbers into a pass/fail verdict. The
defaults are deliberately simple and documented in the README; they are a
screen, not a proof: windows overlap, so the samples are not independent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from trading_bot.backtest.engine import BacktestEngine
from trading_bot.backtest.target_engine import TargetExposureEngine

DEFAULT_STRESS_MULTIPLIERS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class RobustnessCriteria:
    """Thresholds of the verdict (see :meth:`RobustnessReport.verdict`)."""

    # Random starts: share of windows with a positive return, and the
    # median Sharpe across windows, must reach these.
    min_profitable_share: float = 0.6
    min_median_sharpe: float = 0.0
    # Slippage stress: Sharpe at this multiple of the modelled slippage
    # must stay above min_stress_sharpe.
    stress_multiplier: float = 2.0
    min_stress_sharpe: float = 0.0


@dataclass
class RandomStartResult:
    """Metrics of each random window, in start order."""

    starts: list[pd.Timestamp]
    window_bars: int
    seed: int
    samples: list[dict[str, float]]

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"n_samples": 0.0}
        ret = np.array([m.get("total_return_pct", 0.0) for m in self.samples])
        sharpe = np.array([m.get("sharpe_ratio", 0.0) for m in self.samples])
        dd = np.array([m.get("max_drawdown_pct", 0.0) for m in self.samples])
        trades = np.array([_activity(m) for m in self.samples])
        return {
            "n_samples": float(len(self.samples)),
            "window_bars": float(self.window_bars),
            "profitable_share": float((ret > 0).mean()),
            "return_p10_pct": float(np.percentile(ret, 10)),
            "return_median_pct": float(np.median(ret)),
            "return_p90_pct": float(np.percentile(ret, 90)),
            "sharpe_median": float(np.median(sharpe)),
            "sharpe_min": float(sharpe.min()),
            "max_drawdown_worst_pct": float(dd.max()),
            "trades_median": float(np.median(trades)),
        }


@dataclass
class StressPoint:
    multiplier: float
    slippage_pct: float
    metrics: dict[str, float]


@dataclass
class RobustnessReport:
    random_starts: RandomStartResult | None = None
    stress: list[StressPoint] = field(default_factory=list)
    criteria: RobustnessCriteria = field(default_factory=RobustnessCriteria)

    def verdict(self) -> tuple[bool, list[str]]:
        """``(passed, reasons)``; a check that was not run does not pass."""
        c = self.criteria
        reasons: list[str] = []
        passed = True
        if self.random_starts is None or not self.random_starts.samples:
            passed = False
            reasons.append("random starts: not run")
        else:
            s = self.random_starts.summary()
            share, med = s["profitable_share"], s["sharpe_median"]
            ok = share >= c.min_profitable_share and med > c.min_median_sharpe
            passed &= ok
            reasons.append(
                f"random starts: {share:.0%} profitable "
                f"(min {c.min_profitable_share:.0%}), median Sharpe {med:.2f} "
                f"(> {c.min_median_sharpe:.2f}) -> {'ok' if ok else 'FAIL'}"
            )
        point = next(
            (p for p in self.stress if np.isclose(p.multiplier, c.stress_multiplier)),
            None,
        )
        if point is None:
            passed = False
            reasons.append(f"slippage x{c.stress_multiplier:g}: not run")
        else:
            sharpe = point.metrics.get("sharpe_ratio", 0.0)
            ok = sharpe > c.min_stress_sharpe
            passed &= ok
            reasons.append(
                f"slippage x{c.stress_multiplier:g} "
                f"({point.slippage_pct:.3f}%): Sharpe {sharpe:.2f} "
                f"(> {c.min_stress_sharpe:.2f}) -> {'ok' if ok else 'FAIL'}"
            )
        return passed, reasons

    def as_dict(self) -> dict[str, object]:
        passed, reasons = self.verdict()
        rs = self.random_starts
        return {
            "criteria": asdict(self.criteria),
            "passed": passed,
            "reasons": reasons,
            "random_starts": None
            if rs is None
            else {
                "seed": rs.seed,
                "window_bars": rs.window_bars,
                "starts": [str(t) for t in rs.starts],
                "summary": rs.summary(),
                "samples": rs.samples,
            },
            "stress": [asdict(p) for p in self.stress],
        }


def sample_starts(
    n_bars: int, window: int, n_samples: int, min_start: int, seed: int
) -> list[int]:
    """``n_samples`` sorted start positions in ``[min_start, n_bars - window]``.

    Without replacement while there are enough distinct starts.
    """
    if window < 2:
        raise ValueError("window must be at least 2 bars")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    hi = n_bars - window  # last start whose window fits
    if hi < min_start:
        raise ValueError(
            f"not enough data: {n_bars} bars cannot fit a {window}-bar window "
            f"after {min_start} warm-up bars"
        )
    rng = np.random.default_rng(seed)
    choices = hi - min_start + 1
    picks = rng.choice(choices, size=n_samples, replace=n_samples > choices)
    return sorted(int(min_start + p) for p in picks)


def default_window(n_bars: int, min_start: int) -> int:
    """A quarter of the tradable history (at least 2 bars)."""
    return max(2, (n_bars - min_start) // 4)


def random_starts(
    engine: BacktestEngine,
    ohlcv: pd.DataFrame,
    pair: str,
    n_samples: int = 20,
    window_bars: int | None = None,
    seed: int = 42,
    min_start: int | None = None,
) -> RandomStartResult:
    """Launch *engine* flat on random bars and trade ``window_bars`` each.

    Bars before each start are indicator warm-up only (``trade_from``), as
    in walk-forward, so a window sees the same signals the full run would
    from that bar on, but without the full run's open position.
    """
    n = len(ohlcv)
    lo = engine.required_lookback if min_start is None else min_start
    window = window_bars or default_window(n, lo)
    starts = sample_starts(n, window, n_samples, lo, seed)
    samples = [
        engine.run(ohlcv.iloc[: s + window], pair, trade_from=s).metrics for s in starts
    ]
    return RandomStartResult(
        starts=[pd.Timestamp(ohlcv.index[s]) for s in starts],
        window_bars=window,
        seed=seed,
        samples=samples,
    )


def random_starts_target(
    engine: TargetExposureEngine,
    data: Mapping[str, pd.DataFrame],
    n_samples: int = 20,
    window_days: int | None = None,
    seed: int = 42,
    min_start: int | None = None,
) -> RandomStartResult:
    """Launch the target-exposure portfolio in cash on random days.

    Each sample is a bot started on that day (``TargetExposureEngine.run``
    with ``start``): its first open trades the target the full history
    implies, and it runs for ``window_days``. Data after the window is
    dropped (targets are causal, so this changes nothing before it).
    """
    index = pd.DatetimeIndex(sorted(set().union(*(df.index for df in data.values()))))
    lo = engine.min_history if min_start is None else min_start
    window = window_days or default_window(len(index), lo)
    starts = sample_starts(len(index), window, n_samples, lo, seed)
    samples = []
    for s in starts:
        first, last = index[s], index[s + window - 1]
        cut = {p: df.loc[:last] for p, df in data.items()}
        samples.append(engine.run(cut, start=first).metrics(first, last))
    return RandomStartResult(
        starts=[pd.Timestamp(index[s]) for s in starts],
        window_bars=window,
        seed=seed,
        samples=samples,
    )


def slippage_stress(
    evaluate: Callable[[float], dict[str, float]],
    base_slippage_pct: float,
    multipliers: Sequence[float] = DEFAULT_STRESS_MULTIPLIERS,
) -> list[StressPoint]:
    """``evaluate(slippage_pct)`` at each multiple of the modelled slippage.

    *evaluate* builds the engine with that slippage and returns the metrics
    to judge (e.g. chained walk-forward OOS, or the holdout period).
    """
    if base_slippage_pct <= 0:
        raise ValueError("slippage stress needs a positive base slippage to multiply")
    points = []
    for k in multipliers:
        slip = base_slippage_pct * k
        points.append(StressPoint(float(k), slip, evaluate(slip)))
    return points


def _activity(metrics: Mapping[str, float]) -> float:
    """Round trips (BacktestEngine) or rebalancing days (target engine)."""
    if "total_trades" in metrics:
        return float(metrics["total_trades"])
    return float(metrics.get("fills", 0.0))
