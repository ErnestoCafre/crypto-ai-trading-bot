"""Out-of-sample evaluation of the meta-label model.

* Events of every pair are pooled and split on **calendar time** (pairs are
  correlated): a training event is kept only if its label ends ``embargo``
  before the test window starts (walk-forward), or does not overlap any test
  group ± ``embargo`` (CPCV, AFML ch. 12).
* Hard-filter metrics on the labelled events: precision of the accepted
  class, AUC, Brier, calibration error and the net expectancy of accepted vs
  all entries. Uncertainty is computed on chains of overlapping events: a
  cluster bootstrap inside each fold, and a random filter that circularly
  shifts the model's decisions within each fold.
* The real effect is measured with ``BacktestEngine`` and the live
  ``SignalCombiner`` via ``ai_signals_series``: in the bot the prediction is
  a soft filter (a "down" only blocks when confident enough, otherwise it
  shrinks the position).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.meta_events import (
    average_uniqueness,
    entry_events,
    label_events,
    technical_signals,
)
from trading_bot.ai.prediction.meta_features import meta_feature_frame
from trading_bot.ai.prediction.meta_label import (
    MetaModel,
    fit_meta,
    prediction_fields,
    sample_weights,
)
from trading_bot.config.settings import Settings

logger = logging.getLogger(__name__)

_META_COLS = (
    "ret",
    "label",
    "t0",
    "t1",
    "pair",
    "uniqueness",
    "exit_reason",
    "cluster",
)


@dataclass
class PairData:
    pair: str
    ohlcv: pd.DataFrame
    features: pd.DataFrame  # meta features, one row per bar
    events: pd.DataFrame  # labelled events (finished trades)
    candidates: pd.DatetimeIndex  # every entry bar, labelled or not


def build_pair_data(
    pair: str,
    ohlcv: pd.DataFrame,
    pair_code: int,
    settings: Settings | None = None,
    max_bars: int = 96,
    feature_columns: Sequence[str] | None = None,
) -> PairData:
    """Signals, labelled events and meta features for one pair."""
    signals = technical_signals(ohlcv, pair, settings)
    events = entry_events(signals)
    labels = label_events(ohlcv, signals, events, max_bars=max_bars)
    feats = meta_feature_frame(
        FeatureEngine().build_features(ohlcv), signals, pair_code
    )
    if feature_columns is not None:
        feats = feats[list(feature_columns)]
    idx = ohlcv.index
    labels = labels.assign(
        t0=idx[labels["t0"].to_numpy()],
        t1=idx[labels["t1"].to_numpy()],
        pair=pair,
        uniqueness=average_uniqueness(
            labels["t0"].to_numpy(), labels["t1"].to_numpy(), len(ohlcv)
        ),
        cluster=pair_code * 1_000_000
        + overlap_clusters(labels["t0"].to_numpy(), labels["t1"].to_numpy()),
    )
    return PairData(pair, ohlcv, feats, labels, idx[events])


def overlap_clusters(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Ids of chains of overlapping labels (events sorted by ``t0``).

    Consecutive entry bars of one trend share most of their outcome; a chain
    is the unit of independent evidence for the statistics below.
    """
    ids = np.zeros(len(t0), dtype=np.int64)
    reach = -1
    k = -1
    for i, (a, b) in enumerate(zip(t0, t1, strict=True)):
        if a > reach:
            k += 1
        reach = max(reach, int(b))
        ids[i] = k
    return ids


def pool_events(data: Sequence[PairData]) -> pd.DataFrame:
    """Feature rows of every labelled event, all pairs, sorted by ``t0``."""
    frames = []
    for d in data:
        ev = d.events
        x = d.features.loc[ev["t0"]].reset_index(drop=True)
        meta = ev[list(_META_COLS)].reset_index(drop=True)
        frames.append(pd.concat([x, meta], axis=1))
    pooled = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in pooled.columns if c not in _META_COLS]
    pooled = pooled.dropna(subset=feature_cols)
    return pooled.sort_values("t0", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class TimeSplit:
    train: np.ndarray  # event rows
    test: np.ndarray
    windows: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]  # test intervals


def walk_forward_time_splits(
    t0: pd.Series,
    t1: pd.Series,
    n_folds: int,
    test_len: pd.Timedelta,
    embargo: pd.Timedelta,
    end: pd.Timestamp | None = None,
) -> list[TimeSplit]:
    """Expanding walk-forward over the last ``n_folds * test_len``."""
    end = end if end is not None else t0.max() + pd.Timedelta(timedelta(seconds=1))
    step = pd.Timedelta(test_len).to_pytimedelta()
    first = end - n_folds * step
    splits = []
    for k in range(n_folds):
        start = first + k * step
        stop = start + test_len
        test = np.flatnonzero(((t0 >= start) & (t0 < stop)).to_numpy())
        train = np.flatnonzero((t1 + embargo < start).to_numpy())
        splits.append(TimeSplit(train, test, ((start, stop),)))
    return splits


def cpcv_splits(
    t0: pd.Series,
    t1: pd.Series,
    n_groups: int,
    n_test_groups: int,
    embargo: pd.Timedelta,
) -> list[TimeSplit]:
    """Combinatorial purged CV: every choice of ``n_test_groups`` time groups.

    Training events overlapping a test group (by ``[t0, t1]``) are purged,
    as are events starting within ``embargo`` after it.
    """
    edges = pd.date_range(t0.min(), t1.max(), periods=n_groups + 1)
    groups = list(zip(edges[:-1], edges[1:], strict=True))
    splits = []
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(len(t0), dtype=bool)
        keep = np.ones(len(t0), dtype=bool)
        for g in combo:
            start, stop = groups[g]
            last = g == n_groups - 1
            in_group = (t0 >= start) & ((t0 <= stop) if last else (t0 < stop))
            test_mask |= in_group.to_numpy()
            overlaps = (t0 <= stop) & (t1 >= start)
            embargoed = (t0 > stop) & (t0 <= stop + embargo)
            keep &= ~(overlaps | embargoed).to_numpy()
        splits.append(
            TimeSplit(
                np.flatnonzero(keep & ~test_mask),
                np.flatnonzero(test_mask),
                tuple(groups[g] for g in combo),
            )
        )
    return splits


@dataclass
class FoldPrediction:
    fold: int
    rows: np.ndarray
    p: np.ndarray
    threshold: float
    model: MetaModel


def fit_split(
    events: pd.DataFrame,
    feature_cols: Sequence[str],
    split: TimeSplit,
    embargo: pd.Timedelta,
    weight_mode: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> MetaModel:
    tr = events.iloc[split.train]
    w = sample_weights(tr["uniqueness"].to_numpy(), tr["ret"].to_numpy(), weight_mode)
    return fit_meta(
        tr[list(feature_cols)],
        tr["label"].to_numpy(),
        tr["ret"].to_numpy(),
        w,
        tr["t0"].to_numpy(),
        tr["t1"].to_numpy(),
        embargo.to_timedelta64(),
        params,
        seed=seed,
        groups=tr["pair"].to_numpy(),
    )


def predict_splits(
    events: pd.DataFrame,
    feature_cols: Sequence[str],
    splits: Sequence[TimeSplit],
    embargo: pd.Timedelta,
    weight_mode: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[FoldPrediction]:
    out = []
    for k, split in enumerate(splits):
        model = fit_split(
            events, feature_cols, split, embargo, weight_mode, params, seed
        )
        x = events.iloc[split.test][list(feature_cols)]
        out.append(
            FoldPrediction(
                k, split.test, model.predict_proba(x), model.threshold, model
            )
        )
    return out


@dataclass
class EventMetrics:
    n: int
    accepted: int
    acceptance_rate: float
    base_rate: float
    precision: float
    auc: float
    brier: float
    ece: float
    expectancy_all: float
    expectancy_accepted: float
    expectancy_delta: float
    total_all: float
    total_accepted: float
    delta_ci_low: float = float("nan")
    random_filter_p: float = float("nan")
    # Share of folds (with events) whose accepted trades have expectancy > 0;
    # a fold where the filter accepted nothing counts as not positive.
    folds_positive_expectancy: float = float("nan")
    per_fold: list[dict[str, float]] = field(default_factory=list)


def event_metrics(
    events: pd.DataFrame,
    preds: Sequence[FoldPrediction],
    mask: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> EventMetrics:
    """Pooled OOS metrics over ``preds``; ``mask`` selects event rows (a pair).

    These are per-*event* statistics: evidence that ``p`` ranks entries.
    The bot holds one position per pair, so the per-*trade* effect of the
    filter is measured by :func:`backtest_comparison`, not here.

    Events overlap heavily (consecutive entry bars of one trend), so the
    uncertainty is computed on chains of overlapping events: the bootstrap
    resamples chains within each fold, and the random filter circularly
    shifts the model's accept/reject sequence within each fold, keeping its
    acceptance rate and its run structure.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    folds = []
    for fp in preds:
        keep = np.ones(len(fp.rows), bool) if mask is None else mask[fp.rows]
        if keep.any():
            ev = events.iloc[fp.rows[keep]]
            folds.append(
                (
                    fp.p[keep],
                    fp.p[keep] >= fp.threshold,
                    ev["label"].to_numpy(),
                    ev["ret"].to_numpy(),
                    ev["cluster"].to_numpy(),
                )
            )
    if not folds:
        raise ValueError("no out-of-sample events")
    p = np.concatenate([f[0] for f in folds])
    acc = np.concatenate([f[1] for f in folds])
    y = np.concatenate([f[2] for f in folds])
    r = np.concatenate([f[3] for f in folds])
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    delta = _mean(r[acc]) - _mean(r)
    rng = np.random.default_rng(seed)
    triples = [(f[1], f[3], f[4]) for f in folds]
    boot = np.array([_boot_delta(triples, rng) for _ in range(n_boot)])
    rand = np.array([_random_filter_mean(triples, rng) for _ in range(n_boot)])
    model_mean = _mean(r[acc])
    per_fold = [
        {
            "n": float(len(f[3])),
            "accepted": float(f[1].sum()),
            "expectancy_all": _mean(f[3]),
            "expectancy_accepted": _mean(f[3][f[1]]),
        }
        for f in folds
    ]
    positive = [d["accepted"] > 0 and d["expectancy_accepted"] > 0 for d in per_fold]
    return EventMetrics(
        n=len(y),
        accepted=int(acc.sum()),
        acceptance_rate=float(acc.mean()),
        base_rate=float(y.mean()),
        precision=float(y[acc].mean()) if acc.any() else float("nan"),
        auc=auc,
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(p, y),
        expectancy_all=_mean(r),
        expectancy_accepted=model_mean,
        expectancy_delta=delta,
        total_all=float(r.sum()),
        total_accepted=float(r[acc].sum()),
        delta_ci_low=float(np.nanquantile(boot, 0.05)),
        random_filter_p=float((1 + np.sum(rand >= model_mean)) / (1 + len(rand))),
        folds_positive_expectancy=float(np.mean(positive)),
        per_fold=per_fold,
    )


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def _mean(a: np.ndarray) -> float:
    return float(a.mean()) if len(a) else float("nan")


def _boot_delta(
    folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> float:
    """Cluster bootstrap within each fold: resample whole overlap chains."""
    accs, rets = [], []
    for acc, r, cluster in folds:
        ids, inverse = np.unique(cluster, return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
        picks = rng.integers(0, len(ids), size=len(ids))
        idx = np.concatenate([members[k] for k in picks])
        accs.append(acc[idx])
        rets.append(r[idx])
    a, r = np.concatenate(accs), np.concatenate(rets)
    return _mean(r[a]) - _mean(r) if a.any() else float("nan")


def _random_filter_mean(
    folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> float:
    """Mean return under the model's decisions circularly shifted per fold."""
    chosen = []
    for acc, r, _ in folds:
        n = len(acc)
        shift = int(rng.integers(1, n)) if n > 1 else 0
        chosen.append(r[np.roll(acc, shift)])
    picked = np.concatenate(chosen)
    return float(picked.mean()) if len(picked) else float("nan")


# -- Backtest through the live pipeline -------------------------------------


def candidate_signals(
    data: PairData,
    preds_by_window: Sequence[tuple[pd.Timestamp, pd.Timestamp, MetaModel]],
    contract: str = "binary",
) -> pd.Series:
    """AISignals (ML prediction only) for every entry bar in the test windows.

    Each bar is scored by the model of the window containing it — a model
    trained only on events that ended before that window.
    """
    from trading_bot.core.models import AISignals

    out: dict[pd.Timestamp, AISignals] = {}
    for start, stop, model in preds_by_window:
        bars = data.candidates[(data.candidates >= start) & (data.candidates < stop)]
        x = data.features.loc[bars, model.feature_names].dropna()
        if x.empty:
            continue
        for ts, p in zip(x.index, model.predict_proba(x), strict=True):
            direction, conf, ret = prediction_fields(
                float(p), model.threshold, contract
            )
            out[ts] = AISignals(
                prediction_return=ret,
                prediction_confidence=conf,
                prediction_direction=direction,
            )
    return pd.Series(out, dtype=object)


@dataclass
class BacktestComparison:
    trades_base: int
    trades_ml: int
    expectancy_base: float  # mean pnl_pct per trade
    expectancy_ml: float
    return_base: float
    return_ml: float
    max_dd_base: float
    max_dd_ml: float
    sharpe_base: float
    sharpe_ml: float


def backtest_comparison(
    data: PairData,
    ai_series: Mapping[str, pd.Series],
    start: pd.Timestamp,
    settings: Settings,
) -> dict[str, BacktestComparison]:
    """Live pipeline over ``[start, end]`` without, and with each ML series."""
    from trading_bot.backtest.pipeline import build_live_engine

    trade_from = int(data.ohlcv.index.searchsorted(start))

    def run(series: pd.Series | None) -> Any:
        engine = build_live_engine(data.pair, settings=settings)
        return engine.run(
            data.ohlcv, data.pair, ai_signals_series=series, trade_from=trade_from
        )

    def exp(res: Any) -> float:
        return _mean(np.array([t.pnl_pct for t in res.trades]))

    base = run(None)
    out = {}
    for name, series in ai_series.items():
        ml = run(series)
        m0, m1 = base.metrics, ml.metrics
        out[name] = BacktestComparison(
            trades_base=len(base.trades),
            trades_ml=len(ml.trades),
            expectancy_base=exp(base),
            expectancy_ml=exp(ml),
            return_base=float(m0["total_return_pct"]),
            return_ml=float(m1["total_return_pct"]),
            max_dd_base=float(m0["max_drawdown_pct"]),
            max_dd_ml=float(m1["max_drawdown_pct"]),
            sharpe_base=float(m0["sharpe_ratio"]),
            sharpe_ml=float(m1["sharpe_ratio"]),
        )
    return out


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]


def meta_gate(
    pooled: EventMetrics,
    pair: EventMetrics,
    bt: BacktestComparison,
    min_trades: int = 30,
    alpha: float = 0.05,
) -> GateResult:
    """Publish only if the filter adds value beyond chance, without more risk."""
    checks = [
        (
            pooled.delta_ci_low > 0,
            f"pooled delta CI low {pooled.delta_ci_low:+.4f} > 0",
        ),
        (
            pooled.random_filter_p < alpha,
            f"beats random filter (p={pooled.random_filter_p:.3f} < {alpha})",
        ),
        (
            bt.expectancy_ml > bt.expectancy_base,
            f"backtest expectancy {bt.expectancy_ml:+.3f}% "
            f"> {bt.expectancy_base:+.3f}%",
        ),
        (
            bt.max_dd_ml <= bt.max_dd_base,
            f"max DD {bt.max_dd_ml:.2f}% <= {bt.max_dd_base:.2f}%",
        ),
        (bt.trades_ml >= min_trades, f"{bt.trades_ml} trades >= {min_trades}"),
        (
            pair.folds_positive_expectancy > 0.5,
            f"filtered expectancy > 0 in {pair.folds_positive_expectancy:.0%} of folds",
        ),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return GateResult(not failed, failed or [msg for _, msg in checks])
