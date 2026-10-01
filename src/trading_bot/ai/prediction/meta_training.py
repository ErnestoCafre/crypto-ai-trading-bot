"""End-to-end meta-label training run (used by ``scripts/train_models.py``)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.meta_events import evaluate_bar, live_components
from trading_bot.ai.prediction.meta_features import (
    PRICE_FEATURES,
    SIGNAL_FEATURES,
    price_features,
)
from trading_bot.ai.prediction.meta_label import (
    CONTRACTS,
    MetaLabelPredictor,
    MetaModel,
    training_settings,
)
from trading_bot.ai.prediction.meta_validation import (
    BacktestComparison,
    EventMetrics,
    GateResult,
    PairData,
    TimeSplit,
    backtest_comparison,
    build_pair_data,
    candidate_signals,
    cpcv_splits,
    event_metrics,
    fit_split,
    meta_gate,
    pool_events,
    predict_splits,
    walk_forward_time_splits,
)
from trading_bot.config.settings import Settings

logger = logging.getLogger(__name__)

_BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


@dataclass
class MetaConfig:
    timeframe: str = "1h"
    max_bars: int = 96
    embargo_bars: int | None = None  # default: max_bars
    folds: int = 6
    test_days: float = 180.0
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    weight_mode: str = "clipped"
    # Evaluated on events + CPCV only (the backtest runs for weight_mode).
    compare_weight_modes: tuple[str, ...] = ("raw",)
    # Prediction contract published; every contract is backtested. "binary"
    # is the default: with the combiner weights "scaled" only shrinks size.
    contract: str = "binary"
    min_trades: int = 30
    serving_window: int = 250
    parity_samples: int = 25
    n_boot: int = 2000
    seed: int = 42

    @property
    def bar(self) -> pd.Timedelta:
        return pd.Timedelta(timedelta(minutes=_BAR_MINUTES[self.timeframe]))

    @property
    def embargo(self) -> pd.Timedelta:
        bars = self.max_bars if self.embargo_bars is None else self.embargo_bars
        return pd.Timedelta(timedelta(minutes=bars * _BAR_MINUTES[self.timeframe]))


@dataclass
class PairReport:
    pair: str
    events: int
    metrics: EventMetrics
    backtest: BacktestComparison | None  # for MetaConfig.contract
    gate: GateResult | None
    backtests: dict[str, BacktestComparison] = field(default_factory=dict)


@dataclass
class ModeReport:
    weight_mode: str
    pooled: EventMetrics
    pairs: list[PairReport]
    cpcv_delta: list[float] = field(default_factory=list)

    @property
    def cpcv_positive(self) -> float:
        d = [x for x in self.cpcv_delta if np.isfinite(x)]
        return float(np.mean([x > 0 for x in d])) if d else float("nan")


@dataclass
class MetaRun:
    pairs: list[str]
    pair_codes: dict[str, int]
    features: list[str]
    dropped_features: list[str]
    events_per_pair: dict[str, int]
    oos_start: pd.Timestamp
    modes: list[ModeReport]
    final_model: MetaModel | None = None

    @property
    def primary(self) -> ModeReport:
        return self.modes[0]

    def as_dict(self) -> dict[str, Any]:
        def mode(m: ModeReport) -> dict[str, Any]:
            return {
                "weight_mode": m.weight_mode,
                "pooled": asdict(m.pooled),
                "cpcv_delta": m.cpcv_delta,
                "cpcv_positive": m.cpcv_positive,
                "pairs": [
                    {
                        "pair": p.pair,
                        "events": p.events,
                        "metrics": asdict(p.metrics),
                        "backtests": {k: asdict(v) for k, v in p.backtests.items()},
                        "gate": None if p.gate is None else asdict(p.gate),
                    }
                    for p in m.pairs
                ],
            }

        return {
            "pairs": self.pairs,
            "features": self.features,
            "dropped_features": self.dropped_features,
            "events_per_pair": self.events_per_pair,
            "oos_start": str(self.oos_start),
            "threshold": None
            if self.final_model is None
            else self.final_model.threshold,
            "modes": [mode(m) for m in self.modes],
        }


def meta_parity_columns(
    data: PairData,
    columns: Sequence[str],
    window: int,
    samples: int = 25,
    tolerance: float = 0.01,
) -> list[str]:
    """Columns the bot reproduces from its ``window``-bar fetch at entry bars.

    Price features are rebuilt from the trailing window; signal features by
    running the live pipeline on it. A column passes when every sample is
    within ``tolerance`` standard deviations of the full-history value.
    """
    bars = [ts for ts in data.events["t0"] if data.ohlcv.index.get_loc(ts) >= window]
    if not bars:
        raise ValueError(f"{data.pair}: no event with {window} bars of history")
    picks = [bars[int(i)] for i in np.linspace(0, len(bars) - 1, samples)]
    scale = data.features[list(columns)].std().replace(0.0, 1.0).fillna(1.0)
    _, selector, combiner = live_components(data.pair)
    worst = dict.fromkeys(columns, 0.0)
    for ts in picks:
        end = data.ohlcv.index.get_loc(ts)
        win = data.ohlcv.iloc[end - window + 1 : end + 1]
        live = price_features(FeatureEngine().build_features(win)).iloc[-1].to_dict()
        bar = evaluate_bar(selector, combiner, win)
        live.update(
            {
                "tech_confidence": bar.confidence,
                "strategy": bar.strategy,
                "regime": bar.regime,
                "regime_confidence": bar.regime_confidence,
                "volatility_percentile": bar.volatility_percentile,
                "sizing_factor": bar.sizing_factor,
            }
        )
        ref = data.features.loc[ts]
        for col in columns:
            if col == "pair_code" or pd.isna(ref[col]):
                continue
            a = live.get(col, np.nan)
            diff = (
                np.inf if pd.isna(a) else abs(float(a) - float(ref[col])) / scale[col]
            )
            worst[col] = max(worst[col], float(diff))
    return [c for c in columns if worst[c] <= tolerance]


def run_meta_label(
    histories: Mapping[str, pd.DataFrame],
    cfg: MetaConfig,
    settings: Settings,
    fit_final: bool = True,
) -> MetaRun:
    pairs = sorted(histories)
    codes = {p: i for i, p in enumerate(pairs)}
    data = [
        build_pair_data(p, histories[p], codes[p], settings, cfg.max_bars)
        for p in pairs
    ]
    for d in data:
        logger.info("%s: %d labelled entry events", d.pair, len(d.events))

    candidates = [*PRICE_FEATURES, *SIGNAL_FEATURES]
    kept = set(candidates)
    for d in data:
        kept &= set(
            meta_parity_columns(d, candidates, cfg.serving_window, cfg.parity_samples)
        )
    features = [c for c in candidates if c in kept] + ["pair_code"]
    dropped = sorted(set(candidates) - kept)
    if dropped:
        logger.warning("Meta features dropped for train/serve mismatch: %s", dropped)

    events = pool_events(data)
    test_len = pd.Timedelta(timedelta(days=cfg.test_days))
    wf = walk_forward_time_splits(
        events["t0"], events["t1"], cfg.folds, test_len, cfg.embargo
    )
    cpcv = cpcv_splits(
        events["t0"], events["t1"], cfg.cpcv_groups, cfg.cpcv_test_groups, cfg.embargo
    )
    oos_start = wf[0].windows[0][0]
    by_pair = {p: (events["pair"] == p).to_numpy() for p in pairs}

    modes = []
    for mode in (cfg.weight_mode, *cfg.compare_weight_modes):
        modes.append(
            _evaluate_mode(
                mode,
                events,
                features,
                wf,
                cpcv,
                data,
                by_pair,
                oos_start,
                cfg,
                settings,
                backtest=mode == cfg.weight_mode,
            )
        )

    final = None
    if fit_final:
        everything = TimeSplit(np.arange(len(events)), np.array([], int), ())
        final = fit_split(
            events, features, everything, cfg.embargo, cfg.weight_mode, seed=cfg.seed
        )
    return MetaRun(
        pairs=pairs,
        pair_codes=codes,
        features=features,
        dropped_features=dropped,
        events_per_pair={d.pair: len(d.events) for d in data},
        oos_start=oos_start,
        modes=modes,
        final_model=final,
    )


def _evaluate_mode(
    mode: str,
    events: pd.DataFrame,
    features: list[str],
    wf: list[TimeSplit],
    cpcv: list[TimeSplit],
    data: list[PairData],
    by_pair: dict[str, np.ndarray],
    oos_start: pd.Timestamp,
    cfg: MetaConfig,
    settings: Settings,
    backtest: bool = True,
) -> ModeReport:
    preds = predict_splits(events, features, wf, cfg.embargo, mode, seed=cfg.seed)
    pooled = event_metrics(events, preds, n_boot=cfg.n_boot, seed=cfg.seed)
    windows = [
        (s.windows[0][0], s.windows[0][1], fp.model)
        for s, fp in zip(wf, preds, strict=True)
    ]
    reports = []
    for d in data:
        pm = event_metrics(
            events, preds, by_pair[d.pair], n_boot=cfg.n_boot, seed=cfg.seed
        )
        if not backtest:
            reports.append(PairReport(d.pair, len(d.events), pm, None, None))
            continue
        series = {c: candidate_signals(d, windows, c) for c in CONTRACTS}
        bts = backtest_comparison(d, series, oos_start, settings)
        bt = bts[cfg.contract]
        gate = meta_gate(pooled, pm, bt, cfg.min_trades)
        reports.append(PairReport(d.pair, len(d.events), pm, bt, gate, bts))

    cpcv_preds = predict_splits(
        events, features, cpcv, cfg.embargo, mode, seed=cfg.seed
    )
    deltas = []
    for fp in cpcv_preds:
        r = events["ret"].to_numpy()[fp.rows]
        acc = fp.p >= fp.threshold
        deltas.append(float(r[acc].mean() - r.mean()) if acc.any() else float("nan"))
    return ModeReport(mode, pooled, reports, deltas)


def build_predictor(
    run: MetaRun,
    pair: str,
    timeframe: str,
    metadata: dict[str, Any],
    contract: str = "binary",
) -> MetaLabelPredictor:
    if run.final_model is None:
        raise ValueError("run has no final model")
    return MetaLabelPredictor(
        pair,
        timeframe,
        run.final_model,
        run.pair_codes,
        training_settings(),
        metadata,
        contract,
    )
