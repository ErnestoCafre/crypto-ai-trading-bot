"""Leak-free validation for the direction models: purged walk-forward.

Every split is expressed in *bar positions* of the source OHLCV frame. A
training row at bar ``i`` carries a label built from ``close[i + horizon]``,
so it may only be used for a test window starting at bar ``s`` when
``i + horizon + embargo < s``. The same purge applies to the early-stopping
split inside each training window.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor

logger = logging.getLogger(__name__)

# Scale-free oscillators: bounded or price-normalised, so a model trained on
# one price regime reads the same values in another. Price/volume levels
# (EMA, SMA, Bollinger bands, VWAP, Ichimoku lines, OBV, ...) are excluded.
SCALE_FREE_PREFIXES = (
    "RSI_",
    "STOCHk_",
    "STOCHd_",
    "CCI_",
    "WILLR_",
    "ADX_",
    "DMP_",
    "DMN_",
    "BBB_",
    "BBP_",
)
_OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Split:
    """Row selections (positions into the feature matrix) for one fold."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class ValidationReport:
    """Out-of-sample metrics pooled over every walk-forward test window."""

    n_folds: int
    n_test: int
    accuracy: float
    directional_accuracy: float
    auc: float
    log_loss: float
    baseline_accuracy: float
    baseline_directional_accuracy: float
    baseline_log_loss: float
    positive_rate: float
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None
    folds: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "folds"}
        for key in ("test_start", "test_end"):
            out[key] = None if out[key] is None else str(out[key])
        out["folds"] = self.folds
        return out


def candidate_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    """Columns a model may train on, before the train/serve parity check."""
    if feature_set == "scale_free":
        return [c for c in features.columns if c.startswith(SCALE_FREE_PREFIXES)]
    if feature_set == "all":
        return [c for c in features.columns if c not in _OHLCV]
    raise ValueError(f"Unknown feature set: {feature_set!r}")


def serving_parity_columns(
    ohlcv: pd.DataFrame,
    columns: Sequence[str],
    window: int,
    feature_engine: FeatureEngine | None = None,
    samples: int = 20,
    tolerance: float = 0.01,
) -> list[str]:
    """Keep the columns the live bot reproduces from a ``window``-bar fetch.

    The bot computes features over the last ``window`` bars only, while
    training sees the full history. Path-dependent columns (OBV's running
    sum, a 200-bar EMA seeded inside a 250-bar window) then take values the
    model never saw. A column passes when, at ``samples`` bars, the value
    computed from the trailing window matches the full-history one within
    ``tolerance`` standard deviations of the column.
    """
    fe = feature_engine or FeatureEngine()
    full = fe.build_features(ohlcv)
    if len(ohlcv) <= window:
        raise ValueError(f"need more than {window} bars for the parity check")
    ends = np.linspace(window - 1, len(ohlcv) - 1, samples).astype(int)
    scale = full[list(columns)].std().replace(0.0, 1.0)
    worst = pd.Series(0.0, index=list(columns))
    for end in ends:
        live = fe.build_features(ohlcv.iloc[end - window + 1 : end + 1]).iloc[-1]
        ref = full.iloc[end]
        for col in columns:
            a = live.get(col, np.nan)
            b = ref[col]
            if pd.isna(b):
                continue
            diff = np.inf if pd.isna(a) else abs(float(a) - float(b)) / scale[col]
            worst[col] = max(worst[col], diff)
    compared = {c for c in columns if full[c].iloc[ends].notna().any()}
    kept = [c for c in columns if c in compared and worst[c] <= tolerance]
    dropped = sorted(set(columns) - set(kept))
    if dropped:
        logger.info("Dropped for train/serve mismatch: %s", dropped)
    return kept


def walk_forward_splits(
    bar_pos: np.ndarray,
    horizon: int,
    n_folds: int,
    test_bars: int,
    embargo: int | None = None,
    val_fraction: float = 0.15,
    min_train: int = 500,
) -> list[Split]:
    """Expanding-window walk-forward over the last ``n_folds * test_bars`` bars.

    ``bar_pos`` is the (increasing) OHLCV position of every feature row.
    Training rows end ``horizon + embargo`` bars before their test window;
    the early-stopping rows are the tail of the training window, purged the
    same way from the rows that fit the trees.
    """
    if embargo is None:
        embargo = horizon
    gap = horizon + embargo
    last = int(bar_pos[-1]) + 1
    first_test = last - n_folds * test_bars
    splits: list[Split] = []
    for k in range(n_folds):
        start = first_test + k * test_bars
        end = start + test_bars
        test = np.flatnonzero((bar_pos >= start) & (bar_pos < end))
        train_val = np.flatnonzero(bar_pos + gap < start)
        if len(train_val) < min_train:
            raise ValueError(
                f"fold {k}: {len(train_val)} training rows (< {min_train}) — "
                "use more history or fewer/shorter folds"
            )
        split = purged_holdout(bar_pos, train_val, horizon, embargo, val_fraction)
        if len(split[0]) < min_train or len(test) == 0:
            raise ValueError(
                f"fold {k}: {len(split[0])} training rows (< {min_train}) — "
                "use more history or fewer/shorter folds"
            )
        splits.append(Split(train=split[0], val=split[1], test=test))
    return splits


def purged_holdout(
    bar_pos: np.ndarray,
    rows: np.ndarray,
    horizon: int,
    embargo: int,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``rows`` into (fit, val) with the val tail purged from fit."""
    if len(rows) < 2:
        raise ValueError(f"{len(rows)} rows are too few for a train/val split")
    n_val = max(1, int(len(rows) * val_fraction))
    val = rows[-n_val:]
    val_start = bar_pos[val[0]]
    fit = rows[bar_pos[rows] + horizon + embargo < val_start]
    return fit, val


def _frame(x: pd.DataFrame, y: pd.Series, rows: np.ndarray) -> pd.DataFrame:
    """Feature rows with the label as last column (LightGBMPredictor.train)."""
    out = x.iloc[rows].copy()
    out["__label__"] = y.iloc[rows].to_numpy()
    return out


def fit_model(
    x: pd.DataFrame,
    y: pd.Series,
    fit_rows: np.ndarray,
    val_rows: np.ndarray,
    params: dict[str, Any],
    pair: str = "BTC/USDT",
    timeframe: str = "1h",
) -> LightGBMPredictor:
    model = LightGBMPredictor(pair=pair, timeframe=timeframe, params=params)
    model.train(_frame(x, y, fit_rows), _frame(x, y, val_rows))
    return model


def evaluate_walk_forward(
    x: pd.DataFrame,
    y: pd.Series,
    forward_ret: pd.Series,
    splits: Sequence[Split],
    params: dict[str, Any],
) -> ValidationReport:
    """Train one model per fold and score it on that fold's test window only.

    Baselines are "trivial" models fitted on the same training rows: the
    majority class (accuracy), the constant training positive rate
    (log-loss) and the majority sign of the forward return (directional).
    Bars with a zero forward return have no direction and are excluded from
    the directional scores.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rets: list[np.ndarray] = []
    base_cls: list[np.ndarray] = []
    base_prob: list[np.ndarray] = []
    base_dir: list[np.ndarray] = []
    folds: list[dict[str, float]] = []
    for split in splits:
        model = fit_model(x, y, split.train, split.val, params)
        p = model.predict_proba(x.iloc[split.test])
        y_test = y.iloc[split.test].to_numpy()
        r_test = forward_ret.iloc[split.test].to_numpy()
        y_train = y.iloc[split.train].to_numpy()
        r_train = forward_ret.iloc[split.train].to_numpy()
        n = len(split.test)
        pos_rate = float(y_train.mean())
        probs.append(p)
        labels.append(y_test)
        rets.append(r_test)
        base_cls.append(np.full(n, int(pos_rate >= 0.5)))
        base_prob.append(np.full(n, pos_rate))
        base_dir.append(np.full(n, int((r_train > 0).mean() >= 0.5)))
        folds.append(
            {
                "n_train": float(len(split.train)),
                "n_test": float(n),
                "accuracy": float(((p > 0.5) == y_test).mean()),
                "directional_accuracy": _directional(p > 0.5, r_test),
            }
        )

    p_all = np.concatenate(probs)
    y_all = np.concatenate(labels)
    r_all = np.concatenate(rets)
    try:
        auc = float(roc_auc_score(y_all, p_all))
    except ValueError:
        auc = float("nan")
    test_index = x.index[np.concatenate([s.test for s in splits])]
    return ValidationReport(
        n_folds=len(splits),
        n_test=len(y_all),
        accuracy=float(((p_all > 0.5) == y_all).mean()),
        directional_accuracy=_directional(p_all > 0.5, r_all),
        auc=auc,
        log_loss=float(log_loss(y_all, p_all, labels=[0, 1])),
        baseline_accuracy=float((np.concatenate(base_cls) == y_all).mean()),
        baseline_directional_accuracy=_directional(
            np.concatenate(base_dir).astype(bool), r_all
        ),
        baseline_log_loss=float(
            log_loss(y_all, np.concatenate(base_prob), labels=[0, 1])
        ),
        positive_rate=float(y_all.mean()),
        test_start=test_index[0] if len(test_index) else None,
        test_end=test_index[-1] if len(test_index) else None,
        folds=folds,
    )


def _directional(pred_up: np.ndarray, forward_ret: np.ndarray) -> float:
    moved = forward_ret != 0
    if not moved.any():
        return float("nan")
    return float((pred_up[moved] == (forward_ret[moved] > 0)).mean())


def tune_params(
    x: pd.DataFrame,
    y: pd.Series,
    bar_pos: np.ndarray,
    rows: np.ndarray,
    horizon: int,
    embargo: int,
    n_trials: int,
    base_params: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Optuna search on ``rows`` only (callers pass pre-test data).

    Minimises log-loss on a purged holdout at the end of ``rows``.
    """
    if n_trials <= 0:
        return dict(base_params)
    import optuna
    from sklearn.metrics import log_loss

    from trading_bot.ai.prediction.trainer import suggest_lightgbm_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fit_rows, holdout = purged_holdout(bar_pos, rows, horizon, embargo, 0.2)
    # Early stopping gets its own purged tail inside the fitting rows, so
    # the holdout that ranks trials is never seen during training.
    es_fit, es_val = purged_holdout(bar_pos, fit_rows, horizon, embargo, 0.15)
    y_hold = y.iloc[holdout].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_lightgbm_params(trial)}
        model = fit_model(x, y, es_fit, es_val, params)
        p = model.predict_proba(x.iloc[holdout])
        return float(log_loss(y_hold, p, labels=[0, 1]))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**base_params, **study.best_params}
