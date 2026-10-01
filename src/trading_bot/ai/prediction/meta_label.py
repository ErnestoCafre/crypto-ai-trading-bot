"""Meta-label model: should the bot take *this* technical LONG entry?

The technical pipeline (primary model) chooses the side; this model only
estimates the probability that the resulting trade ends with a positive net
return (see ``meta_events``). It is trained on entry events only.

Contract with the bot (``core/bot.py`` and ``SignalCombiner`` unchanged):
``predict`` returns ``predicted_direction="up"`` when the calibrated
probability reaches the decision threshold chosen out of sample, else
``"down"``. ``confidence`` depends on the contract stored in the artifact:
``scaled`` = distance to the threshold scaled to [0, 1]; ``binary`` = 1.0.
The combiner turns "up" into agreement with a LONG and "down" into
disagreement: with the default weights a scaled prediction mostly resizes
the position, a binary one blocks "down" entries (see ``prediction_fields``).
**Outside a LONG entry event there is no prediction**: ``predict`` raises
PredictionUnavailableError, which the bot treats as "no ML signal".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.base import (
    BasePredictor,
    PredictionUnavailableError,
    read_artifact,
    write_artifact,
)
from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.meta_events import evaluate_bar, live_components
from trading_bot.ai.prediction.meta_features import meta_feature_frame
from trading_bot.config.settings import get_settings
from trading_bot.core.models import Prediction

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
# Longest history a meta feature reads (100-bar means + diff).
_MAX_FEATURE_LOOKBACK = 110
# Settings that define the primary model; stored to detect live drift.
PRIMARY_SETTINGS = (
    "USE_REGIME_DETECTION",
    "REGIME_ADX_TRENDING",
    "REGIME_VOLATILITY_HIGH_PCT",
    "REGIME_VOLATILE_SIZING_FACTOR",
    "MIN_CONFIDENCE_THRESHOLD",
    "MEAN_REVERSION_ENTRIES",
)
DEFAULT_PARAMS: dict[str, Any] = {
    # Few hundred to few thousand overlapping events: keep trees small.
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
}


@dataclass
class MetaModel:
    """Fitted classifier + Platt calibration + decision threshold."""

    classifier: Any
    platt: tuple[float, float]  # p = sigmoid(a * logit(raw) + b)
    threshold: float
    feature_names: list[str]
    info: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.classifier.predict_proba(x[self.feature_names]))[:, 1]
        return _platt_apply(raw, self.platt)


def sample_weights(
    uniqueness: np.ndarray, ret: np.ndarray, mode: str, clip_quantile: float = 0.95
) -> np.ndarray:
    """AFML return attribution: uniqueness x |net return|.

    ``mode``: ``"clipped"`` winsorises |ret| at ``clip_quantile`` (a few
    outsized trades would otherwise dominate), ``"raw"`` keeps it,
    ``"uniqueness"`` ignores the return. Normalised to mean 1.
    """
    if mode == "uniqueness":
        w = uniqueness.astype(float)
    else:
        mag = np.abs(ret)
        if mode == "clipped":
            mag = np.minimum(mag, np.quantile(mag, clip_quantile))
        elif mode != "raw":
            raise ValueError(f"Unknown weight mode {mode!r}")
        w = uniqueness * mag
    total = w.sum()
    return w * len(w) / total if total > 0 else np.ones(len(w))


def fit_meta(
    x: pd.DataFrame,
    y: np.ndarray,
    ret: np.ndarray,
    weights: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    embargo: np.timedelta64 | pd.Timedelta,
    params: dict[str, Any] | None = None,
    cal_fraction: float = 0.25,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> MetaModel:
    """Fit on the early events, calibrate + pick the threshold on the late ones.

    ``t0``/``t1`` are event start/end times. The calibration block is the
    last ``cal_fraction`` of events by ``t0``; fitting events whose label
    ends within ``embargo`` of it are purged. The threshold maximises the
    net return of the trades the bot would actually take in the calibration
    block (``sequential_return``: per ``groups`` value, i.e. per pair, an
    accepted entry opens only when flat). Accepting all is a candidate, so a
    useless model ends up passing everything.
    """
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    start, end = pd.DatetimeIndex(t0), pd.DatetimeIndex(t1)
    order = np.argsort(start.asi8, kind="stable")
    n_cal = max(20, int(len(order) * cal_fraction))
    if len(order) - n_cal < 50:
        raise ValueError(f"{len(order)} events are too few to fit and calibrate")
    cal = order[-n_cal:]
    cal_start = start[cal].min()
    fit = order[: len(order) - n_cal]
    fit = fit[np.asarray(end[fit] + pd.Timedelta(embargo) < cal_start)]
    if len(np.unique(y[fit])) < 2 or len(np.unique(y[cal])) < 2:
        raise ValueError("both classes are needed to fit and calibrate")

    p = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}
    clf = lgb.LGBMClassifier(**p)
    clf.fit(x.iloc[fit], y[fit], sample_weight=weights[fit])

    raw_cal = np.asarray(clf.predict_proba(x.iloc[cal]))[:, 1]
    lr = LogisticRegression(C=1e6)
    lr.fit(_logit(raw_cal).reshape(-1, 1), y[cal])
    platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    p_cal = _platt_apply(raw_cal, platt)
    grp = np.zeros(len(y)) if groups is None else np.asarray(groups)
    threshold = choose_threshold(
        p_cal, ret[cal], start.asi8[cal], end.asi8[cal], grp[cal]
    )
    return MetaModel(
        classifier=clf,
        platt=platt,
        threshold=threshold,
        feature_names=list(x.columns),
        info={"n_fit": int(len(fit)), "n_cal": int(len(cal))},
    )


def sequential_return(
    accept: np.ndarray,
    ret: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Summed net return of the trades taken when entries open only if flat.

    Per group (pair), in ``t0`` order: an accepted event opens a trade when
    the previous trade has exited (``t0 >= t1`` of the last one taken), as
    in the bot and BacktestEngine. Overlapping accepted events are skipped.
    """
    total = 0.0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(t0[idx], kind="stable")]
        free_at = np.iinfo(np.int64).min
        for k in idx:
            if accept[k] and t0[k] >= free_at:
                total += float(ret[k])
                free_at = t1[k]
    return total


def choose_threshold(
    p: np.ndarray,
    ret: np.ndarray,
    t0: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    grid: int = 19,
) -> float:
    """Threshold on ``p`` maximising the return of the trades it lets through.

    With ``t0``/``t1`` the objective is :func:`sequential_return` (the bot
    trades one position per pair); without them, the plain sum of ``ret``
    over accepted events.
    """

    def objective(accept: np.ndarray) -> float:
        if t0 is None or t1 is None:
            return float(ret[accept].sum())
        g = np.zeros(len(p)) if groups is None else groups
        return sequential_return(accept, ret, t0, t1, g)

    candidates = np.unique(np.quantile(p, np.linspace(0.05, 0.95, grid)))
    best_t, best = 0.0, objective(np.ones(len(p), dtype=bool))  # 0.0: accept all
    for t in candidates:
        total = objective(p >= t)
        if total > best:
            best_t, best = float(t), total
    return best_t


CONTRACTS = ("scaled", "binary")


def prediction_fields(
    p: float, threshold: float, contract: str = "binary"
) -> tuple[str, float, float]:
    """(direction, confidence, predicted_return) for a calibrated ``p``.

    ``scaled``: confidence = distance to the threshold scaled to [0, 1]; in
    the combiner it mostly resizes the position. ``binary``: confidence 1.0,
    so "up" keeps full size and "down" blocks the entry (combined
    confidence 1/3 with the default weights) — a hard filter.
    """
    if contract not in CONTRACTS:
        raise ValueError(f"Unknown contract {contract!r}")
    if contract == "binary":
        return ("up" if p >= threshold else "down"), 1.0, float(p - threshold)
    if p >= threshold:
        span = 1.0 - threshold
        conf = (p - threshold) / span if span > 0 else 1.0
        return "up", float(min(1.0, conf)), float(p - threshold)
    conf = (threshold - p) / threshold if threshold > 0 else 0.0
    return "down", float(min(1.0, conf)), float(p - threshold)


class MetaLabelPredictor(BasePredictor):
    """Serves a :class:`MetaModel` from the live 250-bar OHLCV window."""

    def __init__(
        self,
        pair: str,
        timeframe: str,
        model: MetaModel,
        pair_codes: dict[str, int],
        training_settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        contract: str = "binary",
    ) -> None:
        if contract not in CONTRACTS:
            raise ValueError(f"Unknown contract {contract!r}")
        self._contract = contract
        self._pair = pair
        self._timeframe = timeframe
        self._model = model
        self._pair_codes = dict(pair_codes)
        self._training_settings = dict(training_settings or {})
        self.metadata = dict(metadata or {})
        self._components: tuple[object, object, object] | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self._model.feature_names)

    @property
    def threshold(self) -> float:
        return self._model.threshold

    @property
    def feature_importance(self) -> dict[str, float]:
        importances = self._model.classifier.feature_importances_
        return dict(
            zip(self._model.feature_names, map(float, importances), strict=False)
        )

    @classmethod
    def from_artifact(
        cls, data: dict[str, Any], pair: str, timeframe: str
    ) -> MetaLabelPredictor:
        if data.get("task") != "meta_label":
            raise ValueError("not a meta-label artifact")
        if data.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported meta-label version {data.get('version')}")
        model = MetaModel(
            classifier=data["classifier"],
            platt=tuple(data["platt"]),  # type: ignore[arg-type]
            threshold=float(data["threshold"]),
            feature_names=list(data["feature_names"]),
        )
        predictor = cls(
            pair,
            timeframe,
            model,
            data["pair_codes"],
            data.get("training_settings"),
            data.get("metadata"),
            # Artifacts without the field predate the "binary" default.
            data.get("contract", "scaled"),
        )
        predictor._warn_on_settings_drift()
        return predictor

    def to_artifact(self) -> dict[str, Any]:
        return {
            "task": "meta_label",
            "version": ARTIFACT_VERSION,
            "classifier": self._model.classifier,
            "platt": list(self._model.platt),
            "threshold": self._model.threshold,
            "feature_names": self._model.feature_names,
            "pair_codes": self._pair_codes,
            "training_settings": self._training_settings,
            "metadata": self.metadata,
            "contract": self._contract,
        }

    def predict_probability(
        self, ohlcv: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> float:
        """Calibrated P(trade opened at the next bar ends net positive)."""
        if self._pair not in self._pair_codes:
            raise PredictionUnavailableError(f"model not trained for {self._pair}")
        if len(ohlcv) < 2:
            raise PredictionUnavailableError("no feature rows")
        feats = features
        if feats is None or "VOL_SMA_20" not in feats.columns:
            feats = FeatureEngine().build_features(ohlcv)
        if self._components is None:
            self._components = live_components(self._pair)
        _, selector, combiner = self._components
        bar = evaluate_bar(selector, combiner, ohlcv)
        if bar.type != "long" or not bar.should_trade:
            # Trained on entry events only; elsewhere a prediction would be
            # out of distribution and, through the combiner, could even turn
            # a sub-threshold technical LONG into a trade.
            raise PredictionUnavailableError("not a technical LONG entry bar")
        signals = pd.DataFrame([bar.__dict__], index=feats.index[-1:])
        row = meta_feature_frame(
            feats.iloc[-1 - _MAX_FEATURE_LOOKBACK :],
            signals,
            self._pair_codes[self._pair],
        ).iloc[[-1]]
        nan_cols = row.columns[row.isna().iloc[0]].tolist()
        if nan_cols:
            raise PredictionUnavailableError(f"NaN in latest row: {nan_cols}")
        return float(self._model.predict_proba(row)[0])

    def predict(
        self, ohlcv: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> Prediction:
        p = self.predict_probability(ohlcv, features)
        direction, confidence, predicted_return = prediction_fields(
            p, self._model.threshold, self._contract
        )
        return Prediction(
            pair=self._pair,
            timeframe=self._timeframe,
            predicted_return=predicted_return,
            predicted_direction=direction,
            confidence=confidence,
            model_type="lightgbm_meta_label",
            model_version=str(self.metadata.get("trained_at", ARTIFACT_VERSION)),
            features_used=self.feature_names,
            timestamp=datetime.now(UTC),
        )

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "meta-label models are event-weighted: use meta_label.fit_meta "
            "(scripts/train_models.py --task meta_label)"
        )

    def load(self, model_path: str) -> None:
        loaded = MetaLabelPredictor.from_artifact(
            read_artifact(model_path), self._pair, self._timeframe
        )
        self.__dict__.update(loaded.__dict__)

    def save(self, model_path: str) -> None:
        write_artifact(model_path, self.to_artifact())

    def is_loaded(self) -> bool:
        return True

    def _warn_on_settings_drift(self) -> None:
        live = get_settings()
        drift = {
            k: (v, getattr(live, k, None))
            for k, v in self._training_settings.items()
            if getattr(live, k, None) != v
        }
        if drift:
            logger.warning(
                "Meta-label model for %s was trained with other primary-model "
                "settings (trained, live): %s",
                self._pair,
                drift,
            )


def training_settings() -> dict[str, Any]:
    s = get_settings()
    return {k: getattr(s, k) for k in PRIMARY_SETTINGS}


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(q / (1 - q)))


def _platt_apply(raw: np.ndarray, platt: tuple[float, float]) -> np.ndarray:
    a, b = platt
    return np.asarray(1.0 / (1.0 + np.exp(-(a * _logit(raw) + b))))
