"""Backtest vs live window parity for ``regime_selector``.

The :class:`BacktestEngine` computes indicators once over the whole history
(:class:`PrecomputedFeatureEngine`) and gives the strategy ``ohlcv[:i+1]``.
The live bot only fetches the last ``OHLCV_LOOKBACK_BARS`` closed candles
and computes everything on that window. Recursive indicators (EMA, Wilder's
RSI/ADX/ATR) depend on where the series starts, so the two can disagree,
and a strategy validated on the full history would then not be the one
that trades. (A public strategy screen hit exactly this: its backtest read
52,800 bars while live bots got 1,000, and it had to cap the backtest.)

:func:`compare_live_window` replays every ``step``-th bar both ways and
reports where the decision differs: signal type, regime, volatile sizing
factor, confidence, and ATR-14 (it sets the stop and therefore the size).
``scripts/window_parity.py`` runs it on cached market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import (
    FeatureEngine,
    PrecomputedFeatureEngine,
)
from trading_bot.backtest.pipeline import build_live_selector
from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import Signal
from trading_bot.core.strategy_selector import StrategySelector


@dataclass
class Decision:
    signal: str | None  # SignalType value, None = no signal
    regime: str
    sizing_factor: float | None
    confidence: float | None
    atr: float | None


@dataclass
class WindowParityResult:
    window: int
    checked: int = 0
    mismatches: list[dict[str, object]] = field(default_factory=list)
    max_confidence_diff: float = 0.0
    max_atr_rel_diff: float = 0.0
    # Bars where only confidence / ATR exceeded their tolerance.
    numeric_only: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, object]:
        return {
            "window": self.window,
            "checked": self.checked,
            "mismatches": len(self.mismatches),
            "decision_mismatches": len(self.mismatches) - self.numeric_only,
            "max_confidence_diff": self.max_confidence_diff,
            "max_atr_rel_diff": self.max_atr_rel_diff,
            "examples": self.mismatches[:20],
        }


def compare_live_window(
    ohlcv: pd.DataFrame,
    pair: str,
    settings: Settings | None = None,
    window: int | None = None,
    step: int = 1,
    start: int | None = None,
    confidence_tol: float = 1e-3,
    atr_rel_tol: float = 1e-3,
) -> WindowParityResult:
    """Compare full-history and live-window decisions on every ``step`` bars.

    ``window`` defaults to ``OHLCV_LOOKBACK_BARS`` (what the bot fetches,
    closed candles only). Checking starts at ``start`` (default: the first
    bar with a full live window). Raises if the window is shorter than the
    strategies' required lookback: the live bot would then never signal.
    """
    s = settings or get_settings()
    w = window or s.OHLCV_LOOKBACK_BARS
    full_fe = PrecomputedFeatureEngine()
    full_sel = build_live_selector(pair, full_fe, s)
    need = full_sel.get_required_lookback()
    if w < need:
        raise ValueError(
            f"live window {w} < required lookback {need}: raise OHLCV_LOOKBACK_BARS"
        )
    first = w - 1 if start is None else max(start, w - 1)
    result = WindowParityResult(window=w)
    if first >= len(ohlcv):
        return result

    full_fe.prime(ohlcv)
    live_fe = PrecomputedFeatureEngine()
    live_sel = build_live_selector(pair, live_fe, s)
    try:
        for i in range(first, len(ohlcv), step):
            full = _decide(full_sel, full_fe, ohlcv.iloc[: i + 1])
            win = ohlcv.iloc[i + 1 - w : i + 1]
            live_fe.prime(win)  # one indicator pass per bar for the live side
            live = _decide(live_sel, live_fe, win)
            result.checked += 1
            _record(result, ohlcv.index[i], full, live, confidence_tol, atr_rel_tol)
    finally:
        full_fe.clear()
        live_fe.clear()
    return result


def _decide(
    selector: StrategySelector, fe: PrecomputedFeatureEngine, frame: pd.DataFrame
) -> Decision:
    signal, cls = selector.generate_signal(frame)
    atr = fe.build_features(frame)[FeatureEngine.ATR_14].iloc[-1]
    return Decision(
        signal=None if signal is None else signal.type.value,
        regime=cls.regime.value,
        sizing_factor=_sizing(signal),
        confidence=None if signal is None else float(signal.confidence),
        atr=None if pd.isna(atr) else float(atr),
    )


def _sizing(signal: Signal | None) -> float | None:
    if signal is None:
        return None
    factor = signal.metadata.get("sizing_factor")
    return None if factor is None else float(factor)  # type: ignore[arg-type]


def _record(
    result: WindowParityResult,
    ts: object,
    full: Decision,
    live: Decision,
    confidence_tol: float,
    atr_rel_tol: float,
) -> None:
    diffs: list[str] = []
    for name in ("signal", "regime", "sizing_factor"):
        if getattr(full, name) != getattr(live, name):
            diffs.append(name)
    decision_diff = bool(diffs)
    if full.confidence is not None and live.confidence is not None:
        d = abs(full.confidence - live.confidence)
        result.max_confidence_diff = max(result.max_confidence_diff, d)
        if d > confidence_tol:
            diffs.append("confidence")
    if full.atr is not None and live.atr is not None and full.atr > 0:
        d = abs(full.atr - live.atr) / full.atr
        result.max_atr_rel_diff = max(result.max_atr_rel_diff, d)
        if d > atr_rel_tol:
            diffs.append("atr")
    elif (full.atr is None) != (live.atr is None):
        diffs.append("atr")
    if not diffs:
        return
    if not decision_diff:
        result.numeric_only += 1
    result.mismatches.append(
        {
            "timestamp": str(ts),
            "fields": diffs,
            "full": _round(full),
            "live": _round(live),
        }
    )


def _round(d: Decision) -> dict[str, object]:
    def r(x: float | None) -> float | None:
        return None if x is None or not np.isfinite(x) else round(x, 6)

    return {
        "signal": d.signal,
        "regime": d.regime,
        "sizing_factor": d.sizing_factor,
        "confidence": r(d.confidence),
        "atr": r(d.atr),
    }
