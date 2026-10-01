"""Meta-labeling events: technical entries and the outcome of their real trade.

The primary model is the live technical pipeline (StrategySelector with
regime routing + the technical-only SignalCombiner gate). An *event* is every
closed bar where it would open a LONG when flat; the ML model only judges
whether that entry is worth taking (López de Prado, AFML ch. 3).

Each event is labelled with the trade the backtest engine would make:
entry at the next bar's open, stop at ``2 x ATR14`` below the fill
(StopLossManager), exit on the strategy's CLOSE/SHORT at the following open,
stops filled intrabar with the gap rule of ``BacktestEngine``. The bot has
no take-profit, so none is simulated: the right tail is where the strategy
earns. ``max_bars`` only caps the label horizon (exit at that bar's close)
so that purging has a bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.config.settings import Settings, get_settings

# BacktestEngine / build_live_engine defaults, per side, in percent.
FEE_PCT = 0.1
SLIPPAGE_PCT = 0.05
STOP_ATR_MULTIPLIER = 2.0

_REGIME_CODE = {"trending": 0, "ranging": 1, "volatile": 2}
_STRATEGY_CODE = {"trend_following": 0, "mean_reversion": 1}


@dataclass(frozen=True)
class BarSignal:
    """Primary-model output at one closed bar."""

    type: str | None  # "long" | "short" | "close" | "neutral" | None
    confidence: float
    should_trade: bool
    strategy: int  # _STRATEGY_CODE, -1 without a signal
    regime: int  # _REGIME_CODE
    regime_confidence: float
    volatility_percentile: float
    sizing_factor: float


def evaluate_bar(selector: object, combiner: object, window: pd.DataFrame) -> BarSignal:
    """Run the live technical pipeline on ``window`` (closed bars only)."""
    signal, cls = selector.generate_signal(window)  # type: ignore[attr-defined]
    should_trade = False
    if signal is not None:
        should_trade = bool(combiner.combine(signal, None).should_trade)  # type: ignore[attr-defined]
    factor = signal.metadata.get("sizing_factor") if signal is not None else None
    return BarSignal(
        type=signal.type.value if signal is not None else None,
        confidence=float(signal.confidence) if signal is not None else 0.0,
        should_trade=should_trade,
        strategy=_STRATEGY_CODE.get(signal.source, -1) if signal is not None else -1,
        regime=_REGIME_CODE[cls.regime.value],
        regime_confidence=float(cls.confidence),
        volatility_percentile=float(cls.volatility_percentile),
        sizing_factor=float(factor) if isinstance(factor, (int, float)) else 1.0,
    )


def live_components(
    pair: str, settings: Settings | None = None
) -> tuple[object, object, object]:
    """(feature_engine, selector, combiner) built like the live bot."""
    from trading_bot.ai.prediction.feature_engine import PrecomputedFeatureEngine
    from trading_bot.backtest.pipeline import build_live_selector
    from trading_bot.core.signal_combiner import SignalCombiner

    s = settings or get_settings()
    fe = PrecomputedFeatureEngine()
    return fe, build_live_selector(pair, fe, s), SignalCombiner(settings=s)


def technical_signals(
    ohlcv: pd.DataFrame, pair: str, settings: Settings | None = None
) -> pd.DataFrame:
    """Primary-model output for every closed bar (one row per bar).

    Bars before the selector's warm-up have ``type`` None. Computed bar by
    bar on ``ohlcv[: i + 1]`` exactly as BacktestEngine does.
    """
    fe, selector, combiner = live_components(pair, settings)
    fe.prime(ohlcv)  # type: ignore[attr-defined]
    try:
        lookback = selector.get_required_lookback()  # type: ignore[attr-defined]
        rows: list[BarSignal | None] = [None] * len(ohlcv)
        for i in range(lookback - 1, len(ohlcv)):
            rows[i] = evaluate_bar(selector, combiner, ohlcv.iloc[: i + 1])
    finally:
        fe.clear()  # type: ignore[attr-defined]
    empty = BarSignal(None, 0.0, False, -1, _REGIME_CODE["ranging"], 0.0, 50.0, 1.0)
    return pd.DataFrame([(r or empty).__dict__ for r in rows], index=ohlcv.index)


def entry_events(signals: pd.DataFrame) -> np.ndarray:
    """Bar positions where the pipeline opens a LONG when flat."""
    mask = (signals["type"] == "long") & signals["should_trade"]
    return np.flatnonzero(mask.to_numpy())


def label_events(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    events: np.ndarray,
    max_bars: int = 96,
    fee_pct: float = FEE_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
) -> pd.DataFrame:
    """Simulate the trade opened by each event; one row per finished trade.

    Columns: ``t0`` (signal bar), ``entry`` (fill bar), ``t1`` (last bar
    whose prices the outcome depends on), ``ret`` (net return of the trade,
    fees and slippage on both sides, as ``BacktestTrade.pnl_pct / 100``),
    ``label`` (``ret > 0``) and ``exit_reason``. Events whose trade is still
    open at the end of the data are dropped: their outcome is unknown.
    """
    opens = ohlcv["open"].to_numpy(dtype=float)
    lows = ohlcv["low"].to_numpy(dtype=float)
    closes = ohlcv["close"].to_numpy(dtype=float)
    feats = FeatureEngine().build_features(ohlcv)
    atr = (
        feats[FeatureEngine.ATR_14].to_numpy(dtype=float)
        if FeatureEngine.ATR_14 in feats.columns
        else np.full(len(ohlcv), np.nan)
    )
    exits = signals["type"].isin(["close", "short"]).to_numpy()
    fee = fee_pct / 100
    slip = slippage_pct / 100
    n = len(ohlcv)

    rows: list[dict[str, object]] = []
    for i in events:
        entry = int(i) + 1
        if entry >= n:
            continue
        fill = opens[entry] * (1 + slip)
        a = atr[i]
        # StopLossManager: ATR of the signal window, 2% fallback.
        distance = (a if np.isfinite(a) else fill * 0.02) * STOP_ATR_MULTIPLIER
        stop = max(0.0, fill - distance)
        raw_exit: float | None = None
        reason = ""
        t1 = entry
        for j in range(entry, n):
            if opens[j] <= stop:
                raw_exit, reason, t1 = opens[j], "stop_loss", j
                break
            if lows[j] <= stop:
                raw_exit, reason, t1 = stop, "stop_loss", j
                break
            if exits[j] and j + 1 < n:
                raw_exit, reason, t1 = opens[j + 1], "signal", j + 1
                break
            if j - entry + 1 >= max_bars:
                raw_exit, reason, t1 = closes[j], "max_bars", j
                break
        if raw_exit is None:
            continue
        exit_fill = raw_exit * (1 - slip)
        ret = (exit_fill - fill - fee * (fill + exit_fill)) / fill
        rows.append(
            {
                "t0": int(i),
                "entry": entry,
                "t1": int(t1),
                "ret": float(ret),
                "label": int(ret > 0),
                "exit_reason": reason,
            }
        )
    out = pd.DataFrame(
        rows, columns=["t0", "entry", "t1", "ret", "label", "exit_reason"]
    )
    out.index = ohlcv.index[out["t0"].to_numpy()] if len(out) else pd.DatetimeIndex([])
    return out


def average_uniqueness(t0: np.ndarray, t1: np.ndarray, n_bars: int) -> np.ndarray:
    """AFML 4.4: mean over each label's bars of 1 / (labels alive on that bar).

    A label spans the bars ``t0 + 1 .. t1`` whose prices decide it.
    """
    start = t0 + 1
    end = np.maximum(t1, start)
    concurrency = np.zeros(n_bars + 1)
    np.add.at(concurrency, start, 1.0)
    np.add.at(concurrency, end + 1, -1.0)
    conc = np.cumsum(concurrency)[:n_bars]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    return np.asarray((cum[end + 1] - cum[start]) / (end - start + 1))
