"""Donchian channel ensemble with volatility targeting (daily bars, long-only).

Zarattini, Pagani & Barbon (2025), "Catching Crypto Trends; A Tactical
Approach for Bitcoin and Altcoins" (SSRN 5209907). Rules are fixed as in the
paper and its pre-registered replication; they are not tuned here.

For each lookback N (default 5, 10, 20, 30, 60, 90, 150, 250, 360 days):

- enter when ``close[t]`` exceeds the maximum of the N *previous* closes;
- trailing stop = max(previous stop, midpoint of the last N closes including
  t), initialised at the midpoint on entry;
- exit when ``close[t] < stop``.

Raw exposure is the mean of the binary positions, in [0, 1]. Volatility
targeting scales it by ``min(1, vol_target / realized_vol)`` where
realized_vol is the annualised (sqrt 365) standard deviation of the last
``vol_window`` daily returns: there is no leverage. The target decided at the
close of day t is held over day t+1.

Each channel's state is rebuilt by walking the whole history passed in, so it
needs no persistence and a restart reproduces the same target. A channel is
known to match any longer history once its close drops below the channel
midpoint (both an open and a flat channel are flat after that bar); see
:attr:`ChannelState.synced`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.core.models import AISignals

DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 10, 20, 30, 60, 90, 150, 250, 360)
DEFAULT_VOL_TARGET = 0.25
DEFAULT_VOL_WINDOW = 90
DAYS_PER_YEAR = 365
# Closed daily bars the live bot fetches: one Binance page. On BTC/ETH/SOL
# 2017-2026 a 720-bar window already rebuilt the same state as the full
# history on every tested day; 450 bars did not.
HISTORY_BARS = 1000
# Exposure gap treated as already on target (float residue).
_EPSILON = 1e-12


@dataclass(frozen=True)
class ChannelState:
    """One lookback's position at the last bar."""

    lookback: int
    in_position: bool
    # Trailing stop while in position, else None.
    stop: float | None
    # Whether the walked history contains a bar that forces this channel flat
    # (close below the channel midpoint): its state does not depend on where
    # the history starts.
    synced: bool


@dataclass(frozen=True)
class EnsembleTarget:
    """Target exposure for the next day, decided at the last closed bar."""

    timestamp: pd.Timestamp | None
    exposure: float  # final weight in [0, 1]
    raw_exposure: float  # mean of the channel positions
    vol_scale: float  # min(1, vol_target / realized_vol); 0 when unknown
    realized_vol: float | None  # annualised
    channels: tuple[ChannelState, ...]
    # False when there is not enough history: exposure is then 0.
    ready: bool

    @property
    def active_stops(self) -> list[float]:
        return [c.stop for c in self.channels if c.stop is not None]


def required_history(
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> int:
    """Closed daily bars the live target needs: longest channel + vol window."""
    return max(lookbacks) + vol_window


def channel_positions(
    close: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Binary position and trailing stop per bar for one lookback.

    Returns ``(position, stop)``; ``stop`` is NaN while flat. Bars before the
    channel has ``lookback`` previous closes are flat.
    """
    s = pd.Series(close, dtype=float)
    upper = s.rolling(lookback).max()
    prev_high = upper.shift(1).to_numpy()
    mid = ((upper + s.rolling(lookback).min()) / 2).to_numpy()

    n = len(close)
    pos = np.zeros(n)
    stops = np.full(n, np.nan)
    in_pos = False
    stop = -np.inf
    for t in range(n):
        if np.isnan(prev_high[t]):
            continue
        if not in_pos and close[t] > prev_high[t]:
            in_pos, stop = True, mid[t]
        elif in_pos:
            stop = max(stop, mid[t])
            if close[t] < stop:
                in_pos = False
        if in_pos:
            pos[t] = 1.0
            stops[t] = stop
    return pos, stops


def realized_vol(close: pd.Series, window: int = DEFAULT_VOL_WINDOW) -> pd.Series:
    """Annualised rolling std of daily returns (sqrt 365: crypto trades daily)."""
    return close.pct_change().rolling(window).std() * np.sqrt(DAYS_PER_YEAR)


def exposure_series(
    close: pd.Series,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    vol_target: float | None = DEFAULT_VOL_TARGET,
    vol_window: int = DEFAULT_VOL_WINDOW,
    min_history: int = 0,
) -> pd.DataFrame:
    """Per-bar ``raw``, ``vol``, ``scale`` and target ``exposure`` (``w``).

    ``exposure`` at bar t is decided at its close and applies to bar t+1.
    Bars with fewer than ``min_history`` closes so far (counting t) and bars
    without a volatility estimate get exposure 0. ``vol_target=None``
    disables volatility targeting.
    """
    values = close.to_numpy(dtype=float)
    raw = np.mean([channel_positions(values, n)[0] for n in lookbacks], axis=0)
    vol = realized_vol(close, vol_window)
    if vol_target:
        scale = (vol_target / vol).clip(upper=1.0).fillna(0.0)
    else:
        scale = pd.Series(1.0, index=close.index)
    w = pd.Series(raw, index=close.index) * scale
    if min_history > 0:
        w.iloc[: min_history - 1] = 0.0
    return pd.DataFrame(
        {"raw": raw, "vol": vol, "scale": scale, "exposure": w.fillna(0.0)},
        index=close.index,
    )


def target_exposure(
    ohlcv: pd.DataFrame,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    vol_target: float = DEFAULT_VOL_TARGET,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> EnsembleTarget:
    """Target exposure from closed daily candles (oldest first).

    Pure: the whole state is rebuilt from ``ohlcv``. With fewer than
    :func:`required_history` bars the target is 0 (``ready=False``).
    """
    close = ohlcv["close"].astype(float)
    ts = (
        close.index[-1]
        if len(close) and isinstance(close.index, pd.DatetimeIndex)
        else None
    )
    if len(close) < required_history(lookbacks, vol_window):
        return EnsembleTarget(
            timestamp=ts,
            exposure=0.0,
            raw_exposure=0.0,
            vol_scale=0.0,
            realized_vol=None,
            channels=(),
            ready=False,
        )

    values = close.to_numpy(dtype=float)
    channels: list[ChannelState] = []
    for n in lookbacks:
        pos, stops = channel_positions(values, n)
        s = pd.Series(values)
        mid = ((s.rolling(n).max() + s.rolling(n).min()) / 2).to_numpy()
        in_pos = bool(pos[-1] > 0)
        channels.append(
            ChannelState(
                lookback=n,
                in_position=in_pos,
                stop=float(stops[-1]) if in_pos else None,
                synced=bool(np.any(values < mid)),  # NaN compares False
            )
        )

    last = exposure_series(close, lookbacks, vol_target, vol_window).iloc[-1]
    vol = float(last["vol"])
    return EnsembleTarget(
        timestamp=ts,
        exposure=float(last["exposure"]),
        raw_exposure=float(last["raw"]),
        vol_scale=float(last["scale"]),
        realized_vol=vol if np.isfinite(vol) else None,
        channels=tuple(channels),
        ready=True,
    )


def rebalance_delta(
    *,
    target_exposure: float,
    held_value: float,
    equity: float,
    allocation: float,
    threshold: float,
) -> float:
    """Signed quote value to trade for one pair (0 = leave it).

    The sleeve is ``equity x allocation``; exposures are fractions of it. The
    gap is traded only when it exceeds ``threshold`` (in exposure units). A
    target of 0 always closes a held position, so no residue is left behind.
    """
    sleeve = equity * allocation
    if sleeve <= 0:
        return 0.0
    target_value = sleeve * max(0.0, min(1.0, target_exposure))
    delta = target_value - held_value
    if target_value <= 0:
        return -held_value if held_value > 0 else 0.0
    gap = abs(delta) / sleeve
    if gap <= _EPSILON or gap <= threshold:
        return 0.0
    return delta


def ai_exposure_multiplier(
    ai: AISignals | None,
    floor: float,
    sentiment_weight: float = 1.0,
    llm_weight: float = 1.0,
) -> tuple[float, bool]:
    """AI adjustment of the target exposure: ``(multiplier, block_increase)``.

    AI can only reduce exposure. Each source contributes its *bearishness* in
    [0, 1] (negative sentiment score; LLM SHORT/CLOSE confidence); bullish or
    neutral readings count as 0, never as a boost. The multiplier is
    ``1 - (1 - floor) x weighted mean bearishness``, so it lies in
    [floor, 1]. LLM consensus on SHORT/CLOSE additionally blocks any increase
    (ADR-006 veto). Without AI sources: ``(1.0, False)``. The ML meta-label
    was trained for discrete entries and is not used in this mode.
    """
    if ai is None:
        return 1.0, False
    bearish: list[tuple[float, float]] = []
    if ai.sentiment_score is not None:
        bearish.append((sentiment_weight, max(0.0, -min(1.0, ai.sentiment_score))))
    llm_short = False
    if ai.llm_recommendation is not None and ai.llm_confidence is not None:
        llm_short = ai.llm_recommendation.upper() in ("SHORT", "CLOSE")
        conf = min(1.0, max(0.0, ai.llm_confidence))
        bearish.append((llm_weight, conf if llm_short else 0.0))
    total = sum(w for w, _ in bearish)
    if total <= 0:
        return 1.0, False
    score = sum(w * b for w, b in bearish) / total
    floor = min(1.0, max(0.0, floor))
    return 1.0 - (1.0 - floor) * score, llm_short and ai.llm_consensus is True
