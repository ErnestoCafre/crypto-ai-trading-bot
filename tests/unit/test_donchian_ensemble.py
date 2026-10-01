"""Tests for the Donchian ensemble target-exposure strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.core.models import AISignals
from trading_bot.core.strategies.donchian_ensemble import (
    DEFAULT_LOOKBACKS,
    ai_exposure_multiplier,
    channel_positions,
    exposure_series,
    rebalance_delta,
    required_history,
    target_exposure,
)

# ---------------------------------------------------------------------------
# Reference: the research script used for the pre-registered study
# (positions() and the weight of strategy()), kept verbatim as an oracle.
# ---------------------------------------------------------------------------


def _ref_positions(close: pd.Series, n: int) -> pd.Series:
    c = close.to_numpy()
    hi = close.rolling(n).max().shift(1).to_numpy()
    mid = ((close.rolling(n).max() + close.rolling(n).min()) / 2).to_numpy()
    pos = np.zeros(len(c))
    inpos = False
    stop = -np.inf
    for t in range(len(c)):
        if np.isnan(hi[t]):
            continue
        if not inpos and c[t] > hi[t]:
            inpos, stop = True, mid[t]
        elif inpos:
            stop = max(stop, mid[t])
            if c[t] < stop:
                inpos = False
        pos[t] = 1.0 if inpos else 0.0
    return pd.Series(pos, index=close.index)


def _ref_weight(close: pd.Series, vol_target: float = 0.25) -> pd.Series:
    raw = pd.concat([_ref_positions(close, n) for n in DEFAULT_LOOKBACKS], axis=1)
    vol = close.pct_change().rolling(90).std() * np.sqrt(365)
    return (raw.mean(axis=1) * (vol_target / vol).clip(upper=1.0)).fillna(0.0)


def _daily(close: np.ndarray | list[float], start: str = "2020-01-01") -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(values), freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "volume": 1.0,
        },
        index=idx,
    )


def _random_walk(n: int, seed: int = 7, drift: float = 0.0005) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.03, n)))


class TestChannelPositions:
    def test_hand_case_entry_trailing_stop_and_exit(self) -> None:
        # N=3. t=3: 13 > max(10,11,12) -> enter, stop = mid(11,12,13) = 12.
        # t=4: 15, mid(12,13,15)=13.5 -> stop 13.5. t=5: 14 >= 13.5 holds,
        # mid(13,15,14)=14 -> stop 14. t=6: 13.9 < stop 14 (mid 14.45 -> 14.45)
        # -> exit. t=7: 16 > max(15,14,13.9) -> re-enter.
        close = np.array([10, 11, 12, 13, 15, 14, 13.9, 16], dtype=float)
        pos, stops = channel_positions(close, 3)
        assert pos.tolist() == [0, 0, 0, 1, 1, 1, 0, 1]
        assert stops[3] == pytest.approx(12.0)
        assert stops[4] == pytest.approx(13.5)
        assert stops[5] == pytest.approx(14.0)
        assert np.isnan(stops[6])
        assert stops[7] == pytest.approx((13.9 + 16.0) / 2)

    def test_stop_never_lowers(self) -> None:
        close = np.array([10, 11, 12, 20, 19, 18.5, 18, 17.9], dtype=float)
        pos, stops = channel_positions(close, 3)
        held = stops[pos > 0]
        assert (np.diff(held) >= 0).all()

    def test_equal_to_previous_high_is_not_a_breakout(self) -> None:
        close = np.array([10, 12, 11, 12], dtype=float)
        pos, _ = channel_positions(close, 3)
        assert pos[-1] == 0

    @pytest.mark.parametrize("n", DEFAULT_LOOKBACKS)
    def test_matches_reference(self, n: int) -> None:
        close = pd.Series(_random_walk(1500))
        pos, _ = channel_positions(close.to_numpy(), n)
        np.testing.assert_array_equal(pos, _ref_positions(close, n).to_numpy())


class TestExposureSeries:
    def test_matches_reference_weight(self) -> None:
        close = pd.Series(_random_walk(1500, seed=3))
        mine = exposure_series(close)["exposure"]
        np.testing.assert_allclose(mine.to_numpy(), _ref_weight(close).to_numpy())

    def test_bounded_and_no_leverage(self) -> None:
        close = pd.Series(_random_walk(1200, seed=11, drift=0.003))
        out = exposure_series(close)
        assert out["exposure"].between(0.0, 1.0).all()
        assert (out["scale"] <= 1.0).all()

    def test_low_volatility_uses_full_raw_exposure(self) -> None:
        # Steady 0.1 %/day uptrend: vol ~0, every channel long -> w = 1.
        close = pd.Series(100 * 1.001 ** np.arange(500))
        out = exposure_series(close)
        assert out["exposure"].iloc[-1] == pytest.approx(1.0)

    def test_min_history_zeroes_warmup(self) -> None:
        close = pd.Series(_random_walk(600, seed=5, drift=0.004))
        out = exposure_series(close, min_history=450)
        assert (out["exposure"].iloc[:449] == 0).all()
        free = exposure_series(close)
        pd.testing.assert_series_equal(
            out["exposure"].iloc[449:], free["exposure"].iloc[449:]
        )


class TestTargetExposure:
    def test_insufficient_history_is_flat(self) -> None:
        df = _daily(100 * 1.01 ** np.arange(required_history() - 1))
        target = target_exposure(df)
        assert not target.ready
        assert target.exposure == 0.0
        assert target.channels == ()

    def test_matches_series_last_bar(self) -> None:
        df = _daily(_random_walk(1000, seed=21, drift=0.002))
        target = target_exposure(df)
        series = exposure_series(df["close"])
        assert target.ready
        assert target.exposure == pytest.approx(series["exposure"].iloc[-1])
        assert target.raw_exposure == pytest.approx(series["raw"].iloc[-1])
        assert target.timestamp == df.index[-1]
        assert len(target.channels) == len(DEFAULT_LOOKBACKS)
        for ch in target.channels:
            assert (ch.stop is not None) == ch.in_position

    def test_restart_mid_trend_rebuilds_same_target(self) -> None:
        # A restart re-fetches the same window: the state is rebuilt from
        # OHLCV, so the target is identical and nothing needs persisting.
        df = _daily(_random_walk(1000, seed=4, drift=0.004))
        first = target_exposure(df)
        assert first.raw_exposure > 0  # in a trend
        again = target_exposure(df.copy())
        assert again == first

    def test_shorter_window_converges_once_synced(self) -> None:
        df = _daily(_random_walk(1600, seed=10, drift=0.001))
        full = target_exposure(df)
        window = target_exposure(df.iloc[-1000:])
        assert all(ch.synced for ch in window.channels)
        assert 0 < full.raw_exposure < 1
        assert window.exposure == pytest.approx(full.exposure)
        assert window.channels == full.channels

    def test_close_below_midpoint_marks_synced(self) -> None:
        # Monotonic uptrend never closes below any midpoint: not synced.
        up = target_exposure(_daily(100 * 1.002 ** np.arange(500)))
        assert not any(ch.synced for ch in up.channels)
        assert up.raw_exposure == 1.0


class TestRebalanceDelta:
    def _delta(self, target: float, held: float, threshold: float = 0.05) -> float:
        return rebalance_delta(
            target_exposure=target,
            held_value=held,
            equity=100_000.0,
            allocation=0.25,
            threshold=threshold,
        )

    def test_buys_gap_to_target_value(self) -> None:
        assert self._delta(0.4, 0.0) == pytest.approx(10_000.0)

    def test_partial_sell(self) -> None:
        assert self._delta(0.2, 10_000.0) == pytest.approx(-5_000.0)

    def test_gap_within_threshold_is_ignored(self) -> None:
        # 0.04 of the 25k sleeve = 1k gap, below 5 %.
        assert self._delta(0.44, 10_000.0) == 0.0
        assert self._delta(0.36, 10_000.0) == 0.0
        assert self._delta(0.46, 10_000.0) == pytest.approx(1_500.0)

    def test_zero_target_always_closes(self) -> None:
        assert self._delta(0.0, 300.0) == -300.0  # 1.2 % of the sleeve
        assert self._delta(0.0, 0.0) == 0.0

    def test_small_entry_below_threshold_is_skipped(self) -> None:
        assert self._delta(0.03, 0.0) == 0.0

    def test_zero_threshold_trades_any_gap(self) -> None:
        assert self._delta(0.401, 10_000.0, threshold=0.0) == pytest.approx(25.0)

    def test_target_clipped_to_allocation(self) -> None:
        assert self._delta(1.7, 0.0) == pytest.approx(25_000.0)


class TestAIExposureMultiplier:
    def test_no_sources_is_neutral(self) -> None:
        assert ai_exposure_multiplier(None, 0.25) == (1.0, False)
        assert ai_exposure_multiplier(AISignals(), 0.25) == (1.0, False)

    def test_bullish_never_raises(self) -> None:
        ai = AISignals(sentiment_score=1.0, llm_recommendation="LONG", llm_confidence=1)
        assert ai_exposure_multiplier(ai, 0.25) == (1.0, False)

    def test_bearish_sentiment_reduces_to_floor(self) -> None:
        mult, block = ai_exposure_multiplier(AISignals(sentiment_score=-1.0), 0.25)
        assert mult == pytest.approx(0.25)
        assert not block

    def test_weighted_mean_of_sources(self) -> None:
        ai = AISignals(
            sentiment_score=-0.5, llm_recommendation="NEUTRAL", llm_confidence=0.9
        )
        mult, _ = ai_exposure_multiplier(ai, 0.0, sentiment_weight=0.2, llm_weight=0.2)
        assert mult == pytest.approx(1 - 0.25)

    def test_llm_consensus_short_blocks_increase(self) -> None:
        ai = AISignals(
            llm_recommendation="SHORT", llm_confidence=0.8, llm_consensus=True
        )
        mult, block = ai_exposure_multiplier(ai, 0.25)
        assert mult == pytest.approx(1 - 0.75 * 0.8)
        assert block

    def test_llm_short_without_consensus_does_not_block(self) -> None:
        ai = AISignals(
            llm_recommendation="SHORT", llm_confidence=0.8, llm_consensus=False
        )
        assert ai_exposure_multiplier(ai, 0.25)[1] is False

    def test_ml_prediction_is_ignored(self) -> None:
        ai = AISignals(prediction_confidence=0.9, prediction_direction="down")
        assert ai_exposure_multiplier(ai, 0.25) == (1.0, False)
