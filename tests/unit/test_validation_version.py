"""Golden results pinned to VALIDATION_VERSION.

If one of these fails, the simulation changed: a result produced now is not
comparable with one produced before. That may be intended (a fix). Then:

1. bump ``VALIDATION_VERSION`` in ``src/trading_bot/backtest/provenance.py``;
2. log the change in ``docs/VALIDATION_CHANGELOG.md`` (date, what, and the
   direction it moves results);
3. update ``GOLDEN`` below with the new values, keyed by the new version.

Do not just update the numbers: the version is how a report says which
simulation produced it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.pipeline import build_live_engine, build_target_engine
from trading_bot.backtest.provenance import VALIDATION_VERSION
from trading_bot.config.settings import Settings

CHANGELOG = Path(__file__).resolve().parents[2] / "docs" / "VALIDATION_CHANGELOG.md"

# (trades, final equity) per scenario, for each validation version.
GOLDEN: dict[str, dict[str, tuple[int, float]]] = {
    "v1": {
        "regime_selector_1h": (14, 135408.30481159964),
        "donchian_portfolio": (55, 104553.1384180237),
        "donchian_daily_halve": (56, 104571.92059750344),
    }
}


def _synth(n: int, freq: str, seed: int, drift: float) -> pd.DataFrame:
    """GBM with regime-switching drift (not market data)."""
    rng = np.random.default_rng(seed)
    per_year = 365 * 24 if freq == "h" else 365
    sigma = 0.65 / np.sqrt(per_year)
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.5], size=n // 60 + 1), 60)[:n]
    close = 30000 * np.exp(
        np.cumsum(rng.normal(regime * 0.8 / per_year * drift, sigma))
    )
    open_ = np.r_[30000, close[:-1]] * np.exp(rng.normal(0, sigma / 4, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sigma / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sigma / 2, n)))
    index = pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def daily() -> dict[str, pd.DataFrame]:
    return {
        "BTC/USDT": _synth(900, "D", 7, drift=1),
        "ETH/USDT": _synth(900, "D", 8, drift=1),
    }


def _golden(name: str) -> tuple[int, float]:
    assert VALIDATION_VERSION in GOLDEN, (
        f"no golden results for {VALIDATION_VERSION}: add them (module docstring)"
    )
    return GOLDEN[VALIDATION_VERSION][name]


def _check(name: str, trades: int, equity: float) -> None:
    exp_trades, exp_equity = _golden(name)
    msg = (
        f"{name}: the simulation changed under {VALIDATION_VERSION} "
        f"(got {trades} trades, equity {equity!r}); bump VALIDATION_VERSION and "
        "log it in docs/VALIDATION_CHANGELOG.md (see this module's docstring)"
    )
    assert trades == exp_trades, msg
    assert equity == pytest.approx(exp_equity, rel=1e-9), msg


def test_regime_selector_golden(settings: Settings) -> None:
    ohlcv = _synth(2000, "h", 11, drift=50)
    result = build_live_engine("BTC/USDT", settings=settings).run(ohlcv, "BTC/USDT")
    reasons = {t.exit_reason for t in result.trades}
    assert {"signal", "stop_loss"} <= reasons  # the scenario exercises both exits
    _check(
        "regime_selector_1h", len(result.trades), float(result.equity_curve.iloc[-1])
    )


def test_donchian_portfolio_golden(
    settings: Settings, daily: dict[str, pd.DataFrame]
) -> None:
    result = build_target_engine(settings=settings).run(daily)
    _check(
        "donchian_portfolio", len(result.trades), float(result.equity_curve.iloc[-1])
    )


def test_donchian_kill_switch_golden(
    settings: Settings, daily: dict[str, pd.DataFrame]
) -> None:
    tight = settings.model_copy(
        update={"MAX_DAILY_LOSS_PCT": 1.0, "MAX_DRAWDOWN_PCT": 8.0}
    )
    result = build_target_engine(settings=tight, daily_loss_action="halve").run(daily)
    assert [kind for _, kind in result.risk_events] == ["daily_halve"]
    _check(
        "donchian_daily_halve", len(result.trades), float(result.equity_curve.iloc[-1])
    )


def test_changelog_documents_current_version() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    assert (
        f"## {VALIDATION_VERSION} " in text or f"## {VALIDATION_VERSION}\n" in text
    ), f"docs/VALIDATION_CHANGELOG.md has no '## {VALIDATION_VERSION}' section"
