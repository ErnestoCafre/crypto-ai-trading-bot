"""Which validator produced a result: version, code, data and settings.

A backtest or training report is a verdict of the simulation that produced
it. When the simulation changes (fill rules, costs, kill-switch, a gate),
an old number is no longer comparable with a new one, and nothing in the
number says so. Every report therefore carries :func:`provenance`:

- ``validation_version``: :data:`VALIDATION_VERSION`, bumped by hand on
  every change that can move a result, and logged with its date and
  direction in ``docs/VALIDATION_CHANGELOG.md``. A golden test
  (``tests/unit/test_validation_version.py``) pins the engines' output on
  synthetic data to this version, so a behaviour change fails CI until the
  version and the changelog are updated.
- ``git_commit`` / ``git_dirty``: the code (``None`` outside a checkout,
  e.g. in the Docker image).
- ``data``: bars, first/last timestamp and a SHA-256 of the OHLCV values of
  each series, so two runs can be checked to have used the same candles.
- ``settings``: the subset of settings the result depends on.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_bot.config.settings import Settings

# Bump on any change that can move a backtest or validation result, and log
# it in docs/VALIDATION_CHANGELOG.md (date, what changed, direction).
VALIDATION_VERSION = "v1"

# Settings a backtest or training result depends on. Secrets never go here.
RESULT_SETTINGS = (
    "STRATEGY_MODE",
    "TRADING_TIMEFRAME",
    "OHLCV_LOOKBACK_BARS",
    "MAX_RISK_PER_TRADE_PCT",
    "MAX_POSITION_PCT",
    "MAX_DRAWDOWN_PCT",
    "MAX_DAILY_LOSS_PCT",
    "MIN_CONFIDENCE_THRESHOLD",
    "MEAN_REVERSION_ENTRIES",
    "USE_REGIME_DETECTION",
    "REGIME_ADX_TRENDING",
    "REGIME_VOLATILITY_HIGH_PCT",
    "REGIME_VOLATILE_SIZING_FACTOR",
    "DONCHIAN_LOOKBACKS",
    "DONCHIAN_VOL_TARGET",
    "DONCHIAN_VOL_WINDOW",
    "REBALANCE_THRESHOLD",
    "DONCHIAN_DAILY_LOSS_ACTION",
    "BACKTEST_FEE_PCT",
    "BACKTEST_SLIPPAGE_PCT",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def provenance(
    settings: Settings | None = None,
    data: Mapping[str, pd.DataFrame] | None = None,
    setting_names: Iterable[str] = RESULT_SETTINGS,
    **extra: object,
) -> dict[str, object]:
    """Provenance block for a report (JSON-serialisable)."""
    commit, dirty = git_state()
    out: dict[str, object] = {
        "validation_version": VALIDATION_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    if data is not None:
        out["data"] = {name: data_fingerprint(df) for name, df in data.items()}
    if settings is not None:
        out["settings"] = {
            name: _jsonable(getattr(settings, name))
            for name in setting_names
            if hasattr(settings, name)
        }
    out.update(extra)
    return out


def data_fingerprint(ohlcv: pd.DataFrame) -> dict[str, object]:
    """Size, span and SHA-256 of the timestamps and OHLCV values."""
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in ohlcv]
    digest = hashlib.sha256()
    digest.update(pd.Index(ohlcv.index).astype(str).str.cat(sep=",").encode())
    digest.update(ohlcv[cols].to_numpy(dtype="float64").tobytes())
    return {
        "bars": len(ohlcv),
        "first": str(ohlcv.index[0]) if len(ohlcv) else None,
        "last": str(ohlcv.index[-1]) if len(ohlcv) else None,
        "sha256": digest.hexdigest(),
    }


def git_state(root: Path = _REPO_ROOT) -> tuple[str | None, bool | None]:
    """``(short commit, uncommitted changes)``, or ``(None, None)``.

    Untracked files count as changes (a new module not yet committed can
    produce the result); ignored files (``var/``, ``reports/``) do not.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def _jsonable(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)
