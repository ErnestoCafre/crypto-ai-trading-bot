"""Tests for the provenance block of reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest.provenance import (
    VALIDATION_VERSION,
    data_fingerprint,
    git_state,
    provenance,
)
from trading_bot.config.settings import Settings


def _ohlcv(n: int = 50) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    v = np.arange(n, dtype=float) + 100
    return pd.DataFrame(
        {"open": v, "high": v + 1, "low": v - 1, "close": v, "volume": 1.0},
        index=index,
    )


def test_fingerprint_is_stable_and_detects_changes() -> None:
    a = _ohlcv()
    fp = data_fingerprint(a)
    assert fp == data_fingerprint(a.copy())
    assert fp["bars"] == 50 and fp["first"] == str(a.index[0])
    b = a.copy()
    b.iloc[10, b.columns.get_loc("close")] += 0.01
    assert data_fingerprint(b)["sha256"] != fp["sha256"]
    shifted = a.copy()
    shifted.index = shifted.index.shift(1)
    assert data_fingerprint(shifted)["sha256"] != fp["sha256"]


def test_provenance_block() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    out = provenance(s, {"BTC/USDT": _ohlcv()}, task="x")
    assert out["validation_version"] == VALIDATION_VERSION
    assert out["task"] == "x"
    assert out["data"]["BTC/USDT"]["bars"] == 50  # type: ignore[index]
    settings = out["settings"]
    assert isinstance(settings, dict)
    assert settings["MAX_POSITION_PCT"] == s.MAX_POSITION_PCT
    assert settings["DONCHIAN_LOOKBACKS"] == list(s.DONCHIAN_LOOKBACKS)
    assert not any("KEY" in k or "SECRET" in k or "TOKEN" in k for k in settings)
    json.dumps(out)


def test_git_state_outside_a_repo(tmp_path: Path) -> None:
    assert git_state(tmp_path) == (None, None)


def test_git_state_counts_untracked_files(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    git("add", ".gitignore")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "x.txt").write_text("x")
    commit, dirty = git_state(tmp_path)
    assert commit and dirty is False  # ignored files do not count
    (tmp_path / "new_module.py").write_text("x = 1\n")
    assert git_state(tmp_path)[1] is True
