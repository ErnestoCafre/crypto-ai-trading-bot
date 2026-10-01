"""Tests for the container health check."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot.config.settings import Settings
from trading_bot.healthcheck import check, risk_state_path
from trading_bot.risk.risk_state import RiskStateStore

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        RISK_STATE_PATH=str(tmp_path / "risk_state_{mode}.json"),
        BOT_MODE="paper",
        TRADING_INTERVAL_MINUTES=60,
    )


def test_path_resolves_mode(tmp_path: Path) -> None:
    assert risk_state_path(_settings(tmp_path)).name == "risk_state_paper.json"


def test_missing_state_is_unhealthy(tmp_path: Path) -> None:
    assert check(_settings(tmp_path), NOW) is not None


def test_recent_heartbeat_is_healthy(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    saved = (NOW - timedelta(minutes=90)).isoformat()
    RiskStateStore(risk_state_path(s)).save({"saved_at": saved})
    assert check(s, NOW) is None


def test_stale_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    saved = (NOW - timedelta(hours=4)).isoformat()
    RiskStateStore(risk_state_path(s)).save({"saved_at": saved})
    assert "last completed cycle" in (check(s, NOW) or "")


def test_corrupt_state_is_unhealthy(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    risk_state_path(s).write_text("{not json", encoding="utf-8")
    assert check(s, NOW) is not None


def test_donchian_mode_uses_its_hourly_cycle(tmp_path: Path) -> None:
    # A short TRADING_INTERVAL_MINUTES left in .env must not flag a healthy
    # donchian_ensemble bot, whose cycle runs hourly regardless.
    s = Settings(
        RISK_STATE_PATH=str(tmp_path / "risk_state_{mode}.json"),
        BOT_MODE="paper",
        TRADING_INTERVAL_MINUTES=5,
        STRATEGY_MODE="donchian_ensemble",
    )
    saved = (NOW - timedelta(minutes=90)).isoformat()
    RiskStateStore(risk_state_path(s)).save({"saved_at": saved})
    assert check(s, NOW) is None
    stale = (NOW - timedelta(hours=4)).isoformat()
    RiskStateStore(risk_state_path(s)).save({"saved_at": stale})
    assert check(s, NOW) is not None
