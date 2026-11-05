"""Tests for config.settings."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.config.settings import Settings


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.BOT_MODE == "paper"
        assert s.EXCHANGE_ID == "binance"
        assert isinstance(s.TRADING_PAIRS, list)

    def test_trading_pairs_parsing_from_csv(self) -> None:
        s = Settings(TRADING_PAIRS="BTC/USDT,ETH/USDT")  # type: ignore[arg-type]
        assert s.TRADING_PAIRS == ["BTC/USDT", "ETH/USDT"]

    def test_trading_pairs_accepts_list(self) -> None:
        s = Settings(TRADING_PAIRS=["SOL/USDT"])
        assert s.TRADING_PAIRS == ["SOL/USDT"]

    def test_trading_pairs_csv_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADING_PAIRS", "BTC/USDT, ETH/USDT,SOL/USDT")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.TRADING_PAIRS == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_trading_pairs_csv_from_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT\n")
        s = Settings(_env_file=env_file)  # type: ignore[call-arg]
        assert s.TRADING_PAIRS == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_list_settings_accept_json_array_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADING_PAIRS", '["BTC/USDT", "ETH/USDT"]')
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.TRADING_PAIRS == ["BTC/USDT", "ETH/USDT"]

    def test_strategy_mode_defaults(self) -> None:
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.STRATEGY_MODE == "regime_selector"
        assert s.DONCHIAN_LOOKBACKS == [5, 10, 20, 30, 60, 90, 150, 250, 360]
        assert s.DONCHIAN_VOL_TARGET == 0.25
        assert s.DONCHIAN_VOL_WINDOW == 90
        assert s.REBALANCE_THRESHOLD == 0.05

    def test_donchian_lookbacks_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRATEGY_MODE", "donchian_ensemble")
        monkeypatch.setenv("DONCHIAN_LOOKBACKS", "20, 60,360")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.STRATEGY_MODE == "donchian_ensemble"
        assert s.DONCHIAN_LOOKBACKS == [20, 60, 360]

    def test_strategy_settings_validated(self) -> None:
        with pytest.raises(ValueError):
            Settings(STRATEGY_MODE="martingale")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            Settings(DONCHIAN_LOOKBACKS=[1, 5])
        with pytest.raises(ValueError):
            Settings(REBALANCE_THRESHOLD=1.5)

    def test_ai_flags_default_disabled(self) -> None:
        s = Settings()
        assert s.USE_SENTIMENT is False
        assert s.USE_LLM_AGENT is False
        assert s.USE_DL_PREDICTION is False
        assert s.USE_RL_AGENT is False

    def test_risk_defaults(self) -> None:
        s = Settings()
        assert s.MAX_RISK_PER_TRADE_PCT == 0.20
        assert s.MAX_POSITION_PCT == 25.0
        assert s.MAX_DRAWDOWN_PCT == 15.0
        assert s.MAX_DAILY_LOSS_PCT == 3.0


class TestSignalWeightValidation:
    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, SENTIMENT_WEIGHT=-0.1)  # type: ignore[call-arg]

    def test_zero_technical_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, TECHNICAL_WEIGHT=0.0)  # type: ignore[call-arg]

    def test_threshold_must_be_a_probability(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, MIN_CONFIDENCE_THRESHOLD=1.5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# .env.example stays in sync with Settings (audit: drift found 2026-11-05)
# ---------------------------------------------------------------------------

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
# Read by docker-compose, not by Settings.
COMPOSE_ONLY = {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"}
# Documented on purpose with a value other than the default.
EXAMPLE_OVERRIDES = {"LLM_MODEL"}  # pinned snapshot of the default alias


def _env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=([^#]*)", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def test_env_example_has_no_unknown_keys() -> None:
    unknown = set(_env_example()) - set(Settings.model_fields) - COMPOSE_ONLY
    assert not unknown, f"keys Settings ignores (extra='ignore'): {sorted(unknown)}"


def test_env_example_matches_the_defaults() -> None:
    """Copying .env.example (README Setup) must not override calibrated defaults."""
    for key, raw in _env_example().items():
        if key in COMPOSE_ONLY | EXAMPLE_OVERRIDES or raw == "":
            continue
        expected = Settings(_env_file=None).model_dump()[key]  # type: ignore[call-arg]
        parsed = Settings(_env_file=None, **{key: raw}).model_dump()[key]  # type: ignore[call-arg]
        assert parsed == expected, f"{key}: .env.example={raw!r}, default={expected!r}"


@pytest.mark.parametrize(
    "key",
    [
        "MAX_RISK_PER_TRADE_PCT",
        "MAX_POSITION_PCT",
        "MAX_DRAWDOWN_PCT",
        "MAX_DAILY_LOSS_PCT",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, 150.0])
def test_risk_percentages_are_validated(key: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{key: value})  # type: ignore[call-arg]
