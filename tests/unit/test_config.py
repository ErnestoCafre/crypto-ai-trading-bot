"""Tests for config.settings."""

from __future__ import annotations

from config.settings import Settings


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

    def test_ai_flags_default_disabled(self) -> None:
        s = Settings()
        assert s.USE_SENTIMENT is False
        assert s.USE_LLM_AGENT is False
        assert s.USE_DL_PREDICTION is False
        assert s.USE_RL_AGENT is False

    def test_risk_defaults(self) -> None:
        s = Settings()
        assert s.MAX_RISK_PER_TRADE_PCT == 1.0
        assert s.MAX_DRAWDOWN_PCT == 15.0
        assert s.MAX_DAILY_LOSS_PCT == 3.0
