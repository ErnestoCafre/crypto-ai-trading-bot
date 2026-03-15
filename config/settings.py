"""Pydantic settings — loads from .env file."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Execution mode ---
    BOT_MODE: str = "paper"
    LOG_LEVEL: str = "INFO"

    # --- Database ---
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/trading_bot"
    DATABASE_ECHO: bool = False

    # --- Exchange ---
    EXCHANGE_ID: str = "binance"
    EXCHANGE_API_KEY: str = ""
    EXCHANGE_API_SECRET: str = ""
    EXCHANGE_SANDBOX: bool = True

    # --- Trading ---
    TRADING_PAIRS: list[str] = Field(default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    # --- Risk Management ---
    MAX_RISK_PER_TRADE_PCT: float = 1.0
    MAX_POSITION_PCT: float = 10.0
    MAX_DRAWDOWN_PCT: float = 15.0
    MAX_DAILY_LOSS_PCT: float = 3.0

    # --- Telegram ---
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # --- AI Feature Flags ---
    USE_SENTIMENT: bool = False
    USE_LLM_AGENT: bool = False
    USE_DL_PREDICTION: bool = False
    USE_RL_AGENT: bool = False

    # --- Sentiment ---
    SENTIMENT_MODEL: str = "kk08/CryptoBERT"
    SENTIMENT_WEIGHT: float = 0.2
    SENTIMENT_INTERVAL_MIN: int = 15

    # --- LLM Agent ---
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_WEIGHT: float = 0.2
    LLM_ANALYSIS_INTERVAL_MIN: int = 15
    LLM_MAX_DAILY_COST_USD: float = 5.0

    # --- API Keys LLM ---
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- Deep Learning ---
    DL_MODEL_TYPE: str = "gru"
    DL_WEIGHT: float = 0.2
    DL_RETRAIN_DAYS: int = 14
    DL_LOOKBACK_BARS: int = 168

    # --- RL ---
    RL_ALGORITHM: str = "PPO"

    # --- Signal Combiner ---
    TECHNICAL_WEIGHT: float = 0.4
    MIN_CONFIDENCE_THRESHOLD: float = 0.5

    # --- Data APIs ---
    COINGECKO_API_KEY: str = ""
    CRYPTOCOMPARE_API_KEY: str = ""

    # --- Social APIs ---
    TWITTER_BEARER_TOKEN: str = ""
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "trading-bot/1.0"

    # --- Infrastructure ---
    PROMETHEUS_ENABLED: bool = False

    @field_validator("TRADING_PAIRS", mode="before")
    @classmethod
    def parse_trading_pairs(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [p.strip() for p in v.split(",")]
        return v


def get_settings() -> Settings:
    return Settings()
