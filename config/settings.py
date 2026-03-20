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

    # --- Regime Detection ---
    USE_REGIME_DETECTION: bool = True
    REGIME_ADX_TRENDING: float = 25.0
    REGIME_VOLATILITY_HIGH_PCT: float = 80.0
    REGIME_VOLATILE_SIZING_FACTOR: float = 0.5

    # --- Backtesting ---
    BACKTEST_INITIAL_CAPITAL: float = 100_000.0
    BACKTEST_FEE_PCT: float = 0.1
    BACKTEST_SLIPPAGE_PCT: float = 0.05
    BACKTEST_WALK_FORWARD_SPLITS: int = 5
    BACKTEST_WALK_FORWARD_TRAIN_RATIO: float = 0.8
    OPTUNA_N_TRIALS: int = 100

    # --- Ensemble ML ---
    USE_ENSEMBLE: bool = False
    ENSEMBLE_MODELS: list[str] = Field(default=["lightgbm"])
    ML_LABEL_HORIZON: int = 1
    ML_LABEL_THRESHOLD: float = 0.001
    ML_TRAIN_RATIO: float = 0.7
    ML_VAL_RATIO: float = 0.15
    OPTUNA_ML_TRIALS: int = 50

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
