"""Pydantic settings — loads from .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
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

    # --- Orchestrator ---
    TRADING_TIMEFRAME: str = "1h"
    TRADING_INTERVAL_MINUTES: int = 60
    OHLCV_LOOKBACK_BARS: int = 250
    # Bounded retries (exponential backoff + jitter) and a total timeout per
    # OHLCV fetch, so an exchange outage cannot stall the trading cycle.
    OHLCV_FETCH_MAX_RETRIES: int = 5
    OHLCV_FETCH_TIMEOUT_S: float = 60.0
    INITIAL_BALANCE: float = 100_000.0
    PAPER_SLIPPAGE_PCT: float = 0.05
    PAPER_FEE_PCT: float = 0.1

    # --- Risk Management ---
    MAX_RISK_PER_TRADE_PCT: float = 1.0
    MAX_POSITION_PCT: float = 10.0
    MAX_DRAWDOWN_PCT: float = 15.0
    MAX_DAILY_LOSS_PCT: float = 3.0
    # Persisted drawdown latch + stops. "{mode}" is replaced by BOT_MODE so a
    # paper session never overwrites the live halt state.
    RISK_STATE_PATH: str = "var/state/risk_state_{mode}.json"
    # Manual reset of a max-drawdown HALT on startup. Remove after use.
    RISK_RESET: bool = False

    # --- Telegram ---
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # --- AI Feature Flags ---
    USE_SENTIMENT: bool = False
    USE_LLM_AGENT: bool = False
    USE_DL_PREDICTION: bool = False
    USE_RL_AGENT: bool = False

    # --- Sentiment ---
    SENTIMENT_MODEL: str = "ElKulako/cryptobert"
    SENTIMENT_WEIGHT: float = Field(default=0.2, ge=0.0)
    SENTIMENT_INTERVAL_MIN: int = 15

    # --- LLM Agent ---
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-haiku-4-5"
    LLM_WEIGHT: float = Field(default=0.2, ge=0.0)
    LLM_ANALYSIS_INTERVAL_MIN: int = 15
    # Daily LLM budget, priced per model (ai/pricing.py). The day's spend
    # persists across restarts; "{mode}" is replaced by BOT_MODE.
    LLM_MAX_DAILY_COST_USD: float = 5.0
    LLM_COST_STATE_PATH: str = "var/state/llm_cost_{mode}.json"

    # --- API Keys LLM ---
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- Deep Learning ---
    # Only "lightgbm" is implemented; "gru"/"cnn_lstm" are stubs.
    DL_MODEL_TYPE: str = "lightgbm"
    DL_WEIGHT: float = Field(default=0.2, ge=0.0)
    DL_RETRAIN_DAYS: int = 14
    DL_LOOKBACK_BARS: int = 168

    # --- RL ---
    RL_ALGORITHM: str = "PPO"

    # --- Signal Combiner ---
    # Weights are renormalised over the sources present (see SignalCombiner).
    TECHNICAL_WEIGHT: float = Field(default=0.4, ge=0.0)
    MIN_CONFIDENCE_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    # Position risk multiplier when AI signals take part: AI_SIZING_FLOOR at
    # MIN_CONFIDENCE_THRESHOLD, 1.0 at full confidence (linear). AI never
    # raises risk above MAX_RISK_PER_TRADE_PCT. Technical-only: 1.0.
    AI_SIZING_FLOOR: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Data APIs ---
    COINGECKO_API_KEY: str = ""
    CRYPTOCOMPARE_API_KEY: str = ""

    # --- Social APIs ---
    TWITTER_BEARER_TOKEN: str = ""
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "trading-bot/1.0"

    # --- Regime Detection ---
    # False: always trend-following at full size (regime still classified).
    USE_REGIME_DETECTION: bool = True
    REGIME_ADX_TRENDING: float = 25.0
    REGIME_VOLATILITY_HIGH_PCT: float = 80.0
    # VOLATILE regime: risk per trade × this factor (confidence unchanged).
    REGIME_VOLATILE_SIZING_FACTOR: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Backtesting ---
    BACKTEST_INITIAL_CAPITAL: float = 100_000.0
    BACKTEST_FEE_PCT: float = 0.1
    BACKTEST_SLIPPAGE_PCT: float = 0.05
    BACKTEST_WALK_FORWARD_SPLITS: int = 5
    BACKTEST_WALK_FORWARD_TRAIN_RATIO: float = 0.8
    OPTUNA_N_TRIALS: int = 100

    # --- Ensemble ML ---
    # Not wired into live trading yet: enabling it only logs a warning and
    # the per-pair LightGBM models are used (see TradingOrchestrator).
    USE_ENSEMBLE: bool = False
    ENSEMBLE_MODELS: list[str] = Field(default=["lightgbm"])
    # Per-pair models: {ML_MODEL_DIR}/{BTC_USDT}_{timeframe}.pkl
    ML_MODEL_DIR: str = "var/models/prediction"
    ML_LABEL_HORIZON: int = 1
    ML_LABEL_THRESHOLD: float = 0.001
    ML_TRAIN_RATIO: float = 0.7
    ML_VAL_RATIO: float = 0.15
    OPTUNA_ML_TRIALS: int = 50

    # --- Infrastructure ---
    PROMETHEUS_ENABLED: bool = False

    @model_validator(mode="after")
    def check_signal_weights(self) -> Settings:
        if self.TECHNICAL_WEIGHT <= 0:
            raise ValueError("TECHNICAL_WEIGHT must be > 0")
        return self

    @field_validator("TRADING_PAIRS", mode="before")
    @classmethod
    def parse_trading_pairs(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [p.strip() for p in v.split(",")]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read from the environment and .env once.

    Call ``get_settings.cache_clear()`` after changing the environment.
    """
    return Settings()
