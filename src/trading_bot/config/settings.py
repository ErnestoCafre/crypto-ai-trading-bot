"""Pydantic settings — loads from .env file."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    # Persist fills, trades and per-cycle snapshots (read by the dashboard).
    # Best effort: an unreachable DB only logs a warning; trading continues.
    DB_PERSISTENCE: bool = True
    DB_TIMEOUT_S: float = 5.0

    # --- Exchange ---
    EXCHANGE_ID: str = "binance"
    EXCHANGE_API_KEY: str = ""
    EXCHANGE_API_SECRET: str = ""
    EXCHANGE_SANDBOX: bool = True

    # --- Trading ---
    # NoDecode: pydantic-settings would otherwise JSON-decode list fields read
    # from the environment and reject the comma-separated form in .env.example.
    TRADING_PAIRS: Annotated[list[str], NoDecode] = Field(
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )

    # --- Strategy mode ---
    # "regime_selector": TrendFollowing / MeanReversion per regime on
    # TRADING_TIMEFRAME, discrete entries with ATR stops. "donchian_ensemble":
    # target exposure from 9 Donchian channels on daily candles, rebalanced
    # with partial buys/sells (README "Modo donchian_ensemble"). In that mode
    # MAX_POSITION_PCT is each pair's allocation: 25 % x 3 pairs caps total
    # exposure at 75 %; 33.33 is the equal-weight portfolio that was tested.
    STRATEGY_MODE: Literal["regime_selector", "donchian_ensemble"] = "regime_selector"
    DONCHIAN_LOOKBACKS: Annotated[list[int], NoDecode] = Field(
        default=[5, 10, 20, 30, 60, 90, 150, 250, 360]
    )
    # Annualised volatility target per pair: w = raw x min(1, target / vol).
    DONCHIAN_VOL_TARGET: float = Field(default=0.25, gt=0.0)
    DONCHIAN_VOL_WINDOW: int = Field(default=90, ge=2)
    # Trade only when |target - held| exceeds this exposure (fraction of the
    # pair's allocation); a target of 0 always closes. In the backtest 0.05
    # cut fills 3-5x vs 0 at a similar Sharpe; 0.10 lowered it (README).
    REBALANCE_THRESHOLD: float = Field(default=0.05, ge=0.0, lt=1.0)

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
    # Sizing: quantity = min(risk budget / stop distance, position cap). The
    # risk budget should set the size and the cap only act as a backstop:
    # with 1 % / 10 % the cap set ~100 % of backtested 1h entries (real risk
    # ~0.18 %, no volatility adjustment). See README "Tamaño de posición".
    # donchian_ensemble: MAX_POSITION_PCT is each pair's allocation instead
    # and MAX_RISK_PER_TRADE_PCT is unused.
    MAX_RISK_PER_TRADE_PCT: float = 0.20
    MAX_POSITION_PCT: float = 25.0
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
    # "cryptobert": local transformer (SENTIMENT_MODEL). "llm": LLM_PROVIDER /
    # LLM_MODEL, asynchronous, charged to LLM_MAX_DAILY_COST_USD (shared with
    # the LLM agents); each article is scored once (per-article cache).
    SENTIMENT_PROVIDER: Literal["cryptobert", "llm"] = "cryptobert"
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
    # News for sentiment. "coindesk": CoinDesk Data API (COINDESK_API_KEY);
    # "cryptocompare": legacy min-api (CRYPTOCOMPARE_API_KEY). Both require a
    # key; without one only the news source is disabled (Fear & Greed stays).
    NEWS_PROVIDER: Literal["coindesk", "cryptocompare"] = "coindesk"
    COINDESK_API_KEY: str = ""
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
    # MeanReversion LONG entries in RANGING regime. Off: backtested on Binance
    # 1h 2024-2026 they lowered return and raised max drawdown on BTC, ETH and
    # SOL (README "Estrategias"). Its CLOSE/SHORT exits stay active either way.
    MEAN_REVERSION_ENTRIES: bool = False

    # --- Backtesting ---
    BACKTEST_INITIAL_CAPITAL: float = 100_000.0
    BACKTEST_FEE_PCT: float = 0.1
    BACKTEST_SLIPPAGE_PCT: float = 0.05
    BACKTEST_WALK_FORWARD_SPLITS: int = 5
    # Warm-up share before the first OOS fold (scripts/run_backtest.py default).
    BACKTEST_WALK_FORWARD_TRAIN_RATIO: float = 0.6
    OPTUNA_N_TRIALS: int = 100

    # --- ML prediction ---
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

    @field_validator("TRADING_PAIRS", "DONCHIAN_LOOKBACKS", mode="before")
    @classmethod
    def parse_str_list(cls, v: str | list[str]) -> list[str]:
        """Accept a comma-separated string or a JSON array for list settings."""
        if isinstance(v, str):
            if v.strip().startswith("["):
                return list(json.loads(v))
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("DONCHIAN_LOOKBACKS")
    @classmethod
    def check_lookbacks(cls, v: list[int]) -> list[int]:
        if not v or any(n < 2 for n in v):
            raise ValueError("DONCHIAN_LOOKBACKS needs lookbacks >= 2")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read from the environment and .env once.

    Call ``get_settings.cache_clear()`` after changing the environment.
    """
    return Settings()
