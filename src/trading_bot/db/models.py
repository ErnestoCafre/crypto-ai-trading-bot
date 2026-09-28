"""ORM models — 8 PostgreSQL tables."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from trading_bot.db.base import Base

JSONVariant = JSON().with_variant(JSONB, "postgresql")
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


# ---------------------------------------------------------------------------
# 1. ohlcv_data — Market data cache
# ---------------------------------------------------------------------------


class OHLCVData(Base):
    __tablename__ = "ohlcv_data"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(50))
    pair: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(10))
    timestamp: Mapped[datetime] = mapped_column()
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8))

    __table_args__ = (
        UniqueConstraint("exchange", "pair", "timeframe", "timestamp"),
        Index("ix_ohlcv_pair_tf_ts", "pair", "timeframe", "timestamp"),
    )


# ---------------------------------------------------------------------------
# 2. orders — Orders sent to exchange
# ---------------------------------------------------------------------------


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(100))
    trade_id: Mapped[uuid.UUID | None] = mapped_column()
    exchange: Mapped[str] = mapped_column(String(50))
    pair: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(20))
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    filled_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    filled_quantity: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    fee: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    fee_currency: Mapped[str | None] = mapped_column(String(10))
    slippage_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    status: Mapped[str] = mapped_column(String(20))
    strategy: Mapped[str] = mapped_column(String(50))
    signal_source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column()

    __table_args__ = (
        Index("ix_orders_pair_status_created", "pair", "status", "created_at"),
        Index("ix_orders_trade_id", "trade_id"),
        Index("ix_orders_strategy_created", "strategy", "created_at"),
    )


# ---------------------------------------------------------------------------
# 3. trades — Completed trades (entry + exit)
# ---------------------------------------------------------------------------


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(default=uuid.uuid4, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(50))
    pair: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(5))
    strategy: Mapped[str] = mapped_column(String(50))
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    entry_time: Mapped[datetime] = mapped_column()
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    exit_time: Mapped[datetime | None] = mapped_column()
    exit_reason: Mapped[str | None] = mapped_column(String(30))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    pnl_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    total_fees: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    total_slippage_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    holding_time_hours: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    ai_sentiment_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ai_prediction_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ai_llm_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    ai_combined_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    status: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column()

    __table_args__ = (
        Index("ix_trades_pair_status", "pair", "status"),
        Index("ix_trades_strategy_created", "strategy", "created_at"),
        Index("ix_trades_status", "status"),
    )


# ---------------------------------------------------------------------------
# 4. portfolio_snapshots — Portfolio state over time
# ---------------------------------------------------------------------------


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column()
    total_value_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    cash_balance_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    positions_value_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    realized_pnl_day: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    drawdown_pct: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    max_drawdown_pct: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    var_95: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    cvar_95: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    open_positions_count: Mapped[int] = mapped_column(Integer)
    snapshot_trigger: Mapped[str] = mapped_column(String(20))

    __table_args__ = (Index("ix_portfolio_timestamp", "timestamp"),)


# ---------------------------------------------------------------------------
# 5. sentiment_history — Sentiment analysis history
# ---------------------------------------------------------------------------


class SentimentHistory(Base):
    __tablename__ = "sentiment_history"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(30))
    model_used: Mapped[str] = mapped_column(String(50))
    score: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    aggregated_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    raw_text: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONVariant)
    analyzed_at: Mapped[datetime] = mapped_column()

    __table_args__ = (
        Index("ix_sentiment_pair_analyzed", "pair", "analyzed_at"),
        Index("ix_sentiment_source_analyzed", "source", "analyzed_at"),
    )


# ---------------------------------------------------------------------------
# 6. predictions — ML/DL model predictions
# ---------------------------------------------------------------------------


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(10))
    model_type: Mapped[str] = mapped_column(String(30))
    model_version: Mapped[str] = mapped_column(String(50))
    predicted_return: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    predicted_direction: Mapped[str] = mapped_column(String(4))
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    actual_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    was_correct: Mapped[bool | None] = mapped_column(Boolean)
    features_snapshot: Mapped[dict | None] = mapped_column(JSONVariant)
    predicted_at: Mapped[datetime] = mapped_column()
    evaluated_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        Index("ix_predictions_pair_model_at", "pair", "model_type", "predicted_at"),
        Index("ix_predictions_model_correct", "model_type", "was_correct"),
    )


# ---------------------------------------------------------------------------
# 7. llm_analysis_log — LLM agent analysis log
# ---------------------------------------------------------------------------


class LLMAnalysisLog(Base):
    __tablename__ = "llm_analysis_log"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(20))
    llm_provider: Mapped[str] = mapped_column(String(20))
    llm_model: Mapped[str] = mapped_column(String(50))
    technical_analysis: Mapped[dict] = mapped_column(JSONVariant)
    sentiment_analysis: Mapped[dict] = mapped_column(JSONVariant)
    risk_evaluation: Mapped[dict] = mapped_column(JSONVariant)
    orchestrator_decision: Mapped[dict] = mapped_column(JSONVariant)
    final_recommendation: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    reasoning: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    latency_ms: Mapped[int] = mapped_column(Integer)
    analyzed_at: Mapped[datetime] = mapped_column()

    __table_args__ = (
        Index("ix_llm_pair_analyzed", "pair", "analyzed_at"),
        Index("ix_llm_analyzed", "analyzed_at"),
    )


# ---------------------------------------------------------------------------
# 8. model_metrics — Trained model performance registry
# ---------------------------------------------------------------------------


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50))
    model_type: Mapped[str] = mapped_column(String(30))
    version: Mapped[str] = mapped_column(String(50))
    pair: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(10))
    train_start: Mapped[datetime] = mapped_column()
    train_end: Mapped[datetime] = mapped_column()
    val_start: Mapped[datetime] = mapped_column()
    val_end: Mapped[datetime] = mapped_column()
    accuracy_directional: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    mse: Mapped[Decimal | None] = mapped_column(Numeric(15, 8))
    mae: Mapped[Decimal | None] = mapped_column(Numeric(15, 8))
    hyperparameters: Mapped[dict | None] = mapped_column(JSONVariant)
    feature_importance: Mapped[dict | None] = mapped_column(JSONVariant)
    model_path: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean)
    trained_at: Mapped[datetime] = mapped_column()
    deactivated_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        Index("ix_model_type_pair_active", "model_type", "pair", "is_active"),
        UniqueConstraint("model_name", "version"),
    )
