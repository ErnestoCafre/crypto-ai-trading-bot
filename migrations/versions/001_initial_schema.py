"""Initial database schema.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-09-16 16:18:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. ohlcv_data
    op.create_table(
        "ohlcv_data",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("high", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("low", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("close", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("volume", sa.Numeric(precision=30, scale=8), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "exchange",
            "pair",
            "timeframe",
            "timestamp",
            name="uq_ohlcv_exchange_pair_tf_ts",
        ),
    )
    op.create_index(
        "ix_ohlcv_pair_tf_ts", "ohlcv_data", ["pair", "timeframe", "timestamp"]
    )

    # 2. orders
    op.create_table(
        "orders",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("exchange_order_id", sa.String(length=100), nullable=True),
        sa.Column("trade_id", sa.Uuid(), nullable=True),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("order_type", sa.String(length=20), nullable=False),
        sa.Column("price", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("filled_price", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("filled_quantity", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("fee", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("fee_currency", sa.String(length=10), nullable=True),
        sa.Column("slippage_pct", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("signal_source", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_orders_pair_status_created", "orders", ["pair", "status", "created_at"]
    )
    op.create_index("ix_orders_trade_id", "orders", ["trade_id"])
    op.create_index("ix_orders_strategy_created", "orders", ["strategy", "created_at"])

    # 3. trades
    op.create_table(
        "trades",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=5), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("entry_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_price", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_reason", sa.String(length=30), nullable=True),
        sa.Column("quantity", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("pnl", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("pnl_pct", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("total_fees", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column(
            "total_slippage_pct", sa.Numeric(precision=10, scale=6), nullable=True
        ),
        sa.Column(
            "holding_time_hours", sa.Numeric(precision=10, scale=2), nullable=True
        ),
        sa.Column(
            "ai_sentiment_score", sa.Numeric(precision=5, scale=4), nullable=True
        ),
        sa.Column(
            "ai_prediction_score", sa.Numeric(precision=5, scale=4), nullable=True
        ),
        sa.Column("ai_llm_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column(
            "ai_combined_confidence", sa.Numeric(precision=5, scale=4), nullable=True
        ),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trades_pair_status", "trades", ["pair", "status"])
    op.create_index("ix_trades_strategy_created", "trades", ["strategy", "created_at"])
    op.create_index("ix_trades_status", "trades", ["status"])

    # 4. portfolio_snapshots
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_value_usd", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column(
            "cash_balance_usd", sa.Numeric(precision=20, scale=8), nullable=False
        ),
        sa.Column(
            "positions_value_usd", sa.Numeric(precision=20, scale=8), nullable=False
        ),
        sa.Column("unrealized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column(
            "realized_pnl_day", sa.Numeric(precision=20, scale=8), nullable=False
        ),
        sa.Column("drawdown_pct", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column(
            "max_drawdown_pct", sa.Numeric(precision=10, scale=6), nullable=False
        ),
        sa.Column("var_95", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("cvar_95", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("open_positions_count", sa.Integer(), nullable=False),
        sa.Column("snapshot_trigger", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_portfolio_timestamp", "portfolio_snapshots", ["timestamp"])

    # 5. sentiment_history
    op.create_table(
        "sentiment_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=30), nullable=False),
        sa.Column("model_used", sa.String(length=50), nullable=False),
        sa.Column("score", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("aggregated_score", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_sentiment_pair_analyzed", "sentiment_history", ["pair", "analyzed_at"]
    )
    op.create_index(
        "ix_sentiment_source_analyzed", "sentiment_history", ["source", "analyzed_at"]
    )

    # 6. predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("model_type", sa.String(length=30), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column(
            "predicted_return", sa.Numeric(precision=10, scale=6), nullable=False
        ),
        sa.Column("predicted_direction", sa.String(length=4), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("actual_return", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("was_correct", sa.Boolean(), nullable=True),
        sa.Column(
            "features_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_predictions_pair_model_at",
        "predictions",
        ["pair", "model_type", "predicted_at"],
    )
    op.create_index(
        "ix_predictions_model_correct", "predictions", ["model_type", "was_correct"]
    )

    # 7. llm_analysis_log
    op.create_table(
        "llm_analysis_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("llm_provider", sa.String(length=20), nullable=False),
        sa.Column("llm_model", sa.String(length=50), nullable=False),
        sa.Column(
            "technical_analysis",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "sentiment_analysis",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "risk_evaluation", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "orchestrator_decision",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("final_recommendation", sa.String(length=10), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_pair_analyzed", "llm_analysis_log", ["pair", "analyzed_at"])
    op.create_index("ix_llm_analyzed", "llm_analysis_log", ["analyzed_at"])

    # 8. model_metrics
    op.create_table(
        "model_metrics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("model_type", sa.String(length=30), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("pair", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("val_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("val_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "accuracy_directional", sa.Numeric(precision=5, scale=4), nullable=True
        ),
        sa.Column("sharpe_ratio", sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column("mse", sa.Numeric(precision=15, scale=8), nullable=True),
        sa.Column("mae", sa.Numeric(precision=15, scale=8), nullable=True),
        sa.Column(
            "hyperparameters", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "feature_importance", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("model_path", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_name", "version", name="uq_model_name_version"),
    )
    op.create_index(
        "ix_model_type_pair_active",
        "model_metrics",
        ["model_type", "pair", "is_active"],
    )


def downgrade() -> None:
    op.drop_table("model_metrics")
    op.drop_table("llm_analysis_log")
    op.drop_table("predictions")
    op.drop_table("sentiment_history")
    op.drop_table("portfolio_snapshots")
    op.drop_table("trades")
    op.drop_table("orders")
    op.drop_table("ohlcv_data")
