"""Domain models — pure dataclasses, no ORM coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    NEUTRAL = "neutral"


@dataclass
class Signal:
    type: SignalType
    pair: str
    confidence: float
    strength: float
    source: str
    metadata: dict[str, object]
    timestamp: datetime


@dataclass
class AISignals:
    sentiment_score: float | None = None
    prediction_return: float | None = None
    prediction_confidence: float | None = None
    llm_recommendation: str | None = None
    llm_confidence: float | None = None


@dataclass
class CombinedSignal:
    type: SignalType
    pair: str
    combined_confidence: float
    technical_signal: Signal | None
    ai_signals: AISignals | None
    weights_used: dict[str, float]
    should_trade: bool
    recommended_sizing_pct: float
    reasoning: str
    timestamp: datetime


@dataclass
class SentimentResult:
    score: float
    confidence: float
    label: str
    model_used: str
    analyzed_text: str


@dataclass
class AggregatedSentiment:
    pair: str
    overall_score: float
    news_score: float
    social_score: float
    fear_greed_score: float
    num_sources_analyzed: int
    time_window: str
    timestamp: datetime


@dataclass
class Prediction:
    pair: str
    timeframe: str
    predicted_return: float
    predicted_direction: str
    confidence: float
    model_type: str
    model_version: str
    features_used: list[str]
    timestamp: datetime


@dataclass
class AgentAnalysis:
    agent_name: str
    recommendation: str
    confidence: float
    reasoning: str
    key_factors: list[str]
    raw_response: str
    tokens_used: int
    latency_ms: int


@dataclass
class OrchestratorDecision:
    pair: str
    final_recommendation: str
    confidence: float
    agent_analyses: list[AgentAnalysis]
    consensus: bool
    reasoning: str
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: int
    timestamp: datetime


@dataclass
class PositionSize:
    pair: str
    quantity: float
    value_usd: float
    risk_pct: float
    sizing_method: str
    ai_adjustment: float
    stop_loss_price: float
    stop_loss_pct: float


@dataclass
class OrderResult:
    order_id: str
    exchange_order_id: str | None
    pair: str
    side: str
    filled_price: float
    filled_quantity: float
    fee: float
    slippage_pct: float
    status: str
    timestamp: datetime


@dataclass
class NewsArticle:
    title: str
    body: str
    source: str
    url: str
    published_at: datetime
    categories: list[str] = field(default_factory=list)
    related_pairs: list[str] = field(default_factory=list)
