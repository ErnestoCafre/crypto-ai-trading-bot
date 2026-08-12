"""Tests for LLM multi-agent system (agents + orchestrator)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from ai.agents.orchestrator import Orchestrator
from ai.agents.risk_evaluator import RiskEvaluatorAgent
from ai.agents.sentiment_analyst import SentimentAnalystAgent
from ai.agents.technical_analyst import TechnicalAnalystAgent
from ai.sentiment.cost_tracker import SentimentCostTracker
from core.models import AgentAnalysis, OrchestratorDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_json(**overrides) -> str:
    """Build a valid agent JSON response."""
    data = {
        "recommendation": "LONG",
        "confidence": 0.85,
        "reasoning": "Strong bullish momentum.",
        "key_factors": ["RSI above 60", "MACD crossover"],
    }
    data.update(overrides)
    return json.dumps(data)


def _make_analysis(
    name: str = "test_agent",
    recommendation: str = "LONG",
    confidence: float = 0.8,
) -> AgentAnalysis:
    return AgentAnalysis(
        agent_name=name,
        recommendation=recommendation,
        confidence=confidence,
        reasoning="test",
        key_factors=["f1"],
        raw_response="{}",
        tokens_used=100,
        latency_ms=50,
    )


def _make_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
    )


def _make_settings(**overrides):
    defaults = {
        "BOT_MODE": "paper",
        "EXCHANGE_ID": "binance",
        "EXCHANGE_API_KEY": "",
        "EXCHANGE_API_SECRET": "",
        "EXCHANGE_SANDBOX": True,
        "TRADING_PAIRS": ["BTC/USDT"],
        "TRADING_TIMEFRAME": "1h",
        "TRADING_INTERVAL_MINUTES": 60,
        "OHLCV_LOOKBACK_BARS": 250,
        "INITIAL_BALANCE": 100_000.0,
        "PAPER_SLIPPAGE_PCT": 0.05,
        "PAPER_FEE_PCT": 0.1,
        "MAX_RISK_PER_TRADE_PCT": 1.0,
        "MAX_POSITION_PCT": 10.0,
        "MAX_DRAWDOWN_PCT": 15.0,
        "MAX_DAILY_LOSS_PCT": 3.0,
        "USE_SENTIMENT": False,
        "USE_LLM_AGENT": False,
        "USE_DL_PREDICTION": False,
        "USE_RL_AGENT": False,
        "USE_ENSEMBLE": False,
        "USE_REGIME_DETECTION": True,
        "REGIME_ADX_TRENDING": 25.0,
        "REGIME_VOLATILITY_HIGH_PCT": 80.0,
        "REGIME_VOLATILE_SIZING_FACTOR": 0.5,
        "SENTIMENT_MODEL": "kk08/CryptoBERT",
        "SENTIMENT_INTERVAL_MIN": 15,
        "SENTIMENT_WEIGHT": 0.2,
        "CRYPTOCOMPARE_API_KEY": "",
        "TECHNICAL_WEIGHT": 0.4,
        "MIN_CONFIDENCE_THRESHOLD": 0.5,
        "ENSEMBLE_MODELS": ["lightgbm"],
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-haiku-4-5-20251001",
        "LLM_WEIGHT": 0.2,
        "LLM_ANALYSIS_INTERVAL_MIN": 15,
        "LLM_MAX_DAILY_COST_USD": 5.0,
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "",
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


# ===========================================================================
# Group 1: TechnicalAnalystAgent
# ===========================================================================


class TestTechnicalAnalystAgent:
    def test_technical_name(self):
        agent = TechnicalAnalystAgent()
        assert agent.name == "technical_analyst"

    def test_technical_builds_prompt(self):
        agent = TechnicalAnalystAgent()
        context = {
            "pair": "BTC/USDT",
            "timeframe": "1h",
            "indicators": {"rsi": 65.3, "macd": 120.5},
            "regime": "TRENDING",
        }
        prompt = agent._build_prompt(context)
        assert "BTC/USDT" in prompt
        assert "rsi: 65.3" in prompt
        assert "macd: 120.5" in prompt
        assert "TRENDING" in prompt

    @pytest.mark.asyncio
    async def test_technical_parses_valid_json(self):
        agent = TechnicalAnalystAgent()
        agent._call_llm_async = AsyncMock(return_value=_valid_json())

        result = await agent.analyze({"pair": "BTC/USDT", "timeframe": "1h"})

        assert isinstance(result, AgentAnalysis)
        assert result.recommendation == "LONG"
        assert result.confidence == 0.85
        assert result.agent_name == "technical_analyst"
        assert len(result.key_factors) == 2

    @pytest.mark.asyncio
    async def test_technical_handles_malformed_json(self):
        agent = TechnicalAnalystAgent()
        agent._call_llm_async = AsyncMock(return_value="not valid json {{{")

        result = await agent.analyze({"pair": "BTC/USDT"})

        assert result.recommendation == "NEUTRAL"
        assert result.confidence == 0.0
        assert result.reasoning == "Parse error"


# ===========================================================================
# Group 2: SentimentAnalystAgent
# ===========================================================================


class TestSentimentAnalystAgent:
    def test_sentiment_name(self):
        agent = SentimentAnalystAgent()
        assert agent.name == "sentiment_analyst"

    def test_sentiment_builds_prompt_with_headlines(self):
        agent = SentimentAnalystAgent()
        context = {
            "pair": "ETH/USDT",
            "sentiment_score": 0.6,
            "fear_greed": 72,
            "headlines": ["ETH hits new high", "DeFi surges"],
        }
        prompt = agent._build_prompt(context)
        assert "ETH/USDT" in prompt
        assert "1. ETH hits new high" in prompt
        assert "2. DeFi surges" in prompt
        assert "0.6" in prompt

    def test_sentiment_empty_headlines(self):
        agent = SentimentAnalystAgent()
        context = {
            "pair": "BTC/USDT",
            "sentiment_score": 0.0,
            "fear_greed": "N/A",
            "headlines": [],
        }
        prompt = agent._build_prompt(context)
        assert "No recent headlines available" in prompt

    @pytest.mark.asyncio
    async def test_sentiment_parses_valid_response(self):
        agent = SentimentAnalystAgent()
        agent._call_llm_async = AsyncMock(
            return_value=_valid_json(recommendation="SHORT", confidence=0.7)
        )

        result = await agent.analyze({"pair": "BTC/USDT"})

        assert result.recommendation == "SHORT"
        assert result.confidence == 0.7
        assert result.agent_name == "sentiment_analyst"


# ===========================================================================
# Group 3: RiskEvaluatorAgent
# ===========================================================================


class TestRiskEvaluatorAgent:
    def test_risk_name(self):
        agent = RiskEvaluatorAgent()
        assert agent.name == "risk_evaluator"

    def test_risk_builds_prompt(self):
        agent = RiskEvaluatorAgent()
        context = {
            "pair": "SOL/USDT",
            "portfolio_value": 100_000.0,
            "drawdown_pct": 5.2,
            "open_positions": 3,
            "max_risk_pct": 1.0,
            "max_position_pct": 10.0,
            "max_drawdown_pct": 15.0,
        }
        prompt = agent._build_prompt(context)
        assert "SOL/USDT" in prompt
        assert "$100,000.00" in prompt
        assert "5.2%" in prompt
        assert "3" in prompt

    @pytest.mark.asyncio
    async def test_risk_parses_valid_response(self):
        agent = RiskEvaluatorAgent()
        agent._call_llm_async = AsyncMock(
            return_value=_valid_json(recommendation="NEUTRAL", confidence=0.9)
        )

        result = await agent.analyze({"pair": "BTC/USDT"})

        assert result.recommendation == "NEUTRAL"
        assert result.confidence == 0.9
        assert result.agent_name == "risk_evaluator"


# ===========================================================================
# Group 4: Shared parsing
# ===========================================================================


class TestSharedParsing:
    @pytest.mark.asyncio
    async def test_clamps_confidence(self):
        agent = TechnicalAnalystAgent()
        # confidence > 1
        agent._call_llm_async = AsyncMock(return_value=_valid_json(confidence=5.0))
        result = await agent.analyze({"pair": "BTC/USDT"})
        assert result.confidence == 1.0

        # confidence < 0
        agent._call_llm_async = AsyncMock(return_value=_valid_json(confidence=-0.5))
        result = await agent.analyze({"pair": "BTC/USDT"})
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_normalizes_recommendation(self):
        agent = TechnicalAnalystAgent()
        # lowercase → LONG
        agent._call_llm_async = AsyncMock(
            return_value=_valid_json(recommendation="long")
        )
        result = await agent.analyze({"pair": "BTC/USDT"})
        assert result.recommendation == "LONG"

        # unknown → NEUTRAL
        agent._call_llm_async = AsyncMock(
            return_value=_valid_json(recommendation="BUY_NOW")
        )
        result = await agent.analyze({"pair": "BTC/USDT"})
        assert result.recommendation == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_strips_markdown_code_blocks(self):
        agent = TechnicalAnalystAgent()
        wrapped = "```json\n" + _valid_json() + "\n```"
        agent._call_llm_async = AsyncMock(return_value=wrapped)

        result = await agent.analyze({"pair": "BTC/USDT"})

        assert result.recommendation == "LONG"
        assert result.confidence == 0.85


# ===========================================================================
# Group 5: Orchestrator Consensus
# ===========================================================================


class TestOrchestratorConsensus:
    @pytest.mark.asyncio
    async def test_consensus_3_of_3(self):
        """All 3 agents agree → consensus=True, full confidence."""
        agents = []
        for name in ("a1", "a2", "a3"):
            agent = MagicMock()
            agent.name = name
            agent.analyze = AsyncMock(return_value=_make_analysis(name, "LONG", 0.8))
            agents.append(agent)

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator(agents, tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is not None
        assert result.final_recommendation == "LONG"
        assert result.consensus is True
        # Unanimous → avg_confidence * 1.0
        assert abs(result.confidence - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_consensus_2_of_3(self):
        """2 LONG + 1 SHORT → LONG with reduced confidence."""
        agents = []
        for name, rec in [("a1", "LONG"), ("a2", "LONG"), ("a3", "SHORT")]:
            agent = MagicMock()
            agent.name = name
            agent.analyze = AsyncMock(return_value=_make_analysis(name, rec, 0.8))
            agents.append(agent)

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator(agents, tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is not None
        assert result.final_recommendation == "LONG"
        assert result.consensus is True
        # 2/3 → avg_confidence * 0.75
        assert abs(result.confidence - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_no_agreement(self):
        """L + S + N → NEUTRAL, confidence=0, no consensus."""
        agents = []
        for name, rec in [("a1", "LONG"), ("a2", "SHORT"), ("a3", "NEUTRAL")]:
            agent = MagicMock()
            agent.name = name
            agent.analyze = AsyncMock(return_value=_make_analysis(name, rec, 0.8))
            agents.append(agent)

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator(agents, tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is not None
        assert result.final_recommendation == "NEUTRAL"
        assert result.confidence == 0.0
        assert result.consensus is False

    @pytest.mark.asyncio
    async def test_partial_failure_2_agents(self):
        """1 agent fails, 2 agree → consensus works."""
        good1 = MagicMock()
        good1.name = "g1"
        good1.analyze = AsyncMock(return_value=_make_analysis("g1", "LONG", 0.9))

        good2 = MagicMock()
        good2.name = "g2"
        good2.analyze = AsyncMock(return_value=_make_analysis("g2", "LONG", 0.7))

        bad = MagicMock()
        bad.name = "bad"
        bad.analyze = AsyncMock(side_effect=RuntimeError("LLM down"))

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator([good1, good2, bad], tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is not None
        assert result.final_recommendation == "LONG"
        assert result.consensus is True
        assert len(result.agent_analyses) == 2

    @pytest.mark.asyncio
    async def test_all_agents_fail(self):
        """All agents raise → returns None."""
        agents = []
        for name in ("a1", "a2", "a3"):
            agent = MagicMock()
            agent.name = name
            agent.analyze = AsyncMock(side_effect=RuntimeError("fail"))
            agents.append(agent)

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator(agents, tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is None


# ===========================================================================
# Group 6: Cost Control
# ===========================================================================


class TestCostControl:
    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_none(self):
        """Budget exceeded → orchestrator returns None without calling agents."""
        agent = MagicMock()
        agent.name = "a1"
        agent.analyze = AsyncMock(return_value=_make_analysis())

        tracker = SentimentCostTracker(max_daily_cost_usd=0.0)  # zero budget
        orch = Orchestrator([agent], tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is None
        agent.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_cost_recorded(self):
        """After successful run, cost is recorded in tracker."""
        agent = MagicMock()
        agent.name = "a1"
        agent.analyze = AsyncMock(return_value=_make_analysis())

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator([agent], tracker)

        await orch.run("BTC/USDT", {})

        assert tracker.daily_spent_usd > 0

    @pytest.mark.asyncio
    async def test_timeout_handled(self):
        """Agent exceeding timeout → None, other agents continue."""

        async def slow_analyze(context):
            await asyncio.sleep(60)  # way over the 15s limit
            return _make_analysis("slow")  # pragma: no cover

        slow_agent = MagicMock()
        slow_agent.name = "slow"
        slow_agent.analyze = slow_analyze

        fast_agent = MagicMock()
        fast_agent.name = "fast"
        fast_agent.analyze = AsyncMock(return_value=_make_analysis("fast", "LONG", 0.8))

        tracker = SentimentCostTracker(max_daily_cost_usd=10.0)
        orch = Orchestrator([slow_agent, fast_agent], tracker)

        result = await orch.run("BTC/USDT", {})

        assert result is not None
        assert len(result.agent_analyses) == 1
        assert result.agent_analyses[0].agent_name == "fast"


# ===========================================================================
# Group 7: Integration with TradingOrchestrator
# ===========================================================================


class TestBotIntegration:
    @pytest.mark.asyncio
    async def test_llm_cycle_caches_decision(self):
        """_run_llm_cycle populates _latest_llm_decision."""
        from core.bot import TradingOrchestrator

        s = _make_settings(USE_LLM_AGENT=True)
        orch = TradingOrchestrator(settings=s)

        decision = OrchestratorDecision(
            pair="BTC/USDT",
            final_recommendation="LONG",
            confidence=0.8,
            agent_analyses=[_make_analysis()],
            consensus=True,
            reasoning="test",
            total_tokens=300,
            total_cost_usd=0.001,
            total_latency_ms=100,
            timestamp=datetime.now(UTC),
        )

        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value=decision)
        orch._llm_orchestrator = mock_orch
        orch._fetcher = AsyncMock()
        orch._feature_engine = None  # skip indicator building

        await orch._run_llm_cycle()

        assert "BTC/USDT" in orch._latest_llm_decision
        assert orch._latest_llm_decision["BTC/USDT"].final_recommendation == "LONG"

    def test_llm_disabled_no_init(self):
        """USE_LLM_AGENT=False → no orchestrator."""
        from core.bot import TradingOrchestrator

        s = _make_settings(USE_LLM_AGENT=False)
        orch = TradingOrchestrator(settings=s)

        assert orch._llm_orchestrator is None
        assert orch._llm_cost_tracker is None

    def test_build_ai_signals_includes_llm(self):
        """Cached LLM decision flows into AISignals."""
        from core.bot import TradingOrchestrator

        s = _make_settings(USE_LLM_AGENT=True)
        orch = TradingOrchestrator(settings=s)

        orch._latest_llm_decision["BTC/USDT"] = OrchestratorDecision(
            pair="BTC/USDT",
            final_recommendation="SHORT",
            confidence=0.75,
            agent_analyses=[],
            consensus=True,
            reasoning="test",
            total_tokens=200,
            total_cost_usd=0.001,
            total_latency_ms=80,
            timestamp=datetime.now(UTC),
        )

        ohlcv = _make_ohlcv()
        result = orch._build_ai_signals("BTC/USDT", ohlcv)

        assert result is not None
        assert result.llm_recommendation == "SHORT"
        assert result.llm_confidence == 0.75
