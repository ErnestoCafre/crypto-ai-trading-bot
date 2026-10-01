"""Tests for TradingOrchestrator (core.bot)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_bot.core.bot import TradingOrchestrator
from trading_bot.core.models import (
    AggregatedSentiment,
    AISignals,
    CombinedSignal,
    OrderResult,
    PositionSize,
    Signal,
    SignalType,
)
from trading_bot.execution.paper_executor import PaperExecutor
from trading_bot.risk.drawdown import TradingState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RISK_STATE_PATH = "unset"
_DECISION_LOG_PATH = "unset"


@pytest.fixture(autouse=True)
def _isolate_risk_state(tmp_path, monkeypatch):
    """Keep every orchestrator's state files inside the test's tmp_path."""
    monkeypatch.setattr(
        sys.modules[__name__],
        "_RISK_STATE_PATH",
        str(tmp_path / "risk_state_{mode}.json"),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_DECISION_LOG_PATH",
        str(tmp_path / "donchian_decisions_{mode}.jsonl"),
    )


def _make_settings(**overrides):
    """Build a Settings mock with defaults + overrides."""
    defaults = {
        "BOT_MODE": "paper",
        "EXCHANGE_ID": "binance",
        "EXCHANGE_API_KEY": "",
        "EXCHANGE_API_SECRET": "",
        "EXCHANGE_SANDBOX": True,
        "TRADING_PAIRS": ["BTC/USDT", "ETH/USDT"],
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
        "USE_REGIME_DETECTION": True,
        "REGIME_ADX_TRENDING": 25.0,
        "REGIME_VOLATILITY_HIGH_PCT": 80.0,
        "REGIME_VOLATILE_SIZING_FACTOR": 0.5,
        "SENTIMENT_MODEL": "kk08/CryptoBERT",
        "SENTIMENT_INTERVAL_MIN": 15,
        "SENTIMENT_WEIGHT": 0.2,
        "LLM_WEIGHT": 0.2,
        "DL_WEIGHT": 0.2,
        "NEWS_PROVIDER": "coindesk",
        "SENTIMENT_PROVIDER": "cryptobert",
        "DB_PERSISTENCE": False,
        "DB_TIMEOUT_S": 5.0,
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "COINDESK_API_KEY": "",
        "CRYPTOCOMPARE_API_KEY": "",
        "TECHNICAL_WEIGHT": 0.4,
        "MIN_CONFIDENCE_THRESHOLD": 0.5,
        "AI_SIZING_FLOOR": 0.25,
        "RISK_STATE_PATH": _RISK_STATE_PATH,
        "DONCHIAN_DECISION_LOG_PATH": _DECISION_LOG_PATH,
        "RISK_RESET": False,
        "LLM_ANALYSIS_INTERVAL_MIN": 15,
        "OHLCV_FETCH_MAX_RETRIES": 5,
        "OHLCV_FETCH_TIMEOUT_S": 60.0,
        "DL_MODEL_TYPE": "lightgbm",
        "ML_MODEL_DIR": "var/models/prediction",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _make_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 50000 + np.cumsum(rng.normal(0, 100, n))
    index = pd.date_range(end="2026-01-10", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        index=index,
        data={
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        },
    )


def _make_signal(
    pair: str = "BTC/USDT",
    signal_type: SignalType = SignalType.LONG,
    confidence: float = 0.75,
) -> Signal:
    return Signal(
        type=signal_type,
        pair=pair,
        confidence=confidence,
        strength=0.6,
        source="trend_following",
        metadata={},
        timestamp=datetime.now(UTC),
    )


def _make_combined(
    pair: str = "BTC/USDT",
    signal_type: SignalType = SignalType.LONG,
    confidence: float = 0.75,
    should_trade: bool = True,
    ai_signals: AISignals | None = None,
) -> CombinedSignal:
    return CombinedSignal(
        type=signal_type,
        pair=pair,
        combined_confidence=confidence,
        technical_signal=_make_signal(pair, signal_type, confidence),
        ai_signals=ai_signals,
        weights_used={"technical": 0.4},
        should_trade=should_trade,
        reasoning="test",
        timestamp=datetime.now(UTC),
    )


def _make_position_size(pair: str = "BTC/USDT", quantity: float = 0.01) -> PositionSize:
    return PositionSize(
        pair=pair,
        quantity=quantity,
        value_usd=500.0,
        risk_pct=1.0,
        sizing_method="atr_stop",
        ai_adjustment=1.0,
        stop_loss_price=49000.0,
        stop_loss_pct=2.0,
    )


def _make_order_result(pair: str = "BTC/USDT", side: str = "buy") -> OrderResult:
    return OrderResult(
        order_id="test-001",
        exchange_order_id=None,
        pair=pair,
        side=side,
        filled_price=50000.0,
        filled_quantity=0.01,
        fee=0.5,
        slippage_pct=0.01,
        status="filled",
        timestamp=datetime.now(UTC),
    )


def _setup_orchestrator(settings=None, **overrides) -> TradingOrchestrator:
    """Create an orchestrator with mocked internals for unit testing."""
    s = settings or _make_settings(**overrides)
    orch = TradingOrchestrator(settings=s)
    return orch


def _wire_mocks(orch: TradingOrchestrator, ohlcv=None):
    """Wire mock components into the orchestrator for testing."""
    from trading_bot.core.regime import MarketRegime, RegimeClassification

    ohlcv = ohlcv if ohlcv is not None else _make_ohlcv()

    orch._executor = AsyncMock()
    orch._executor.get_balance = AsyncMock(return_value={"total": 100_000.0})
    orch._executor.place_order = AsyncMock(return_value=_make_order_result())
    orch._executor.close_all_positions = AsyncMock(return_value=[])
    orch._executor.get_position = AsyncMock(return_value=0.0)
    orch._executor.update_prices = MagicMock()

    orch._fetcher = AsyncMock()
    orch._fetcher.fetch_ohlcv = AsyncMock(return_value=ohlcv)
    orch._fetcher.__aexit__ = AsyncMock()

    orch._feature_engine = MagicMock()

    # Strategy selectors for each pair
    regime_cls = RegimeClassification(
        regime=MarketRegime.TRENDING,
        confidence=0.7,
        adx_value=30.0,
        volatility_percentile=50.0,
        trend_strength=0.6,
        timestamp=datetime.now(UTC),
    )
    for pair in orch._settings.TRADING_PAIRS:
        selector_mock = MagicMock()
        selector_mock.generate_signal = MagicMock(
            return_value=(_make_signal(pair), regime_cls)
        )
        orch._strategy_selectors[pair] = selector_mock

    orch._signal_combiner = MagicMock()
    orch._signal_combiner.combine = MagicMock(
        side_effect=lambda sig, ai: _make_combined(pair=sig.pair if sig else "BTC/USDT")
    )

    orch._position_sizer = MagicMock()
    orch._position_sizer.calculate = MagicMock(
        side_effect=lambda pair, **kw: _make_position_size(pair=pair)
    )

    orch._drawdown_monitor = MagicMock()
    orch._drawdown_monitor.update = MagicMock(return_value=None)
    orch._drawdown_monitor.state = TradingState.ACTIVE
    orch._drawdown_monitor.entries_allowed = True
    orch._drawdown_monitor.halt_reason = None
    orch._drawdown_monitor.current_drawdown_pct = 2.0

    orch._ai_confidence_filter = MagicMock()
    orch._ai_confidence_filter.should_trade = MagicMock(return_value=True)


# ===========================================================================
# Group 1: Initialisation
# ===========================================================================


class TestInitialisation:
    def test_init_paper_mode(self):
        """PaperExecutor created when BOT_MODE=paper."""
        orch = _setup_orchestrator(BOT_MODE="paper")
        assert orch._settings.BOT_MODE == "paper"
        # executor not created yet (lazy)
        assert orch._executor is None

    @pytest.mark.asyncio
    async def test_init_creates_strategy_per_pair(self):
        """Strategy selectors created for each pair."""
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        _wire_mocks(orch)
        # _wire_mocks populates selectors for all pairs in settings
        assert len(orch._strategy_selectors) == 3
        assert "BTC/USDT" in orch._strategy_selectors
        assert "ETH/USDT" in orch._strategy_selectors
        assert "SOL/USDT" in orch._strategy_selectors

    def test_init_sentiment_when_enabled(self):
        """USE_SENTIMENT=True -> orchestrator is configured for it."""
        orch = _setup_orchestrator(USE_SENTIMENT=True)
        assert orch._settings.USE_SENTIMENT is True
        # Components created during _init_ai_components (not constructor)
        assert orch._sentiment_analyzer is None  # until start()

    def test_init_no_ai_by_default(self):
        """Feature flags False -> no AI components."""
        orch = _setup_orchestrator()
        assert orch._sentiment_analyzer is None
        assert orch._predictors == {}
        assert orch._news_fetcher is None
        assert orch._fear_greed is None

    def test_init_drawdown_monitor_with_initial_balance(self):
        """DrawdownMonitor configured from settings."""
        orch = _setup_orchestrator(INITIAL_BALANCE=50_000.0)
        _wire_mocks(orch)
        # After wiring, drawdown monitor is a mock — check the setting
        assert orch._settings.INITIAL_BALANCE == 50_000.0


# ===========================================================================
# Group 2: Trading Cycle
# ===========================================================================


class TestTradingCycle:
    @pytest.mark.asyncio
    async def test_cycle_processes_all_pairs(self):
        """Executor place_order called for each pair with signal."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 2  # BTC + ETH

    @pytest.mark.asyncio
    async def test_cycle_skips_pair_on_fetch_error(self):
        """One pair fails fetch, others continue."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)

        call_count = 0
        original_ohlcv = _make_ohlcv()

        async def side_effect_fetch(pair, **kw):
            nonlocal call_count
            call_count += 1
            if pair == "BTC/USDT":
                raise ConnectionError("Network error")
            return original_ohlcv

        orch._fetcher.fetch_ohlcv = AsyncMock(side_effect=side_effect_fetch)

        await orch.run_trading_cycle()

        # BTC fails, ETH succeeds
        assert orch._executor.place_order.call_count == 1

    @pytest.mark.asyncio
    async def test_cycle_skips_insufficient_data(self):
        """<50 bars -> no signal generated."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)

        short_ohlcv = _make_ohlcv(n=30)
        orch._fetcher.fetch_ohlcv = AsyncMock(return_value=short_ohlcv)

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_cycle_respects_should_trade_false(self):
        """Combined signal says no -> executor not called."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._signal_combiner.combine = MagicMock(
            return_value=_make_combined(should_trade=False)
        )

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_cycle_respects_confidence_filter(self):
        """AIConfidenceFilter blocks -> no execution."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._ai_confidence_filter.should_trade = MagicMock(return_value=False)

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_cycle_skips_neutral_signal(self):
        """NEUTRAL signal -> no trade."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._signal_combiner.combine = MagicMock(
            return_value=_make_combined(
                signal_type=SignalType.NEUTRAL, should_trade=True
            )
        )

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_cycle_handles_close_signal(self):
        """CLOSE signal with an open position -> sell order."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._executor.get_position = AsyncMock(return_value=0.5)
        orch._signal_combiner.combine = MagicMock(
            side_effect=lambda sig, ai: _make_combined(
                pair=sig.pair if sig else "BTC/USDT",
                signal_type=SignalType.CLOSE,
            )
        )

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 2
        for call in orch._executor.place_order.call_args_list:
            kwargs = call.kwargs
            assert kwargs.get("side") == "sell"

    @pytest.mark.asyncio
    async def test_cycle_skips_zero_quantity(self):
        """quantity=0 -> no order."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._position_sizer.calculate = MagicMock(
            return_value=_make_position_size(quantity=0.0)
        )

        await orch.run_trading_cycle()

        assert orch._executor.place_order.call_count == 0


# ===========================================================================
# Group 3: Kill Switch
# ===========================================================================


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_drawdown_kills_trading(self):
        """Drawdown exceeded -> close_all + skip pairs."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._drawdown_monitor.update = MagicMock(return_value=TradingState.HALTED)

        await orch.run_trading_cycle()

        orch._executor.close_all_positions.assert_called_once()
        assert orch._executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_no_kill_in_normal_range(self):
        """Normal drawdown -> trading continues."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        await orch.run_trading_cycle()

        orch._executor.close_all_positions.assert_not_called()
        assert orch._executor.place_order.call_count > 0


# ===========================================================================
# Group 4: Sentiment Cycle
# ===========================================================================


class TestSentimentCycle:
    @pytest.mark.asyncio
    async def test_sentiment_populates_cache(self):
        """After sentiment cycle, _latest_sentiment has entries."""
        orch = _setup_orchestrator(USE_SENTIMENT=True)
        _wire_mocks(orch)

        from trading_bot.core.models import NewsArticle, SentimentResult

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_batch = MagicMock(
            return_value=[
                SentimentResult(
                    score=0.5,
                    confidence=0.8,
                    label="positive",
                    model_used="CryptoBERT",
                    analyzed_text="test",
                )
            ]
        )

        orch._sentiment_analyzer = mock_analyzer
        orch._sentiment_aggregator = MagicMock()
        orch._sentiment_aggregator.aggregate = MagicMock(
            return_value=AggregatedSentiment(
                pair="BTC/USDT",
                overall_score=0.5,
                news_score=0.5,
                social_score=0.0,
                fear_greed_score=0.0,
                num_sources_analyzed=1,
                time_window="1h",
                timestamp=datetime.now(UTC),
            )
        )

        mock_news = AsyncMock()
        mock_news.fetch_for_pairs = AsyncMock(
            return_value={
                "BTC/USDT": [
                    NewsArticle(
                        title="BTC pumps",
                        body="Bitcoin pumps hard",
                        source="test",
                        url="http://test.com",
                        published_at=datetime.now(UTC),
                    )
                ]
            }
        )
        orch._news_fetcher = mock_news

        mock_fg = AsyncMock()
        mock_fg.fetch_latest = AsyncMock(
            return_value={"value": 75, "classification": "Greed"}
        )
        orch._fear_greed = mock_fg

        await orch._run_sentiment_cycle()

        assert len(orch._latest_sentiment) > 0
        # The headlines behind the sentiment reach the LLM sentiment agent...
        assert orch._latest_headlines["BTC/USDT"] == ["BTC pumps"]
        context = await orch._build_llm_context("BTC/USDT")
        assert context["headlines"] == ["BTC pumps"]

        # ...and expire with it.
        stale = orch._latest_sentiment["BTC/USDT"]
        stale.timestamp = datetime.now(UTC) - timedelta(hours=2)
        context = await orch._build_llm_context("BTC/USDT")
        assert context["headlines"] == []

    @pytest.mark.asyncio
    async def test_sentiment_skipped_when_disabled(self):
        """USE_SENTIMENT=False -> nothing happens."""
        orch = _setup_orchestrator(USE_SENTIMENT=False)
        _wire_mocks(orch)
        orch._sentiment_analyzer = None

        await orch._run_sentiment_cycle()

        assert len(orch._latest_sentiment) == 0

    def test_ai_signals_from_cache(self):
        """_build_ai_signals returns AISignals from cached sentiment."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._latest_sentiment["BTC/USDT"] = AggregatedSentiment(
            pair="BTC/USDT",
            overall_score=0.6,
            news_score=0.5,
            social_score=0.0,
            fear_greed_score=0.7,
            num_sources_analyzed=3,
            time_window="1h",
            timestamp=datetime.now(UTC),
        )

        ohlcv = _make_ohlcv()
        result = orch._build_ai_signals("BTC/USDT", ohlcv)

        assert result is not None
        assert result.sentiment_score == 0.6


# ===========================================================================
# Group 5: Shutdown
# ===========================================================================


class TestShutdown:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """stop() -> _running=False."""
        orch = _setup_orchestrator()
        orch._running = True

        await orch.stop()

        assert orch._running is False

    @pytest.mark.asyncio
    async def test_stop_closes_fetcher(self):
        """stop() -> fetcher.__aexit__ called."""
        orch = _setup_orchestrator()
        orch._running = True
        orch._fetcher = AsyncMock()
        orch._fetcher.__aexit__ = AsyncMock()

        await orch.stop()

        orch._fetcher.__aexit__.assert_called_once()


# ===========================================================================
# Group 6: Integration
# ===========================================================================


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_paper_end_to_end(self):
        """Mock only fetcher, everything else real with synthetic OHLCV."""
        settings = _make_settings(
            TRADING_PAIRS=["BTC/USDT"],
            INITIAL_BALANCE=100_000.0,
            MIN_CONFIDENCE_THRESHOLD=0.0,  # Accept any signal
        )

        orch = TradingOrchestrator(settings=settings)

        # Real components
        from trading_bot.ai.prediction.feature_engine import FeatureEngine
        from trading_bot.core.regime import RegimeDetector
        from trading_bot.core.signal_combiner import SignalCombiner
        from trading_bot.core.strategies.mean_reversion import MeanReversion
        from trading_bot.core.strategies.trend_following import TrendFollowing
        from trading_bot.core.strategy_selector import StrategySelector
        from trading_bot.execution.paper_executor import PaperExecutor
        from trading_bot.risk.ai_confidence import AIConfidenceFilter
        from trading_bot.risk.drawdown import DrawdownMonitor
        from trading_bot.risk.position_sizer import PositionSizer
        from trading_bot.risk.stop_loss import StopLossManager

        fe = FeatureEngine()
        rd = RegimeDetector(feature_engine=fe)
        trend = TrendFollowing(pair="BTC/USDT", feature_engine=fe)
        reversion = MeanReversion(pair="BTC/USDT", feature_engine=fe)
        selector = StrategySelector(trend, reversion, rd)

        orch._feature_engine = fe
        orch._strategy_selectors = {"BTC/USDT": selector}
        orch._signal_combiner = SignalCombiner(settings=settings)
        orch._position_sizer = PositionSizer(
            stop_loss_mgr=StopLossManager(), settings=settings
        )
        orch._drawdown_monitor = DrawdownMonitor(
            initial_value=100_000.0, settings=settings
        )
        orch._ai_confidence_filter = AIConfidenceFilter(settings=settings)
        orch._executor = PaperExecutor(
            initial_balance=100_000.0, slippage_pct=0.05, fee_pct=0.1
        )

        # Mock only the fetcher
        ohlcv = _make_ohlcv(n=250)
        orch._fetcher = AsyncMock()
        orch._fetcher.fetch_ohlcv = AsyncMock(return_value=ohlcv)

        # Run one cycle — should not raise
        await orch.run_trading_cycle()

        balance = await orch._executor.get_balance()
        # Balance may or may not change depending on signal direction
        assert balance["total"] > 0

    @pytest.mark.asyncio
    async def test_volatile_regime_reduces_size_not_confidence(self):
        """VOLATILE: the entry goes through, with the regime risk factor."""
        from dataclasses import replace

        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        selector = orch._strategy_selectors["BTC/USDT"]
        signal, regime = selector.generate_signal.return_value
        tagged = replace(signal, metadata={"regime": "volatile", "sizing_factor": 0.5})
        selector.generate_signal.return_value = (tagged, regime)

        await orch._process_pair("BTC/USDT", _make_ohlcv(), 100_000.0, 50_000.0)

        kwargs = orch._position_sizer.calculate.call_args.kwargs
        assert kwargs["regime_size_multiplier"] == 0.5
        orch._executor.place_order.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("consensus", [True, False])
    async def test_llm_consensus_short_vetoes_long_entry(self, consensus):
        """Cached LLM consensus SHORT blocks a technical LONG (ADR-006).

        LLM_WEIGHT is tiny so confidence alone stays above the threshold:
        only the veto can block the entry.
        """
        from trading_bot.core.models import OrchestratorDecision
        from trading_bot.core.signal_combiner import SignalCombiner

        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"], LLM_WEIGHT=0.01)
        _wire_mocks(orch)
        orch._signal_combiner = SignalCombiner(settings=orch._settings)
        orch._latest_llm_decision["BTC/USDT"] = OrchestratorDecision(
            pair="BTC/USDT",
            final_recommendation="SHORT",
            confidence=0.8,
            agent_analyses=[],
            consensus=consensus,
            reasoning="test",
            total_tokens=0,
            total_cost_usd=0.0,
            total_latency_ms=0,
            timestamp=datetime.now(UTC),
        )

        await orch._process_pair("BTC/USDT", _make_ohlcv(), 100_000.0, 50_000.0)

        if consensus:
            orch._executor.place_order.assert_not_called()
        else:
            orch._executor.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_with_ai_signals(self):
        """USE_SENTIMENT + cached sentiment -> flows through to SignalCombiner."""
        orch = _setup_orchestrator(
            USE_SENTIMENT=True,
            TRADING_PAIRS=["BTC/USDT"],
        )
        _wire_mocks(orch)

        # Pre-populate sentiment cache
        orch._latest_sentiment["BTC/USDT"] = AggregatedSentiment(
            pair="BTC/USDT",
            overall_score=0.7,
            news_score=0.6,
            social_score=0.0,
            fear_greed_score=0.8,
            num_sources_analyzed=2,
            time_window="1h",
            timestamp=datetime.now(UTC),
        )

        await orch.run_trading_cycle()

        # Signal combiner should have been called with ai_signals != None
        combine_call = orch._signal_combiner.combine.call_args
        ai_arg = combine_call[0][1]  # second positional arg
        assert ai_arg is not None
        assert ai_arg.sentiment_score == 0.7


# ===========================================================================
# Group 7: Position awareness (spot-only)
# ===========================================================================


def _set_signal(orch: TradingOrchestrator, signal_type: SignalType) -> None:
    orch._signal_combiner.combine = MagicMock(
        side_effect=lambda sig, ai: _make_combined(
            pair=sig.pair if sig else "BTC/USDT", signal_type=signal_type
        )
    )


def _paper_executor() -> PaperExecutor:
    return PaperExecutor(initial_balance=100_000.0, slippage_pct=0.0, fee_pct=0.0)


class TestPositionAwareness:
    @pytest.mark.asyncio
    async def test_long_without_position_buys_sized_quantity(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        _set_signal(orch, SignalType.LONG)

        await orch.run_trading_cycle()

        orch._executor.place_order.assert_awaited_once()
        kwargs = orch._executor.place_order.call_args.kwargs
        assert kwargs["side"] == "buy"
        assert kwargs["quantity"] == 0.01

    @pytest.mark.asyncio
    async def test_repeated_long_does_not_duplicate_position(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        _set_signal(orch, SignalType.LONG)
        orch._executor = _paper_executor()

        await orch.run_trading_cycle()
        await orch.run_trading_cycle()
        await orch.run_trading_cycle()

        assert await orch._executor.get_position("BTC/USDT") == pytest.approx(0.01)
        assert orch._position_sizer.calculate.call_count == 1

    @pytest.mark.asyncio
    async def test_close_sells_exact_position_quantity(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        _set_signal(orch, SignalType.CLOSE)
        orch._executor.get_position = AsyncMock(return_value=0.37)

        await orch.run_trading_cycle()

        orch._executor.place_order.assert_awaited_once()
        kwargs = orch._executor.place_order.call_args.kwargs
        assert kwargs["side"] == "sell"
        assert kwargs["quantity"] == 0.37
        orch._position_sizer.calculate.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_paper_executor_flattens_position(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        orch._executor = _paper_executor()
        await orch._executor.place_order(
            "BTC/USDT", "buy", "market", 0.37, price=50_000.0
        )
        _set_signal(orch, SignalType.CLOSE)

        await orch.run_trading_cycle()

        assert await orch._executor.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio
    async def test_close_without_position_does_nothing(self):
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        _set_signal(orch, SignalType.CLOSE)

        await orch.run_trading_cycle()

        orch._executor.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_without_position_does_nothing(self):
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        _set_signal(orch, SignalType.SHORT)

        await orch.run_trading_cycle()

        orch._executor.place_order.assert_not_called()
        orch._position_sizer.calculate.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_with_position_closes_it(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        _set_signal(orch, SignalType.SHORT)
        orch._executor.get_position = AsyncMock(return_value=0.2)

        await orch.run_trading_cycle()

        kwargs = orch._executor.place_order.call_args.kwargs
        assert kwargs["side"] == "sell"
        assert kwargs["quantity"] == 0.2

    @pytest.mark.asyncio
    async def test_cycle_updates_executor_prices(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        ohlcv = _make_ohlcv()
        _wire_mocks(orch, ohlcv=ohlcv)

        await orch.run_trading_cycle()

        orch._executor.update_prices.assert_called_once_with(
            {"BTC/USDT": float(ohlcv["close"].iloc[-1])}
        )

    @pytest.mark.asyncio
    async def test_price_drop_triggers_kill_switch_in_paper(self):
        from trading_bot.risk.drawdown import DrawdownMonitor

        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        ohlcv = _make_ohlcv()
        ohlcv["close"] = 30_000.0  # crash from 50k entry
        _wire_mocks(orch, ohlcv=ohlcv)
        orch._executor = _paper_executor()
        await orch._executor.place_order(
            "BTC/USDT", "buy", "market", 1.5, price=50_000.0
        )
        orch._drawdown_monitor = DrawdownMonitor(initial_value=100_000.0)

        await orch.run_trading_cycle()

        # 25k cash + 1.5 BTC @ 30k = 70k -> 30% drawdown > 15% limit
        assert orch._drawdown_monitor.current_drawdown_pct == pytest.approx(30.0)
        assert await orch._executor.get_position("BTC/USDT") == 0.0
        balance = await orch._executor.get_balance()
        assert balance["cash"] == pytest.approx(70_000.0)

    @pytest.mark.asyncio
    async def test_balance_failure_skips_cycle_without_kill_switch(self):
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._executor.get_balance = AsyncMock(side_effect=ConnectionError("down"))

        await orch.run_trading_cycle()

        orch._drawdown_monitor.update.assert_not_called()
        orch._executor.close_all_positions.assert_not_called()
        orch._executor.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_drawdown_baseline_uses_exchange_balance(self, monkeypatch):
        import ccxt.async_support as ccxt_async

        import trading_bot.core.bot as bot_module
        from trading_bot.execution.live_executor import LiveExecutor

        orch = _setup_orchestrator(
            BOT_MODE="live",
            TELEGRAM_BOT_TOKEN="",
            TELEGRAM_CHAT_ID="",
        )
        monkeypatch.setattr(ccxt_async, "binance", MagicMock(), raising=False)
        fetcher = MagicMock()
        fetcher.__aenter__ = AsyncMock(return_value=fetcher)
        monkeypatch.setattr(bot_module, "CCXTFetcher", MagicMock(return_value=fetcher))
        monkeypatch.setattr(
            LiveExecutor,
            "get_balance",
            AsyncMock(return_value={"total": 1_234.0, "cash": 1_234.0}),
        )

        await orch._init_components()

        assert isinstance(orch._executor, LiveExecutor)
        assert orch._executor._pairs == ["BTC/USDT", "ETH/USDT"]
        assert orch._drawdown_monitor is not None
        assert not orch._drawdown_monitor.check(1_234.0)
        assert orch._drawdown_monitor.check(1_000.0)  # 19% below real baseline


# ===========================================================================
# Group 8: Stop-loss execution
# ===========================================================================


def _flat_ohlcv(n: int = 60, price: float = 50_000.0) -> pd.DataFrame:
    index = pd.date_range(end="2026-01-10", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 100,
            "low": price - 100,
            "close": price,
            "volume": 1000.0,
        },
        index=index,
    )


async def _paper_with_btc(qty: float = 1.0, price: float = 50_000.0):
    ex = _paper_executor()
    await ex.place_order("BTC/USDT", "buy", "market", qty, price=price)
    return ex


def _live_exchange(btc: float = 0.5):
    exchange = MagicMock()
    exchange.has = {}
    exchange.fetch_balance = AsyncMock(
        return_value={
            "total": {"USDT": 1_000.0, "BTC": btc},
            "free": {"USDT": 1_000.0, "BTC": 0.0},
            "used": {"BTC": btc},
        }
    )
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50_000.0})
    exchange.fetch_tickers = AsyncMock(return_value={"BTC/USDT": {"last": 50_000.0}})
    exchange.load_markets = AsyncMock(return_value={})
    exchange.market = MagicMock(return_value={"limits": {}})
    exchange.amount_to_precision = MagicMock(
        side_effect=lambda s, a: f"{int(a * 1e5) / 1e5:.5f}"
    )
    exchange.price_to_precision = MagicMock(
        side_effect=lambda s, p: f"{int(p * 1e2) / 1e2:.2f}"
    )
    exchange.fetch_order = AsyncMock(return_value={"id": "stop-1", "status": "open"})
    exchange.cancel_order = AsyncMock(return_value={})
    exchange.create_order = AsyncMock(
        return_value={"id": "sell-1", "status": "closed", "filled": btc}
    )
    return exchange


class TestStopLoss:
    def _orch(self, ohlcv: pd.DataFrame) -> TradingOrchestrator:
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=ohlcv)
        _set_signal(orch, SignalType.NEUTRAL)
        orch._notifier = AsyncMock()
        return orch

    @pytest.mark.asyncio
    async def test_long_fill_registers_watched_stop_in_paper(self):
        ohlcv = _flat_ohlcv()
        orch = self._orch(ohlcv)
        _set_signal(orch, SignalType.LONG)
        orch._executor = _paper_executor()

        await orch.run_trading_cycle()

        stop = orch._stop_tracker.get("BTC/USDT")
        assert stop is not None
        assert stop.stop_price == 49_000.0  # from _make_position_size
        assert stop.quantity == pytest.approx(0.01)
        assert stop.order_id is None
        # The candle forming at entry (last closed + 1h) is never checked.
        entry_candle = ohlcv.index[-1] + pd.Timedelta(hours=1)
        assert stop.last_checked == entry_candle.isoformat()

    @pytest.mark.asyncio
    async def test_paper_stop_hit_by_low_sells_at_stop(self):
        ohlcv = _flat_ohlcv()
        ohlcv.iloc[-2, ohlcv.columns.get_loc("low")] = 47_500.0
        orch = self._orch(ohlcv)
        orch._executor = await _paper_with_btc()
        orch._stop_tracker.register(
            "BTC/USDT", 48_000.0, 1.0, None, entry_candle=ohlcv.index[-5]
        )

        await orch.run_trading_cycle()

        assert await orch._executor.get_position("BTC/USDT") == 0.0
        assert (await orch._executor.get_balance())["cash"] == pytest.approx(98_000.0)
        assert "BTC/USDT" not in orch._stop_tracker
        orch._notifier.send_alert.assert_awaited_once()
        assert "STOP-LOSS" in orch._notifier.send_alert.call_args.args[0]

    @pytest.mark.asyncio
    async def test_paper_gap_below_stop_sells_at_open(self):
        ohlcv = _flat_ohlcv()
        ohlcv.iloc[-2, ohlcv.columns.get_loc("open")] = 46_000.0
        ohlcv.iloc[-2, ohlcv.columns.get_loc("low")] = 45_000.0
        orch = self._orch(ohlcv)
        orch._executor = await _paper_with_btc()
        orch._stop_tracker.register(
            "BTC/USDT", 48_000.0, 1.0, None, entry_candle=ohlcv.index[-5]
        )

        await orch.run_trading_cycle()

        assert await orch._executor.get_position("BTC/USDT") == 0.0
        assert (await orch._executor.get_balance())["cash"] == pytest.approx(96_000.0)

    @pytest.mark.asyncio
    async def test_paper_stop_not_touched_keeps_position(self):
        ohlcv = _flat_ohlcv()
        # A still-forming candle (opens now) breaching the stop is dropped
        # before stops run: its low is not final.
        forming = pd.DataFrame(
            {
                "open": [50_000.0],
                "high": [50_100.0],
                "low": [40_000.0],
                "close": [49_000.0],
                "volume": [1.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").floor("h")]),
        )
        ohlcv = pd.concat([ohlcv, forming])
        orch = self._orch(ohlcv)
        orch._executor = await _paper_with_btc()
        orch._stop_tracker.register(
            "BTC/USDT", 48_000.0, 1.0, None, entry_candle=ohlcv.index[-5]
        )

        await orch.run_trading_cycle()

        assert await orch._executor.get_position("BTC/USDT") == 1.0
        assert "BTC/USDT" in orch._stop_tracker
        orch._notifier.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_still_run_while_halted(self):
        ohlcv = _flat_ohlcv()
        ohlcv.iloc[-2, ohlcv.columns.get_loc("low")] = 47_500.0
        orch = self._orch(ohlcv)
        orch._drawdown_monitor.state = TradingState.HALTED
        orch._drawdown_monitor.entries_allowed = False
        orch._executor = await _paper_with_btc()
        orch._stop_tracker.register(
            "BTC/USDT", 48_000.0, 1.0, None, entry_candle=ohlcv.index[-5]
        )

        await orch.run_trading_cycle()

        assert await orch._executor.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio
    async def test_live_stop_placed_with_precision(self):
        from trading_bot.execution.live_executor import LiveExecutor

        exchange = _live_exchange(btc=0.0123456)
        exchange.create_order = AsyncMock(return_value={"id": "stop-1"})
        orch = self._orch(_flat_ohlcv())
        orch._executor = LiveExecutor(exchange=exchange, pairs=["BTC/USDT"])

        await orch._register_stop("BTC/USDT", 48_000.1299, _flat_ohlcv())

        exchange.create_order.assert_awaited_once_with(
            symbol="BTC/USDT",
            type="market",
            side="sell",
            amount=0.01234,
            price=None,
            params={"stopLossPrice": 48_000.12},
        )
        stop = orch._stop_tracker.get("BTC/USDT")
        assert stop is not None and stop.order_id == "stop-1"

    @pytest.mark.asyncio
    async def test_live_stop_failure_falls_back_to_watching(self):
        from trading_bot.execution.live_executor import LiveExecutor

        exchange = _live_exchange(btc=0.5)
        exchange.create_order = AsyncMock(side_effect=Exception("rejected"))
        orch = self._orch(_flat_ohlcv())
        orch._executor = LiveExecutor(exchange=exchange, pairs=["BTC/USDT"])

        await orch._register_stop("BTC/USDT", 48_000.0, _flat_ohlcv())

        stop = orch._stop_tracker.get("BTC/USDT")
        assert stop is not None and stop.order_id is None

    @pytest.mark.asyncio
    async def test_close_cancels_native_stop_before_selling(self):
        from trading_bot.execution.live_executor import LiveExecutor

        exchange = _live_exchange(btc=0.5)
        calls = MagicMock()
        calls.attach_mock(exchange.cancel_order, "cancel_order")
        calls.attach_mock(exchange.create_order, "create_order")
        orch = self._orch(_flat_ohlcv())
        _set_signal(orch, SignalType.CLOSE)
        orch._executor = LiveExecutor(exchange=exchange, pairs=["BTC/USDT"])
        orch._stop_tracker.register("BTC/USDT", 48_000.0, 0.5, "stop-1", None)

        await orch.run_trading_cycle()

        names = [c[0] for c in calls.mock_calls]
        assert names == ["cancel_order", "create_order"]
        assert calls.mock_calls[0].args == ("stop-1", "BTC/USDT")
        assert calls.mock_calls[1].kwargs["side"] == "sell"
        assert calls.mock_calls[1].kwargs["amount"] == 0.5
        assert "BTC/USDT" not in orch._stop_tracker

    @pytest.mark.asyncio
    async def test_filled_native_stop_is_cleared_and_notified(self):
        from trading_bot.execution.live_executor import LiveExecutor

        exchange = _live_exchange(btc=0.0)
        exchange.fetch_order = AsyncMock(
            return_value={
                "id": "stop-1",
                "status": "closed",
                "average": 47_990.0,
                "filled": 0.5,
            }
        )
        orch = self._orch(_flat_ohlcv())
        orch._executor = LiveExecutor(exchange=exchange, pairs=["BTC/USDT"])
        orch._stop_tracker.register("BTC/USDT", 48_000.0, 0.5, "stop-1", None)

        await orch.run_trading_cycle()

        exchange.fetch_order.assert_awaited_once_with("stop-1", "BTC/USDT")
        assert "BTC/USDT" not in orch._stop_tracker
        orch._notifier.send_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_externally_cancelled_native_stop_is_watched(self):
        from trading_bot.execution.live_executor import LiveExecutor

        exchange = _live_exchange(btc=0.5)
        exchange.fetch_order = AsyncMock(
            return_value={"id": "stop-1", "status": "canceled"}
        )
        orch = self._orch(_flat_ohlcv())
        orch._executor = LiveExecutor(exchange=exchange, pairs=["BTC/USDT"])
        orch._stop_tracker.register("BTC/USDT", 48_000.0, 0.5, "stop-1", None)

        await orch.run_trading_cycle()

        stop = orch._stop_tracker.get("BTC/USDT")
        assert stop is not None and stop.order_id is None


# ===========================================================================
# Group 9: Exits bypass confidence filters
# ===========================================================================


class TestExitFilters:
    @pytest.mark.asyncio
    async def test_low_confidence_close_still_sells(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        orch._signal_combiner.combine = MagicMock(
            return_value=_make_combined(
                signal_type=SignalType.CLOSE, confidence=0.1, should_trade=False
            )
        )
        orch._ai_confidence_filter.should_trade = MagicMock(return_value=False)
        orch._executor.get_position = AsyncMock(return_value=0.5)

        await orch.run_trading_cycle()

        kwargs = orch._executor.place_order.call_args.kwargs
        assert kwargs["side"] == "sell"
        assert kwargs["quantity"] == 0.5

    @pytest.mark.asyncio
    async def test_low_confidence_long_is_blocked(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        orch._signal_combiner.combine = MagicMock(
            return_value=_make_combined(
                signal_type=SignalType.LONG, confidence=0.1, should_trade=False
            )
        )

        await orch.run_trading_cycle()

        orch._executor.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_ai_filter_blocks_long_only(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        orch._ai_confidence_filter.should_trade = MagicMock(return_value=False)

        await orch.run_trading_cycle()

        orch._executor.place_order.assert_not_called()


# ===========================================================================
# Group 10: Latching kill-switch and persisted risk state
# ===========================================================================


class TestKillSwitchLatch:
    @pytest.mark.asyncio
    async def test_drawdown_halts_liquidates_and_notifies_once(self):
        from trading_bot.risk.drawdown import DrawdownMonitor

        ohlcv = _flat_ohlcv(price=30_000.0)  # crash from 50k entry
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=ohlcv)
        _set_signal(orch, SignalType.LONG)
        orch._notifier = AsyncMock()
        paper = await _paper_with_btc(qty=1.5)
        paper.close_all_positions = AsyncMock(wraps=paper.close_all_positions)
        paper.place_order = AsyncMock(wraps=paper.place_order)
        orch._executor = paper
        orch._drawdown_monitor = DrawdownMonitor(initial_value=100_000.0)

        for _ in range(3):
            await orch.run_trading_cycle()

        assert orch._drawdown_monitor.state == TradingState.HALTED
        paper.close_all_positions.assert_awaited_once()
        kill_alerts = [
            c
            for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") == "kill_switch"
        ]
        assert len(kill_alerts) == 1
        # Only the liquidation sell; no BUY despite LONG signals afterwards.
        sides = [c.kwargs["side"] for c in paper.place_order.call_args_list]
        assert sides == ["sell"]
        assert await paper.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio
    async def test_daily_loss_blocks_entries_until_next_utc_day(self):
        from trading_bot.risk.drawdown import DrawdownMonitor

        class Clock:
            now = datetime(2026, 3, 1, 12, tzinfo=UTC)

            def __call__(self):
                return self.now

        clock = Clock()
        ohlcv = _flat_ohlcv(price=45_000.0)
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=ohlcv)
        orch._notifier = AsyncMock()
        paper = await _paper_with_btc(qty=0.8)  # 40k exposure
        paper.place_order = AsyncMock(wraps=paper.place_order)
        orch._executor = paper
        orch._drawdown_monitor = DrawdownMonitor(initial_value=100_000.0, clock=clock)

        # 60k cash + 0.8 * 45k = 96k -> -4% daily loss
        await orch.run_trading_cycle()
        assert orch._drawdown_monitor.state == TradingState.DAILY_PAUSED
        assert await paper.get_position("BTC/USDT") == 0.0

        _set_signal(orch, SignalType.LONG)
        await orch.run_trading_cycle()
        assert await paper.get_position("BTC/USDT") == 0.0  # still paused

        clock.now = datetime(2026, 3, 2, 0, 5, tzinfo=UTC)
        await orch.run_trading_cycle()
        assert orch._drawdown_monitor.state == TradingState.ACTIVE
        assert await paper.get_position("BTC/USDT") == pytest.approx(0.01)
        kill_alerts = [
            c
            for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") == "kill_switch"
        ]
        assert len(kill_alerts) == 1


def _drop_paper_account(path) -> None:
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["paper_account"]
    path.write_text(json.dumps(state), encoding="utf-8")


class TestRiskStatePersistence:
    @pytest.mark.asyncio
    async def test_state_survives_orchestrator_restart(self, tmp_path):
        path = tmp_path / "risk_{mode}.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path), BOT_MODE="live")
        orch1._executor = await _paper_with_btc()
        await orch1._load_risk_state(100_000.0)
        orch1._stop_tracker.register("BTC/USDT", 48_000.0, 1.0, None, None)
        assert orch1._drawdown_monitor.update(80_000.0) == TradingState.HALTED
        orch1._save_risk_state()
        assert (tmp_path / "risk_live.json").exists()

        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path), BOT_MODE="live")
        orch2._executor = await _paper_with_btc()
        await orch2._load_risk_state(80_000.0)

        assert orch2._drawdown_monitor.state == TradingState.HALTED
        assert not orch2._drawdown_monitor.entries_allowed
        assert orch2._drawdown_monitor.to_dict()["peak_value"] == 100_000.0
        stop = orch2._stop_tracker.get("BTC/USDT")
        assert stop is not None and stop.stop_price == 48_000.0

    @pytest.mark.asyncio
    async def test_stops_of_flat_pairs_are_dropped_on_load(self, tmp_path):
        path = tmp_path / "risk.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch1._executor = await _paper_with_btc()
        await orch1._load_risk_state(100_000.0)
        orch1._stop_tracker.register("BTC/USDT", 48_000.0, 1.0, None, None)
        orch1._save_risk_state()
        _drop_paper_account(path)

        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch2._executor = _paper_executor()  # no BTC held
        await orch2._load_risk_state(100_000.0)

        assert len(orch2._stop_tracker) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content",
        ["{not json", "[1, 2]", '{"drawdown": {"peak_value": "abc"}}', "{}"],
    )
    async def test_corrupt_state_starts_fresh(self, tmp_path, content):
        path = tmp_path / "risk.json"
        path.write_text(content, encoding="utf-8")
        orch = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch._executor = _paper_executor()

        await orch._load_risk_state(100_000.0)

        assert orch._drawdown_monitor.state == TradingState.ACTIVE
        assert orch._drawdown_monitor.to_dict()["peak_value"] == 100_000.0
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["drawdown"]["state"] == "active"

    @pytest.mark.asyncio
    async def test_risk_reset_clears_halt(self, tmp_path):
        path = tmp_path / "risk.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch1._executor = _paper_executor()
        await orch1._load_risk_state(100_000.0)
        orch1._drawdown_monitor.update(80_000.0)
        orch1._save_risk_state()
        # The 80k is simulated: the saved paper account would say 100k.
        _drop_paper_account(path)

        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path), RISK_RESET=True)
        orch2._executor = _paper_executor()
        await orch2._load_risk_state(80_000.0)

        assert orch2._drawdown_monitor.state == TradingState.ACTIVE
        assert orch2._drawdown_monitor.entries_allowed
        assert orch2._drawdown_monitor.update(80_000.0) is None
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["drawdown"]["state"] == "active"

    @pytest.mark.asyncio
    async def test_paper_restart_keeps_latch_but_rebaselines(self, tmp_path):
        path = tmp_path / "risk.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch1._executor = _paper_executor()
        await orch1._load_risk_state(100_000.0)
        orch1._drawdown_monitor.update(120_000.0)  # profitable session
        orch1._drawdown_monitor.update(119_000.0)
        orch1._save_risk_state()
        _drop_paper_account(path)  # file written before the account was saved

        # Paper account restarts at INITIAL_BALANCE (100k): not a drawdown.
        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch2._executor = _paper_executor()
        await orch2._load_risk_state(100_000.0)

        assert orch2._drawdown_monitor.update(100_000.0) is None
        assert orch2._drawdown_monitor.state == TradingState.ACTIVE

    @pytest.mark.asyncio
    async def test_paper_restart_restores_account_and_peak(self, tmp_path):
        path = tmp_path / "risk.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch1._executor = await _paper_with_btc(qty=1.0, price=50_000.0)
        await orch1._load_risk_state(100_000.0)
        orch1._executor.update_prices({"BTC/USDT": 70_000.0})
        orch1._drawdown_monitor.update(120_000.0)
        orch1._stop_tracker.register("BTC/USDT", 48_000.0, 1.0, None, None)
        orch1._save_risk_state()

        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch2._executor = _paper_executor()  # fresh process: 100k cash
        await orch2._load_risk_state(100_000.0)

        assert await orch2._executor.get_position("BTC/USDT") == 1.0
        balance = await orch2._executor.get_balance()
        assert balance["cash"] == pytest.approx(50_000.0)
        assert balance["total"] == pytest.approx(120_000.0)  # saved mark
        assert orch2._drawdown_monitor.to_dict()["peak_value"] == 120_000.0
        assert orch2._stop_tracker.get("BTC/USDT") is not None

    @pytest.mark.asyncio
    async def test_invalid_paper_account_starts_at_initial_balance(self, tmp_path):
        path = tmp_path / "risk.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch1._executor = await _paper_with_btc()
        await orch1._load_risk_state(100_000.0)
        state = json.loads(path.read_text(encoding="utf-8"))
        state["paper_account"]["cash"] = "nan"
        path.write_text(json.dumps(state), encoding="utf-8")

        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch2._executor = _paper_executor()
        await orch2._load_risk_state(100_000.0)

        assert await orch2._executor.get_position("BTC/USDT") == 0.0
        assert (await orch2._executor.get_balance())["total"] == 100_000.0

    @pytest.mark.asyncio
    async def test_paper_restart_keeps_halt(self, tmp_path):
        path = tmp_path / "risk.json"
        orch1 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch1._executor = _paper_executor()
        await orch1._load_risk_state(100_000.0)
        orch1._drawdown_monitor.update(80_000.0)
        orch1._save_risk_state()

        orch2 = _setup_orchestrator(RISK_STATE_PATH=str(path))
        orch2._executor = _paper_executor()
        await orch2._load_risk_state(100_000.0)

        assert orch2._drawdown_monitor.state == TradingState.HALTED


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_stop_closes_executor_once(self):
        orch = _setup_orchestrator()
        orch._executor = AsyncMock()
        orch._fetcher = AsyncMock()

        await orch.stop()
        await orch.stop()

        orch._executor.close.assert_awaited_once()
        orch._fetcher.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_context_counts_open_positions(self):
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT", "ETH/USDT"])
        _wire_mocks(orch)
        orch._executor.get_position = AsyncMock(
            side_effect=lambda pair: 0.5 if pair == "BTC/USDT" else 0.0
        )

        context = await orch._build_llm_context("BTC/USDT")

        assert context["open_positions"] == 1


# ===========================================================================
# Group 11: Market data, ML predictors, sentiment, cache expiry
# ===========================================================================


class TestMarketData:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("tf", "minutes"), [("15m", 15), ("1h", 60), ("4h", 240)])
    async def test_since_matches_timeframe(self, tf, minutes):
        orch = _setup_orchestrator(TRADING_TIMEFRAME=tf, OHLCV_LOOKBACK_BARS=100)
        orch._fetcher = AsyncMock()
        orch._fetcher.fetch_ohlcv = AsyncMock(return_value=_make_ohlcv())

        before = datetime.now(UTC)
        await orch._fetch_ohlcv("BTC/USDT")

        kwargs = orch._fetcher.fetch_ohlcv.call_args.kwargs
        # 100 closed bars + the forming one, plus one bar of slack.
        expected = before - pd.Timedelta(minutes=minutes * 102)
        assert abs((kwargs["since"] - expected).total_seconds()) < 5
        assert kwargs["limit"] == 101
        assert kwargs["timeframe"] == tf

    @pytest.mark.asyncio
    async def test_cycle_drops_open_candle_but_prices_it(self):
        closed = _flat_ohlcv(price=50_000.0)
        forming = pd.DataFrame(
            {
                "open": [50_000.0],
                "high": [52_000.0],
                "low": [49_000.0],
                "close": [51_234.0],
                "volume": [1.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC").floor("h")]),
        )
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=pd.concat([closed, forming]))

        await orch.run_trading_cycle()

        # Signals saw closed candles only; valuation/fills got the live price.
        seen = orch._strategy_selectors["BTC/USDT"].generate_signal.call_args.args[0]
        assert seen.index[-1] == closed.index[-1]
        orch._executor.update_prices.assert_called_once_with({"BTC/USDT": 51_234.0})
        assert orch._executor.place_order.call_args.kwargs["price"] == 51_234.0

    @pytest.mark.asyncio
    async def test_fetcher_uses_configured_exchange(self, monkeypatch):
        import trading_bot.core.bot as bot_module

        orch = _setup_orchestrator(
            EXCHANGE_ID="kraken",
            OHLCV_FETCH_MAX_RETRIES=2,
            OHLCV_FETCH_TIMEOUT_S=7.0,
        )
        fetcher = MagicMock()
        fetcher.__aenter__ = AsyncMock(return_value=fetcher)
        factory = MagicMock(return_value=fetcher)
        monkeypatch.setattr(bot_module, "CCXTFetcher", factory)

        await orch._init_components()

        kwargs = factory.call_args.kwargs
        assert kwargs["exchange_id"] == "kraken"
        assert kwargs["max_retries"] == 2
        assert kwargs["timeout_s"] == 7.0


class TestPredictors:
    def _train_and_save(self, path):
        from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor

        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (120, 3))
        y = (x[:, 0] > 0).astype(int)
        df = pd.DataFrame(x, columns=["a", "b", "c"])
        df["label"] = y
        model = LightGBMPredictor(
            params={"early_stopping_rounds": 0, "n_estimators": 10}
        )
        model.train(df.iloc[:100], df.iloc[100:])
        model.save(str(path))

    def test_one_predictor_per_pair_loaded_by_path(self, tmp_path):
        from trading_bot.ai.prediction.base import model_path

        self._train_and_save(model_path(tmp_path, "BTC/USDT", "1h"))
        orch = _setup_orchestrator(
            TRADING_PAIRS=["BTC/USDT", "ETH/USDT"],
            DL_MODEL_TYPE="lightgbm",
            ML_MODEL_DIR=str(tmp_path),
        )

        orch._init_predictors()

        assert list(orch._predictors) == ["BTC/USDT"]  # ETH has no model file
        assert orch._predictors["BTC/USDT"].is_loaded()

    def test_stub_model_type_disables_prediction(self, tmp_path):
        from trading_bot.ai.prediction.base import model_path

        self._train_and_save(model_path(tmp_path, "BTC/USDT", "1h"))
        orch = _setup_orchestrator(DL_MODEL_TYPE="gru", ML_MODEL_DIR=str(tmp_path))

        orch._init_predictors()

        assert orch._predictors == {}

    def test_prediction_unavailable_means_no_ai_prediction(self):
        from trading_bot.ai.prediction.base import PredictionUnavailableError

        orch = _setup_orchestrator()
        _wire_mocks(orch)
        predictor = MagicMock()
        predictor.is_loaded.return_value = True
        predictor.predict.side_effect = PredictionUnavailableError("NaN")
        orch._predictors = {"BTC/USDT": predictor}

        assert orch._build_ai_signals("BTC/USDT", _make_ohlcv()) is None

    def test_each_pair_uses_its_own_predictor(self):
        from trading_bot.core.models import Prediction

        orch = _setup_orchestrator()
        _wire_mocks(orch)

        def predictor(conf):
            p = MagicMock()
            p.is_loaded.return_value = True
            p.predict.return_value = Prediction(
                pair="x",
                timeframe="1h",
                predicted_return=0.001,
                predicted_direction="up",
                confidence=conf,
                model_type="lightgbm",
                model_version="1.0",
                features_used=[],
                timestamp=datetime.now(UTC),
            )
            return p

        orch._predictors = {"BTC/USDT": predictor(0.3), "ETH/USDT": predictor(0.8)}

        btc = orch._build_ai_signals("BTC/USDT", _make_ohlcv())
        eth = orch._build_ai_signals("ETH/USDT", _make_ohlcv())
        assert btc.prediction_confidence == 0.3
        assert eth.prediction_confidence == 0.8


class TestSentimentAndCaches:
    @pytest.mark.asyncio
    async def test_sentiment_inference_does_not_block_event_loop(self):
        import asyncio
        import time

        from trading_bot.core.models import NewsArticle, SentimentResult

        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)

        def slow_batch(texts):
            time.sleep(0.3)  # CPU-bound transformer inference stand-in
            return [
                SentimentResult(
                    score=0.5,
                    confidence=0.9,
                    label="bullish",
                    model_used="m",
                    analyzed_text=t,
                )
                for t in texts
            ]

        analyzer = MagicMock()
        analyzer.analyze_batch = MagicMock(side_effect=slow_batch)
        orch._sentiment_analyzer = analyzer
        orch._sentiment_aggregator = MagicMock()
        orch._news_fetcher = AsyncMock()
        orch._news_fetcher.fetch_for_pairs = AsyncMock(
            return_value={
                "BTC/USDT": [
                    NewsArticle(
                        title=f"t{i}",
                        body="b",
                        source="s",
                        url="u",
                        published_at=datetime.now(UTC),
                    )
                    for i in range(3)
                ]
            }
        )
        orch._fear_greed = None

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await orch._run_sentiment_cycle()
        tick_task.cancel()

        assert analyzer.analyze_batch.call_count == 1  # one batched call
        assert len(analyzer.analyze_batch.call_args.args[0]) == 3
        assert ticks >= 5  # the loop kept running during inference

    def _sentiment(self, age_min):
        return AggregatedSentiment(
            pair="BTC/USDT",
            overall_score=0.6,
            news_score=0.5,
            social_score=0.0,
            fear_greed_score=0.7,
            num_sources_analyzed=3,
            time_window="1h",
            timestamp=datetime.now(UTC) - pd.Timedelta(minutes=age_min),
        )

    def test_stale_sentiment_is_ignored(self):
        orch = _setup_orchestrator(SENTIMENT_INTERVAL_MIN=15)
        _wire_mocks(orch)

        orch._latest_sentiment["BTC/USDT"] = self._sentiment(age_min=29)
        assert orch._build_ai_signals("BTC/USDT", _make_ohlcv()) is not None

        orch._latest_sentiment["BTC/USDT"] = self._sentiment(age_min=31)
        assert orch._build_ai_signals("BTC/USDT", _make_ohlcv()) is None

    def test_stale_llm_decision_is_ignored(self):
        from trading_bot.core.models import OrchestratorDecision

        orch = _setup_orchestrator(LLM_ANALYSIS_INTERVAL_MIN=15)
        _wire_mocks(orch)

        def decision(age_min):
            return OrchestratorDecision(
                pair="BTC/USDT",
                final_recommendation="BUY",
                confidence=0.8,
                consensus=True,
                agent_analyses=[],
                reasoning="r",
                total_tokens=0,
                total_cost_usd=0.0,
                total_latency_ms=0,
                timestamp=datetime.now(UTC) - pd.Timedelta(minutes=age_min),
            )

        orch._latest_llm_decision["BTC/USDT"] = decision(age_min=10)
        assert orch._build_ai_signals("BTC/USDT", _make_ohlcv()).llm_confidence == 0.8

        orch._latest_llm_decision["BTC/USDT"] = decision(age_min=45)
        assert orch._build_ai_signals("BTC/USDT", _make_ohlcv()) is None

    @pytest.mark.asyncio
    async def test_llm_context_ignores_stale_sentiment(self):
        orch = _setup_orchestrator(SENTIMENT_INTERVAL_MIN=15)
        _wire_mocks(orch)
        orch._latest_sentiment["BTC/USDT"] = self._sentiment(age_min=120)

        context = await orch._build_llm_context("BTC/USDT")

        assert context["sentiment_score"] == 0.0


# ===========================================================================
# Group 12: Sentiment cycle — independent sources, per-article cache
# ===========================================================================


def _news(title: str, article_id: str = "") -> object:
    from trading_bot.core.models import NewsArticle

    return NewsArticle(
        title=title,
        body="b",
        source="s",
        url="u",
        published_at=datetime.now(UTC) - timedelta(minutes=5),
        article_id=article_id,
    )


def _scoring_analyzer(score: float = 0.5) -> MagicMock:
    from trading_bot.core.models import SentimentResult

    analyzer = MagicMock()
    analyzer.analyze_batch = MagicMock(
        side_effect=lambda texts: [
            SentimentResult(
                score=score,
                confidence=0.9,
                label="bullish",
                model_used="cryptobert",
                analyzed_text=t,
            )
            for t in texts
        ]
    )
    return analyzer


def _sentiment_orch(news_by_pair=None, fg=75, **overrides) -> TradingOrchestrator:
    from trading_bot.ai.sentiment.aggregator import SentimentAggregator

    orch = _setup_orchestrator(USE_SENTIMENT=True, **overrides)
    _wire_mocks(orch)
    orch._sentiment_analyzer = _scoring_analyzer()
    orch._sentiment_aggregator = SentimentAggregator()
    orch._news_fetcher = AsyncMock()
    orch._news_fetcher.fetch_for_pairs = AsyncMock(return_value=news_by_pair or {})
    orch._fear_greed = AsyncMock()
    orch._fear_greed.fetch_latest = AsyncMock(
        return_value={"value": fg, "classification": "Greed"}
    )
    return orch


class TestSentimentSources:
    @pytest.mark.asyncio
    async def test_news_401_keeps_fear_greed(self):
        import httpx

        orch = _sentiment_orch()
        request = httpx.Request("GET", "https://data-api.coindesk.com/x")
        orch._news_fetcher.fetch_for_pairs = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401", request=request, response=httpx.Response(401, request=request)
            )
        )

        await orch._run_sentiment_cycle()

        for pair in orch._settings.TRADING_PAIRS:
            sentiment = orch._latest_sentiment[pair]
            assert sentiment.fear_greed_score == pytest.approx(0.5)
            assert sentiment.num_sources_analyzed == 1

    @pytest.mark.asyncio
    async def test_fear_greed_failure_keeps_news(self):
        orch = _sentiment_orch({"BTC/USDT": [_news("up", "1")]})
        orch._fear_greed.fetch_latest = AsyncMock(side_effect=RuntimeError("down"))

        await orch._run_sentiment_cycle()

        sentiment = orch._latest_sentiment["BTC/USDT"]
        assert sentiment.news_score == pytest.approx(0.5)
        assert sentiment.num_sources_analyzed == 1
        assert "ETH/USDT" not in orch._latest_sentiment  # no source at all

    @pytest.mark.asyncio
    async def test_fear_greed_and_news_fetched_once_per_cycle(self):
        orch = _sentiment_orch(
            {"BTC/USDT": [_news("a", "1")]},
            TRADING_PAIRS=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        )

        await orch._run_sentiment_cycle()

        orch._fear_greed.fetch_latest.assert_awaited_once()
        orch._news_fetcher.fetch_for_pairs.assert_awaited_once()
        assert set(orch._latest_sentiment) == set(orch._settings.TRADING_PAIRS)

    @pytest.mark.asyncio
    async def test_articles_scored_once_across_cycles(self):
        shared = _news("shared", "1")
        orch = _sentiment_orch(
            {"BTC/USDT": [shared], "ETH/USDT": [shared, _news("eth", "2")]}
        )

        await orch._run_sentiment_cycle()
        await orch._run_sentiment_cycle()

        analyzer = orch._sentiment_analyzer
        assert analyzer.analyze_batch.call_count == 1
        assert len(analyzer.analyze_batch.call_args.args[0]) == 2
        assert orch._latest_sentiment["ETH/USDT"].num_sources_analyzed == 3

        # A new article is the only text scored on the next cycle.
        orch._news_fetcher.fetch_for_pairs.return_value = {
            "BTC/USDT": [shared, _news("fresh", "3")]
        }
        await orch._run_sentiment_cycle()
        assert analyzer.analyze_batch.call_count == 2
        assert len(analyzer.analyze_batch.call_args.args[0]) == 1
        assert set(orch._latest_headlines["BTC/USDT"]) == {"shared", "fresh"}

    @pytest.mark.asyncio
    async def test_news_outage_reuses_cached_scores(self):
        orch = _sentiment_orch({"BTC/USDT": [_news("a", "1")]}, fg=50)
        await orch._run_sentiment_cycle()

        orch._news_fetcher.fetch_for_pairs = AsyncMock(side_effect=RuntimeError())
        await orch._run_sentiment_cycle()

        assert orch._latest_sentiment["BTC/USDT"].news_score == pytest.approx(0.5)


class TestNewsProviderSetup:
    def test_missing_key_disables_only_news(self):
        from loguru import logger

        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING")
        try:
            orch = _setup_orchestrator(USE_SENTIMENT=True, COINDESK_API_KEY="")
            assert orch._build_news_fetcher() is None
        finally:
            logger.remove(sink)
        assert any("COINDESK_API_KEY is empty" in m for m in messages)

    @pytest.mark.asyncio
    async def test_provider_selects_client(self):
        from trading_bot.data.news.coindesk import CoinDeskNews
        from trading_bot.data.news.cryptocompare import CryptoCompareNews

        orch = _setup_orchestrator(COINDESK_API_KEY="k")
        fetcher = orch._build_news_fetcher()
        assert isinstance(fetcher, CoinDeskNews)
        await fetcher.close()

        orch = _setup_orchestrator(
            NEWS_PROVIDER="cryptocompare", CRYPTOCOMPARE_API_KEY="k"
        )
        fetcher = orch._build_news_fetcher()
        assert isinstance(fetcher, CryptoCompareNews)
        await fetcher.close()


# ===========================================================================
# Group 13: DB persistence (TradeJournal on aiosqlite)
# ===========================================================================


@pytest.fixture
async def journal_db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from trading_bot.core.trade_journal import TradeJournal
    from trading_bot.db.base import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield TradeJournal(factory, exchange="binance"), factory
    await engine.dispose()


async def _db_rows(factory):
    from sqlalchemy import select

    from trading_bot.db.models import Order, PortfolioSnapshot, Trade

    async with factory() as session:
        trades = list((await session.execute(select(Trade))).scalars())
        orders = list((await session.execute(select(Order))).scalars())
        snaps = list(
            (
                await session.execute(
                    select(PortfolioSnapshot).order_by(PortfolioSnapshot.id)
                )
            ).scalars()
        )
    return trades, orders, snaps


class TestPersistence:
    @pytest.mark.asyncio
    async def test_entry_exit_and_snapshots_are_recorded(self, journal_db):
        journal, factory = journal_db
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=_flat_ohlcv())
        orch._executor = _paper_executor()
        orch._journal = journal

        _set_signal(orch, SignalType.LONG)
        await orch.run_trading_cycle()
        trades, orders, snaps = await _db_rows(factory)
        assert [(t.status, t.strategy) for t in trades] == [("open", "trend_following")]
        assert [o.side for o in orders] == ["buy"]
        assert orders[0].trade_id == trades[0].id
        assert snaps[-1].open_positions_count == 1
        assert snaps[-1].snapshot_trigger == "cycle"

        _set_signal(orch, SignalType.CLOSE)
        await orch.run_trading_cycle()
        trades, orders, snaps = await _db_rows(factory)
        assert [(t.status, t.exit_reason) for t in trades] == [("closed", "signal")]
        assert [o.side for o in orders] == ["buy", "sell"]
        assert len(snaps) == 2
        assert snaps[-1].open_positions_count == 0

    @pytest.mark.asyncio
    async def test_bot_side_stop_closes_trade(self, journal_db):
        journal, factory = journal_db
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        orch._executor = _paper_executor()
        orch._journal = journal
        result = await orch._execute_order("BTC/USDT", "buy", 0.5, 50_000.0)
        await journal.record_entry(result, strategy="trend_following")

        await orch._execute_stop("BTC/USDT", 48_000.0)

        trades, orders, _ = await _db_rows(factory)
        assert trades[0].exit_reason == "stop_loss"
        assert float(trades[0].exit_price) == pytest.approx(48_000.0)
        assert [o.strategy for o in orders] == ["trend_following", "stop_loss"]

    @pytest.mark.asyncio
    async def test_exchange_stop_fill_closes_trade(self, journal_db):
        journal, factory = journal_db
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch)
        orch._journal = journal
        await journal.record_entry(_make_order_result(), strategy="trend_following")
        orch._stop_tracker.register("BTC/USDT", 49_000.0, 0.01, "ex-1", None)
        orch._executor.get_order = AsyncMock(
            return_value={
                "status": "closed",
                "average": 48_900.0,
                "filled": 0.01,
                "fee": {"cost": 0.4, "currency": "USDT"},
            }
        )

        await orch._check_exchange_stop("BTC/USDT", "ex-1")

        trades, orders, _ = await _db_rows(factory)
        assert trades[0].exit_reason == "stop_loss"
        assert orders[-1].exchange_order_id == "ex-1"

    @pytest.mark.asyncio
    async def test_kill_switch_liquidation_is_recorded(self, journal_db):
        from trading_bot.risk.drawdown import DrawdownMonitor

        journal, factory = journal_db
        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=_flat_ohlcv(price=30_000.0))
        paper = _paper_executor()
        entry = await paper.place_order("BTC/USDT", "buy", "market", 1.5, 50_000.0)
        await journal.record_entry(entry, strategy="trend_following")
        orch._executor = paper
        orch._journal = journal
        orch._drawdown_monitor = DrawdownMonitor(initial_value=100_000.0)

        await orch.run_trading_cycle()

        trades, _, snaps = await _db_rows(factory)
        assert [(t.status, t.exit_reason) for t in trades] == [
            ("closed", "kill_switch")
        ]
        assert float(trades[0].pnl) < 0
        assert snaps[-1].snapshot_trigger == "halted"
        assert float(snaps[-1].realized_pnl_day) == pytest.approx(float(trades[0].pnl))

    @pytest.mark.asyncio
    async def test_db_failures_never_stop_trading(self):
        from trading_bot.core.trade_journal import TradeJournal

        def broken_factory():
            raise ConnectionError("db down")

        orch = _setup_orchestrator(TRADING_PAIRS=["BTC/USDT"])
        _wire_mocks(orch, ohlcv=_flat_ohlcv())
        orch._executor = _paper_executor()
        orch._journal = TradeJournal(broken_factory, exchange="binance")  # type: ignore[arg-type]

        _set_signal(orch, SignalType.LONG)
        await orch.run_trading_cycle()
        assert await orch._executor.get_position("BTC/USDT") > 0
        _set_signal(orch, SignalType.CLOSE)
        await orch.run_trading_cycle()
        assert await orch._executor.get_position("BTC/USDT") == 0.0

    @pytest.mark.asyncio
    async def test_unreachable_db_at_startup_disables_persistence(self, tmp_path):
        from trading_bot.core.trade_journal import TradeJournal

        url = f"sqlite+aiosqlite:///{tmp_path}/missing/dir/bot.db"
        assert await TradeJournal.connect(url, exchange="binance") is None

    @pytest.mark.asyncio
    async def test_open_trades_restored_after_restart(self, journal_db):
        from trading_bot.core.trade_journal import TradeJournal

        journal, factory = journal_db
        await journal.record_entry(_make_order_result(), strategy="trend_following")

        restarted = TradeJournal(factory, exchange="binance")
        await restarted.restore()
        sell = _make_order_result(side="sell")
        await restarted.record_exit(sell, strategy="signal_combiner", reason="signal")

        trades, _, _ = await _db_rows(factory)
        assert [(t.status, t.exit_reason) for t in trades] == [("closed", "signal")]


# ===========================================================================
# Group 14: SENTIMENT_PROVIDER=llm
# ===========================================================================


class TestLLMSentimentProvider:
    def test_llm_provider_shares_the_agents_budget(self, tmp_path):
        from trading_bot.ai.sentiment.llm_news import AsyncLLMSentimentAnalyzer

        orch = _setup_orchestrator(
            SENTIMENT_PROVIDER="llm",
            LLM_PROVIDER="anthropic",
            LLM_MODEL="claude-haiku-4-5",
            ANTHROPIC_API_KEY="k",
            LLM_MAX_DAILY_COST_USD=5.0,
            LLM_COST_STATE_PATH=str(tmp_path / "cost_{mode}.json"),
        )
        analyzer = orch._build_sentiment_analyzer()
        assert isinstance(analyzer, AsyncLLMSentimentAnalyzer)
        assert analyzer._tracker is orch._shared_llm_cost_tracker()

    def test_llm_provider_without_key_falls_back_to_cryptobert(self, monkeypatch):
        import trading_bot.ai.sentiment.cryptobert as cryptobert

        monkeypatch.setattr(cryptobert, "_require_backend", lambda: None)
        orch = _setup_orchestrator(
            SENTIMENT_PROVIDER="llm", LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=""
        )
        assert isinstance(
            orch._build_sentiment_analyzer(), cryptobert.CryptoBERTAnalyzer
        )

    @pytest.mark.asyncio
    async def test_failed_llm_scores_are_retried_next_cycle(self, tmp_path):
        from trading_bot.ai.sentiment.cost_tracker import SentimentCostTracker
        from trading_bot.ai.sentiment.llm_news import AsyncLLMSentimentAnalyzer
        from trading_bot.core.models import SentimentResult

        orch = _sentiment_orch(
            {"BTC/USDT": [_news("a", "1"), _news("b", "2")]},
            TRADING_PAIRS=["BTC/USDT"],
        )
        analyzer = AsyncLLMSentimentAnalyzer(SentimentCostTracker(5.0), api_key="k")
        ok = SentimentResult(
            score=0.4, confidence=0.9, label="bullish", model_used="m", analyzed_text=""
        )
        analyzer.analyze_batch_async = AsyncMock(return_value=[ok, None])
        orch._sentiment_analyzer = analyzer

        await orch._run_sentiment_cycle()
        assert len(orch._news_cache) == 1

        analyzer.analyze_batch_async = AsyncMock(return_value=[ok])
        await orch._run_sentiment_cycle()
        assert analyzer.analyze_batch_async.await_args.args[0] == ["b. b"]
        assert len(orch._news_cache) == 2


class TestJournalReconcile:
    @pytest.mark.asyncio
    async def test_flat_pairs_abandon_stale_open_trades(self, journal_db):
        from trading_bot.core.trade_journal import TradeJournal

        journal, factory = journal_db
        await journal.record_entry(_make_order_result(), strategy="trend_following")
        await journal.record_entry(
            _make_order_result(pair="ETH/USDT"), strategy="trend_following"
        )

        # Paper restart: the simulated account is flat again.
        restarted = TradeJournal(factory, exchange="binance")
        await restarted.restore()
        await restarted.reconcile({"BTC/USDT": 0.0, "ETH/USDT": 0.01})

        trades, _, _ = await _db_rows(factory)
        status = {t.pair: (t.status, t.exit_reason) for t in trades}
        assert status == {
            "BTC/USDT": ("abandoned", "no_position"),
            "ETH/USDT": ("open", None),
        }
        assert restarted.open_trade_id("BTC/USDT") is None
        assert restarted.open_trade_id("ETH/USDT") is not None

    @pytest.mark.asyncio
    async def test_new_entry_abandons_unclosed_previous_trade(self, journal_db):
        journal, factory = journal_db
        await journal.record_entry(_make_order_result(), strategy="trend_following")
        await journal.record_entry(_make_order_result(), strategy="mean_reversion")

        trades, _, _ = await _db_rows(factory)
        assert sorted(t.status for t in trades) == ["abandoned", "open"]
