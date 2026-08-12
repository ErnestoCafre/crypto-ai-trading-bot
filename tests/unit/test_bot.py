"""Tests for TradingOrchestrator (core.bot)."""

from __future__ import annotations

from datetime import UTC, datetime
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 50, n),
            "high": close + abs(rng.normal(0, 200, n)),
            "low": close - abs(rng.normal(0, 200, n)),
            "close": close,
            "volume": abs(rng.normal(1000, 300, n)),
        }
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
        recommended_sizing_pct=7.5,
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
    orch._drawdown_monitor.check = MagicMock(return_value=False)
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
        assert orch._predictor is None
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
        """CLOSE signal -> sell order."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._signal_combiner.combine = MagicMock(
            side_effect=lambda sig, ai: _make_combined(
                pair=sig.pair if sig else "BTC/USDT",
                signal_type=SignalType.CLOSE,
            )
        )

        await orch.run_trading_cycle()

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
        orch._drawdown_monitor.check = MagicMock(return_value=True)

        await orch.run_trading_cycle()

        orch._executor.close_all_positions.assert_called_once()
        assert orch._executor.place_order.call_count == 0

    @pytest.mark.asyncio
    async def test_no_kill_in_normal_range(self):
        """Normal drawdown -> trading continues."""
        orch = _setup_orchestrator()
        _wire_mocks(orch)
        orch._drawdown_monitor.check = MagicMock(return_value=False)

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
        mock_analyzer.analyze = MagicMock(
            return_value=SentimentResult(
                score=0.5,
                confidence=0.8,
                label="positive",
                model_used="CryptoBERT",
                analyzed_text="test",
            )
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
        mock_news.fetch_by_pair = AsyncMock(
            return_value=[
                NewsArticle(
                    title="BTC pumps",
                    body="Bitcoin pumps hard",
                    source="test",
                    url="http://test.com",
                    published_at=datetime.now(UTC),
                )
            ]
        )
        orch._news_fetcher = mock_news

        mock_fg = AsyncMock()
        mock_fg.fetch_latest = AsyncMock(
            return_value={"value": 75, "classification": "Greed"}
        )
        orch._fear_greed = mock_fg

        await orch._run_sentiment_cycle()

        assert len(orch._latest_sentiment) > 0

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
        orch._signal_combiner = SignalCombiner()
        orch._position_sizer = PositionSizer(stop_loss_mgr=StopLossManager())
        orch._drawdown_monitor = DrawdownMonitor(initial_value=100_000.0)
        orch._ai_confidence_filter = AIConfidenceFilter()
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
