"""TradingOrchestrator — main bot loop connecting all components."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
from loguru import logger

from trading_bot.ai.agents.orchestrator import Orchestrator
from trading_bot.ai.prediction.base import BasePredictor
from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.sentiment.aggregator import SentimentAggregator
from trading_bot.ai.sentiment.base import BaseSentimentAnalyzer
from trading_bot.ai.sentiment.cost_tracker import SentimentCostTracker
from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import (
    AggregatedSentiment,
    AISignals,
    CombinedSignal,
    OrchestratorDecision,
    OrderResult,
    PositionSize,
    SentimentResult,
    SignalType,
)
from trading_bot.core.regime import RegimeDetector
from trading_bot.core.signal_combiner import SignalCombiner
from trading_bot.core.strategies.mean_reversion import MeanReversion
from trading_bot.core.strategies.trend_following import TrendFollowing
from trading_bot.core.strategy_selector import StrategySelector
from trading_bot.data.market.binance import BinanceFetcher
from trading_bot.data.news.cryptocompare import CryptoCompareNews
from trading_bot.data.onchain.fear_greed import FearGreedIndex
from trading_bot.execution.base import BaseExecutor
from trading_bot.execution.paper_executor import PaperExecutor
from trading_bot.risk.ai_confidence import AIConfidenceFilter
from trading_bot.risk.drawdown import DrawdownMonitor
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.stop_loss import StopLossManager


class TradingOrchestrator:
    """Connects all trading components in an automated loop with scheduling."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scheduler: object | None = None
        self._running: bool = False

        # Core components — initialised in start()
        self._fetcher: BinanceFetcher | None = None
        self._executor: BaseExecutor | None = None
        self._feature_engine: FeatureEngine | None = None
        self._strategy_selectors: dict[str, StrategySelector] = {}
        self._signal_combiner: SignalCombiner | None = None
        self._position_sizer: PositionSizer | None = None
        self._drawdown_monitor: DrawdownMonitor | None = None
        self._ai_confidence_filter: AIConfidenceFilter | None = None

        # AI components — optional, feature-flagged
        self._sentiment_analyzer: BaseSentimentAnalyzer | None = None
        self._sentiment_aggregator: SentimentAggregator | None = None
        self._news_fetcher: CryptoCompareNews | None = None
        self._fear_greed: FearGreedIndex | None = None
        self._predictor: BasePredictor | None = None

        # LLM multi-agent components — optional, feature-flagged
        self._llm_orchestrator: Orchestrator | None = None
        self._llm_cost_tracker: SentimentCostTracker | None = None

        # State cache
        self._latest_sentiment: dict[str, AggregatedSentiment] = {}
        self._latest_llm_decision: dict[str, OrchestratorDecision] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all components, start scheduler, run first cycle, then loop."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        logger.info("TradingOrchestrator starting…")

        await self._init_components()
        await self._init_ai_components()
        self._setup_signal_handlers()

        self._running = True

        scheduler = AsyncIOScheduler()
        self._scheduler = scheduler

        # Main trading job
        scheduler.add_job(
            self.run_trading_cycle,
            trigger=IntervalTrigger(
                minutes=self._settings.TRADING_INTERVAL_MINUTES,
            ),
            id="trading_cycle",
            max_instances=1,
        )

        # Sentiment job (separate cadence)
        if self._settings.USE_SENTIMENT and self._sentiment_analyzer is not None:
            scheduler.add_job(
                self._run_sentiment_cycle,
                trigger=IntervalTrigger(
                    minutes=self._settings.SENTIMENT_INTERVAL_MIN,
                ),
                id="sentiment_cycle",
                max_instances=1,
            )

        # LLM multi-agent job
        if self._settings.USE_LLM_AGENT and self._llm_orchestrator is not None:
            scheduler.add_job(
                self._run_llm_cycle,
                trigger=IntervalTrigger(
                    minutes=self._settings.LLM_ANALYSIS_INTERVAL_MIN,
                ),
                id="llm_cycle",
                max_instances=1,
            )

        scheduler.start()

        # First cycle immediately
        try:
            if self._settings.USE_SENTIMENT and self._sentiment_analyzer is not None:
                await self._run_sentiment_cycle()
            if self._settings.USE_LLM_AGENT and self._llm_orchestrator is not None:
                await self._run_llm_cycle()
            await self.run_trading_cycle()
        except Exception:
            logger.exception("Error in initial cycle")

        # Keep alive
        while self._running:
            await asyncio.sleep(1)

        logger.info("TradingOrchestrator stopped.")

    async def stop(self) -> None:
        """Graceful shutdown: stop scheduler, close fetcher, set running=False."""
        logger.info("Stopping TradingOrchestrator…")
        self._running = False

        if self._scheduler is not None:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            if isinstance(self._scheduler, AsyncIOScheduler):
                self._scheduler.shutdown(wait=False)

        if self._fetcher is not None:
            await self._fetcher.__aexit__(None, None, None)

        if self._news_fetcher is not None:
            await self._news_fetcher.close()

        if self._fear_greed is not None:
            await self._fear_greed.close()

    async def run_trading_cycle(self) -> None:
        """One complete trading cycle: check drawdown, process each pair."""
        assert self._executor is not None
        assert self._drawdown_monitor is not None

        balance = await self._executor.get_balance()
        portfolio_value = balance["total"]
        logger.info(
            "Trading cycle — portfolio: ${:,.2f}",
            portfolio_value,
        )

        if self._drawdown_monitor.check(portfolio_value):
            logger.warning(
                "KILL SWITCH triggered — drawdown {:.1f}%",
                self._drawdown_monitor.current_drawdown_pct,
            )
            await self._executor.close_all_positions()
            return

        for pair in self._settings.TRADING_PAIRS:
            try:
                await self._process_pair(pair, portfolio_value)
            except Exception:
                logger.exception("Error processing {}", pair)
                continue

    # ------------------------------------------------------------------
    # Private — initialisation
    # ------------------------------------------------------------------

    async def _init_components(self) -> None:
        """Initialise fetcher, executor, strategies per-pair, risk components."""
        s = self._settings

        # Feature engine (shared)
        self._feature_engine = FeatureEngine()

        # Regime detector (shared)
        regime_detector = RegimeDetector(
            adx_trending_threshold=s.REGIME_ADX_TRENDING,
            volatility_high_percentile=s.REGIME_VOLATILITY_HIGH_PCT,
            feature_engine=self._feature_engine,
        )

        # Strategy selectors per pair
        self._strategy_selectors = {}
        for pair in s.TRADING_PAIRS:
            trend = TrendFollowing(pair=pair, feature_engine=self._feature_engine)
            reversion = MeanReversion(pair=pair, feature_engine=self._feature_engine)
            self._strategy_selectors[pair] = StrategySelector(
                trend, reversion, regime_detector, s.REGIME_VOLATILE_SIZING_FACTOR
            )

        # Risk components
        self._position_sizer = PositionSizer(stop_loss_mgr=StopLossManager())
        self._drawdown_monitor = DrawdownMonitor(initial_value=s.INITIAL_BALANCE)
        self._ai_confidence_filter = AIConfidenceFilter()
        self._signal_combiner = SignalCombiner()

        # Executor
        if s.BOT_MODE == "live":
            import ccxt.async_support as ccxt_async

            exchange_cls = getattr(ccxt_async, s.EXCHANGE_ID)
            exchange = exchange_cls(
                {
                    "apiKey": s.EXCHANGE_API_KEY,
                    "secret": s.EXCHANGE_API_SECRET,
                    "enableRateLimit": True,
                }
            )
            if s.EXCHANGE_SANDBOX:
                exchange.set_sandbox_mode(True)

            from trading_bot.execution.live_executor import LiveExecutor

            self._executor = LiveExecutor(exchange=exchange)
        else:
            self._executor = PaperExecutor(
                initial_balance=s.INITIAL_BALANCE,
                slippage_pct=s.PAPER_SLIPPAGE_PCT,
                fee_pct=s.PAPER_FEE_PCT,
            )

        # Data fetcher
        self._fetcher = BinanceFetcher(
            api_key=s.EXCHANGE_API_KEY,
            api_secret=s.EXCHANGE_API_SECRET,
            sandbox=s.EXCHANGE_SANDBOX,
        )
        await self._fetcher.__aenter__()

        logger.info(
            "Components initialised — mode={}, pairs={}",
            s.BOT_MODE,
            s.TRADING_PAIRS,
        )

    async def _init_ai_components(self) -> None:
        """Initialise sentiment / prediction based on feature flags."""
        s = self._settings

        if s.USE_SENTIMENT:
            try:
                from trading_bot.ai.sentiment.cryptobert import CryptoBERTAnalyzer

                self._sentiment_analyzer = CryptoBERTAnalyzer(
                    model_name=s.SENTIMENT_MODEL,
                )
                self._sentiment_aggregator = SentimentAggregator()
                self._news_fetcher = CryptoCompareNews(api_key=s.CRYPTOCOMPARE_API_KEY)
                self._fear_greed = FearGreedIndex()
                logger.info("Sentiment components initialised")
            except Exception:
                logger.exception("Failed to init sentiment — continuing without AI")
                self._sentiment_analyzer = None

        if s.USE_LLM_AGENT:
            try:
                from trading_bot.ai.agents.risk_evaluator import RiskEvaluatorAgent
                from trading_bot.ai.agents.sentiment_analyst import (
                    SentimentAnalystAgent,
                )
                from trading_bot.ai.agents.technical_analyst import (
                    TechnicalAnalystAgent,
                )

                api_key = (
                    s.ANTHROPIC_API_KEY
                    if s.LLM_PROVIDER == "anthropic"
                    else s.OPENAI_API_KEY
                )
                agents = [
                    TechnicalAnalystAgent(s.LLM_PROVIDER, s.LLM_MODEL, api_key),
                    SentimentAnalystAgent(s.LLM_PROVIDER, s.LLM_MODEL, api_key),
                    RiskEvaluatorAgent(s.LLM_PROVIDER, s.LLM_MODEL, api_key),
                ]
                self._llm_cost_tracker = SentimentCostTracker(s.LLM_MAX_DAILY_COST_USD)
                self._llm_orchestrator = Orchestrator(
                    agents, self._llm_cost_tracker, s.LLM_PROVIDER, s.LLM_MODEL
                )
                logger.info("LLM multi-agent orchestrator initialised")
            except Exception:
                logger.exception("Failed to init LLM agents — continuing without LLM")
                self._llm_orchestrator = None

        if s.USE_ENSEMBLE:
            try:
                from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor

                self._predictor = LightGBMPredictor(
                    pair=s.TRADING_PAIRS[0],
                    timeframe=s.TRADING_TIMEFRAME,
                )
                logger.info("ML predictor initialised")
            except Exception:
                logger.exception("Failed to init predictor — continuing without ML")
                self._predictor = None

    # ------------------------------------------------------------------
    # Private — per-pair pipeline
    # ------------------------------------------------------------------

    async def _process_pair(self, pair: str, portfolio_value: float) -> None:
        """Full pipeline for one pair: fetch -> signal -> risk -> execute."""
        assert self._signal_combiner is not None
        assert self._ai_confidence_filter is not None
        assert self._position_sizer is not None

        # A. Fetch OHLCV
        ohlcv = await self._fetch_ohlcv(pair)
        if len(ohlcv) < 50:
            logger.warning(
                "{}: insufficient data ({} bars), skipping", pair, len(ohlcv)
            )
            return

        # B. Generate technical signal via StrategySelector
        selector = self._strategy_selectors[pair]
        signal, regime = selector.generate_signal(ohlcv)

        # C. Build AI signals from cache
        ai_signals = self._build_ai_signals(pair, ohlcv)

        # D. Combine signals
        combined = self._signal_combiner.combine(signal, ai_signals)

        # E. Risk filters
        if not combined.should_trade:
            logger.debug("{}: should_trade=False, skipping", pair)
            return

        if not self._ai_confidence_filter.should_trade(combined):
            logger.debug("{}: AI confidence filter blocked", pair)
            return

        if combined.type == SignalType.NEUTRAL:
            logger.debug("{}: NEUTRAL signal, skipping", pair)
            return

        # F. Position sizing
        entry_price = float(ohlcv["close"].iloc[-1])
        side = self._signal_side(combined.type)
        if side is None:
            return

        position_size = self._position_sizer.calculate(
            pair=pair,
            signal_confidence=combined.combined_confidence,
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            side=side,
            ai_confidence=(ai_signals.prediction_confidence if ai_signals else None),
            ohlcv=ohlcv,
        )
        if position_size.quantity <= 0:
            logger.debug("{}: position size zero, skipping", pair)
            return

        # G. Execute
        result = await self._execute_signal(pair, combined, position_size, entry_price)
        if result:
            logger.info(
                "{}: {} {} qty={:.6f} @ ${:,.2f} (fee=${:.2f})",
                pair,
                result.side,
                result.status,
                result.filled_quantity,
                result.filled_price,
                result.fee,
            )

    async def _fetch_ohlcv(self, pair: str) -> pd.DataFrame:
        """Fetch recent OHLCV from the exchange."""
        assert self._fetcher is not None
        s = self._settings
        since = datetime.now(UTC) - timedelta(
            hours=s.OHLCV_LOOKBACK_BARS
            * (60 if s.TRADING_TIMEFRAME == "1h" else 1)
            / 60
        )
        return await self._fetcher.fetch_ohlcv(
            pair=pair,
            timeframe=s.TRADING_TIMEFRAME,
            since=since,
            limit=s.OHLCV_LOOKBACK_BARS,
        )

    def _build_ai_signals(self, pair: str, ohlcv: pd.DataFrame) -> AISignals | None:
        """Build AISignals from cached sentiment + inline ML prediction + LLM."""
        sentiment = self._latest_sentiment.get(pair)
        prediction_conf: float | None = None
        prediction_ret: float | None = None

        if self._predictor is not None and self._predictor.is_loaded():
            try:
                pred = self._predictor.predict(ohlcv)
                prediction_conf = pred.confidence
                prediction_ret = pred.predicted_return
            except Exception:
                logger.warning("{}: ML prediction failed", pair)

        # LLM multi-agent decision from cache
        llm_decision = self._latest_llm_decision.get(pair)
        llm_rec: str | None = None
        llm_conf: float | None = None
        if llm_decision is not None:
            llm_rec = llm_decision.final_recommendation
            llm_conf = llm_decision.confidence

        if sentiment is None and prediction_conf is None and llm_rec is None:
            return None

        return AISignals(
            sentiment_score=(sentiment.overall_score if sentiment else None),
            prediction_return=prediction_ret,
            prediction_confidence=prediction_conf,
            llm_recommendation=llm_rec,
            llm_confidence=llm_conf,
        )

    async def _run_llm_cycle(self) -> None:
        """Separate job: run LLM agents for each pair, cache decisions."""
        if self._llm_orchestrator is None:
            return

        logger.debug("LLM cycle starting…")

        for pair in self._settings.TRADING_PAIRS:
            try:
                context = await self._build_llm_context(pair)
                decision = await self._llm_orchestrator.run(pair, context)
                if decision is not None:
                    self._latest_llm_decision[pair] = decision
                    logger.debug(
                        "{}: LLM decision={} conf={:.2f} consensus={}",
                        pair,
                        decision.final_recommendation,
                        decision.confidence,
                        decision.consensus,
                    )
            except Exception:
                logger.exception("LLM cycle failed for {}", pair)

    async def _build_llm_context(self, pair: str) -> dict[str, object]:
        """Build unified context dict for all LLM agents."""
        s = self._settings
        context: dict[str, object] = {"pair": pair, "timeframe": s.TRADING_TIMEFRAME}

        # Technical indicators
        try:
            ohlcv = await self._fetch_ohlcv(pair)
            if self._feature_engine is not None and len(ohlcv) >= 50:
                features = self._feature_engine.build_features(ohlcv)
                last_row = features.iloc[-1]
                indicator_keys = [
                    "rsi",
                    "macd",
                    "ema_9",
                    "ema_21",
                    "ema_50",
                    "adx",
                    "bb_upper",
                    "bb_lower",
                    "atr",
                    "close",
                    "volume",
                ]
                context["indicators"] = {
                    k: round(float(last_row[k]), 4)
                    for k in indicator_keys
                    if k in last_row.index and pd.notna(last_row[k])
                }

                # Regime
                selector = self._strategy_selectors.get(pair)
                if selector is not None:
                    _, regime_cls = selector.generate_signal(ohlcv)
                    context["regime"] = regime_cls.regime.value
        except Exception:
            logger.warning("{}: failed to build technical context", pair)
            context["indicators"] = {}
            context["regime"] = "UNKNOWN"

        # Sentiment from cache
        sentiment = self._latest_sentiment.get(pair)
        context["sentiment_score"] = sentiment.overall_score if sentiment else 0.0
        context["fear_greed"] = sentiment.fear_greed_score if sentiment else "N/A"
        context["headlines"] = []

        # Risk data
        if self._executor is not None:
            try:
                balance = await self._executor.get_balance()
                context["portfolio_value"] = balance["total"]
            except Exception:
                context["portfolio_value"] = s.INITIAL_BALANCE
        else:
            context["portfolio_value"] = s.INITIAL_BALANCE

        dd = self._drawdown_monitor
        context["drawdown_pct"] = dd.current_drawdown_pct if dd else 0.0
        context["open_positions"] = 0
        context["max_risk_pct"] = s.MAX_RISK_PER_TRADE_PCT
        context["max_position_pct"] = s.MAX_POSITION_PCT
        context["max_drawdown_pct"] = s.MAX_DRAWDOWN_PCT

        return context

    async def _run_sentiment_cycle(self) -> None:
        """Separate job: fetch news/fear-greed, analyse, cache results."""
        if self._sentiment_analyzer is None or self._sentiment_aggregator is None:
            return

        logger.debug("Sentiment cycle starting…")

        for pair in self._settings.TRADING_PAIRS:
            try:
                sentiment_results = []

                # News sentiment
                if self._news_fetcher is not None:
                    since = datetime.now(UTC) - timedelta(hours=24)
                    articles = await self._news_fetcher.fetch_by_pair(
                        pair, since=since, limit=10
                    )
                    for article in articles:
                        text = f"{article.title}. {article.body}"
                        result = self._sentiment_analyzer.analyze(text)
                        sentiment_results.append(result)

                # Fear & Greed
                if self._fear_greed is not None:
                    fg_data = await self._fear_greed.fetch_latest()
                    if fg_data:
                        fg_value = int(str(fg_data.get("value", 50)))
                        # Normalise 0-100 to [-1, 1]
                        normalised = (fg_value - 50) / 50
                        sentiment_results.append(
                            SentimentResult(
                                score=normalised,
                                confidence=0.7,
                                label="fear_greed",
                                model_used="fear_greed_index",
                                analyzed_text=f"Fear & Greed: {fg_value}",
                            )
                        )

                # Aggregate
                if sentiment_results:
                    aggregated = self._sentiment_aggregator.aggregate(
                        sentiment_results, pair=pair
                    )
                    self._latest_sentiment[pair] = aggregated
                    logger.debug(
                        "{}: sentiment score={:.2f} ({} sources)",
                        pair,
                        aggregated.overall_score,
                        aggregated.num_sources_analyzed,
                    )
            except Exception:
                logger.exception("Sentiment cycle failed for {}", pair)

    async def _execute_signal(
        self,
        pair: str,
        combined: CombinedSignal,
        position_size: PositionSize,
        price: float,
    ) -> OrderResult | None:
        """Execute an order via the executor."""
        assert self._executor is not None
        side = self._signal_side(combined.type)
        if side is None:
            return None

        return await self._executor.place_order(
            pair=pair,
            side=side,
            order_type="market",
            quantity=position_size.quantity,
            price=price,
        )

    # ------------------------------------------------------------------
    # Private — signal handling
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM for graceful shutdown (Windows-compatible)."""

        def _handle_shutdown(*_: object) -> None:
            logger.info("Shutdown signal received")
            self._running = False

        if sys.platform == "win32":
            signal.signal(signal.SIGINT, _handle_shutdown)
            signal.signal(signal.SIGTERM, _handle_shutdown)
        else:
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, _handle_shutdown)
            except RuntimeError:
                signal.signal(signal.SIGINT, _handle_shutdown)
                signal.signal(signal.SIGTERM, _handle_shutdown)

    @staticmethod
    def _signal_side(signal_type: SignalType) -> str | None:
        """Map SignalType to order side string."""
        if signal_type == SignalType.LONG:
            return "buy"
        if signal_type in (SignalType.SHORT, SignalType.CLOSE):
            return "sell"
        return None
