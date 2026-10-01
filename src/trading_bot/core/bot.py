"""TradingOrchestrator — main bot loop connecting all components."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from trading_bot.ai.agents.orchestrator import Orchestrator
from trading_bot.ai.prediction.base import BasePredictor, PredictionUnavailableError
from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.sentiment.aggregator import SentimentAggregator
from trading_bot.ai.sentiment.base import BaseSentimentAnalyzer
from trading_bot.ai.sentiment.cost_tracker import EXHAUSTED, SentimentCostTracker
from trading_bot.ai.sentiment.llm_news import AsyncLLMSentimentAnalyzer
from trading_bot.ai.sentiment.news_cache import NewsSentimentCache
from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import (
    AggregatedSentiment,
    AISignals,
    OrchestratorDecision,
    OrderResult,
    SentimentResult,
    Signal,
    SignalType,
)
from trading_bot.core.regime import RegimeDetector
from trading_bot.core.signal_combiner import SignalCombiner
from trading_bot.core.strategies.mean_reversion import MeanReversion
from trading_bot.core.strategies.trend_following import TrendFollowing
from trading_bot.core.strategy_selector import StrategySelector
from trading_bot.core.trade_journal import TradeJournal
from trading_bot.data.market.candles import (
    split_closed_candles,
    timeframe_to_timedelta,
)
from trading_bot.data.market.ccxt_fetcher import CCXTFetcher
from trading_bot.data.news.base import BaseNewsFetcher
from trading_bot.data.onchain.fear_greed import FearGreedIndex
from trading_bot.execution.base import BaseExecutor
from trading_bot.execution.paper_executor import PaperExecutor
from trading_bot.notifications.base import BaseNotifier
from trading_bot.risk.ai_confidence import AIConfidenceFilter
from trading_bot.risk.drawdown import DrawdownMonitor, TradingState
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.risk_state import RiskStateStore
from trading_bot.risk.stop_loss import StopLossManager
from trading_bot.risk.stop_tracker import StopTracker

# News window scored for sentiment, and articles kept per pair per fetch.
NEWS_WINDOW = timedelta(hours=24)
NEWS_PER_PAIR = 10


def _regime_size_factor(signal: Signal | None) -> float | None:
    """Risk multiplier the StrategySelector attached (VOLATILE regime)."""
    if signal is None:
        return None
    factor = signal.metadata.get("sizing_factor")
    return float(factor) if isinstance(factor, (int, float)) else None


class TradingOrchestrator:
    """Connects all trading components in an automated loop with scheduling."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scheduler: object | None = None
        self._running: bool = False
        self._closed: bool = False

        # Core components — initialised in start()
        self._fetcher: CCXTFetcher | None = None
        self._executor: BaseExecutor | None = None
        self._feature_engine: FeatureEngine | None = None
        self._strategy_selectors: dict[str, StrategySelector] = {}
        self._signal_combiner: SignalCombiner | None = None
        self._position_sizer: PositionSizer | None = None
        self._drawdown_monitor: DrawdownMonitor | None = None
        self._stop_tracker = StopTracker()
        self._risk_store: RiskStateStore | None = None
        self._ai_confidence_filter: AIConfidenceFilter | None = None
        self._notifier: BaseNotifier | None = None
        self._journal: TradeJournal | None = None

        # AI components — optional, feature-flagged
        self._sentiment_analyzer: BaseSentimentAnalyzer | None = None
        self._sentiment_aggregator: SentimentAggregator | None = None
        self._news_fetcher: BaseNewsFetcher | None = None
        self._news_cache = NewsSentimentCache(ttl=NEWS_WINDOW)
        self._fear_greed: FearGreedIndex | None = None
        self._predictors: dict[str, BasePredictor] = {}

        # LLM multi-agent components — optional, feature-flagged
        self._llm_orchestrator: Orchestrator | None = None
        self._llm_cost_tracker: SentimentCostTracker | None = None

        # State cache
        self._latest_sentiment: dict[str, AggregatedSentiment] = {}
        # Headlines behind each cached sentiment; they expire with it.
        self._latest_headlines: dict[str, list[str]] = {}
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

        await self.stop()
        logger.info("TradingOrchestrator stopped.")

    async def stop(self) -> None:
        """Graceful shutdown: stop scheduler, close fetcher, set running=False."""
        self._running = False
        if self._closed:
            return
        self._closed = True
        logger.info("Stopping TradingOrchestrator…")

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

        if self._executor is not None:
            await self._executor.close()

        if self._journal is not None:
            await self._journal.close()

    async def run_trading_cycle(self) -> None:
        """One complete trading cycle: price, check drawdown, process each pair."""
        assert self._executor is not None
        assert self._drawdown_monitor is not None

        # Fetch market data first so the portfolio is valued at current prices.
        # Signals/stops use closed candles only; the forming candle's close is
        # the freshest price and drives valuation and fills.
        market_data: dict[str, pd.DataFrame] = {}
        prices: dict[str, float] = {}
        for pair in self._settings.TRADING_PAIRS:
            try:
                closed, price = await self._fetch_market(pair)
            except Exception:
                logger.exception("Error fetching market data for {}", pair)
                continue
            market_data[pair] = closed
            if price is not None:
                prices[pair] = price
        self._executor.update_prices(prices)

        # Stops run before valuation and signals, and regardless of halts.
        await self._process_stops(market_data)

        try:
            balance = await self._executor.get_balance()
        except Exception:
            # Never feed a guessed value to the drawdown monitor.
            logger.exception("Could not value portfolio — skipping cycle")
            return
        portfolio_value = balance["total"]
        logger.info(
            "Trading cycle — portfolio: ${:,.2f} ({})",
            portfolio_value,
            self._drawdown_monitor.state.value,
        )

        # Latching kill-switch: liquidate and notify only on the transition.
        triggered = self._drawdown_monitor.update(portfolio_value)
        # Persist the latch before liquidating: a crash mid-liquidation must
        # still restart HALTED/paused.
        self._save_risk_state()
        if triggered is not None:
            await self._handle_risk_trigger(triggered)
            self._save_risk_state()
            await self._record_snapshot(prices, trigger=triggered.value.lower())
            return

        for pair, ohlcv in market_data.items():
            if pair not in prices:
                continue
            try:
                await self._process_pair(pair, ohlcv, portfolio_value, prices[pair])
            except Exception:
                logger.exception("Error processing {}", pair)
                continue

        await self._record_snapshot(prices)

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
            reversion = MeanReversion(
                pair=pair,
                feature_engine=self._feature_engine,
                enable_entries=s.MEAN_REVERSION_ENTRIES,
            )
            self._strategy_selectors[pair] = StrategySelector(
                trend,
                reversion,
                regime_detector,
                s.REGIME_VOLATILE_SIZING_FACTOR,
                use_regime=s.USE_REGIME_DETECTION,
            )

        # Risk components
        self._position_sizer = PositionSizer(
            stop_loss_mgr=StopLossManager(), settings=s
        )
        self._ai_confidence_filter = AIConfidenceFilter(settings=s)
        self._signal_combiner = SignalCombiner(settings=s)

        # Notifier
        if s.TELEGRAM_BOT_TOKEN and s.TELEGRAM_CHAT_ID:
            from trading_bot.notifications.telegram_notifier import TelegramNotifier

            self._notifier = TelegramNotifier(
                bot_token=s.TELEGRAM_BOT_TOKEN,
                chat_id=s.TELEGRAM_CHAT_ID,
            )
            logger.info("Telegram notifier initialised")

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

            self._executor = LiveExecutor(exchange=exchange, pairs=s.TRADING_PAIRS)
        else:
            self._executor = PaperExecutor(
                initial_balance=s.INITIAL_BALANCE,
                slippage_pct=s.PAPER_SLIPPAGE_PCT,
                fee_pct=s.PAPER_FEE_PCT,
            )

        # Drawdown baseline is the executor's real portfolio value (in live
        # mode the exchange account, not INITIAL_BALANCE), unless a persisted
        # risk state exists.
        initial_balance = await self._executor.get_balance()
        await self._load_risk_state(initial_balance["total"])

        if s.DB_PERSISTENCE:
            self._journal = await TradeJournal.connect(
                s.DATABASE_URL, exchange=s.EXCHANGE_ID, timeout_s=s.DB_TIMEOUT_S
            )
            if self._journal is not None:
                try:
                    held = {
                        pair: await self._executor.get_position(pair)
                        for pair in s.TRADING_PAIRS
                    }
                except Exception:
                    logger.exception("Could not read positions to reconcile trades")
                else:
                    await self._journal.reconcile(held)

        # Data fetcher
        # Same exchange as execution: valuation and stops must see its prices.
        self._fetcher = CCXTFetcher(
            exchange_id=s.EXCHANGE_ID,
            api_key=s.EXCHANGE_API_KEY,
            api_secret=s.EXCHANGE_API_SECRET,
            sandbox=s.EXCHANGE_SANDBOX,
            max_retries=s.OHLCV_FETCH_MAX_RETRIES,
            timeout_s=s.OHLCV_FETCH_TIMEOUT_S,
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
                self._sentiment_analyzer = self._build_sentiment_analyzer()
                self._sentiment_aggregator = SentimentAggregator()
                self._news_fetcher = self._build_news_fetcher()
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

                api_key = self._llm_api_key()
                agents = [
                    TechnicalAnalystAgent(s.LLM_PROVIDER, s.LLM_MODEL, api_key),
                    SentimentAnalystAgent(s.LLM_PROVIDER, s.LLM_MODEL, api_key),
                    RiskEvaluatorAgent(s.LLM_PROVIDER, s.LLM_MODEL, api_key),
                ]
                self._llm_orchestrator = Orchestrator(
                    agents, self._shared_llm_cost_tracker(), s.LLM_PROVIDER, s.LLM_MODEL
                )
                logger.info("LLM multi-agent orchestrator initialised")
            except Exception:
                logger.exception("Failed to init LLM agents — continuing without LLM")
                self._llm_orchestrator = None

        if s.USE_DL_PREDICTION:
            self._init_predictors()

    def _llm_api_key(self) -> str:
        s = self._settings
        return (
            s.ANTHROPIC_API_KEY if s.LLM_PROVIDER == "anthropic" else s.OPENAI_API_KEY
        )

    def _shared_llm_cost_tracker(self) -> SentimentCostTracker:
        """One daily LLM budget for the agents and LLM news sentiment."""
        if self._llm_cost_tracker is None:
            s = self._settings
            self._llm_cost_tracker = SentimentCostTracker(
                s.LLM_MAX_DAILY_COST_USD,
                state_path=self._mode_path(s.LLM_COST_STATE_PATH),
            )
        return self._llm_cost_tracker

    def _build_sentiment_analyzer(self) -> BaseSentimentAnalyzer:
        """News scorer for SENTIMENT_PROVIDER (CryptoBERT unless "llm")."""
        s = self._settings
        if str(s.SENTIMENT_PROVIDER).lower() == "llm":
            api_key = self._llm_api_key()
            if api_key:
                logger.info(
                    "News sentiment via LLM {}/{} (budget ${:.2f}/day, shared)",
                    s.LLM_PROVIDER,
                    s.LLM_MODEL,
                    s.LLM_MAX_DAILY_COST_USD,
                )
                return AsyncLLMSentimentAnalyzer(
                    self._shared_llm_cost_tracker(),
                    provider=s.LLM_PROVIDER,
                    model=s.LLM_MODEL,
                    api_key=api_key,
                )
            logger.warning(
                "SENTIMENT_PROVIDER=llm but no API key for LLM_PROVIDER={} — "
                "falling back to CryptoBERT",
                s.LLM_PROVIDER,
            )
        from trading_bot.ai.sentiment.cryptobert import CryptoBERTAnalyzer

        return CryptoBERTAnalyzer(model_name=s.SENTIMENT_MODEL)

    def _build_news_fetcher(self) -> BaseNewsFetcher | None:
        """News client for NEWS_PROVIDER; None (news off) without an API key."""
        from trading_bot.data.news import build_news_fetcher

        return build_news_fetcher(self._settings)

    def _init_predictors(self) -> None:
        """Load one trained model per pair; pairs without one get no ML."""
        s = self._settings
        self._predictors = {}
        if str(s.DL_MODEL_TYPE).lower() != "lightgbm":
            logger.warning(
                "DL_MODEL_TYPE={} is not implemented (only 'lightgbm') — "
                "ML prediction disabled",
                s.DL_MODEL_TYPE,
            )
            return

        from trading_bot.ai.prediction.base import model_path
        from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor

        for pair in s.TRADING_PAIRS:
            path = model_path(s.ML_MODEL_DIR, pair, s.TRADING_TIMEFRAME)
            if not path.exists():
                logger.info("{}: no ML model at {} — no prediction", pair, path)
                continue
            predictor = LightGBMPredictor(pair=pair, timeframe=s.TRADING_TIMEFRAME)
            try:
                predictor.load(str(path))
            except Exception:
                logger.exception("{}: failed to load ML model {}", pair, path)
                continue
            self._predictors[pair] = predictor
            logger.info("{}: ML model loaded from {}", pair, path)

    # ------------------------------------------------------------------
    # Private — per-pair pipeline
    # ------------------------------------------------------------------

    async def _process_pair(
        self,
        pair: str,
        ohlcv: pd.DataFrame,
        portfolio_value: float,
        price: float,
    ) -> None:
        """Full pipeline for one pair: signal -> risk -> position -> execute."""
        assert self._executor is not None
        assert self._signal_combiner is not None
        assert self._ai_confidence_filter is not None
        assert self._position_sizer is not None

        # A. Validate OHLCV (closed candles, fetched by run_trading_cycle)
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

        if combined.type == SignalType.NEUTRAL:
            logger.debug("{}: NEUTRAL signal, skipping", pair)
            return

        # E. Map the signal onto the current spot position
        entry_price = price
        position_qty = await self._executor.get_position(pair)

        if combined.type in (SignalType.CLOSE, SignalType.SHORT):
            if position_qty <= 0:
                if combined.type == SignalType.SHORT:
                    logger.info(
                        "{}: SHORT ignored — spot-only bot with no position", pair
                    )
                else:
                    logger.debug("{}: CLOSE with no position, skipping", pair)
                return
            # Exits skip the confidence filters: a weak exit signal must never
            # keep a position open.
            await self._exit_position(pair, position_qty, entry_price)
            return

        # F. Entry (LONG): only when flat, trading allowed and confident
        if position_qty > 0:
            logger.debug(
                "{}: LONG but already holding {:.8f}, skipping", pair, position_qty
            )
            return

        assert self._drawdown_monitor is not None
        if not self._drawdown_monitor.entries_allowed:
            logger.info(
                "{}: LONG blocked — risk state {} ({})",
                pair,
                self._drawdown_monitor.state.value,
                self._drawdown_monitor.halt_reason,
            )
            return

        if not combined.should_trade:
            logger.debug("{}: should_trade=False, skipping", pair)
            return

        if not self._ai_confidence_filter.should_trade(combined):
            logger.debug("{}: AI confidence filter blocked", pair)
            return

        position_size = self._position_sizer.calculate(
            pair=pair,
            signal_confidence=combined.combined_confidence,
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            side="buy",
            ai_size_multiplier=self._position_sizer.ai_size_multiplier(combined),
            regime_size_multiplier=_regime_size_factor(signal),
            ohlcv=ohlcv,
        )
        if position_size.quantity <= 0:
            logger.debug("{}: position size zero, skipping", pair)
            return

        # G. Execute and protect with a stop
        result = await self._execute_order(
            pair, "buy", position_size.quantity, entry_price
        )
        if not await self._report_fill(result, "signal_combiner"):
            return
        if self._journal is not None:
            await self._journal.record_entry(
                result,
                strategy=signal.source if signal is not None else "signal_combiner",
                confidence=combined.combined_confidence,
            )
        await self._register_stop(pair, position_size.stop_loss_price, ohlcv)

    async def _fetch_ohlcv(self, pair: str) -> pd.DataFrame:
        """Fetch the lookback window plus the forming candle (may be partial)."""
        assert self._fetcher is not None
        s = self._settings
        bars = s.OHLCV_LOOKBACK_BARS + 1  # +1: the forming candle is dropped
        since = datetime.now(UTC) - (bars + 1) * timeframe_to_timedelta(
            s.TRADING_TIMEFRAME
        )
        return await self._fetcher.fetch_ohlcv(
            pair=pair,
            timeframe=s.TRADING_TIMEFRAME,
            since=since,
            limit=bars,
        )

    async def _fetch_market(self, pair: str) -> tuple[pd.DataFrame, float | None]:
        """Closed candles for signals and the latest traded price."""
        raw = await self._fetch_ohlcv(pair)
        return split_closed_candles(
            raw, self._settings.TRADING_TIMEFRAME, datetime.now(UTC)
        )

    def _build_ai_signals(self, pair: str, ohlcv: pd.DataFrame) -> AISignals | None:
        """Build AISignals from cached sentiment + inline ML prediction + LLM."""
        sentiment = self._fresh_sentiment(pair)
        prediction_conf: float | None = None
        prediction_ret: float | None = None
        prediction_dir: str | None = None

        predictor = self._predictors.get(pair)
        if predictor is not None and predictor.is_loaded():
            try:
                features = (
                    self._feature_engine.build_features(ohlcv)
                    if self._feature_engine is not None
                    else None
                )
                pred = predictor.predict(ohlcv, features=features)
                prediction_conf = pred.confidence
                prediction_ret = pred.predicted_return
                prediction_dir = pred.predicted_direction
            except PredictionUnavailableError as exc:
                logger.debug("{}: no ML prediction ({})", pair, exc)
            except Exception:
                logger.exception("{}: ML prediction failed", pair)

        # LLM multi-agent decision from cache
        llm_decision = self._fresh_llm_decision(pair)
        llm_rec: str | None = None
        llm_conf: float | None = None
        llm_consensus: bool | None = None
        if llm_decision is not None:
            llm_rec = llm_decision.final_recommendation
            llm_conf = llm_decision.confidence
            llm_consensus = llm_decision.consensus

        if sentiment is None and prediction_conf is None and llm_rec is None:
            return None

        return AISignals(
            sentiment_score=(sentiment.overall_score if sentiment else None),
            prediction_return=prediction_ret,
            prediction_confidence=prediction_conf,
            prediction_direction=prediction_dir,
            llm_recommendation=llm_rec,
            llm_confidence=llm_conf,
            llm_consensus=llm_consensus,
        )

    def _is_fresh(self, pair: str, kind: str, ts: datetime, interval_min: int) -> bool:
        """Cached AI results expire after 2x their job interval."""
        age = datetime.now(UTC) - ts
        if age <= timedelta(minutes=2 * interval_min):
            return True
        logger.debug("{}: ignoring stale {} ({} old)", pair, kind, age)
        return False

    def _fresh_sentiment(self, pair: str) -> AggregatedSentiment | None:
        sentiment = self._latest_sentiment.get(pair)
        if sentiment is None or not self._is_fresh(
            pair,
            "sentiment",
            sentiment.timestamp,
            self._settings.SENTIMENT_INTERVAL_MIN,
        ):
            return None
        return sentiment

    def _fresh_llm_decision(self, pair: str) -> OrchestratorDecision | None:
        decision = self._latest_llm_decision.get(pair)
        if decision is None or not self._is_fresh(
            pair,
            "LLM decision",
            decision.timestamp,
            self._settings.LLM_ANALYSIS_INTERVAL_MIN,
        ):
            return None
        return decision

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
            await self._notify_llm_budget()

    async def _notify_llm_budget(self) -> None:
        """Budget alerts, once per UTC day each: 50 % log, 80 % and 100 % Telegram."""
        tracker = self._llm_cost_tracker
        if tracker is None:
            return
        for threshold in tracker.pop_alerts():
            spent, cap = tracker.daily_spent_usd, tracker.max_daily_usd
            if threshold >= EXHAUSTED:
                message = (
                    f"LLM budget exhausted: ${spent:.2f} of ${cap:.2f} spent today. "
                    "LLM analysis paused until the next UTC day."
                )
            else:
                message = (
                    f"LLM spend at {threshold:.0%} of the daily budget: "
                    f"${spent:.2f} of ${cap:.2f}."
                )
            if threshold < 0.8:
                logger.info(message)
                continue
            logger.warning(message)
            if self._notifier is not None:
                try:
                    await self._notifier.send_alert(message, level="warning")
                except Exception:
                    logger.exception("Failed to send LLM budget alert")

    async def _build_llm_context(self, pair: str) -> dict[str, object]:
        """Build unified context dict for all LLM agents."""
        s = self._settings
        context: dict[str, object] = {"pair": pair, "timeframe": s.TRADING_TIMEFRAME}

        # Technical indicators
        try:
            ohlcv, _ = await self._fetch_market(pair)
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

        # Sentiment from cache (expired entries are ignored)
        sentiment = self._fresh_sentiment(pair)
        context["sentiment_score"] = sentiment.overall_score if sentiment else 0.0
        context["fear_greed"] = sentiment.fear_greed_score if sentiment else "N/A"
        context["headlines"] = (
            list(self._latest_headlines.get(pair, [])) if sentiment else []
        )

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
        context["open_positions"] = await self._count_open_positions()
        context["max_risk_pct"] = s.MAX_RISK_PER_TRADE_PCT
        context["max_position_pct"] = s.MAX_POSITION_PCT
        context["max_drawdown_pct"] = s.MAX_DRAWDOWN_PCT

        return context

    async def _count_open_positions(self) -> int:
        if self._executor is None:
            return 0
        count = 0
        for pair in self._settings.TRADING_PAIRS:
            try:
                if await self._executor.get_position(pair) > 0:
                    count += 1
            except Exception:
                logger.warning("{}: could not read position", pair)
        return count

    async def _run_sentiment_cycle(self) -> None:
        """Separate job: fetch news/fear-greed, analyse, cache results.

        News and Fear & Greed fail independently. Fear & Greed is global, so
        it is fetched once per cycle; news comes in one request for all pairs
        and only articles not seen before are scored (NewsSentimentCache).
        """
        if self._sentiment_analyzer is None or self._sentiment_aggregator is None:
            return

        logger.debug("Sentiment cycle starting…")
        pairs = list(self._settings.TRADING_PAIRS)
        now = datetime.now(UTC)

        fear_greed = await self._fetch_fear_greed()
        await self._update_news_sentiment(pairs, now)

        for pair in pairs:
            try:
                sentiment_results = self._news_cache.results_for(pair, now)
                if fear_greed is not None:
                    sentiment_results.append(fear_greed)
                if not sentiment_results:
                    continue
                aggregated = self._sentiment_aggregator.aggregate(
                    sentiment_results, pair=pair
                )
                self._latest_sentiment[pair] = aggregated
                self._latest_headlines[pair] = self._news_cache.headlines_for(
                    pair, limit=NEWS_PER_PAIR, now=now
                )
                logger.debug(
                    "{}: sentiment score={:.2f} ({} sources)",
                    pair,
                    aggregated.overall_score,
                    aggregated.num_sources_analyzed,
                )
            except Exception:
                logger.exception("Sentiment aggregation failed for {}", pair)

    async def _fetch_fear_greed(self) -> SentimentResult | None:
        """Global Fear & Greed as a sentiment result in [-1, 1]; None on failure."""
        if self._fear_greed is None:
            return None
        try:
            fg_data = await self._fear_greed.fetch_latest()
        except Exception:
            logger.exception("Fear & Greed fetch failed")
            return None
        if not fg_data:
            return None
        fg_value = int(str(fg_data.get("value", 50)))
        return SentimentResult(
            score=(fg_value - 50) / 50,
            confidence=0.7,
            label="fear_greed",
            model_used="fear_greed_index",
            analyzed_text=f"Fear & Greed: {fg_value}",
        )

    async def _update_news_sentiment(self, pairs: list[str], now: datetime) -> None:
        """Fetch news once for all pairs and score only unseen articles."""
        assert self._sentiment_analyzer is not None
        self._news_cache.purge(now)
        if self._news_fetcher is None:
            return
        try:
            by_pair = await self._news_fetcher.fetch_for_pairs(
                pairs, since=now - NEWS_WINDOW, limit_per_pair=NEWS_PER_PAIR
            )
        except Exception:
            # Cached scores stay valid until they expire.
            logger.exception("News fetch failed — using cached news sentiment")
            return

        self._news_cache.tag(by_pair)
        pending = self._news_cache.missing(by_pair)
        if not pending:
            return
        keys = list(pending)
        texts = [f"{pending[k][0].title}. {pending[k][0].body}" for k in keys]
        analyzer = self._sentiment_analyzer
        results: list[SentimentResult | None]
        try:
            if isinstance(analyzer, AsyncLLMSentimentAnalyzer):
                results = await analyzer.analyze_batch_async(texts)
            else:
                # Transformer inference is CPU-bound and synchronous: keep it
                # off the event loop (trading cycle, stops).
                results = list(await asyncio.to_thread(analyzer.analyze_batch, texts))
        except Exception:
            logger.exception("News sentiment analysis failed")
            return
        finally:
            await self._notify_llm_budget()
        for key, result in zip(keys, results, strict=False):
            if result is None:  # failed or over budget: retried next cycle
                continue
            article, article_pairs = pending[key]
            self._news_cache.store(key, article, result, article_pairs)
        logger.debug(
            "News sentiment: {} new articles scored, {} cached",
            len(keys),
            len(self._news_cache),
        )

    async def _execute_order(
        self, pair: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Send a market order via the executor."""
        assert self._executor is not None
        return await self._executor.place_order(
            pair=pair,
            side=side,
            order_type="market",
            quantity=quantity,
            price=price,
        )

    async def _report_fill(self, result: OrderResult, strategy: str) -> bool:
        """Log and notify a fill; False if nothing was filled."""
        if result.filled_quantity <= 0:
            logger.warning(
                "{}: {} order not filled (status={})",
                result.pair,
                result.side,
                result.status,
            )
            return False
        logger.info(
            "{}: {} {} qty={:.6f} @ ${:,.2f} (fee=${:.2f}) [{}]",
            result.pair,
            result.side,
            result.status,
            result.filled_quantity,
            result.filled_price,
            result.fee,
            strategy,
        )
        if self._notifier is not None:
            await self._notifier.send_trade_notification(
                {
                    "side": result.side,
                    "pair": result.pair,
                    "filled_price": result.filled_price,
                    "filled_quantity": result.filled_quantity,
                    "status": result.status,
                    "strategy": strategy,
                }
            )
        return True

    async def _record_snapshot(
        self, prices: dict[str, float], trigger: str = "cycle"
    ) -> None:
        """One portfolio snapshot per cycle; skipped without a journal."""
        if self._journal is None or self._executor is None:
            return
        try:
            balance = await self._executor.get_balance()
            held = {
                pair: await self._executor.get_position(pair)
                for pair in self._settings.TRADING_PAIRS
            }
        except Exception:
            logger.exception("Could not value portfolio for the snapshot")
            return
        positions_value = balance.get("positions_value")
        if positions_value is None:
            positions_value = sum(
                qty * prices.get(pair, 0.0) for pair, qty in held.items() if qty > 0
            )
        dd = self._drawdown_monitor
        await self._journal.record_snapshot(
            total_value=balance["total"],
            positions_value=float(positions_value),
            prices=prices,
            drawdown_pct=dd.current_drawdown_pct if dd else 0.0,
            open_positions=sum(1 for qty in held.values() if qty > 0),
            trigger=trigger,
        )

    # ------------------------------------------------------------------
    # Private — stops
    # ------------------------------------------------------------------

    async def _register_stop(
        self, pair: str, stop_price: float, ohlcv: pd.DataFrame
    ) -> None:
        """Protect a freshly filled entry: native stop if possible, else watched."""
        assert self._executor is not None
        if stop_price <= 0:
            logger.warning("{}: no valid stop price, position unprotected", pair)
            return
        quantity = await self._executor.get_position(pair)
        if quantity <= 0:
            return
        order_id = await self._executor.place_stop_loss(pair, quantity, stop_price)
        # The candle forming at entry (last closed + one bar) is never checked:
        # its low may predate the fill.
        entry_candle = (
            ohlcv.index[-1] + timeframe_to_timedelta(self._settings.TRADING_TIMEFRAME)
            if isinstance(ohlcv.index, pd.DatetimeIndex) and len(ohlcv) > 0
            else None
        )
        self._stop_tracker.register(pair, stop_price, quantity, order_id, entry_candle)
        logger.info(
            "{}: stop-loss @ ${:,.2f} for {:.8f} ({})",
            pair,
            stop_price,
            quantity,
            f"exchange order {order_id}" if order_id else "watched by bot",
        )
        self._save_risk_state()

    async def _process_stops(self, market_data: dict[str, pd.DataFrame]) -> None:
        """Detect stops filled on the exchange or hit by candles; exit if so."""
        for pair in self._stop_tracker.pairs():
            stop = self._stop_tracker.get(pair)
            if stop is None:
                continue
            try:
                if stop.order_id is not None:
                    await self._check_exchange_stop(pair, stop.order_id)
                    continue
                ohlcv = market_data.get(pair)
                if ohlcv is None:
                    continue
                fill_price = self._stop_tracker.check_hit(pair, ohlcv)
                if fill_price is not None:
                    await self._execute_stop(pair, fill_price)
            except Exception:
                logger.exception("{}: stop processing failed", pair)
        self._save_risk_state()

    async def _check_exchange_stop(self, pair: str, order_id: str) -> None:
        assert self._executor is not None
        order = await self._executor.get_order(order_id, pair)
        status = str(order.get("status") or "")
        if status == "open":
            return
        if status == "closed":
            self._stop_tracker.remove(pair)
            price = float(order.get("average") or order.get("price") or 0.0)
            qty = float(order.get("filled") or 0.0)
            if self._journal is not None:
                fee = order.get("fee")
                await self._journal.record_exit(
                    OrderResult(
                        order_id=order_id,
                        exchange_order_id=order_id,
                        pair=pair,
                        side="sell",
                        filled_price=price,
                        filled_quantity=qty,
                        fee=float(fee.get("cost") or 0.0)
                        if isinstance(fee, dict)
                        else 0.0,
                        slippage_pct=0.0,
                        status="closed",
                        timestamp=datetime.now(UTC),
                    ),
                    strategy="stop_loss",
                    reason="stop_loss",
                )
            await self._notify_stop(pair, qty, price)
            return
        # Canceled/expired/rejected outside the bot: keep protecting it.
        stop = self._stop_tracker.get(pair)
        if stop is not None and await self._executor.get_position(pair) > 0:
            logger.warning(
                "{}: exchange stop {} is {}; bot will watch the stop",
                pair,
                order_id,
                status or "unknown",
            )
            stop.order_id = None
        else:
            self._stop_tracker.remove(pair)

    async def _execute_stop(self, pair: str, fill_price: float) -> None:
        """Bot-side stop: sell the whole position at ``fill_price``."""
        assert self._executor is not None
        quantity = await self._executor.get_position(pair)
        if quantity <= 0:
            self._stop_tracker.remove(pair)
            return
        logger.warning("{}: STOP-LOSS hit @ ${:,.2f}", pair, fill_price)
        result = await self._execute_order(pair, "sell", quantity, fill_price)
        if not await self._report_fill(result, "stop_loss"):
            return
        if self._journal is not None:
            await self._journal.record_exit(
                result, strategy="stop_loss", reason="stop_loss"
            )
        self._stop_tracker.remove(pair)
        await self._notify_stop(pair, result.filled_quantity, result.filled_price)

    async def _notify_stop(self, pair: str, quantity: float, price: float) -> None:
        logger.warning(
            "{}: stop-loss executed — {:.8f} @ ${:,.2f}", pair, quantity, price
        )
        if self._notifier is not None:
            await self._notifier.send_alert(
                f"STOP-LOSS executed on {pair}: sold {quantity:.8f} @ ${price:,.2f}.",
                level="warning",
            )

    async def _exit_position(self, pair: str, quantity: float, price: float) -> None:
        """Signal exit: cancel the native stop first, then sell everything."""
        assert self._executor is not None
        stop = self._stop_tracker.get(pair)
        cancelled_native = False
        if stop is not None and stop.order_id is not None:
            # The stop locks the balance; the sell fails while it is open.
            cancelled_native = await self._executor.cancel_order(stop.order_id, pair)
            if not cancelled_native:
                logger.warning(
                    "{}: could not cancel stop {}; sell may fail", pair, stop.order_id
                )

        try:
            result = await self._execute_order(pair, "sell", quantity, price)
        except Exception:
            if cancelled_native and stop is not None:
                stop.order_id = None  # keep watching it bot-side
                self._save_risk_state()
            raise

        if await self._report_fill(result, "signal_combiner"):
            self._stop_tracker.remove(pair)
            if self._journal is not None:
                await self._journal.record_exit(
                    result, strategy="signal_combiner", reason="signal"
                )
        elif cancelled_native and stop is not None:
            stop.order_id = None
        self._save_risk_state()

    async def _reconcile_stops(self, native_orders_cancelled: bool) -> None:
        """Drop stops of flat pairs; demote cancelled native stops to watched."""
        assert self._executor is not None
        for pair in self._stop_tracker.pairs():
            stop = self._stop_tracker.get(pair)
            if stop is None:
                continue
            try:
                held = await self._executor.get_position(pair)
            except Exception:
                logger.exception("{}: could not reconcile stop", pair)
                continue
            if held <= 0:
                self._stop_tracker.remove(pair)
            elif native_orders_cancelled:
                stop.order_id = None

    # ------------------------------------------------------------------
    # Private — kill-switch and persisted risk state
    # ------------------------------------------------------------------

    async def _handle_risk_trigger(self, state: TradingState) -> None:
        """Liquidate and notify once, on entering DAILY_PAUSED or HALTED."""
        assert self._executor is not None
        assert self._drawdown_monitor is not None
        reason = self._drawdown_monitor.halt_reason
        if state == TradingState.HALTED:
            message = (
                f"KILL SWITCH — {reason}. Emergency liquidation in progress. "
                "Trading HALTED until manual reset (RISK_RESET=true)."
            )
        else:
            message = (
                f"DAILY LOSS LIMIT — {reason}. Liquidating; new entries paused "
                "until the next UTC day."
            )
        logger.warning(message)
        if self._notifier is not None:
            await self._notifier.send_alert(message, level="kill_switch")
        try:
            results = await self._executor.close_all_positions()
        finally:
            await self._reconcile_stops(native_orders_cancelled=True)
        if self._journal is not None:
            reason = (
                "kill_switch" if state == TradingState.HALTED else "daily_loss_limit"
            )
            for result in results:
                if result.filled_quantity > 0:
                    await self._journal.record_exit(
                        result, strategy="risk_manager", reason=reason
                    )

    def _mode_path(self, template: str) -> Path:
        """State file path; "{mode}" keeps paper and live state apart."""
        return Path(str(template).replace("{mode}", str(self._settings.BOT_MODE)))

    def _risk_state_path(self) -> Path:
        return self._mode_path(self._settings.RISK_STATE_PATH)

    async def _load_risk_state(self, portfolio_value: float) -> None:
        """Restore drawdown latch and stops from disk, or start fresh."""
        self._risk_store = RiskStateStore(self._risk_state_path())
        data = self._risk_store.load()

        monitor: DrawdownMonitor | None = None
        tracker = StopTracker()
        if data is not None:
            try:
                monitor = DrawdownMonitor.from_dict(
                    data["drawdown"], settings=self._settings
                )
                tracker = StopTracker.from_dict(data.get("stops") or {})
            except (KeyError, TypeError, ValueError, AttributeError):
                logger.warning(
                    "Risk state at {} is invalid — starting fresh",
                    self._risk_store.path,
                )
                monitor, tracker = None, StopTracker()

        if monitor is None:
            monitor = DrawdownMonitor(
                initial_value=portfolio_value, settings=self._settings
            )
            logger.info("Drawdown baseline: ${:,.2f}", portfolio_value)
        else:
            logger.info(
                "Risk state restored from {} — state={}, stops={}",
                self._risk_store.path,
                monitor.state.value,
                len(tracker),
            )
            if self._settings.BOT_MODE != "live":
                # The paper account restarts at INITIAL_BALANCE: an old peak
                # would read as a drawdown. Keep the latch, re-anchor values.
                monitor.rebaseline(portfolio_value)

        if self._settings.RISK_RESET is True:
            logger.warning(
                "RISK_RESET=true — clearing risk state {} and re-baselining at "
                "${:,.2f}. Remove RISK_RESET from the environment.",
                monitor.state.value,
                portfolio_value,
            )
            monitor.reset(portfolio_value)

        self._drawdown_monitor = monitor
        self._stop_tracker = tracker
        await self._reconcile_stops(native_orders_cancelled=False)
        self._save_risk_state()

    def _save_risk_state(self) -> None:
        if self._risk_store is None or self._drawdown_monitor is None:
            return
        try:
            self._risk_store.save(
                {
                    "version": 1,
                    "mode": str(self._settings.BOT_MODE),
                    "saved_at": datetime.now(UTC).isoformat(),
                    "drawdown": self._drawdown_monitor.to_dict(),
                    "stops": self._stop_tracker.to_dict(),
                }
            )
        except Exception:
            logger.exception("Failed to persist risk state")

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
