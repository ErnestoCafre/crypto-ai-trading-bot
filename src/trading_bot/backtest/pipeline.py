"""Build the live technical pipeline for backtesting.

Mirrors ``TradingOrchestrator._init_components`` (``core/bot.py``): per-pair
:class:`StrategySelector` over TrendFollowing + MeanReversion routed by
:class:`RegimeDetector`, ATR stops through :class:`PositionSizer`, and the
:class:`SignalCombiner` + :class:`AIConfidenceFilter` entry gates, and
regime / AI size multipliers (a VOLATILE regime shrinks the position via
``signal.metadata["sizing_factor"]``; ``USE_REGIME_DETECTION=false`` always
routes to trend following). All components share one
:class:`PrecomputedFeatureEngine`, so indicators are computed once per run.

Not simulated: AI signals (sentiment / ML / LLM — pass
``ai_signals_series`` to ``BacktestEngine.run`` if you have point-in-time
history), the drawdown kill-switch and exchange-side stop orders. Live
features are computed over a rolling ``OHLCV_LOOKBACK_BARS`` window rather
than the full history; for the indicators the strategies use (EMA<=50,
RSI/ADX/ATR 14, BB 20) the difference is negligible after warm-up.
"""

from __future__ import annotations

from typing import Any

from trading_bot.ai.prediction.feature_engine import PrecomputedFeatureEngine
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.regime import RegimeDetector
from trading_bot.core.signal_combiner import SignalCombiner
from trading_bot.core.strategies.mean_reversion import MeanReversion
from trading_bot.core.strategies.trend_following import TrendFollowing
from trading_bot.core.strategy_selector import StrategySelector
from trading_bot.risk.ai_confidence import AIConfidenceFilter
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.stop_loss import StopLossManager

TREND_PARAMS = (
    "ema_fast",
    "ema_mid",
    "ema_slow",
    "rsi_entry_min",
    "rsi_exit_below",
    "adx_min",
)
REVERSION_PARAMS = ("rsi_oversold", "rsi_overbought", "rsi_exit_low", "rsi_exit_high")


def build_live_selector(
    pair: str,
    feature_engine: PrecomputedFeatureEngine,
    settings: Settings | None = None,
    **strategy_params: Any,
) -> StrategySelector:
    """StrategySelector configured like the live bot.

    ``strategy_params`` override TrendFollowing / MeanReversion thresholds
    (see ``TREND_PARAMS`` and ``REVERSION_PARAMS``); unknown names raise.
    """
    s = settings or get_settings()
    unknown = set(strategy_params) - set(TREND_PARAMS) - set(REVERSION_PARAMS)
    if unknown:
        raise TypeError(f"unknown strategy parameters: {sorted(unknown)}")

    trend_kw = {k: v for k, v in strategy_params.items() if k in TREND_PARAMS}
    rev_kw = {k: v for k, v in strategy_params.items() if k in REVERSION_PARAMS}
    trend = TrendFollowing(pair=pair, feature_engine=feature_engine, **trend_kw)
    rev_kw.setdefault("enable_entries", s.MEAN_REVERSION_ENTRIES)
    reversion = MeanReversion(pair=pair, feature_engine=feature_engine, **rev_kw)
    detector = RegimeDetector(
        adx_trending_threshold=s.REGIME_ADX_TRENDING,
        volatility_high_percentile=s.REGIME_VOLATILITY_HIGH_PCT,
        feature_engine=feature_engine,
    )
    return StrategySelector(
        trend,
        reversion,
        detector,
        s.REGIME_VOLATILE_SIZING_FACTOR,
        use_regime=s.USE_REGIME_DETECTION,
    )


def build_live_engine(
    pair: str,
    initial_capital: float = 100_000.0,
    fee_pct: float = 0.1,
    slippage_pct: float = 0.05,
    settings: Settings | None = None,
    **strategy_params: Any,
) -> BacktestEngine:
    """BacktestEngine running the same technical pipeline as the live bot."""
    ema_periods = [
        int(strategy_params[k])
        for k in ("ema_fast", "ema_mid", "ema_slow")
        if k in strategy_params
    ]
    s = settings or get_settings()
    fe = PrecomputedFeatureEngine(extra_ema_periods=ema_periods)
    selector = build_live_selector(pair, fe, s, **strategy_params)
    return BacktestEngine(
        strategy=selector,
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        position_sizer=PositionSizer(stop_loss_mgr=StopLossManager(), settings=s),
        signal_combiner=SignalCombiner(settings=s),
        ai_confidence_filter=AIConfidenceFilter(settings=s),
        feature_engine=fe,
        allow_short=False,
    )
