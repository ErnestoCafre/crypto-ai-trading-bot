"""Backtesting engine — bar-by-bar simulation of the live technical pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from trading_bot.ai.prediction.feature_engine import PrecomputedFeatureEngine
from trading_bot.backtest.data_types import BacktestResult, BacktestTrade
from trading_bot.backtest.metrics import MetricsCalculator
from trading_bot.core.models import AISignals, Signal, SignalType
from trading_bot.core.strategies.base import BaseStrategy
from trading_bot.core.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


@dataclass
class _Position:
    side: str  # "long" / "short"
    quantity: float
    entry_price: float
    entry_time: datetime
    entry_fee: float
    entry_slippage: float
    stop_price: float | None
    regime: str | None
    entry_risk_pct: float = 0.0
    size_capped: bool = False


class BacktestEngine:
    """Event-driven backtest engine.

    Anti look-ahead guarantee: the signal at bar *i* is generated from
    ``ohlcv[:i+1]`` (closed candles only, as in live trading); execution
    happens at the **open** of bar ``i+1``. Stops are checked on every bar
    after the entry: a bar that opens through the stop (gap) fills at its
    open, otherwise a touch fills at the stop — the same rule as
    ``risk.stop_tracker.StopTracker``.

    Position rules mirror the spot-only live bot (``core.bot``): LONG opens
    only when flat, CLOSE/SHORT close the whole position, SHORT while flat is
    ignored and LONG while holding is ignored. ``allow_short=True`` enables a
    symmetric short book for research; it is not what the live bot does.

    ``strategy`` may be a single :class:`BaseStrategy` or the live
    :class:`StrategySelector` (regime routing + volatile-regime reduction).
    When ``signal_combiner`` is given, entries additionally require
    ``combine(...).should_trade`` — the live confidence gate. Use
    :func:`trading_bot.backtest.pipeline.build_live_engine` to get the same
    components the bot builds.
    """

    def __init__(
        self,
        strategy: BaseStrategy | StrategySelector,
        initial_capital: float = 100_000.0,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
        position_sizer: Any = None,
        stop_loss_manager: Any = None,
        signal_combiner: Any = None,
        feature_engine: PrecomputedFeatureEngine | None = None,
        allow_short: bool = False,
        ai_confidence_filter: Any = None,
    ) -> None:
        self._strategy = strategy
        self._initial_capital = initial_capital
        self._fee_pct = fee_pct
        self._slippage_pct = slippage_pct
        self._position_sizer = position_sizer
        self._stop_loss_manager = stop_loss_manager
        self._signal_combiner = signal_combiner
        self._feature_engine = feature_engine
        self._allow_short = allow_short
        self._ai_confidence_filter = ai_confidence_filter

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    @property
    def required_lookback(self) -> int:
        """Bars the strategy needs before its first signal (warm-up)."""
        return int(self._strategy.get_required_lookback())

    @property
    def slippage_pct(self) -> float:
        return self._slippage_pct

    def run(
        self,
        ohlcv: pd.DataFrame,
        pair: str,
        ai_signals_series: pd.Series | None = None,
        trade_from: int = 0,
    ) -> BacktestResult:
        """Run a backtest on *ohlcv* and return trades, equity and metrics.

        Bars before ``trade_from`` are warm-up only: indicators see them but
        no signal is acted on and the equity curve, trades and metrics cover
        ``ohlcv[trade_from:]`` exclusively (walk-forward out-of-sample).
        ``ai_signals_series`` optionally maps a bar timestamp to the
        :class:`AISignals` available at that bar's close.
        """
        n = len(ohlcv)
        if not 0 <= trade_from < max(n, 1):
            raise ValueError(f"trade_from={trade_from} outside 0..{n - 1}")
        if n - trade_from < 2:
            return self._empty_result(ohlcv, pair, trade_from)

        if self._feature_engine is not None:
            self._feature_engine.prime(ohlcv)
        try:
            return self._run(ohlcv, pair, ai_signals_series, trade_from)
        finally:
            if self._feature_engine is not None:
                self._feature_engine.clear()

    def _run(
        self,
        ohlcv: pd.DataFrame,
        pair: str,
        ai_signals_series: pd.Series | None,
        trade_from: int,
    ) -> BacktestResult:
        n = len(ohlcv)
        opens = ohlcv["open"].to_numpy(dtype=float)
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        closes = ohlcv["close"].to_numpy(dtype=float)

        cash = self._initial_capital
        pos: _Position | None = None
        trades: list[BacktestTrade] = []
        equity_values: list[float] = []
        lookback = self._strategy.get_required_lookback()

        for i in range(trade_from, n):
            # -- Stop-loss on bar i (every bar after the entry fill) --
            if pos is not None and pos.stop_price is not None:
                fill = _stop_fill(pos, opens[i], highs[i], lows[i])
                if fill is not None:
                    trade, cash_back = self._close(
                        pos, fill, _bar_time(ohlcv, i), "stop_loss", pair
                    )
                    trades.append(trade)
                    cash += cash_back
                    pos = None

            equity_values.append(cash + _position_value(pos, closes[i]))

            # -- Signal on closed bar i, executed at the open of bar i+1 --
            if i + 1 < lookback or i + 1 >= n:
                continue

            window = ohlcv.iloc[: i + 1]
            signal, regime = self._generate(window)
            ai = _lookup_ai(ai_signals_series, ohlcv.index[i])

            combined: Any = None
            if self._signal_combiner is not None:
                if signal is None and ai is None:
                    continue
                combined = self._signal_combiner.combine(signal, ai)
                sig_type = combined.type
                confidence = float(combined.combined_confidence)
                entry_allowed = bool(combined.should_trade) and (
                    self._ai_confidence_filter is None
                    or bool(self._ai_confidence_filter.should_trade(combined))
                )
            elif signal is not None:
                sig_type = signal.type
                confidence = signal.confidence
                entry_allowed = True
            else:
                continue

            next_open = opens[i + 1]
            next_time = _bar_time(ohlcv, i + 1)
            entry_side = self._entry_side(sig_type)

            if pos is not None:
                if pos.side in _closes(sig_type):
                    trade, cash_back = self._close(
                        pos, next_open, next_time, "signal", pair
                    )
                    trades.append(trade)
                    cash += cash_back
                    pos = None
                continue

            if entry_side is None or not entry_allowed:
                continue

            equity = cash  # flat: all equity is cash
            pos, cost = self._open(
                entry_side,
                next_open,
                next_time,
                equity,
                confidence,
                window,
                pair,
                ai_multiplier=self._ai_multiplier(combined),
                regime_multiplier=_regime_size_factor(signal),
            )
            if pos is None:
                continue
            pos.regime = regime
            cash -= cost

        # -- Close any remaining position at the last close --
        if pos is not None:
            trade, cash_back = self._close(
                pos, closes[-1], _bar_time(ohlcv, n - 1), "end_of_data", pair
            )
            trades.append(trade)
            cash += cash_back
            equity_values[-1] = cash

        equity_curve = pd.Series(
            equity_values, index=ohlcv.index[trade_from:], name="equity"
        )
        metrics = MetricsCalculator.calculate(trades, equity_curve)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            parameters=self._parameters(trade_from),
            start_date=_bar_time(ohlcv, trade_from),
            end_date=_bar_time(ohlcv, n - 1),
            pair=pair,
            strategy_name=self._strategy.name,
        )

    # ------------------------------------------------------------------
    # Signal / position helpers
    # ------------------------------------------------------------------

    def _generate(self, window: pd.DataFrame) -> tuple[Signal | None, str | None]:
        if isinstance(self._strategy, StrategySelector):
            signal, classification = self._strategy.generate_signal(window)
            return signal, classification.regime.value
        return self._strategy.generate_signal(window), None

    def _entry_side(self, sig_type: SignalType) -> str | None:
        if sig_type == SignalType.LONG:
            return "long"
        if sig_type == SignalType.SHORT and self._allow_short:
            return "short"
        return None

    def _open(
        self,
        side: str,
        raw_price: float,
        bar_time: datetime,
        equity: float,
        confidence: float,
        window: pd.DataFrame,
        pair: str,
        ai_multiplier: float | None = None,
        regime_multiplier: float | None = None,
    ) -> tuple[_Position | None, float]:
        """Size and fill an entry; returns the position and the cash it uses."""
        slip = raw_price * (self._slippage_pct / 100)
        fill = raw_price + slip if side == "long" else raw_price - slip
        if fill <= 0:
            return None, 0.0

        qty, stop, capped = self._size(
            side,
            fill,
            equity,
            confidence,
            window,
            pair,
            ai_multiplier,
            regime_multiplier,
        )
        fee_rate = self._fee_pct / 100
        if qty * fill * (1 + fee_rate) > equity:
            qty = equity / (fill * (1 + fee_rate))
        if qty <= 0:
            return None, 0.0

        fee = fill * qty * fee_rate
        risk_pct = (
            abs(fill - stop) * qty / equity * 100
            if stop is not None and equity > 0
            else 0.0
        )
        position = _Position(
            side=side,
            quantity=qty,
            entry_price=fill,
            entry_time=bar_time,
            entry_fee=fee,
            entry_slippage=slip * qty,
            stop_price=stop,
            regime=None,
            entry_risk_pct=risk_pct,
            size_capped=capped,
        )
        return position, fill * qty + fee

    def _close(
        self,
        pos: _Position,
        raw_price: float,
        bar_time: datetime,
        reason: str,
        pair: str,
    ) -> tuple[BacktestTrade, float]:
        """Fill an exit; returns the trade and the cash released."""
        slip = raw_price * (self._slippage_pct / 100)
        if pos.side == "long":
            fill = raw_price - slip
            gross = (fill - pos.entry_price) * pos.quantity
        else:
            fill = raw_price + slip
            gross = (pos.entry_price - fill) * pos.quantity
        exit_fee = fill * pos.quantity * (self._fee_pct / 100)
        # Net of both fees: the entry fee already left the cash balance.
        pnl = gross - exit_fee - pos.entry_fee
        notional = pos.entry_price * pos.quantity

        trade = BacktestTrade(
            entry_time=pos.entry_time,
            exit_time=bar_time,
            pair=pair,
            side=pos.side,
            strategy=self._strategy.name,
            regime=pos.regime,
            entry_price=pos.entry_price,
            exit_price=fill,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl / notional * 100 if notional > 0 else 0.0,
            fees=pos.entry_fee + exit_fee,
            slippage=pos.entry_slippage + slip * pos.quantity,
            exit_reason=reason,
            entry_risk_pct=pos.entry_risk_pct,
            size_capped=pos.size_capped,
        )
        return trade, notional + gross - exit_fee

    def _size(
        self,
        side: str,
        price: float,
        equity: float,
        confidence: float,
        window: pd.DataFrame,
        pair: str,
        ai_multiplier: float | None,
        regime_multiplier: float | None,
    ) -> tuple[float, float | None, bool]:
        """Quantity, stop price and cap flag (external sizer, else 10% equity).

        The multipliers only shrink the risk budget, as in the live bot.
        """
        if self._position_sizer is not None:
            ps = self._position_sizer.calculate(
                pair=pair,
                signal_confidence=confidence,
                portfolio_value=equity,
                entry_price=price,
                side="buy" if side == "long" else "sell",
                ai_size_multiplier=ai_multiplier,
                regime_size_multiplier=regime_multiplier,
                ohlcv=window,
            )
            stop = float(ps.stop_loss_price) if ps.stop_loss_price else None
            if stop is None:
                stop = self._calculate_stop(price, side, window, pair)
            return float(ps.quantity), stop, bool(getattr(ps, "capped", False))

        scale = min(1.0, max(0.0, ai_multiplier if ai_multiplier is not None else 1.0))
        scale *= min(
            1.0, max(0.0, regime_multiplier if regime_multiplier is not None else 1.0)
        )
        qty = equity * 0.1 * scale / price
        return qty, self._calculate_stop(price, side, window, pair), False

    def _ai_multiplier(self, combined: Any) -> float | None:
        """Live AI sizing (``PositionSizer.ai_size_multiplier``), if available."""
        if combined is None or self._position_sizer is None:
            return None
        fn = getattr(self._position_sizer, "ai_size_multiplier", None)
        return fn(combined) if fn is not None else None

    def _calculate_stop(
        self, entry_price: float, side: str, ohlcv: pd.DataFrame, pair: str
    ) -> float | None:
        """Stop price from the external manager, else a 2% stop."""
        if self._stop_loss_manager is not None:
            return float(
                self._stop_loss_manager.calculate_stop(
                    pair=pair,
                    side="buy" if side == "long" else "sell",
                    entry_price=entry_price,
                    ohlcv=ohlcv,
                )
            )
        distance = entry_price * 0.02
        if side == "long":
            return entry_price - distance
        return entry_price + distance

    def _parameters(self, trade_from: int) -> dict[str, object]:
        return {
            "initial_capital": self._initial_capital,
            "fee_pct": self._fee_pct,
            "slippage_pct": self._slippage_pct,
            "allow_short": self._allow_short,
            "signal_combiner": self._signal_combiner is not None,
            "ai_confidence_filter": self._ai_confidence_filter is not None,
            "trade_from": trade_from,
        }

    def _empty_result(
        self, ohlcv: pd.DataFrame, pair: str, trade_from: int
    ) -> BacktestResult:
        n = len(ohlcv)
        start = _bar_time(ohlcv, trade_from) if n > 0 else datetime.now(UTC)
        end = _bar_time(ohlcv, n - 1) if n > 0 else datetime.now(UTC)
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            metrics=MetricsCalculator.calculate([], pd.Series(dtype=float)),
            parameters=self._parameters(trade_from),
            start_date=start,
            end_date=end,
            pair=pair,
            strategy_name=self._strategy.name,
        )


def _closes(sig_type: SignalType) -> frozenset[str]:
    """Position sides a signal closes (spot: SHORT means sell it all)."""
    if sig_type == SignalType.CLOSE:
        return frozenset({"long", "short"})
    if sig_type == SignalType.SHORT:
        return frozenset({"long"})
    if sig_type == SignalType.LONG:
        return frozenset({"short"})
    return frozenset()


def _stop_fill(pos: _Position, open_: float, high: float, low: float) -> float | None:
    """Raw fill price if the bar hits the stop; a gap through it fills at open."""
    stop = pos.stop_price
    if stop is None:
        return None
    if pos.side == "long":
        if open_ <= stop:
            return open_
        return stop if low <= stop else None
    if open_ >= stop:
        return open_
    return stop if high >= stop else None


def _position_value(pos: _Position | None, close: float) -> float:
    """Cash-equivalent value of the open position marked at *close*."""
    if pos is None:
        return 0.0
    if pos.side == "long":
        return pos.quantity * close
    # Short: the entry notional was set aside as collateral.
    return pos.quantity * (2 * pos.entry_price - close)


def _regime_size_factor(signal: Signal | None) -> float | None:
    """Risk multiplier the StrategySelector attached (VOLATILE regime)."""
    if signal is None:
        return None
    factor = signal.metadata.get("sizing_factor")
    return float(factor) if isinstance(factor, (int, float)) else None


def _lookup_ai(series: pd.Series | None, ts: object) -> AISignals | None:
    if series is None:
        return None
    value = series.get(ts)
    return value if isinstance(value, AISignals) else None


def _bar_time(ohlcv: pd.DataFrame, idx: int) -> datetime:
    """Extract a datetime from the DataFrame index or create one."""
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        return ohlcv.index[idx].to_pydatetime()  # type: ignore[no-any-return]
    return datetime(2020, 1, 1, tzinfo=UTC)
