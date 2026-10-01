"""Risk-aware position sizer."""

from __future__ import annotations

from collections import deque

import pandas as pd
from loguru import logger

from trading_bot.config.settings import Settings, get_settings
from trading_bot.core.models import CombinedSignal, PositionSize
from trading_bot.risk.stop_loss import StopLossManager

# Cap-binding check: warn when MAX_POSITION_PCT sizes most recent entries,
# i.e. the risk budget no longer adapts to volatility (see README, sizing).
CAP_WARN_WINDOW = 20
CAP_WARN_MIN_SAMPLES = 10
CAP_WARN_RATE = 0.5


def _unit(multiplier: float | None) -> float:
    """None -> 1.0; otherwise clipped to [0, 1]."""
    return 1.0 if multiplier is None else min(1.0, max(0.0, multiplier))


class PositionSizer:
    def __init__(
        self,
        stop_loss_mgr: StopLossManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._sl = stop_loss_mgr or StopLossManager()
        s = settings or get_settings()
        self._max_risk_pct = s.MAX_RISK_PER_TRADE_PCT
        self._max_position_pct = s.MAX_POSITION_PCT
        self._min_confidence = s.MIN_CONFIDENCE_THRESHOLD
        self._ai_floor = s.AI_SIZING_FLOOR
        self._recent_capped: deque[bool] = deque(maxlen=CAP_WARN_WINDOW)
        self._cap_warned = False

    @property
    def cap_hit_rate(self) -> float | None:
        """Share of the last sized entries set by the position cap."""
        if not self._recent_capped:
            return None
        return sum(self._recent_capped) / len(self._recent_capped)

    def _track_cap(self, capped: bool) -> None:
        self._recent_capped.append(capped)
        rate = self.cap_hit_rate
        if len(self._recent_capped) < CAP_WARN_MIN_SAMPLES or rate is None:
            return
        if rate > CAP_WARN_RATE and not self._cap_warned:
            self._cap_warned = True
            logger.warning(
                "MAX_POSITION_PCT={}% capped {:.0%} of the last {} entries: "
                "sizing no longer follows MAX_RISK_PER_TRADE_PCT={}% and the "
                "ATR stop. Lower the risk per trade or raise the cap.",
                self._max_position_pct,
                rate,
                len(self._recent_capped),
                self._max_risk_pct,
            )
        elif rate <= CAP_WARN_RATE:
            self._cap_warned = False

    def ai_size_multiplier(self, signal: CombinedSignal) -> float | None:
        """Risk multiplier in [AI_SIZING_FLOOR, 1] from the combined confidence.

        None when no AI source took part (technical-only sizing is unchanged).
        Linear from the floor at MIN_CONFIDENCE_THRESHOLD to 1.0 at full
        confidence. ``combined_confidence`` already nets out each source's
        agreement with the trade direction, so an opposing prediction shrinks
        the position instead of sizing it by its own strength. AI can only
        reduce the per-trade risk, never raise it.
        """
        if set(signal.weights_used) <= {"technical"}:
            return None
        span = 1.0 - self._min_confidence
        if span <= 0:
            return 1.0
        frac = (signal.combined_confidence - self._min_confidence) / span
        frac = min(1.0, max(0.0, frac))
        return self._ai_floor + (1.0 - self._ai_floor) * frac

    def calculate(
        self,
        pair: str,
        signal_confidence: float,
        portfolio_value: float,
        entry_price: float | None = None,
        side: str = "buy",
        ai_size_multiplier: float | None = None,
        ohlcv: pd.DataFrame | None = None,
        regime_size_multiplier: float | None = None,
    ) -> PositionSize:
        risk_amount = portfolio_value * (self._max_risk_pct / 100.0)
        max_position = portfolio_value * (self._max_position_pct / 100.0)

        # AI and regime adjustments: may shrink the risk budget, never grow it
        ai_adj = _unit(ai_size_multiplier)
        regime_adj = _unit(regime_size_multiplier)
        adjusted_risk = risk_amount * ai_adj * regime_adj

        # Stop-loss price
        stop_price = 0.0
        stop_pct = 0.0
        if entry_price is not None and entry_price > 0:
            stop_price = self._sl.calculate_stop(
                pair=pair,
                side=side,
                entry_price=entry_price,
                ohlcv=ohlcv,
            )
            distance = abs(entry_price - stop_price)
            if distance > 0:
                quantity = adjusted_risk / distance
            else:
                quantity = adjusted_risk / entry_price
            stop_pct = distance / entry_price * 100
        else:
            ep = entry_price if entry_price and entry_price > 0 else 1.0
            quantity = adjusted_risk / ep

        # Cap to max position
        capped = False
        if entry_price and entry_price > 0 and quantity * entry_price > max_position:
            quantity = max_position / entry_price
            capped = True
        if entry_price and entry_price > 0 and adjusted_risk > 0:
            self._track_cap(capped)

        quantity = max(0.0, quantity)
        value_usd = quantity * (entry_price if entry_price and entry_price > 0 else 0.0)

        return PositionSize(
            pair=pair,
            quantity=quantity,
            value_usd=value_usd,
            risk_pct=self._max_risk_pct,
            sizing_method="atr_stop" if entry_price else "simple_risk",
            ai_adjustment=ai_adj,
            regime_adjustment=regime_adj,
            stop_loss_price=stop_price,
            stop_loss_pct=stop_pct,
            capped=capped,
        )
