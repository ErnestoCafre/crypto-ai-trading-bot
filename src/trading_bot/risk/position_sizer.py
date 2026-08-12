"""Risk-aware position sizer."""

from __future__ import annotations

import pandas as pd

from trading_bot.config.settings import get_settings
from trading_bot.core.models import PositionSize
from trading_bot.risk.stop_loss import StopLossManager


class PositionSizer:
    def __init__(self, stop_loss_mgr: StopLossManager | None = None) -> None:
        self._sl = stop_loss_mgr or StopLossManager()

    def calculate(
        self,
        pair: str,
        signal_confidence: float,
        portfolio_value: float,
        entry_price: float | None = None,
        side: str = "buy",
        ai_confidence: float | None = None,
        ohlcv: pd.DataFrame | None = None,
    ) -> PositionSize:
        settings = get_settings()
        risk_amount = portfolio_value * (settings.MAX_RISK_PER_TRADE_PCT / 100.0)
        max_position = portfolio_value * (settings.MAX_POSITION_PCT / 100.0)

        # AI adjustment
        ai_adj = ai_confidence if ai_confidence is not None else 1.0
        adjusted_risk = risk_amount * ai_adj

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
        if entry_price and entry_price > 0 and quantity * entry_price > max_position:
            quantity = max_position / entry_price

        quantity = max(0.0, quantity)
        value_usd = quantity * (entry_price if entry_price and entry_price > 0 else 0.0)

        return PositionSize(
            pair=pair,
            quantity=quantity,
            value_usd=value_usd,
            risk_pct=settings.MAX_RISK_PER_TRADE_PCT,
            sizing_method="atr_stop" if entry_price else "simple_risk",
            ai_adjustment=ai_adj,
            stop_loss_price=stop_price,
            stop_loss_pct=stop_pct,
        )
