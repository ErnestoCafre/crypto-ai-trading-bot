"""Per-pair stop-loss state and candle-based stop detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class StopState:
    pair: str
    stop_price: float
    quantity: float
    # Exchange stop order id; None means the bot watches candles itself.
    order_id: str | None = None
    # Open time (ISO) of the newest candle already evaluated against the stop.
    last_checked: str | None = None


class StopTracker:
    def __init__(self) -> None:
        self._stops: dict[str, StopState] = {}

    def register(
        self,
        pair: str,
        stop_price: float,
        quantity: float,
        order_id: str | None,
        entry_candle: pd.Timestamp | None,
    ) -> StopState:
        """Track a stop for a freshly filled entry.

        Candles up to and including ``entry_candle`` are never evaluated:
        their lows may predate the entry.
        """
        stop = StopState(
            pair=pair,
            stop_price=stop_price,
            quantity=quantity,
            order_id=order_id,
            last_checked=entry_candle.isoformat() if entry_candle is not None else None,
        )
        self._stops[pair] = stop
        return stop

    def get(self, pair: str) -> StopState | None:
        return self._stops.get(pair)

    def remove(self, pair: str) -> None:
        self._stops.pop(pair, None)

    def pairs(self) -> list[str]:
        return list(self._stops)

    def __contains__(self, pair: object) -> bool:
        return pair in self._stops

    def __len__(self) -> int:
        return len(self._stops)

    def check_hit(self, pair: str, ohlcv: pd.DataFrame) -> float | None:
        """Fill price if a closed candle since the last check hit the stop.

        The last row of ``ohlcv`` is the still-forming candle and is skipped.
        A candle that opens at or below the stop (gap) fills at its open;
        otherwise a low at or below the stop fills at the stop price.
        """
        stop = self._stops.get(pair)
        if stop is None:
            return None
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            logger.warning("{}: OHLCV has no timestamps, cannot check stop", pair)
            return None

        closed = ohlcv.iloc[:-1]
        if stop.last_checked is not None:
            closed = closed[closed.index > pd.Timestamp(stop.last_checked)]
        if closed.empty:
            return None
        stop.last_checked = closed.index[-1].isoformat()

        for bar_open, bar_low in zip(closed["open"], closed["low"], strict=True):
            if bar_open <= stop.stop_price:
                return float(bar_open)
            if bar_low <= stop.stop_price:
                return stop.stop_price
        return None

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {pair: asdict(stop) for pair, stop in self._stops.items()}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> StopTracker:
        """Rebuild from :meth:`to_dict` output; raises on malformed data."""
        tracker = cls()
        for pair, raw in data.items():
            order_id = raw.get("order_id")
            last_checked = raw.get("last_checked")
            tracker._stops[pair] = StopState(
                pair=str(raw["pair"]),
                stop_price=float(raw["stop_price"]),
                quantity=float(raw["quantity"]),
                order_id=str(order_id) if order_id is not None else None,
                last_checked=str(last_checked) if last_checked is not None else None,
            )
        return tracker
