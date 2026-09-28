"""Candle helpers: timeframe durations and closed/forming candle split."""

from __future__ import annotations

from datetime import datetime, timedelta

import ccxt
import pandas as pd


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Duration of one candle, e.g. "15m" -> 15 minutes (ccxt semantics)."""
    return timedelta(seconds=int(ccxt.Exchange.parse_timeframe(timeframe)))


def split_closed_candles(
    ohlcv: pd.DataFrame, timeframe: str, now: datetime
) -> tuple[pd.DataFrame, float | None]:
    """Split fetched OHLCV into closed candles and the latest traded price.

    Exchanges return the still-forming candle last. Its OHLC values are
    partial, so signals, stops and stored history must use closed candles
    only (open time + duration <= now). The forming candle's close is still
    the freshest price, which is what valuation and fills need.

    Returns ``(closed, last_price)``; ``last_price`` is None when empty.
    Expects a UTC DatetimeIndex of candle open times; other indexes are
    returned unchanged (treated as all closed).
    """
    if ohlcv.empty:
        return ohlcv, None
    last_price = float(ohlcv["close"].iloc[-1])
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        return ohlcv, last_price
    closes_at = ohlcv.index + timeframe_to_timedelta(timeframe)
    closed = ohlcv[closes_at <= pd.Timestamp(now)]
    return closed, last_price
