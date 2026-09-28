"""Unit tests for the dashboard module."""

from __future__ import annotations

from trading_bot.dashboard.app import build_candlestick_chart, generate_mock_ohlcv


def test_generate_mock_ohlcv() -> None:
    df = generate_mock_ohlcv("BTC/USDT", n_bars=50)
    assert len(df) == 50
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert (df["high"] >= df["low"]).all()


def test_build_candlestick_chart() -> None:
    df = generate_mock_ohlcv("ETH/USDT", n_bars=30)
    fig = build_candlestick_chart(df, "ETH/USDT")
    assert fig is not None
    assert len(fig.data) >= 3  # Candlestick, EMAs, Volume
