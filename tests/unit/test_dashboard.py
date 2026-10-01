"""Unit tests for the dashboard: real data or "sin datos", never placeholders."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.dashboard import data as dash_data
from trading_bot.dashboard.app import build_candlestick_chart
from trading_bot.dashboard.data import DbView, parse_risk_state

APP = Path(__file__).parents[2] / "src/trading_bot/dashboard/app.py"
NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _ohlcv(n: int = 30) -> pd.DataFrame:
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 10.0),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
    )


def _state() -> dict[str, object]:
    return {
        "version": 1,
        "mode": "paper",
        "saved_at": (NOW - timedelta(minutes=30)).isoformat(),
        "drawdown": {
            "peak_value": 10_000.0,
            "day_start_value": 9_800.0,
            "current_day": "2026-01-01",
            "last_value": 9_500.0,
            "state": "active",
            "halt_reason": None,
            "halted_at": None,
        },
        "stops": {
            "BTC/USDT": {
                "pair": "BTC/USDT",
                "stop_price": 60_000.0,
                "quantity": 0.01,
                "order_id": None,
                "last_checked": None,
            }
        },
    }


def test_build_candlestick_chart() -> None:
    fig = build_candlestick_chart(_ohlcv(), "ETH/USDT")
    assert len(fig.data) >= 3  # Candlestick, EMAs, Volume


class TestRiskState:
    def test_parse(self) -> None:
        view = parse_risk_state(_state(), now=NOW)
        assert view is not None
        assert view.drawdown_pct == pytest.approx(5.0)
        assert view.daily_pnl_pct == pytest.approx(-300 / 9_800 * 100)
        assert view.heartbeat_age == timedelta(minutes=30)
        assert [s.pair for s in view.stops] == ["BTC/USDT"]

    @pytest.mark.parametrize("data", [None, {}, {"drawdown": {"state": "x"}}])
    def test_missing_or_malformed_is_none(self, data: dict[str, object] | None) -> None:
        assert parse_risk_state(data) is None


def test_win_rate_from_closed_trades() -> None:
    view = DbView(
        trades=pd.DataFrame({"PnL": [10.0, -5.0, 3.0, -1.0]}),
        open_trades=pd.DataFrame(),
        equity=pd.Series(dtype=float),
    )
    assert view.win_rate == pytest.approx(0.5)
    empty = DbView(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))
    assert empty.win_rate is None


def test_unreachable_database_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom() -> DbView:
        raise OSError("connection refused")

    monkeypatch.setattr(dash_data, "_read_db", boom)
    assert dash_data.load_db_view() is None


class TestApp:
    def _run(
        self, monkeypatch: pytest.MonkeyPatch, risk: object, db: object, ohlcv: object
    ):  # type: ignore[no-untyped-def]
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        st.cache_data.clear()
        monkeypatch.setattr(dash_data, "load_risk_view", lambda s: risk)
        monkeypatch.setattr(dash_data, "load_db_view", lambda: db)
        monkeypatch.setattr(dash_data, "load_ohlcv", lambda s, p, tf: ohlcv)
        at = AppTest.from_file(str(APP), default_timeout=30)
        at.run()
        assert not at.exception
        return at

    def test_without_data_shows_sin_datos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at = self._run(monkeypatch, risk=None, db=None, ohlcv=None)
        values = [m.value for m in at.metric]
        assert values and all(v == "sin datos" for v in values)
        rendered = " ".join(
            str(e.value) for e in [*at.markdown, *at.caption, *at.warning, *at.info]
        )
        for placeholder in ("63.5%", "+2.4%", "LONG", "78%", "ord-001"):
            assert placeholder not in rendered

    def test_with_data_shows_real_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        risk = parse_risk_state(_state(), now=NOW)
        db = DbView(
            trades=pd.DataFrame({"PnL": [10.0, -5.0]}),
            open_trades=pd.DataFrame(),
            equity=pd.Series(
                [1.0, 2.0], index=pd.date_range("2026", periods=2, tz="UTC")
            ),
        )
        at = self._run(monkeypatch, risk=risk, db=db, ohlcv=_ohlcv())
        values = [m.value for m in at.metric]
        assert "$9,500.00" in values
        assert "5.00%" in values
        assert "50.0%" in values
        assert "ACTIVE" in values

    def test_donchian_mode_shows_its_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRATEGY_MODE", "donchian_ensemble")
        monkeypatch.setenv("TRADING_TIMEFRAME", "1h")
        at = self._run(monkeypatch, risk=None, db=None, ohlcv=None)
        assert at.sidebar.selectbox[1].value == "1d"
        sidebar = " ".join(str(t.value) for t in at.sidebar.text)
        assert "Asignación por par" in sidebar
        assert "Riesgo máx./trade" not in sidebar
        assert "REBALANCE_THRESHOLD" in str(at.json[0].value)
