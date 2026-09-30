"""Streamlit interactive dashboard for the AI Crypto Trading Bot."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading_bot.config import get_settings
from trading_bot.config.settings import Settings
from trading_bot.dashboard.data import (
    DbView,
    RiskView,
    load_db_view,
    load_ohlcv,
    load_risk_view,
)
from trading_bot.healthcheck import risk_state_path


def build_candlestick_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """Build an interactive candlestick chart with EMA and Volume subplots."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLCV",
        ),
        row=1,
        col=1,
    )

    # Technical Overlays
    ema_20 = df["close"].ewm(span=20).mean()
    ema_50 = df["close"].ewm(span=50).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=ema_20, line=dict(color="#f39c12", width=1.5), name="EMA 20"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=ema_50, line=dict(color="#3498db", width=1.5), name="EMA 50"
        ),
        row=1,
        col=1,
    )

    # Volume
    colors = [
        "#2ecc71" if c >= o else "#e74c3c"
        for c, o in zip(df["close"], df["open"], strict=False)
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{pair} Market Overview",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=550,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


NO_DATA = "sin datos"


@st.cache_data(ttl=60, show_spinner=False)
def _cached_ohlcv(pair: str, timeframe: str) -> pd.DataFrame | None:
    return load_ohlcv(get_settings(), pair, timeframe)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_db() -> DbView | None:
    return load_db_view()


def _fmt_age(age: timedelta | None) -> str:
    if age is None:
        return NO_DATA
    minutes = int(age.total_seconds() // 60)
    return (
        f"hace {minutes // 60}h {minutes % 60}m"
        if minutes >= 60
        else f"hace {minutes}m"
    )


def render_header(settings: Settings, risk: RiskView | None, db: DbView | None) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if risk is not None:
            st.metric(
                "Portfolio (último ciclo)",
                f"${risk.last_value:,.2f}",
                f"{risk.daily_pnl_pct:+.2f}% hoy",
            )
        else:
            st.metric("Portfolio (último ciclo)", NO_DATA)
    with col2:
        if risk is not None:
            st.metric(
                "Drawdown actual",
                f"{risk.drawdown_pct:.2f}%",
                f"Límite: {settings.MAX_DRAWDOWN_PCT}%",
                delta_color="off",
            )
        else:
            st.metric("Drawdown actual", NO_DATA)
    with col3:
        win_rate = db.win_rate if db is not None else None
        if win_rate is not None and db is not None:
            st.metric(
                "Win rate (trades cerrados)",
                f"{win_rate:.1%}",
                f"{len(db.trades)} trades",
                delta_color="off",
            )
        else:
            st.metric("Win rate (trades cerrados)", NO_DATA)
    with col4:
        st.metric("Estado de riesgo", risk.state.upper() if risk else NO_DATA)
        if risk is not None and risk.halt_reason:
            st.caption(risk.halt_reason)


def main() -> None:
    st.set_page_config(
        page_title="AI Crypto Trading Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = get_settings()
    risk = load_risk_view(settings)
    db = _cached_db()

    # --- Sidebar ---
    st.sidebar.title("🤖 Trading Bot")
    st.sidebar.markdown(f"**Modo:** `{settings.BOT_MODE.upper()}`")
    st.sidebar.markdown(f"**Exchange:** `{settings.EXCHANGE_ID.capitalize()}`")

    selected_pair = st.sidebar.selectbox(
        "Par seleccionado", settings.TRADING_PAIRS, index=0
    )
    timeframes = ["15m", "1h", "4h", "1d"]
    default_tf = (
        timeframes.index(settings.TRADING_TIMEFRAME)
        if settings.TRADING_TIMEFRAME in timeframes
        else 1
    )
    timeframe = st.sidebar.selectbox("Timeframe", timeframes, index=default_tf)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Parámetros de riesgo")
    st.sidebar.text(f"Riesgo máx./trade: {settings.MAX_RISK_PER_TRADE_PCT}%")
    st.sidebar.text(f"Drawdown máx.: {settings.MAX_DRAWDOWN_PCT}%")
    st.sidebar.text(f"Pérdida diaria máx.: {settings.MAX_DAILY_LOSS_PCT}%")
    st.sidebar.text(
        f"Último ciclo: {_fmt_age(risk.heartbeat_age) if risk else NO_DATA}"
    )
    if st.sidebar.button("Refrescar"):
        st.cache_data.clear()

    # --- Header Metrics ---
    st.title("📊 Crypto AI Trading Terminal")
    st.caption(
        f"Renderizado: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        + (f" · estado guardado: {risk.saved_at}" if risk and risk.saved_at else "")
    )
    if risk is None:
        st.info(
            "No hay estado de riesgo del bot en "
            f"`{risk_state_path(settings)}`: el bot no corrió todavía en modo "
            f"`{settings.BOT_MODE}` o el volumen no está montado."
        )
    render_header(settings, risk, db)

    st.markdown("---")

    # --- Market chart ---
    df_bars = _cached_ohlcv(selected_pair, timeframe)
    if df_bars is not None:
        st.plotly_chart(
            build_candlestick_chart(df_bars, selected_pair), width="stretch"
        )
    else:
        st.warning(f"Datos de mercado de {selected_pair}: {NO_DATA}")

    if db is not None and not db.equity.empty:
        st.subheader("Equity (snapshots)")
        st.line_chart(db.equity)

    # --- Activity tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Trades cerrados", "💼 Posiciones abiertas", "🛑 Stops", "⚙️ Configuración"]
    )

    with tab1:
        if db is None:
            st.write(f"Base de datos: {NO_DATA} (no disponible)")
        elif db.trades.empty:
            st.write(NO_DATA)
        else:
            st.dataframe(db.trades, width="stretch")

    with tab2:
        if db is None or db.open_trades.empty:
            st.write(NO_DATA)
        else:
            st.dataframe(db.open_trades, width="stretch")

    with tab3:
        if risk is None or not risk.stops:
            st.write(NO_DATA)
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Par": s.pair,
                            "Stop": s.stop_price,
                            "Cantidad": s.quantity,
                            "Orden en exchange": s.exchange_order,
                        }
                        for s in risk.stops
                    ]
                ),
                width="stretch",
            )

    with tab4:
        st.json(
            {
                "BOT_MODE": settings.BOT_MODE,
                "EXCHANGE": settings.EXCHANGE_ID,
                "TRADING_TIMEFRAME": settings.TRADING_TIMEFRAME,
                "USE_SENTIMENT": settings.USE_SENTIMENT,
                "USE_LLM_AGENT": settings.USE_LLM_AGENT,
                "USE_DL_PREDICTION": settings.USE_DL_PREDICTION,
                "USE_RL_AGENT": settings.USE_RL_AGENT,
                "USE_REGIME_DETECTION": settings.USE_REGIME_DETECTION,
            }
        )


if __name__ == "__main__":
    main()
