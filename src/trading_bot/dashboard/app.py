"""Streamlit interactive dashboard for the AI Crypto Trading Bot."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading_bot.config import get_settings


def generate_mock_ohlcv(pair: str, n_bars: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for dashboard visualization when offline."""
    rng = np.random.default_rng(42)
    base_price = 65000.0 if "BTC" in pair else (3400.0 if "ETH" in pair else 140.0)
    returns = rng.normal(0.0005, 0.015, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(UTC), periods=n_bars, freq="1h")
    highs = prices * (1 + rng.uniform(0.001, 0.01, n_bars))
    lows = prices * (1 - rng.uniform(0.001, 0.01, n_bars))
    opens = np.roll(prices, 1)
    opens[0] = base_price
    volumes = rng.uniform(50, 500, n_bars) * (base_price / 1000)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


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


def main() -> None:
    st.set_page_config(
        page_title="AI Crypto Trading Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = get_settings()

    # --- Sidebar ---
    st.sidebar.title("🤖 Trading Bot Control")
    st.sidebar.markdown(f"**Modo:** `{settings.BOT_MODE.upper()}`")
    st.sidebar.markdown(f"**Exchange:** `{settings.EXCHANGE_ID.capitalize()}`")

    selected_pair = st.sidebar.selectbox(
        "Par seleccionado", settings.TRADING_PAIRS, index=0
    )
    _ = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Risk Parameters")
    st.sidebar.text(f"Max Risk/Trade: {settings.MAX_RISK_PER_TRADE_PCT}%")
    st.sidebar.text(f"Max Drawdown: {settings.MAX_DRAWDOWN_PCT}%")
    st.sidebar.text("Kill-Switch: Activo")

    # --- Header Metrics ---
    st.title("📊 Crypto AI Trading Terminal")
    st.caption(
        f"Última actualización: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Total", f"${settings.INITIAL_BALANCE:,.2f}", "+2.4% (24h)")
    with col2:
        st.metric("Drawdown Actual", "3.2%", f"Límite: {settings.MAX_DRAWDOWN_PCT}%")
    with col3:
        st.metric("Win Rate Estimado", "63.5%", "+1.2%")
    with col4:
        st.metric("Kill Switch", "NORMAL", delta_color="normal")

    st.markdown("---")

    # --- Main Content Grid ---
    chart_col, ai_col = st.columns([3, 1])

    with chart_col:
        df_bars = generate_mock_ohlcv(selected_pair)
        fig = build_candlestick_chart(df_bars, selected_pair)
        st.plotly_chart(fig, use_container_width=True)

    with ai_col:
        st.subheader("🧠 Señales de IA")

        # Sentiment Indicator
        st.markdown("**1. Sentimiento (CryptoBERT)**")
        sentiment_score = 0.68
        st.progress(sentiment_score)
        st.caption(f"Score: `{sentiment_score:.2f}` (Alcista moderado)")

        # Prediction Indicator
        st.markdown("**2. Predicción ML (LightGBM)**")
        st.metric(label="Retorno esperado 1h", value="+0.84%", delta="BUY")

        # LLM Multi-Agent Decision
        st.markdown("**3. Consenso Agentes LLM**")
        st.success("Recomendación: **LONG** (Confianza: 78%)")
        st.caption(
            "Los indicadores técnicos confirman tendencia alcista "
            "con soporte respetado y sentimiento positivo."
        )

    st.markdown("---")

    # --- Recent Activity Tabs ---
    tab1, tab2, tab3 = st.tabs(
        ["📋 Órdenes Recientes", "💼 Posiciones Abiertas", "⚙️ Configuración del Bot"]
    )

    with tab1:
        mock_trades = pd.DataFrame(
            [
                {
                    "ID": "ord-001",
                    "Par": "BTC/USDT",
                    "Lado": "BUY",
                    "Precio": 64850.0,
                    "Cantidad": 0.15,
                    "Fee": "$9.72",
                    "Estado": "FILLED",
                },
                {
                    "ID": "ord-002",
                    "Par": "ETH/USDT",
                    "Lado": "BUY",
                    "Precio": 3410.5,
                    "Cantidad": 2.5,
                    "Fee": "$8.52",
                    "Estado": "FILLED",
                },
                {
                    "ID": "ord-003",
                    "Par": "SOL/USDT",
                    "Lado": "SELL",
                    "Precio": 142.2,
                    "Cantidad": 20.0,
                    "Fee": "$2.84",
                    "Estado": "FILLED",
                },
            ]
        )
        st.dataframe(mock_trades, use_container_width=True)

    with tab2:
        mock_positions = pd.DataFrame(
            [
                {
                    "Par": "BTC/USDT",
                    "Tamaño": 0.15,
                    "Precio Entrada": 64850.0,
                    "Precio Actual": 65320.0,
                    "PnL No Realizado": "+$70.50 (+0.72%)",
                },
                {
                    "Par": "ETH/USDT",
                    "Tamaño": 2.5,
                    "Precio Entrada": 3410.5,
                    "Precio Actual": 3425.0,
                    "PnL No Realizado": "+$36.25 (+0.42%)",
                },
            ]
        )
        st.dataframe(mock_positions, use_container_width=True)

    with tab3:
        st.json(
            {
                "BOT_MODE": settings.BOT_MODE,
                "EXCHANGE": settings.EXCHANGE_ID,
                "USE_SENTIMENT": settings.USE_SENTIMENT,
                "USE_LLM_AGENT": settings.USE_LLM_AGENT,
                "USE_DL_PREDICTION": settings.USE_DL_PREDICTION,
                "USE_RL_AGENT": settings.USE_RL_AGENT,
                "USE_REGIME_DETECTION": settings.USE_REGIME_DETECTION,
            }
        )


if __name__ == "__main__":
    main()
