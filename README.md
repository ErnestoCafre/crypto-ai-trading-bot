# Trading Bot Crypto + AI

Bot de trading de criptomonedas con inteligencia artificial integrada, construido en Python.

## Features

- Trading exclusivo de criptomonedas (spot + futuros perpetuos)
- Análisis de sentimiento con CryptoBERT/FinBERT
- Predicción de precio con deep learning (GRU, CNN-LSTM)
- Agente LLM multi-agente (Claude/GPT) como analista de mercado
- Reinforcement Learning (PPO/DQN) para optimización de ejecución
- Backtesting con modelado realista de slippage y comisiones
- Gestión de riesgo profesional (kill-switch, position sizing dinámico)
- Dashboard interactivo (Streamlit)
- Alertas 24/7 via Telegram

## Stack

- **Python 3.12+** | **uv** | **PostgreSQL** | **SQLAlchemy 2.0**
- **CCXT** | **pandas-ta-classic** | **vectorbt** | **LightGBM**
- **PyTorch** | **Hugging Face Transformers** | **Stable-Baselines3**
- **Streamlit** | **Docker** | **GitHub Actions**

## Setup

```bash
uv sync
cp .env.example .env
# Editar .env con tus API keys
uv run python __main__.py
```

## Arquitectura

Hexagonal / Clean Architecture con módulo AI completamente opcional (graceful degradation).

```
trading-bot/
├── core/          # Estrategias y lógica de negocio
├── ai/            # Sentiment, prediction, LLM agents, RL
├── data/          # Fetchers (CCXT, CoinGecko, noticias)
├── backtest/      # vectorbt + Backtesting.py
├── execution/     # Paper trading + Live (CCXT)
├── risk/          # Position sizing, stops, kill-switch
├── dashboard/     # Streamlit
├── notifications/ # Telegram Bot
├── db/            # SQLAlchemy ORM + Alembic
└── config/        # Pydantic settings
```

## License

MIT
