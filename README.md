# Trading Bot Crypto + AI

Bot de trading de criptomonedas con inteligencia artificial integrada, construido en Python.

## Features

- Trading exclusivo de criptomonedas (spot + futuros perpetuos)
- Análisis de sentimiento con CryptoBERT/FinBERT
- Predicción de precio con deep learning (GRU, CNN-LSTM) y LightGBM
- Agente LLM multi-agente (Claude/GPT) como analista de mercado
- Reinforcement Learning (PPO/DQN) para optimización de ejecución
- Backtesting con modelado realista de slippage y comisiones
- Gestión de riesgo profesional (kill-switch, position sizing dinámico)
- Dashboard interactivo (Streamlit)
- Alertas 24/7 vía Telegram

Cada capa de AI es opcional y se activa por feature flag (`USE_SENTIMENT`,
`USE_LLM_AGENT`, `USE_DL_PREDICTION`, `USE_RL_AGENT`). Si una falla, el bot
sigue operando con señales técnicas.

## Stack

- **Python 3.12+** | **uv** | **PostgreSQL** | **SQLAlchemy 2.0** | **Alembic**
- **CCXT** | **pandas-ta-classic** | **vectorbt** | **LightGBM** | **Optuna**
- **PyTorch** | **Hugging Face Transformers** | **Stable-Baselines3**
- **Streamlit** | **Docker** | **GitHub Actions**

## Setup

```bash
uv sync --extra dev          # dev = pytest, ruff, mypy, pre-commit
cp .env.example .env         # editar con tus API keys
uv run alembic upgrade head  # aplicar migraciones
```

## Uso

```bash
uv run trading-bot                                    # ejecutar el bot
uv run python -m trading_bot                          # equivalente
uv run streamlit run src/trading_bot/dashboard/app.py # dashboard
```

## Desarrollo

```bash
uv run pytest              # tests
uv run pytest --cov        # tests con coverage
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy .              # type check
```

## Arquitectura

Hexagonal / Clean Architecture con el módulo AI completamente opcional
(graceful degradation).

```
trading-bot/
├── src/trading_bot/
│   ├── cli.py         # Entry point (console script `trading-bot`)
│   ├── core/          # Estrategias, regime detection y lógica de negocio
│   ├── ai/            # Sentiment, prediction, LLM agents, RL
│   ├── data/          # Fetchers (CCXT, CoinGecko, noticias, social, on-chain)
│   ├── backtest/      # vectorbt + Backtesting.py
│   ├── execution/     # Paper trading + Live (CCXT)
│   ├── risk/          # Position sizing, stops, kill-switch
│   ├── dashboard/     # Streamlit
│   ├── notifications/ # Telegram Bot
│   ├── db/            # SQLAlchemy ORM
│   └── config/        # Pydantic settings
├── migrations/        # Alembic
├── scripts/           # Utilidades (descarga de históricos)
├── tests/             # unit/, integration/, ai/
└── var/               # Artefactos de runtime (logs, modelos, cache) — gitignored
```

El layout `src/` hace que los tests corran contra el paquete instalado y no
contra el directorio de trabajo, así que un `__init__.py` faltante o un import
roto falla en los tests en vez de pasar desapercibido.

## Docker

```bash
docker compose up -d    # bot + dashboard + PostgreSQL
```

## License

MIT
