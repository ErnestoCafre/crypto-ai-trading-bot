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

## Gestión de riesgo

- **Stop-loss**: cada compra registra un stop (ATR x2). En live se coloca una
  orden `STOP_LOSS` nativa en el exchange (ccxt `stopLossPrice`); si el mercado
  no la soporta o la orden es rechazada, el bot vigila el stop con las velas
  cerradas. En paper siempre se vigila por velas: si el `low` toca el stop se
  vende al precio del stop, o al `open` si la vela abrió por debajo (gap).
- **Pérdida diaria** (`MAX_DAILY_LOSS_PCT`): liquida y notifica una vez; las
  entradas nuevas quedan bloqueadas hasta el siguiente día UTC (se levanta sola).
- **Drawdown máximo** (`MAX_DRAWDOWN_PCT`): liquida y notifica una vez y deja el
  bot en estado `HALTED`: no abre posiciones nuevas, ni tras reiniciar, hasta un
  reset manual. Stops y salidas siguen funcionando.
- El estado se persiste en `RISK_STATE_PATH` (por defecto
  `var/state/risk_state_{mode}.json`). Si falta o está corrupto, el bot arranca
  desde el balance actual y deja un warning. En Docker vive en el volumen
  `botstate` (`/app/var/state`), así sobrevive a recrear el contenedor.
- En live, **un retiro de fondos cuenta como pérdida** para el drawdown: tras
  retirar, reiniciar una vez con `RISK_RESET=true`.
- En paper la cuenta simulada se reinicia en `INITIAL_BALANCE` en cada arranque:
  se conserva el estado (pausa/HALT) pero pico e inicio de día se re-anclan al
  balance nuevo.
- `RISK_RESET=true` actúa en **cada** arranque mientras esté puesto: quitarlo
  después de usarlo, o el HALT dejaría de sobrevivir a los reinicios.

Reset manual de un `HALTED`:

```bash
RISK_RESET=true uv run trading-bot   # una sola vez; re-baselinea al balance actual
# o bien: detener el bot y borrar var/state/risk_state_<modo>.json
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
