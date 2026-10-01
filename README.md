# Trading Bot Crypto + AI

Bot de trading de criptomonedas con inteligencia artificial integrada, construido en Python.

## Features

Implementado:

- Trading de criptomonedas **spot** (long-only): entra solo sin posición,
  CLOSE/SHORT venden la posición completa
- Estrategias técnicas por régimen de mercado (trend following / mean reversion)
- Análisis de sentimiento con CryptoBERT (extra `sentiment`)
- Predicción de dirección con LightGBM (un modelo por par)
- Agente LLM multi-agente (Claude/GPT) como analista de mercado
- Backtesting del mismo pipeline que corre en vivo, sin lookahead, con
  walk-forward out-of-sample, slippage y comisiones
- Gestión de riesgo (stops ATR, kill-switch enclavado, límite de pérdida diaria)
- Dashboard (Streamlit) con datos reales del bot
- Alertas vía Telegram

Planificado (hay stubs con `NotImplementedError`, no operativos):

- Futuros perpetuos y posiciones short
- Predicción con deep learning (GRU, CNN-LSTM) — extra `dl`
- Reinforcement Learning (PPO/DQN) para ejecución — extra `rl`

Cada capa de AI es opcional y se activa por feature flag (`USE_SENTIMENT`,
`USE_LLM_AGENT`, `USE_DL_PREDICTION`, `USE_RL_AGENT`). Si una falla, el bot
sigue operando con señales técnicas.

## Stack

- **Python 3.12** (`.python-version`) | **uv** | **PostgreSQL** | **SQLAlchemy 2.1** | **Alembic**
- **CCXT** | **pandas-ta-classic** | **LightGBM** | **Optuna** | **quantstats**
- **Anthropic / OpenAI SDKs** | opcional: **Transformers + PyTorch** (`sentiment`)
- **Streamlit** | **Docker** | **GitHub Actions**

## Setup

```bash
uv sync --extra dev          # dev = pytest, ruff, mypy, pre-commit
cp .env.example .env         # editar con tus API keys
uv run alembic upgrade head  # aplicar migraciones
```

Extras opcionales: `sentiment` (transformers + torch, necesario con
`USE_SENTIMENT=true`), `dl` y `rl` (reservados para los stubs planificados),
`ta-lib`. Ejemplo: `uv sync --extra dev --extra sentiment`.

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

### Tamaño de posición

Cantidad = `min(riesgo / distancia al stop, tope)`: el riesgo
(`MAX_RISK_PER_TRADE_PCT`, pérdida si salta el stop ATR x2) fija el tamaño y
`MAX_POSITION_PCT` es solo un tope de seguridad. Los multiplicadores de IA y
régimen reducen el riesgo, nunca lo suben.

- **Fase 1 (actual): 0.20 % / 25 %.** Con los valores anteriores (1 % / 10 %)
  el tope fijaba ~100 % de las entradas en 1h (backtest BTC/ETH/SOL
  2024-01 → 2026-09): posición fija del 10 %, riesgo real ~0.184 % y sin ajuste
  por volatilidad. Con 0.20 % / 25 % el riesgo real queda en ~0.170 % (casi el
  de antes) y el tope actúa solo en el 18 % (BTC), 5 % (ETH) y 0 % (SOL) de los
  trades.
- **Fase 2 (siguiente paso, no activado): 0.25 % / 30 %.** No 0.25 % / 25 %:
  con esa relación riesgo/tope el tope vuelve a actuar en ~32 % de los trades
  de BTC; con 30 % baja a ~21 % (BTC), 6 % (ETH) y 0 % (SOL).
- **Advertencia**: el Sharpe fue negativo en todas las configuraciones
  probadas (2024-2026, retorno ≈ 0). La calibración ajusta el riesgo, no la
  rentabilidad: no operar en real hasta tener una estrategia con ventaja
  demostrada.
- Validar cualquier cambio con
  `scripts/run_backtest.py --max-risk-pct X --max-position-pct Y`: la línea
  "Cap binding" debe quedar muy por debajo del 50 % (el script avisa si no).
  En vivo, `PositionSizer` deja un warning si el tope fija la mayoría de las
  últimas 20 entradas.

Reset manual de un `HALTED`:

```bash
RISK_RESET=true uv run trading-bot   # una sola vez; re-baselinea al balance actual
# o bien: detener el bot y borrar var/state/risk_state_<modo>.json
```

## Señales de IA y datos de mercado

- **Velas**: señales, stops y ML usan solo velas cerradas; la vela en
  formación se descarta (`data/market/candles.py`), pero su close es el precio
  de mercado para valorar el portfolio y ejecutar. La descarga histórica
  tampoco guarda la vela abierta.
- **Fetcher**: reintentos acotados (`OHLCV_FETCH_MAX_RETRIES`) con backoff
  exponencial + jitter y timeout total (`OHLCV_FETCH_TIMEOUT_S`); los datos se
  piden al mismo exchange de ejecución (`EXCHANGE_ID`).
- **Sentimiento**: `ElKulako/cryptobert` por defecto (ver ADR-003);
  `SENTIMENT_PROVIDER=llm` usa `LLM_PROVIDER`/`LLM_MODEL` con el mismo
  presupuesto diario que los agentes (`LLM_MAX_DAILY_COST_USD`). Cada noticia
  se puntúa una sola vez (caché de 24 h por artículo). Noticias:
  `NEWS_PROVIDER=coindesk` (CoinDesk Data API, requiere `COINDESK_API_KEY`;
  una petición por ciclo para todos los pares). Sin key, solo Fear & Greed.
- **Persistencia**: fills, trades (entradas, salidas, stops, liquidación del
  kill-switch) y un snapshot por ciclo van a `DATABASE_URL` (lo lee el
  dashboard). Es best effort: si la DB no responde al arrancar, warning y el
  bot sigue sin persistencia; un fallo posterior solo se registra en el log.
  `DB_PERSISTENCE=false` la desactiva.
- **Predicción ML**: un modelo LightGBM por par, cargado al arrancar desde
  `{ML_MODEL_DIR}/{BTC_USDT}_{timeframe}.pkl` (lo publica `ModelTrainer`). Sin
  archivo, ese par opera sin predicción. Los `.pkl` usan pickle: cargar solo
  archivos generados por este proyecto; si existe el sidecar `.sha256` se
  verifica.
- Sentimiento y decisiones LLM cacheadas vencen a 2x el intervalo de su job.

## Backtest

```bash
uv run python scripts/download_historical.py     # llena var/cache/<exchange>/<PAIR>/<tf>.parquet
uv run python scripts/run_backtest.py --pair BTC/USDT --timeframe 1h
uv run python scripts/run_backtest.py --csv datos.csv --splits 5 --html
```

`build_live_engine()` (`backtest/pipeline.py`) arma los mismos componentes que el
bot: `StrategySelector` por régimen (con reducción en régimen volátil), gate de
confianza del `SignalCombiner`, `PositionSizer` con stop ATR x2 y reglas spot.

Qué garantiza:

- **Sin lookahead**: la señal de la barra *i* usa solo velas cerradas hasta *i*
  y se ejecuta en el open de *i+1*. Todas las features son causales (el chikou
  span de Ichimoku, que usa closes futuros, está excluido); un test altera las
  barras futuras y verifica que las features pasadas no cambian.
- **Stops realistas**: si la vela abre por debajo del stop (gap) se llena al
  open, igual que en vivo. PnL neto de fees de entrada y salida y slippage.
- **Out-of-sample**: `WalkForwardValidator.run` usa el train solo como warm-up;
  trades y métricas son del test, y la curva OOS se encadena rebasada.
  `StrategyOptimizer.walk_forward` optimiza en el train de cada fold y evalúa en
  su test (optimización anidada); `optimize()` es in-sample y no sirve para
  reportar performance.
- **Métricas 24/7**: Sharpe/Sortino anualizados con la frecuencia real de las
  barras (8760/año en 1h) y rf por barra; duración de drawdown en días.

No simula: señales de IA históricas (salvo que se pasen en
`ai_signals_series`), el kill-switch de drawdown ni el rechazo de órdenes.

## Desarrollo

```bash
uv run pytest              # tests
uv run pytest --cov        # tests con coverage
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy src            # type check (igual que CI)
pre-commit install         # hooks: ruff (versión del lock), EOL LF
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
│   ├── backtest/      # Engine barra a barra, walk-forward, Optuna, métricas
│   ├── execution/     # Paper trading + Live (CCXT)
│   ├── risk/          # Position sizing, stops, kill-switch
│   ├── dashboard/     # Streamlit
│   ├── notifications/ # Telegram Bot
│   ├── db/            # SQLAlchemy ORM
│   └── config/        # Pydantic settings
├── migrations/        # Alembic
├── scripts/           # Descarga de históricos, backtest
├── tests/             # unit/, integration/, ai/
└── var/               # Artefactos de runtime (logs, modelos, cache) — gitignored
```

El layout `src/` hace que los tests corran contra el paquete instalado y no
contra el directorio de trabajo, así que un `__init__.py` faltante o un import
roto falla en los tests en vez de pasar desapercibido.

## Docker

```bash
# .env con POSTGRES_PASSWORD (y POSTGRES_USER / POSTGRES_DB si se cambian)
docker compose up -d --build   # db → migrate (alembic) → bot + dashboard
docker compose build --build-arg EXTRAS="--extra sentiment"   # con CryptoBERT
```

- El contenedor corre como usuario no-root `bot` (uid 10001); el servicio
  `volume-perms` ajusta el dueño de volúmenes creados por versiones anteriores.
- Postgres no publica puertos; el dashboard (sin autenticación) escucha solo en
  `127.0.0.1:8501`.
- `HEALTHCHECK`: healthy mientras el bot complete ciclos
  (`python -m trading_bot.healthcheck`, heartbeat en el estado de riesgo).
- Volúmenes: `botstate` (`var/state`, latch y stops), `botlogs` (`var/logs`),
  `pgdata`.
- **Volumen `pgdata` existente**: Postgres solo aplica `POSTGRES_PASSWORD` al
  inicializar un volumen vacío. Si el volumen se creó con la contraseña anterior
  (`trading_bot`), cambiarla antes de usar la nueva en `.env`:
  `docker compose exec db psql -U trading_bot -c "ALTER USER trading_bot PASSWORD '<nueva>'"`.

## License

MIT
