# Data Model — Esquema de Base de Datos

*PostgreSQL + SQLAlchemy 2.0 + Alembic*

## Diagrama de Relaciones

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  ohlcv_data │     │   trades     │     │ portfolio_snapshots│
│─────────────│     │──────────────│     │──────────────────│
│ exchange    │     │ trade_id (PK)│     │ snapshot_id (PK) │
│ pair        │     │ strategy     │◄────│ timestamp        │
│ timeframe   │     │ pair         │     │ total_value      │
│ timestamp   │     │ side         │     │ cash_balance     │
│ open        │     │ entry/exit   │     │ positions_value  │
│ high        │     │ pnl          │     │ drawdown_pct     │
│ low         │     │ fees         │     │ var_95           │
│ close       │     │ slippage     │     │ cvar_95          │
│ volume      │     │ ai_signals   │     └──────────────────┘
└─────────────┘     └──────────────┘
                                          ┌──────────────────┐
┌─────────────┐     ┌──────────────┐     │  model_metrics   │
│  orders     │     │  sentiment   │     │──────────────────│
│─────────────│     │  _history    │     │ model_name       │
│ order_id(PK)│     │──────────────│     │ version          │
│ trade_id(FK)│     │ id (PK)      │     │ trained_at       │
│ exchange    │     │ pair         │     │ accuracy         │
│ pair        │     │ source       │     │ sharpe           │
│ side        │     │ score        │     │ params (JSON)    │
│ type        │     │ raw_text     │     └──────────────────┘
│ price       │     │ model_used   │
│ quantity    │     │ timestamp    │     ┌──────────────────┐
│ status      │     └──────────────┘     │ llm_analysis_log │
│ timestamp   │                          │──────────────────│
└─────────────┘     ┌──────────────┐     │ id (PK)          │
                    │  predictions │     │ pair             │
                    │──────────────│     │ agents_output    │
                    │ id (PK)      │     │ decision         │
                    │ pair         │     │ confidence       │
                    │ model_type   │     │ tokens_used      │
                    │ predicted_ret│     │ cost_usd         │
                    │ confidence   │     │ timestamp        │
                    │ actual_ret   │     └──────────────────┘
                    │ timestamp    │
                    └──────────────┘
```

---

## Tablas Detalladas

### 1. `ohlcv_data` — Cache de datos de mercado

Almacena datos OHLCV descargados para evitar re-descargas y para backtesting.

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `BIGINT` (PK, auto) | No | ID interno |
| `exchange` | `VARCHAR(50)` | No | Exchange origen (binance, bybit, etc.) |
| `pair` | `VARCHAR(20)` | No | Par de trading (BTC/USDT, ETH/USDT) |
| `timeframe` | `VARCHAR(10)` | No | Timeframe (1m, 5m, 15m, 1h, 4h, 1d) |
| `timestamp` | `TIMESTAMP WITH TZ` | No | Timestamp de la vela (UTC) |
| `open` | `NUMERIC(20,8)` | No | Precio de apertura |
| `high` | `NUMERIC(20,8)` | No | Precio máximo |
| `low` | `NUMERIC(20,8)` | No | Precio mínimo |
| `close` | `NUMERIC(20,8)` | No | Precio de cierre |
| `volume` | `NUMERIC(30,8)` | No | Volumen en base currency |

**Índices**:
- `UNIQUE (exchange, pair, timeframe, timestamp)` — evita duplicados
- `INDEX (pair, timeframe, timestamp)` — queries frecuentes por par y rango temporal

**Notas**: usar `NUMERIC` en lugar de `FLOAT` para evitar errores de precisión en cálculos financieros.

---

### 2. `orders` — Órdenes enviadas al exchange

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `UUID` (PK) | No | ID interno |
| `exchange_order_id` | `VARCHAR(100)` | Sí | ID asignado por el exchange |
| `trade_id` | `UUID` (FK → trades) | Sí | Trade al que pertenece (null si no matcheó) |
| `exchange` | `VARCHAR(50)` | No | Exchange donde se envió |
| `pair` | `VARCHAR(20)` | No | Par de trading |
| `side` | `VARCHAR(4)` | No | `buy` o `sell` |
| `order_type` | `VARCHAR(20)` | No | `market`, `limit`, `stop_loss`, `trailing_stop` |
| `price` | `NUMERIC(20,8)` | Sí | Precio solicitado (null para market orders) |
| `quantity` | `NUMERIC(20,8)` | No | Cantidad en base currency |
| `filled_price` | `NUMERIC(20,8)` | Sí | Precio real de ejecución |
| `filled_quantity` | `NUMERIC(20,8)` | Sí | Cantidad real ejecutada |
| `fee` | `NUMERIC(20,8)` | Sí | Comisión pagada |
| `fee_currency` | `VARCHAR(10)` | Sí | Moneda de la comisión |
| `slippage_pct` | `NUMERIC(10,6)` | Sí | Slippage real (filled_price vs price) |
| `status` | `VARCHAR(20)` | No | `pending`, `open`, `filled`, `partially_filled`, `cancelled`, `failed` |
| `strategy` | `VARCHAR(50)` | No | Estrategia que generó la orden |
| `signal_source` | `VARCHAR(50)` | Sí | Origen de la señal: `technical`, `ai_hybrid`, `rl_agent` |
| `created_at` | `TIMESTAMP WITH TZ` | No | Timestamp de creación (UTC) |
| `updated_at` | `TIMESTAMP WITH TZ` | No | Última actualización |

**Índices**:
- `INDEX (pair, status, created_at)`
- `INDEX (trade_id)`
- `INDEX (strategy, created_at)`

---

### 3. `trades` — Trades completos (entry + exit)

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `UUID` (PK) | No | ID interno |
| `exchange` | `VARCHAR(50)` | No | Exchange |
| `pair` | `VARCHAR(20)` | No | Par de trading |
| `side` | `VARCHAR(5)` | No | `long` o `short` |
| `strategy` | `VARCHAR(50)` | No | Estrategia que generó el trade |
| `entry_price` | `NUMERIC(20,8)` | No | Precio de entrada (promedio si pyramiding) |
| `entry_time` | `TIMESTAMP WITH TZ` | No | Timestamp de entrada |
| `exit_price` | `NUMERIC(20,8)` | Sí | Precio de salida (null si abierto) |
| `exit_time` | `TIMESTAMP WITH TZ` | Sí | Timestamp de salida |
| `exit_reason` | `VARCHAR(30)` | Sí | `signal`, `stop_loss`, `trailing_stop`, `take_profit`, `kill_switch`, `manual` |
| `quantity` | `NUMERIC(20,8)` | No | Tamaño de la posición |
| `pnl` | `NUMERIC(20,8)` | Sí | Profit/Loss neto (después de fees) |
| `pnl_pct` | `NUMERIC(10,6)` | Sí | PnL como porcentaje |
| `total_fees` | `NUMERIC(20,8)` | Sí | Fees totales (entry + exit) |
| `total_slippage_pct` | `NUMERIC(10,6)` | Sí | Slippage total |
| `holding_time_hours` | `NUMERIC(10,2)` | Sí | Duración del trade en horas |
| `ai_sentiment_score` | `NUMERIC(5,4)` | Sí | Score de sentimiento al momento de entry (-1 a +1) |
| `ai_prediction_score` | `NUMERIC(5,4)` | Sí | Predicción del modelo DL al momento de entry |
| `ai_llm_confidence` | `NUMERIC(5,4)` | Sí | Confianza del agente LLM al momento de entry |
| `ai_combined_confidence` | `NUMERIC(5,4)` | Sí | Confianza combinada del signal combiner |
| `status` | `VARCHAR(10)` | No | `open` o `closed` |
| `created_at` | `TIMESTAMP WITH TZ` | No | Timestamp de creación |

**Índices**:
- `INDEX (pair, status)`
- `INDEX (strategy, created_at)`
- `INDEX (status)` — para queries de posiciones abiertas

---

### 4. `portfolio_snapshots` — Estado del portfolio en el tiempo

Snapshot periódico (cada 1h o en cada trade) para equity curve y análisis de riesgo.

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `BIGINT` (PK, auto) | No | ID interno |
| `timestamp` | `TIMESTAMP WITH TZ` | No | Momento del snapshot (UTC) |
| `total_value_usd` | `NUMERIC(20,8)` | No | Valor total del portfolio en USD |
| `cash_balance_usd` | `NUMERIC(20,8)` | No | Cash disponible |
| `positions_value_usd` | `NUMERIC(20,8)` | No | Valor de posiciones abiertas |
| `unrealized_pnl` | `NUMERIC(20,8)` | No | PnL no realizado |
| `realized_pnl_day` | `NUMERIC(20,8)` | No | PnL realizado del día |
| `drawdown_pct` | `NUMERIC(10,6)` | No | Drawdown actual desde peak |
| `max_drawdown_pct` | `NUMERIC(10,6)` | No | Max drawdown histórico |
| `var_95` | `NUMERIC(20,8)` | Sí | Value at Risk 95% |
| `cvar_95` | `NUMERIC(20,8)` | Sí | Conditional VaR 95% |
| `open_positions_count` | `INTEGER` | No | Número de posiciones abiertas |
| `snapshot_trigger` | `VARCHAR(20)` | No | `scheduled`, `trade_entry`, `trade_exit`, `kill_switch` |

**Índices**:
- `INDEX (timestamp)` — queries por rango temporal para equity curve

---

### 5. `sentiment_history` — Histórico de análisis de sentimiento

Almacena cada análisis de sentimiento para backtesting y auditoría.

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `BIGINT` (PK, auto) | No | ID interno |
| `pair` | `VARCHAR(20)` | No | Par relacionado (o `MARKET` para sentimiento general) |
| `source` | `VARCHAR(30)` | No | `cryptocompare`, `reddit`, `twitter`, `fear_greed` |
| `model_used` | `VARCHAR(50)` | No | `CryptoBERT`, `FinBERT`, `LLM`, `fear_greed_index` |
| `score` | `NUMERIC(5,4)` | No | Score de sentimiento (-1.0 a +1.0) |
| `confidence` | `NUMERIC(5,4)` | Sí | Confianza del modelo (0.0 a 1.0) |
| `aggregated_score` | `NUMERIC(5,4)` | Sí | Score ponderado final (output del aggregator) |
| `raw_text` | `TEXT` | Sí | Texto analizado (truncado a 500 chars para storage) |
| `metadata` | `JSONB` | Sí | Datos extra: título de noticia, autor, URL, subreddit, etc. |
| `analyzed_at` | `TIMESTAMP WITH TZ` | No | Timestamp del análisis (UTC) |

**Índices**:
- `INDEX (pair, analyzed_at)` — queries por par y rango temporal
- `INDEX (source, analyzed_at)` — filtrar por fuente

---

### 6. `predictions` — Predicciones de modelos ML/DL

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `BIGINT` (PK, auto) | No | ID interno |
| `pair` | `VARCHAR(20)` | No | Par predicho |
| `timeframe` | `VARCHAR(10)` | No | Timeframe de la predicción |
| `model_type` | `VARCHAR(30)` | No | `lightgbm`, `gru`, `cnn_lstm`, `rl_ppo` |
| `model_version` | `VARCHAR(50)` | No | Versión/hash del modelo usado |
| `predicted_return` | `NUMERIC(10,6)` | No | Retorno predicho para siguiente barra |
| `predicted_direction` | `VARCHAR(4)` | No | `up`, `down`, `flat` |
| `confidence` | `NUMERIC(5,4)` | No | Confianza de la predicción (0.0 a 1.0) |
| `actual_return` | `NUMERIC(10,6)` | Sí | Retorno real (se llena después para evaluar modelo) |
| `was_correct` | `BOOLEAN` | Sí | Si la dirección predicha fue correcta |
| `features_snapshot` | `JSONB` | Sí | Features usados para la predicción (auditoría) |
| `predicted_at` | `TIMESTAMP WITH TZ` | No | Timestamp de la predicción |
| `evaluated_at` | `TIMESTAMP WITH TZ` | Sí | Timestamp de evaluación (cuando se conoce actual_return) |

**Índices**:
- `INDEX (pair, model_type, predicted_at)`
- `INDEX (model_type, was_correct)` — para calcular accuracy por modelo

---

### 7. `llm_analysis_log` — Log de análisis del agente LLM

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `BIGINT` (PK, auto) | No | ID interno |
| `pair` | `VARCHAR(20)` | No | Par analizado |
| `llm_provider` | `VARCHAR(20)` | No | `anthropic`, `openai`, `ollama` |
| `llm_model` | `VARCHAR(50)` | No | Modelo específico usado |
| `technical_analysis` | `JSONB` | No | Output del agente analista técnico |
| `sentiment_analysis` | `JSONB` | No | Output del agente analista de sentimiento |
| `risk_evaluation` | `JSONB` | No | Output del agente risk manager |
| `orchestrator_decision` | `JSONB` | No | Decisión final del orquestador |
| `final_recommendation` | `VARCHAR(10)` | No | `LONG`, `SHORT`, `NEUTRAL`, `SKIP` |
| `confidence` | `NUMERIC(5,4)` | No | Confianza de la decisión (0.0 a 1.0) |
| `reasoning` | `TEXT` | No | Chain-of-thought del orquestador |
| `input_tokens` | `INTEGER` | No | Tokens de input consumidos |
| `output_tokens` | `INTEGER` | No | Tokens de output generados |
| `cost_usd` | `NUMERIC(10,6)` | No | Costo de la llamada en USD |
| `latency_ms` | `INTEGER` | No | Latencia de la llamada en ms |
| `analyzed_at` | `TIMESTAMP WITH TZ` | No | Timestamp del análisis |

**Índices**:
- `INDEX (pair, analyzed_at)`
- `INDEX (analyzed_at)` — para calcular costo diario acumulado (circuit breaker)

---

### 8. `model_metrics` — Registro de modelos entrenados y su rendimiento

| Columna | Tipo | Nullable | Descripción |
|---------|------|----------|-------------|
| `id` | `BIGINT` (PK, auto) | No | ID interno |
| `model_name` | `VARCHAR(50)` | No | Nombre del modelo (e.g., `gru_btcusdt_1h`) |
| `model_type` | `VARCHAR(30)` | No | `lightgbm`, `gru`, `cnn_lstm`, `ppo` |
| `version` | `VARCHAR(50)` | No | Hash o versión del modelo |
| `pair` | `VARCHAR(20)` | No | Par para el que fue entrenado |
| `timeframe` | `VARCHAR(10)` | No | Timeframe de entrenamiento |
| `train_start` | `TIMESTAMP WITH TZ` | No | Inicio del período de training |
| `train_end` | `TIMESTAMP WITH TZ` | No | Fin del período de training |
| `val_start` | `TIMESTAMP WITH TZ` | No | Inicio del período de validación |
| `val_end` | `TIMESTAMP WITH TZ` | No | Fin del período de validación |
| `accuracy_directional` | `NUMERIC(5,4)` | Sí | Accuracy de dirección (%) |
| `sharpe_ratio` | `NUMERIC(10,4)` | Sí | Sharpe del modelo en validación |
| `mse` | `NUMERIC(15,8)` | Sí | Mean Squared Error |
| `mae` | `NUMERIC(15,8)` | Sí | Mean Absolute Error |
| `hyperparameters` | `JSONB` | Sí | Hiperparámetros usados |
| `feature_importance` | `JSONB` | Sí | Importancia de features (si aplica) |
| `model_path` | `VARCHAR(255)` | No | Ruta al archivo del modelo serializado |
| `is_active` | `BOOLEAN` | No | Si este modelo está en uso actualmente |
| `trained_at` | `TIMESTAMP WITH TZ` | No | Timestamp de entrenamiento |
| `deactivated_at` | `TIMESTAMP WITH TZ` | Sí | Timestamp de desactivación |

**Índices**:
- `INDEX (model_type, pair, is_active)` — para cargar modelo activo rápidamente
- `UNIQUE (model_name, version)` — evitar duplicados

---

## Convenciones

- **Timestamps**: siempre `TIMESTAMP WITH TIME ZONE` en UTC.
- **Precios**: `NUMERIC(20,8)` para soportar crypto con muchos decimales (e.g., SHIB).
- **Porcentajes**: `NUMERIC(10,6)` almacenados como decimal (0.15 = 15%).
- **IDs**: `UUID` para entidades de negocio (trades, orders), `BIGINT` auto para datos de serie temporal.
- **JSON**: usar `JSONB` (no `JSON`) para datos semi-estructurados — permite indexación y queries.
- **Soft delete**: no se implementa. Los datos financieros no se borran, se marcan con status.
