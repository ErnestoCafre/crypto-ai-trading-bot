# Trading Bot Crypto + AI

Bot de trading de criptomonedas con inteligencia artificial integrada, construido en Python.

## Features

Implementado:

- Trading de criptomonedas **spot** (long-only): entra solo sin posición,
  CLOSE/SHORT venden la posición completa
- Estrategias técnicas por régimen de mercado: trend following; las entradas
  de mean reversion existen pero están apagadas por defecto (ver Estrategias)
- Modo alternativo `STRATEGY_MODE=donchian_ensemble`: ensemble de canales
  Donchian en velas diarias con volatilidad objetivo, operado por exposición
  objetivo (ver Estrategias). Pendiente de validar en paper
- Análisis de sentimiento con CryptoBERT (extra `sentiment`)
- Filtro ML por *meta-labeling* (LightGBM): estima si una entrada técnica
  terminará con retorno neto positivo; ningún modelo pasa hoy el gate de
  publicación, así que no se usa (ver Entrenamiento)
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
  En `STRATEGY_MODE=donchian_ensemble` no hay stop por trade: la salida es el
  stop de cada canal al cierre diario (ver Modo donchian_ensemble).
- **Pérdida diaria** (`MAX_DAILY_LOSS_PCT`): notifica una vez y bloquea las
  entradas nuevas hasta el siguiente día UTC (se levanta sola). En
  `regime_selector` además liquida; en `donchian_ensemble` la acción la elige
  `DONCHIAN_DAILY_LOSS_ACTION` (por defecto no liquida, ver Modo
  donchian_ensemble).
- **Drawdown máximo** (`MAX_DRAWDOWN_PCT`): liquida y notifica una vez y deja el
  bot en estado `HALTED`: no abre posiciones nuevas, ni tras reiniciar, hasta un
  reset manual. Stops y salidas siguen funcionando.
- El estado se persiste en `RISK_STATE_PATH` (por defecto
  `var/state/risk_state_{mode}.json`). Si falta o está corrupto, el bot arranca
  desde el balance actual y deja un warning. En Docker vive en el volumen
  `botstate` (`/app/var/state`), así sobrevive a recrear el contenedor.
- **Una sola instancia por estado**: al arrancar, el bot toma un lock del
  sistema operativo sobre `<RISK_STATE_PATH>.lock`. Una segunda instancia sobre
  el mismo estado (otro proceso u otro contenedor con el mismo volumen) sale
  con código 1 sin leer ni escribir nada; si no, las dos operarían las mismas
  velas y se pisarían la cuenta paper y los stops. Paper y live usan archivos
  distintos y pueden correr a la vez. El lock se libera solo si el proceso
  muere, así que un crash no deja un lock colgado. Es un lock *advisory*
  (`flock`) local: no protege entre máquinas ni en sistemas de archivos que no
  lo respetan (NFS, algunos bind mounts de Docker Desktop en macOS/Windows), y
  solo cubre `RISK_STATE_PATH`: dos instancias con distinto `RISK_STATE_PATH`
  no deben compartir `DONCHIAN_DECISION_LOG_PATH`.
- En live, **un retiro de fondos cuenta como pérdida** para el drawdown: tras
  retirar, reiniciar una vez con `RISK_RESET=true`.
- En live el bot lleva un **libro de posición propio** (`live_ledger` en el
  archivo de estado): solo opera, valora y liquida las monedas que *él*
  compró de cada par, nunca más que el saldo real del exchange. Monedas tuyas
  del mismo activo en la cuenta se ignoran (se avisa en el log al arrancar) y
  el kill-switch solo cancela las órdenes stop que colocó el bot. Sin libro
  guardado (primer arranque en live o estado de una versión anterior) adopta
  solo lo que cubren sus stops guardados; con
  `LIVE_ADOPT_EXISTING_BALANCE=true` adopta todo el saldo de cada par (usarlo
  una vez, con una cuenta dedicada). El saldo en USDT sigue contando entero
  para el drawdown: lo recomendable sigue siendo una cuenta o subcuenta
  dedicada.
- En paper la cuenta simulada (efectivo, posiciones y último precio) se
  guarda en el mismo archivo, en la misma escritura atómica que los stops, así
  que un reinicio sigue desde donde estaba: posiciones, pico y drawdown
  incluidos. Para empezar de nuevo en `INITIAL_BALANCE`, borrar el archivo. Si
  el archivo no trae la cuenta (versiones anteriores) o la trae inválida, la
  cuenta arranca en `INITIAL_BALANCE`: se conserva el estado (pausa/HALT) pero
  pico e inicio de día se re-anclan al balance nuevo.
- `RISK_RESET=true` actúa en **cada** arranque mientras esté puesto: quitarlo
  después de usarlo, o el HALT dejaría de sobrevivir a los reinicios.

### Tamaño de posición

Cantidad = `min(riesgo / distancia al stop, tope)`: el riesgo
(`MAX_RISK_PER_TRADE_PCT`, pérdida si salta el stop ATR x2) fija el tamaño y
`MAX_POSITION_PCT` es solo un tope de seguridad. Los multiplicadores de IA y
régimen reducen el riesgo, nunca lo suben. Esto vale para `regime_selector`;
en `donchian_ensemble`, `MAX_POSITION_PCT` es la asignación por par y
`MAX_RISK_PER_TRADE_PCT` no se usa.

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

## Estrategias

`StrategySelector` elige por régimen: TRENDING y VOLATILE → trend following,
RANGING → mean reversion. Las dos estrategias usan la misma escala de
confianza (`graded_confidence`, `core/strategies/base.py`): 0,5 = condiciones
de entrada recién cumplidas, 1,0 = lectura fuerte (ADX, RSI y distancia en
ATR más allá de cada umbral). Así una señal técnica válida supera sola
`MIN_CONFIDENCE_THRESHOLD=0.5` y las fuentes de IA deciden si se mantiene.

Antes, la confianza de mean reversion era `(30 − RSI) / 30` (necesitaba
RSI ≤ 15 para llegar a 0,5), así que nunca operaba, y la de trend following
valía siempre 1,0. Con la escala común, las entradas de mean reversion
empeoran el backtest (Binance 1h, 2024-01 → 2026-09, riesgo 0,20 % / 25 %):

| Par | Trades | Retorno | Max DD | Sharpe |
|---|---|---|---|---|
| BTC/USDT sin / con MR | 259 / 334 | +0,10 / −5,27 % | 9,84 / 12,96 % | −1,17 / −1,74 |
| ETH/USDT sin / con MR | 236 / 301 | +1,94 / −0,33 % | 5,28 / 7,28 % | −0,93 / −1,15 |
| SOL/USDT sin / con MR | 252 / 315 | −0,98 / −4,02 % | 7,09 / 9,74 % | −1,42 / −1,77 |

Por eso `MEAN_REVERSION_ENTRIES=false` por defecto: en RANGING el bot no
abre posiciones, pero los CLOSE/SHORT de mean reversion siguen cerrando las
abiertas (sin cambios respecto del comportamiento previo). Reactivarlas exige
rediseñar la estrategia (salidas, stop) y validarla en el backtest.

### Modo donchian_ensemble

`STRATEGY_MODE=donchian_ensemble` reemplaza el pipeline anterior por el
ensemble de canales Donchian de Zarattini, Pagani y Barbon (2025, *Catching
Crypto Trends*, SSRN 5209907), con las reglas del paper y de su réplica
pre-registrada, sin ajustar (`core/strategies/donchian_ensemble.py`). El
default sigue siendo `regime_selector`.

- **Señal**: velas diarias cerradas. Para cada N de `DONCHIAN_LOOKBACKS`
  (5, 10, 20, 30, 60, 90, 150, 250, 360): entra cuando el close supera el
  máximo de los N closes anteriores; stop móvil = máx(stop previo, punto medio
  del canal de N closes) y sale si el close cae debajo. Exposición cruda =
  promedio de las 9 posiciones; `w = cruda × min(1, DONCHIAN_VOL_TARGET /
  vol)`, con la vol anualizada (√365) de `DONCHIAN_VOL_WINDOW` días. Sin
  apalancamiento.
- **Estado**: se reconstruye en cada ciclo recorriendo el OHLCV (el bot pide
  al menos 1000 velas diarias, o 360 + 90 + la ventana más larga si se
  configuran ventanas mayores), así que no se persiste y un reinicio da la
  misma exposición. Con menos de 360 + 90 velas la exposición es 0 si el par
  está plano; si hay posición, el bot **no la vende**: la mantiene, avisa (log
  y Telegram, una vez por vela) y reintenta cada hora, porque una historia
  corta es un problema de datos y no una señal. En BTC/ETH/SOL
  2017-2026, una ventana de 720 velas ya reproducía el estado de la historia
  completa en todos los días probados (450 no). Si algún canal no tiene en la
  ventana un close debajo de su punto medio, el bot avisa con un warning.
- **Ejecución**: por par, valor objetivo = equity × `MAX_POSITION_PCT` × w.
  Se opera la diferencia con lo que se tiene (compras y ventas parciales)
  solo si supera `REBALANCE_THRESHOLD` (0,05 de la asignación). Un objetivo 0
  siempre cierra. Hay un solo rebalanceo por vela diaria: la vela ya operada
  se guarda en el risk_state, y los demás ciclos del día no hacen nada. Las
  ventas van antes que las compras, y una compra usa como máximo el 99 % del
  efectivo libre (el backtest usa todo el efectivo: solo difiere cuando la
  caja es la que limita). En paper la cuenta y la marca se restauran juntas
  tras un reinicio; solo si la cuenta no se pudo restaurar se borra la marca
  y vuelve a comprar en el mismo día. En live se respetan
  `amount_to_precision` y los mínimos del mercado.
- **Horario**: el ciclo corre cada hora en el minuto 2 UTC
  (`TRADING_INTERVAL_MINUTES` no se usa en este modo). El de las 00:02 opera
  el cierre diario cerca del open siguiente, como el backtest; los demás
  valoran la cartera, vigilan el kill-switch y reintentan un rebalanceo
  fallido: tanto si la orden dio error como si volvió sin ejecutarse
  (rechazada o `expired`); en el decision log queda `order_error` o
  `unfilled`.
- **Asignación**: en este modo `MAX_POSITION_PCT` es la asignación de cada
  par. Con 25 % y 3 pares, la exposición total máxima es 75 % y la cartera
  rinde ~¾ de la 1/N testeada (mismo Sharpe, menos CAGR y menos drawdown).
  `MAX_POSITION_PCT=33.33` reproduce la cartera 1/N del estudio.
- **Stops**: la salida es el stop del canal, evaluado al cierre diario. No se
  usan el stop ATR ni `place_stop_loss`, y al arrancar se cancelan los stops
  que hubiera dejado el otro modo. No hay stop de catástrofe en el exchange:
  el backtest no lo simula, y uno intradía cambiaría la estrategia validada.
- **Riesgo**: el HALT por drawdown máximo liquida y deja la exposición en
  0 hasta un reset manual. El límite de **pérdida diaria** hace lo que diga
  `DONCHIAN_DAILY_LOSS_ACTION`:
  - `block_increases` (**por defecto**): no vende; ese día la exposición solo
    puede bajar. Las salidas siguen siendo los stops de canal.
  - `liquidate`: vende todo (el comportamiento anterior, igual que
    `regime_selector`); el rebalanceo del día siguiente vuelve a comprar.
  - `halve`: vende la mitad de cada posición y bloquea subas.
  - `off`: sin límite diario en este modo (el HALT sigue).

  Por qué no liquidar por defecto: es una salida intradía con la que la
  estrategia no se validó. En Bitcoin los retornos de 1-24 h tienden a
  revertir tras caídas grandes, y un stop solo agrega valor si hay momentum
  en su horizonte (Kaminski & Lo, 2014). En un escenario sintético de estrés
  (caídas intradía del 12 % que cierran cerca del open), `liquidate` y
  `halve` realizaban las pérdidas en los mínimos hasta llegar al HALT del
  15 % (Sharpe 0,85 → 0,04), mientras `block_increases` igualaba al bot sin
  límite. Comparar con datos reales antes de operar:
  `scripts/run_backtest.py --strategy-mode donchian_ensemble --compare-daily-loss`.
  En el journal, cada venta parcial queda como un trade cerrado con su P&L.
- **IA**: solo puede reducir w. Sentimiento negativo y LLM SHORT/CLOSE dan
  un multiplicador entre `AI_SIZING_FLOOR` y 1; las lecturas alcistas no
  suben nada. Un consenso SHORT del LLM impide aumentar la exposición. Sin
  fuentes de IA, el multiplicador es 1. El meta-label ML se entrenó para
  entradas discretas en `TRADING_TIMEFRAME`, así que no aplica a este modo
  y no se carga.

**Evidencia** (Binance spot, costos 0,15 % por lado, desarrollo 2018-2023,
holdout 2024-01 → 2026-09 abierto una sola vez para todos los candidatos,
reglas fijadas antes de ver resultados). Sharpe con retornos diarios:

| | 1h actual | 4h actual | Diario actual | Donchian | Buy&hold |
|---|---|---|---|---|---|
| Desarrollo BTC/ETH/SOL | 0,45/0,56/0,56 | 1,18/1,08/1,41 | 1,12/0,57/1,28 | 1,21/1,18/2,12 | 0,63/0,67/1,42 |
| Holdout BTC/ETH/SOL | 0,01/0,16/−0,11 | 0,31/1,08/−0,27 | 0,56/0,36/−0,42 | 0,70/0,59/0,47 | 0,76/0,42/0,46 |

Cartera Donchian 1/N: Sharpe 1,69, CAGR 18,9 % y MaxDD −12,5 % en
desarrollo (buy&hold: 0,90 y −88 %); 0,69, 7,0 % y −12,4 % en holdout
(buy&hold: 0,56 y −64 %). Exposición media del 11-16 %.

`backtest/target_engine.py` reproduce el script del estudio: con ejecución
al cierre, umbral 0 y sin costos, el retorno diario coincide a 1e-15; con
costos, las métricas coinciden a 2 decimales. Con lo que hace el bot
(ejecución al open siguiente, umbral 0,05, fee 0,1 % + slippage 0,05 %):

| Sharpe / CAGR / MaxDD | Desarrollo | Holdout |
|---|---|---|
| BTC/USDT | 1,19 / 16,8 % / −20,3 % | 0,76 / 10,2 % / −15,3 % |
| ETH/USDT | 1,12 / 15,2 % / −17,5 % | 0,79 / 9,2 % / −13,8 % |
| SOL/USDT | 2,11 / 31,9 % / −11,6 % | 0,52 / 5,5 % / −18,2 % |
| Cartera 1/3 por par | 1,65 / 16,6 % / −12,5 % | 0,83 / 8,7 % / −12,4 % |
| Cartera 25 % por par | 1,66 / 12,4 % / −9,6 % | 0,83 / 6,6 % / −9,4 % |

Sensibilidad de la cartera 1/3 al umbral (Sharpe en desarrollo / holdout,
días con operación en holdout): 0 → 1,68 / 0,69, 907 días; 0,05 →
1,65 / 0,83, 328 días; 0,10 → 1,43 / 0,81, 166 días. La rotación con 0,05
es de 3-5× el capital por año. Esas filas usan el calentamiento del script
(exposición desde que cada canal tiene datos). Con la regla del bot (450
velas), el holdout no cambia, pero SOL en desarrollo baja a Sharpe 1,06,
porque su historia empieza en 2020-08 y se pierde 2021.

Cautelas: con ~2,75 años de holdout, el error estándar del Sharpe es ~0,6.
La ventaja sobre buy&hold está más en el drawdown (−12 % contra −64 %) que
en el retorno: en holdout, buy&hold de BTC rindió más. El backtest no
simula IA ni mínimos del exchange. El kill-switch (HALT y pérdida diaria) se
simula desde la auditoría de 2026-11 con una aproximación pesimista (precio
lineal del open al low, mínimos simultáneos); `--no-kill-switch` lo apaga. **No operar en real sin un
período previo de paper trading** en este modo (`BOT_MODE=paper`,
`STRATEGY_MODE=donchian_ensemble`), comparando sus fills con el backtest.

### Verificación contra el backtest (paper o live)

Cada vez que calcula el objetivo diario de un par, el bot agrega una línea a
`DONCHIAN_DECISION_LOG_PATH` (por defecto
`var/state/donchian_decisions_{mode}.jsonl`): vela, objetivo de la
estrategia, ajuste de IA, bloqueos de riesgo, exposición previa y fill. Un
reintento del mismo día agrega otra línea, y vale la última. Si el log falla,
el bot sigue operando.

```bash
uv run python scripts/paper_parity.py            # BOT_MODE de .env
uv run python scripts/paper_parity.py --mode live --cache
```

El script descarga la historia diaria completa (endpoints públicos, o el
caché con `--cache`), recalcula los objetivos con `TargetExposureEngine` y
termina con código 1 si algo no coincide:

- **Objetivo**: el `w` del bot (antes de la IA) igual al del backtest
  (tolerancia 1e-6). Una diferencia indica otras velas, una ventana corta
  (canales `unsynced` en el log) u otros settings.
- **Ejecución**: slippage adverso de cada fill contra el open de t+1, y fee
  sobre el nocional. El promedio no debe superar `BACKTEST_SLIPPAGE_PCT` ni
  `BACKTEST_FEE_PCT`.
- **Continuidad**: una decisión por par y día desde la primera. Un día en que
  el bot mantuvo una posición sin historia suficiente (`held_not_ready`) se
  reporta como discrepancia.
- **Demora**: primera decisión de la vela a más de 60 minutos del cierre
  (`--max-lag-min`). Los reintentos horarios mientras el riesgo bloquea una
  suba no cuentan como demora.

El día de arranque suele fallar por demora: el bot decide a la hora en que
se lo arranca. `--since AAAA-MM-DD` ignora las velas anteriores a esa fecha.

Correrlo cada día después de las 00:10 UTC (antes, la decisión de ayer
todavía puede faltar). El paper trading no mide si la estrategia gana: en
pocas semanas hay pocas decisiones. Lo que valida es que el bot ejecute lo
mismo que se testeó.

## Señales de IA y datos de mercado

- **Velas**: señales, stops y ML usan solo velas cerradas; la vela en
  formación se descarta (`data/market/candles.py`), pero su close es el precio
  de mercado para valorar el portfolio y ejecutar. La descarga histórica
  tampoco guarda la vela abierta.
- **Origen de los datos**: en live, el mismo exchange que ejecuta
  (`EXCHANGE_SANDBOX=true` = testnet, con sus precios). En paper, siempre el
  mercado real por endpoints públicos, sin API keys: el testnet de Binance se
  borra más o menos una vez por mes (no alcanza la historia diaria de
  `donchian_ensemble`) y sus precios no son los del mercado.
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
- **Predicción ML**: un artefacto LightGBM por par, cargado al arrancar desde
  `{ML_MODEL_DIR}/{BTC_USDT}_{timeframe}.pkl` (lo publica
  `scripts/train_models.py`). El meta-label es un único modelo compartido,
  publicado en la ruta de cada par que pasa el gate. Sin archivo, ese par
  opera sin predicción. Los `.pkl` usan pickle: cargar solo
  archivos generados por este proyecto; si existe el sidecar `.sha256` se
  verifica.
- **Entrenamiento** (`scripts/train_models.py`, `--task meta_label` por
  defecto): *meta-labeling* sobre las entradas LONG del pipeline técnico. La
  estrategia decide el lado; el modelo estima si **esa** entrada termina con
  retorno neto positivo. La etiqueta es el trade real: entrada al open
  siguiente, stop 2×ATR y salida por CLOSE de la estrategia, con fees y
  slippage. Se entrena un solo modelo para todos los pares (features
  scale-free) con el historial completo. La validación es walk-forward + CPCV
  purgados por el fin de cada trade. Un par se publica solo si el filtro
  supera a un filtro aleatorio y a la estrategia sin filtro en el backtest
  real, sin aumentar el drawdown. El artefacto (`task: "meta_label"`) se
  carga con la misma ruta canónica; fuera de una entrada LONG su predicción
  no se usa. Contrato `--contract binary` (por defecto): confianza 1,0, así
  que "down" bloquea la entrada (filtro duro) y "up" la deja con tamaño
  completo. `--contract scaled`: la confianza es la distancia al umbral, y con
  los pesos actuales del combiner eso casi solo achica posiciones. El reporte
  hace el backtest de ambos.

  ```bash
  uv run python scripts/train_models.py --pairs BTC/USDT ETH/USDT SOL/USDT --publish
  ```

  La predicción de dirección de la próxima vela (`--task direction`) se
  descartó. Fuera de muestra (BTC/ETH/SOL 1h, 2026-04 a 2026-09) acertó
  49,9 / 50,3 / 49,4 % de las direcciones: azar, frente a un baseline trivial
  de 50,3 / 50,3 / 50,9 %. Además, su etiqueta
  (retorno > `ML_LABEL_THRESHOLD` = 0,1 %) no coincide con la lectura
  p > 0,5 = "up" del bot: el modelo de BTC decía "up" en el 1,2 % de las
  barras, cuando subió el 50,3 %.
- Sentimiento y decisiones LLM cacheadas vencen a 2x el intervalo de su job.

## Backtest

```bash
uv run python scripts/download_historical.py     # llena var/cache/<exchange>/<PAIR>/<tf>.parquet
uv run python scripts/run_backtest.py --pair BTC/USDT --timeframe 1h
uv run python scripts/run_backtest.py --csv datos.csv --splits 5 --html
uv run python scripts/run_backtest.py --strategy-mode donchian_ensemble \
    --pairs BTC/USDT,ETH/USDT,SOL/USDT --holdout-start 2024-01-01
```

Con `donchian_ensemble` el script usa velas 1d y `TargetExposureEngine`
(`build_target_engine()`): imprime cada par como sleeve completo y la cartera
con `MAX_POSITION_PCT` por par (Sharpe con rf = 0, como el estudio).
`--rebalance-threshold` y `--execution close` permiten las variantes de
sensibilidad y de paridad. No hay walk-forward porque no hay parámetros
ajustados.

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
  (`python -m trading_bot.healthcheck`, heartbeat en el estado de riesgo):
  unhealthy tras `HEALTH_MAX_CYCLES` (3) intervalos sin ciclo, con intervalo
  `TRADING_INTERVAL_MINUTES` o 1 hora en `donchian_ensemble`.
- Volúmenes: `botstate` (`var/state`: latch, stops, cuenta paper, log de
  decisiones y lock de instancia), `botlogs` (`var/logs`),
  `pgdata`.
- **Volumen `pgdata` existente**: Postgres solo aplica `POSTGRES_PASSWORD` al
  inicializar un volumen vacío. Si el volumen se creó con la contraseña anterior
  (`trading_bot`), cambiarla antes de usar la nueva en `.env`:
  `docker compose exec db psql -U trading_bot -c "ALTER USER trading_bot PASSWORD '<nueva>'"`.

## License

MIT
