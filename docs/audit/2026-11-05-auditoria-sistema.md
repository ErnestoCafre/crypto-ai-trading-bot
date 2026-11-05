# Auditoría del sistema — 2026-11-05

Alcance: pruebas automatizadas, revisión de correctness de `src/` (estrategia
Donchian, riesgo y tamaño de posición, órdenes y estado, lock de instancia
única, decision log, migraciones Alembic), backtest corto y paridad
paper/backtest. Rama auditada: `main` @ `7a64a98`.

Restricciones respetadas: **el bot no se ejecutó en paper ni en vivo** y no se
usaron claves de exchange. Todo lo de abajo corre sobre tests, datos
sintéticos y una base PostgreSQL local desechable.

Notas operativas para repetir esta auditoría en otra sesión:
[`NOTAS_SESIONES.md`](NOTAS_SESIONES.md). Scripts de apoyo: [`tools/`](tools/).

---

## 1. Resumen ejecutivo

| Área | Estado |
|---|---|
| Suite (`tests/unit`, `tests/ai`, `tests/integration`) | **700 ok, 1 fallo de entorno** (red) → 1.ª ronda: 703 ok, 1 skip → **2.ª ronda: 747 ok, 1 skip** |
| Cobertura de líneas | **90 %** (6800 sentencias tras la 2.ª ronda) |
| Warnings de la suite | 74 → **30** (los 30 restantes son de aiosqlite, librería de terceros) |
| ruff / ruff format / mypy | Sin errores |
| Migraciones Alembic en PostgreSQL 16 real | `upgrade head`, `downgrade base`, `upgrade head` ok; `alembic check`: **sin drift** modelo↔migración |
| TradeJournal sobre PostgreSQL real | ok (entradas, scale-in, venta parcial, cierre, snapshot) |
| Backtest `donchian_ensemble` y `regime_selector` | Corren de punta a punta (datos **sintéticos**: no hay acceso a exchanges) |
| Paridad (`scripts/paper_parity.py`) | **OK** sobre un decision log generado offline (200 días, 0 discrepancias de target) |

Hallazgos: **0 críticos, 2 altos, 5 medios, 7 bajos**. En la 1.ª ronda se
corrigieron 4 defectos menores (§4). En la **2.ª ronda (§7)** se corrigió
todo salvo M-4. Una **auditoría de consistencia** (§8) encontró y corrigió 4
problemas más. En la **3.ª ronda (§9)** se resolvió M-4 y se simuló el
kill-switch en el backtest.

Ya no quedan hallazgos de código abiertos. Antes de operar en vivo faltan
las validaciones operativas que este entorno no permitió (§5 y §6): datos
reales, paper ≥ 30 días y testnet.

---

## 2. Resultados de las pruebas

### 2.1 Entorno

- `uv sync --extra dev` ok (Python 3.12.3, ccxt 4.5.84, pandas 2.3.3,
  numpy 2.5.3, SQLAlchemy 2.1.x, alembic 1.18.x).
- Docker: el cliente existe pero **no hay daemon** (`/var/run/docker.sock`
  ausente), así que `docker compose up db` no es posible. Se levantó un
  PostgreSQL 16 local con los binarios del sistema (`initdb` + `pg_ctl`,
  puerto 55432) — ver `NOTAS_SESIONES.md`.
- Red: el proxy de salida **rechaza** `api.binance.com`,
  `data-api.binance.vision`, `api.binance.us` y `www.okx.com` (HTTP 403 en
  CONNECT). PyPI sí es accesible. Sin datos públicos de mercado.

### 2.2 Suite completa (antes de los fixes)

```
uv run pytest tests --cov=trading_bot
1 failed, 700 passed, 74 warnings in 118.62s
FAILED tests/integration/test_data_pipeline.py::TestCCXTIntegration::test_fetch_small_ohlcv
```

El único fallo es `ExchangeNotAvailable` (DNS de `api.binance.com`): el test
hace una llamada real a Binance. No es un defecto del código. Se cambió para
que haga *skip* cuando el exchange no es alcanzable (commit `a0b7d61`).

Los tests de integración **no necesitan Postgres**: usan
`sqlite+aiosqlite:///:memory:`. Por eso se validó Postgres aparte (§2.4).

### 2.3 Cobertura

Total **90 %**. Módulos por debajo del 85 % relevantes para trading:

| Módulo | Cobertura | Comentario |
|---|---|---|
| `core/bot.py` | 84 % | Lo no cubierto: arranque (`start`/`stop`), señales, ramas de LLM/sentimiento |
| `state_lock.py` | 82 % | Rama Windows (`msvcrt`) no ejecutable en Linux |
| `execution/live_executor.py` | 87 % | Errores de `fetch_balance`/tickers en liquidación |
| `db/base.py` | 77 % | `get_session` y atributos perezosos |
| `cli.py`, `__main__.py`, `backtest/report.py` | 0 % | Puntos de entrada y reporte HTML |

Stubs planificados (GRU, CNN-LSTM, RL, `ai_hybrid`) en 0 %, esperado.

### 2.4 Migraciones y PostgreSQL

```
alembic upgrade head      -> 001_initial_schema ok
alembic check             -> No new upgrade operations detected.
alembic downgrade base && alembic upgrade head -> ok
```

`docs/audit/tools/pg_smoke.py` ejercita `TradeJournal` sobre asyncpg: compra,
scale-in (precio medio 60 666,67), venta parcial (trade cerrado propio con
P&L 110,48), cierre (P&L 320,87), snapshot con `realized_pnl_day` = 431,35.
Tipos `Numeric`, `TIMESTAMPTZ` y `JSONB` se comportan como en SQLite.

### 2.5 Backtest corto (datos sintéticos)

Datos: GBM con deriva por regímenes, volatilidad ~65 % anual
(`tools/synth_ohlcv.py`, semilla 7 diaria / 11 horaria). **Las métricas no
dicen nada sobre el mercado real**; solo prueban que el pipeline funciona.

`donchian_ensemble` (2200 velas diarias, holdout desde 2023-06-01):

```
SYN/USDT              full     sharpe  0.57  CAGR   5.34%  maxDD 13.31%  exposure  9.5%  fills 214
SYN/USDT              holdout  sharpe  0.86  CAGR  10.08%  maxDD 10.72%  exposure 13.6%  fills 73
portfolio (25%/pair)  full     sharpe  0.52  CAGR   1.28%  maxDD  3.50%  exposure  2.4%  fills 220
```

`regime_selector` 1h (6000 velas, 3 folds walk-forward): retorno −1,60 %,
Sharpe −2,66, 63 trades, riesgo medio por entrada 0,168 %, tope
`MAX_POSITION_PCT` activo en 0 % de los trades. OOS encadenado +0,12 %.
Coherente con la advertencia del README (Sharpe ≈ 0 o negativo).

Tras el fix de huecos (§4, F-2) el backtest Donchian sobre datos sin huecos da
**exactamente los mismos números**.

### 2.6 Paridad paper ↔ backtest

No hay decision log real (el bot no corrió). Se generó uno **offline** con
`tools/make_decisions.py`: para cada uno de los últimos 200 días calcula
`target_exposure()` sobre las 1000 velas cerradas previas — lo mismo que hace
el bot (`HISTORY_BARS = 1000`) — y simula el fill a la apertura siguiente con
los costes del backtest. Después corre `scripts/paper_parity.py --cache`:

```
decisions: 200  fills: 199  mean slippage: 0.050%  mean fee: 0.100%
OK: the run matches the backtest.
```

Con ventana de 450 velas tampoco hubo discrepancias de target en esta serie.
Esto valida el script de paridad y la reconstrucción de estado por ventana; no
sustituye la paridad sobre un paper real (pendiente, ver §6).

---

## 3. Hallazgos priorizados

Severidad: **Alta** = puede causar pérdidas o liquidaciones no deseadas en un
escenario realista; **Media** = comportamiento incorrecto acotado o que
degrada la fidelidad backtest/live; **Baja** = robustez, observabilidad o
casos límite.

### Alta

**H-1 — En live, la posición del bot es todo el saldo del activo base.**
`src/trading_bot/execution/live_executor.py:126` (`get_position`),
`:151` (`close_all_positions`), `:84` (`get_balance`).
Escenario: la cuenta tiene 0,5 BTC propios del usuario y el bot opera
`BTC/USDT` en `donchian_ensemble`. `held = 0,5 BTC` se interpreta como
exposición del bot: si el target es 0,1 el bot **vende 0,4 BTC del usuario**;
un kill-switch vende los 0,5. En `regime_selector`, un LONG se ignora porque
"ya hay posición". El docstring dice que solo se gestionan los pares
configurados, pero dentro de un par no se distingue el origen del saldo.
Recomendación: cuenta/subcuenta dedicada (documentarlo como requisito
bloqueante para live), o llevar un libro interno de cantidad propia y
reconciliar contra el exchange. No se cambió: es lógica de trading.

**H-2 — Una orden de rebalanceo no ejecutada se marca como hecha.**
`src/trading_bot/core/bot.py:1515` (`_fill_rebalance` retorna sin excepción)
y `:1351` (`self._rebalanced[pair] = candle`).
Escenario: en live Binance devuelve una orden de mercado `EXPIRED` con
`filled = 0` (protección de precio, STP, falta de liquidez), o en paper
`place_order` devuelve `rejected` (sin precio). No hay excepción, así que la
vela queda marcada y los ciclos horarios siguientes **no reintentan**; el par
queda fuera de target hasta la vela siguiente (24 h). El docstring de
`_trading_trigger` dice que los ciclos horarios "reintentan un rebalanceo
fallido", pero solo ocurre cuando hay excepción.
Recomendación: que `_fill_rebalance` devuelva si hubo fill y no marcar la
vela cuando `filled_quantity <= 0` (dejando constancia en el decision log).

### Media

**M-1 — Sin historia suficiente el target es 0 y se vende lo que haya.**
`src/trading_bot/core/strategies/donchian_ensemble.py:174` +
`core/bot.py:1376-1403`.
Escenario: el exchange devuelve menos de 450 velas (par recién listado,
respuesta truncada sin error, cambio de símbolo). `ready=False` →
`exposure=0` → `rebalance_delta` cierra la posición entera. Es consistente con
el backtest (que tampoco expone antes de 450 barras) pero en vivo una anomalía
de datos se convierte en una liquidación. Recomendación: si `ready=False` y
hay posición, no operar y alertar en lugar de vender.

**M-2 — `filled=None` se interpreta como fill completo.**
`src/trading_bot/execution/live_executor.py:57`.
Escenario: el exchange responde a `create_order` sin `filled` (ccxt lo deja
en `None` en algunos mercados/respuestas `ACK`). El bot registra la cantidad
pedida como ejecutada, el journal y el decision log reflejan un fill que
quizá no ocurrió. Recomendación: si `filled is None`, consultar
`fetch_order(id)` antes de reportar, o reportar `status` y cantidad 0.

**M-3 — Un stop de vela "visto" pero no ejecutado no se reevalúa.**
`src/trading_bot/risk/stop_tracker.py:85` + `core/bot.py:1237`.
`check_hit` avanza `last_checked` antes de que la venta se ejecute. Si la
venta falla (excepción en live, `rejected` en paper), esa vela no se vuelve a
evaluar. Solo se re-dispara si una vela posterior abre o toca bajo el stop;
si el precio rebota, la posición queda abierta sin haber salido en el stop.
Recomendación: avanzar `last_checked` solo tras un fill o reintentar la
salida mientras `stop` siga registrado y el precio esté por debajo.

**M-4 — Kill-switch diario y rebalanceo diario se contradicen.**
`core/bot.py:313-321` y `backtest/target_engine.py` (docstring: el
kill-switch no se simula).
Escenario: `DAILY_PAUSED` liquida todo; al día UTC siguiente el ensemble
vuelve a pedir la misma exposición y recompra (doble coste). El backtest no
lo refleja, así que la paridad no lo detecta y el rendimiento en vivo puede
quedar por debajo del backtest en días volátiles. Recomendación: decidir si en
`donchian_ensemble` la pérdida diaria debe solo bloquear aumentos (no
liquidar), y simular el kill-switch en el motor.

**M-5 — Dependencia: numpy 2.5 + pandas 2.3.3 emiten DeprecationWarning en
`pd.Timedelta(...)`.**
`src/trading_bot/backtest/parity.py:27` (y cualquier `pd.Timedelta`).
numpy 2.5 deprecó la unidad genérica de `timedelta64` y avisa que en el
futuro será un error; pandas 2.3.3 la usa internamente al construir
`Timedelta(days=1)`. Hoy son solo warnings (74 en la suite), pero un
`numpy` futuro podría romper `pd.Timedelta`, que el bot usa para velas,
stops y paridad. `pyproject.toml` pide `numpy>=1.26` sin tope.
Recomendación: acotar `numpy<2.6` hasta que pandas publique compatibilidad,
y añadir `-W error::DeprecationWarning` en CI para enterarse antes.

### Baja

- **B-1 — Tope de `MAX_POSITION_PCT` sin validar.**
  `config/settings.py:84`. Live acepta 0, negativos o >100; el backtest exige
  `allocation ∈ (0, 1]` y falla. Con 150 en `donchian_ensemble` el bot apunta
  a 1,5× el equity por par (las compras quedan limitadas por la caja).
  Recomendación: `Field(gt=0, le=100)` (también `MAX_DRAWDOWN_PCT`,
  `MAX_DAILY_LOSS_PCT > 0`).
- **B-2 — Decision log: una línea a medio escribir corrompe la siguiente.**
  `core/decision_log.py:33`. Si el proceso muere a mitad de `write`, la línea
  parcial queda sin `\n` y la siguiente se concatena: ambas se descartan como
  malformadas. Recomendación: al abrir, si el archivo no termina en `\n`,
  escribir uno antes de anexar.
- **B-3 — Decision log y estado no son atómicos entre sí.**
  `core/bot.py:1352-1353`. El estado (con la vela marcada) se guarda antes de
  escribir las decisiones; un crash entre ambos deja el día sin registro y la
  paridad lo reporta como "missing".
- **B-4 — `atomic_write_json` no hace fsync del directorio.**
  `state_file.py:24`. Tras `os.replace`, un corte de energía puede perder el
  rename en algunos FS (ext4 con ciertas opciones). Riesgo bajo; añadir
  `os.fsync` del directorio padre.
- **B-5 — El lock de instancia es advisory y local.**
  `state_lock.py`. `flock` no se respeta entre hosts ni en algunos bind mounts
  (NFS, Docker Desktop en macOS/Windows). Además solo protege
  `RISK_STATE_PATH`: dos instancias con distinto `RISK_STATE_PATH` y el mismo
  `DONCHIAN_DECISION_LOG_PATH` escribirían el mismo log. Documentar.
- **B-6 — Valoración con precio viejo si falla el fetch de un par.**
  `core/bot.py:285-294`. El paper valora con la última marca conocida; el
  drawdown se evalúa con ella. Aceptable, pero conviene loguear la antigüedad
  de la marca.
- **B-7 — `LiveExecutor.place_order` envía `price` en órdenes de mercado.**
  `execution/live_executor.py:44`. ccxt lo ignora en Binance spot hoy; en
  otros exchanges puede convertirlo en límite o rechazarlo. Pasar `None` en
  `market`.

### Verificado sin hallazgos

- **Estrategia Donchian** (`donchian_ensemble.py`): entrada con el máximo de
  los N cierres *anteriores*, stop = máx(stop previo, punto medio incluyendo
  t), salida con `close < stop`, exposición = media × min(1, 25 %/vol√365),
  sin apalancamiento; el target de t se aplica en t+1. Sin lookahead.
  `rebalance_delta` cierra siempre con target 0 y respeta el umbral.
- **Tamaño de posición** (`position_sizer.py`): `min(riesgo/distancia, tope)`;
  multiplicadores de IA y régimen acotados a [0, 1]: nunca suben el riesgo.
- **Stops nativos**: con ccxt 4.5.84, `type="market"` + `stopLossPrice` en
  spot se traduce a `STOP_LOSS` (verificado en
  `ccxt/async_support/binance.py`, `create_order_request`).
- **Paginación OHLCV**: `limit` de página acotado a 1000 (máximo de
  `/api/v3/klines` de Binance spot); `limit` final se aplica con `tail`.
- **Lock de instancia**: se toma antes de leer estado, se libera al morir el
  proceso, archivo separado del estado (no lo pisa `os.replace`).
- **Drawdown**: latch HALTED persistido antes de liquidar; DAILY_PAUSED se
  levanta solo al cambiar el día UTC; `RISK_RESET` re-ancla.
- **Migraciones**: sin drift; tipos y nombres de índices/constraints
  coinciden con los modelos.

---

## 4. Correcciones aplicadas (commits separados)

| Commit | Tipo | Cambio | ¿Afecta a trading? |
|---|---|---|---|
| `356353a` | fix | `ai_exposure_multiplier`: el bearishness del sentimiento se acota a [0, 1]. Antes `min(1, score)` estaba del lado equivocado y un score < −1 bajaba el multiplicador por debajo de `AI_SIZING_FLOOR` (o a negativo). | Solo con entradas fuera de rango; las fuentes actuales ya acotan a [−1, 1]. Comportamiento en rango idéntico (tests existentes sin cambios). |
| `42026ba` | fix | `TargetExposureEngine`: un día sin vela para un par ya no decide target 0. Antes un solo hueco liquidaba el par a la apertura siguiente y lo recompraba después (verificado: exposición 0,21 → 0,00). El bot en vivo no opera ese par ese día; ahora el backtest tampoco. | Solo el **backtest**. Resultados idénticos sobre datos sin huecos. |
| `68ce922` | fix | `CCXTFetcher.__aenter__` cierra la sesión HTTP si `load_markets()` falla (antes se filtraba: `Unclosed client session`). | No. |
| `a0b7d61` | test | El test de integración contra Binance hace *skip* si el exchange no es alcanzable. | No. |

Cada fix trae su test de regresión (el de huecos falla sin el fix).

---

## 5. Riesgos

1. **Sin validación con datos reales en esta sesión**: backtests y paridad se
   hicieron con datos sintéticos por el bloqueo de red. Las cifras del README
   (BTC/ETH/SOL 2024-2026) no se pudieron reproducir aquí.
2. **Rendimiento**: el propio README reporta Sharpe negativo en
   `regime_selector`; `donchian_ensemble` está "pendiente de validar en
   paper". No hay evidencia en esta auditoría que justifique capital real.
3. **Cuenta compartida en live** (H-1): el riesgo operativo más serio.
4. **Paridad incompleta**: AI, kill-switch y mínimos del exchange no se
   simulan; `paper_parity.py` compara el target *pre-AI*, así que un efecto
   de IA mal aplicado no aparecería como discrepancia.
5. **Dependencias**: numpy sin tope superior (M-5); `ccxt>=4.5` sin tope:
   un cambio de mapeo de `stopLossPrice` rompería los stops nativos de forma
   silenciosa (el bot cae a stops vigilados, pero con peor ejecución).

---

## 6. Recomendaciones (en orden)

1. Exigir cuenta/subcuenta dedicada para live o un libro de posición propio
   (H-1) — **bloqueante para live**.
2. No marcar la vela como rebalanceada sin fill (H-2) y no liquidar con
   `ready=False` (M-1); ambos con test.
3. Resolver `filled=None` con `fetch_order` (M-2) y la reevaluación de stops
   (M-3).
4. Decidir la política de pérdida diaria en `donchian_ensemble` y simularla
   (M-4).
5. Acotar `numpy<2.6` y `ccxt<5`, y validar `MAX_POSITION_PCT` (M-5, B-1).
6. Correr un paper real ≥ 30 días y `scripts/paper_parity.py` diario desde
   cron; exigir 0 discrepancias de target antes de live.
7. En un entorno con red: `scripts/download_historical.py` y repetir §2.5 con
   BTC/ETH/SOL reales; comparar con las cifras del README.
8. Subir cobertura de `core/bot.py` en `start`/`stop` y ramas de error de
   `live_executor.py`.

---

## 7. Segunda ronda: correcciones de los hallazgos

Criterio: corregir la causa, no el síntoma, con un test de regresión por
cambio (cada test falla sin su fix) y sin alterar la estrategia validada. Los
backtests sobre los mismos datos sintéticos dan **resultados idénticos**
antes y después, y la paridad sigue en OK.

| Hallazgo | Estado | Commit | Qué cambió |
|---|---|---|---|
| H-1 cuenta compartida en live | **Corregido** | `cf3d586`, `3747bf9` | `LiveExecutor` lleva un libro de posición propio (`live_ledger` en el risk state): posición = min(libro, saldo real). Valoración, salidas y liquidación usan solo eso; el kill-switch cancela solo los stops del bot. Sin libro guardado adopta lo cubierto por sus stops (`LIVE_ADOPT_EXISTING_BALANCE=true` adopta todo). El estado se guarda tras cada fill. El pico de drawdown se re-ancla al crear el libro, para que no haya un HALT falso. |
| H-2 orden sin fill marcada como hecha | **Corregido** | `f1d49c4` | Si una orden vuelve sin ejecutar, la vela no se marca y se reintenta al ciclo siguiente (`unfilled` en el decision log). "Bajo el mínimo del mercado" sigue contando como hecho. |
| M-1 liquidación sin historia | **Corregido** | `f1d49c4` | Con posición y `ready=False` el bot mantiene la posición, alerta una vez por vela y reintenta (`held_not_ready`). |
| M-2 `filled=None` | **Corregido** | `2fc2b40`, `3747bf9` | Se hace `fetch_order`; si sigue sin dato, el fill se mide por la variación del saldo base y queda `unconfirmed`. |
| M-3 stop consumido sin salir | **Corregido** | `c223f17` | Una vela que tocó el stop se reevalúa hasta que la salida se ejecuta. |
| M-4 kill-switch diario vs rebalanceo | **Corregido** (3.ª ronda, §9) | `7d9d48f`, `4fd7fa4` | `DONCHIAN_DAILY_LOSS_ACTION` (por defecto `block_increases`) y simulación del kill-switch en el backtest. |
| M-5 numpy/pandas | **Corregido** | `f899be4` | `numpy<2.6`, `ccxt<5`; `parity.DAY` y los tests usan unidades explícitas; pytest convierte en error las deprecaciones originadas en `trading_bot`. |
| B-1 settings sin validar | **Corregido** | `f306de0` | Límites en porcentajes, balances, costos e intervalos. |
| B-2 línea rota en el decision log | **Corregido** | `b2e9c9a` | El siguiente registro arranca en línea nueva. |
| B-3 orden log/estado | **Corregido** | `b2e9c9a` | Primero las decisiones, después el estado. |
| B-4 fsync del directorio | **Corregido** | `b2e9c9a` | `atomic_write_json` sincroniza el directorio (POSIX). |
| B-5 límites del lock | **Documentado** | `3ea22a3` | README: es advisory y local, y no cubre el decision log. |
| B-6 precio viejo | **Corregido** | `b2e9c9a` | Warning por par sin precio en el ciclo. |
| B-7 `price` en market | **Corregido** | `2fc2b40` | No se envía `price` en órdenes de mercado. |

Hallazgo nuevo de esta ronda: **`.env.example` contradecía los defaults**
(`f306de0`). Fijaba `MAX_RISK_PER_TRADE_PCT=1.0` / `MAX_POSITION_PCT=10.0`,
la calibración que el README descarta, y el README indica copiarlo a `.env`.
Además listaba 4 variables que ningún código lee y le faltaban las
`POSTGRES_*` que exige docker-compose. Un test nuevo impide que vuelva a
divergir.

## 8. Auditoría de consistencia de los cambios

Revisión del diff completo (`origin/main..HEAD`) y de los módulos que
consumen lo modificado: el bot, la paridad, el dashboard y el healthcheck
(que leen el risk state), los scripts y la documentación de contratos.

| # | Problema encontrado | Severidad | Corrección |
|---|---|---|---|
| C-1 | El libro sumaba el `filled` bruto en compras; Binance sin BNB cobra la comisión en la moneda comprada. Con monedas del dueño en la cuenta, el bot se atribuía la comisión con monedas ajenas. | Media | `3747bf9`: las compras acreditan lo recibido neto de `fee`/`fees` en moneda base. |
| C-2 | Un fill no informado se asumía completo; el recorte del libro al saldo no lo corrige cuando hay monedas ajenas. | Media | `3747bf9`: se mide por la variación del saldo; solo se asume si no se puede leer el saldo. |
| C-3 | El registro `held_not_ready` (nuevo, M-1) no tiene `target`, y `check_parity` hacía `record["target"]`: `paper_parity.py` fallaba con `KeyError`. Reproducido antes de corregir. | Alta para la herramienta (rompía la paridad) | `1708732`: la fila queda sin target y se reporta como discrepancia. |
| C-4 | `test_defaults`, `test_risk_defaults` y el nuevo test de `.env.example` dependían del entorno: fallaban con `BOT_MODE=live` exportado en CI. Problema preexistente que destapó el test nuevo. | Baja | `799747c`: fixture que limpia las variables de `Settings`. |

Verificado sin problemas:

- Orden de inicialización en live: el libro se carga antes de
  `_reconcile_stops`, de `_drop_trade_stops` y de la reconciliación del
  journal, así que todos ven la posición del bot.
- Los stops nativos llegan al libro: `get_order` olvida las órdenes ya
  cerradas y `cancel_order` las canceladas, y `record_external_fill` descuenta
  el fill del stop.
- El dashboard y el healthcheck solo leen `drawdown`/`stops`/`saved_at`: la
  clave nueva `live_ledger` no les afecta.
- La validación nueva de `Settings` coincide con lo que exige el backtest
  (`allocation ∈ (0, 1]`).
- README y `docs/API_CONTRACTS.md` reflejan ahora el comportamiento nuevo:
  posición sin historia, reintento sin fill, límites del lock, libro y fills.

Riesgos residuales del libro (documentados):

- Un crash entre el fill en el exchange y el guardado del estado deja esas
  monedas fuera del libro. El bot nunca las vende; se reportan como ajenas al
  arrancar y se pueden adoptar.
- El saldo en USDT sigue contando entero en el equity y el drawdown: la
  recomendación sigue siendo una cuenta o subcuenta dedicada.
- Con comisiones en BNB y `BNB/USDT` como par operado, el consumo de BNB por
  comisiones no se descuenta del libro (lo corrige el recorte al saldo).

Verificación final: **747 ok, 1 skip** (red), cobertura 90 %, ruff, format
y mypy limpios. Alembic en PostgreSQL 16: `check` sin drift y
downgrade/upgrade ok. Backtests idénticos a los de la 1.ª ronda y paridad OK.

## 9. Tercera ronda: M-4 (pérdida diaria) y simulación del kill-switch

**Decisión.** En `donchian_ensemble` el límite de pérdida diaria ya **no
liquida por defecto**. `DONCHIAN_DAILY_LOSS_ACTION` admite:

- `block_increases` (**por defecto**): notifica, no vende, y la exposición
  solo puede bajar hasta el siguiente día UTC.
- `liquidate`: el comportamiento anterior.
- `halve`: vende la mitad de cada posición y bloquea subas.
- `off`: sin límite diario.

El HALT por drawdown máximo y `regime_selector` siguen liquidando siempre.

**Fundamento:**

1. Liquidar por una caída intradía es una salida que la estrategia validada
   no tiene: sus salidas son los stops de canal al cierre diario.
2. Kaminski & Lo (2014): un stop-loss solo agrega valor si hay momentum en
   el horizonte en que actúa; sin él, solo baja el retorno esperado.
3. En Bitcoin, los retornos de 1-24 h muestran reversión a la media, mayor
   tras caídas grandes (De Nicola, *Ledger* 2021; estudio de shocks
   intradía, 2022). Vender en esa caída es vender en el horizonte que
   tiende a revertir.
4. Backtest de estrés sintético (`synth_ohlcv.py ... 0.40 0.02`: vol 40 %,
   caídas intradía del 12 % en ~2 % de los días que cierran cerca del open;
   un par al 100 %):

| Política | Sharpe | CAGR | maxDD | Eventos diarios | HALT |
|---|---|---|---|---|---|
| sin kill-switch | 0,85 | 9,05 % | 13,8 % | 0 | no |
| `liquidate` | 0,04 | 0,07 % | 15,0 % | 6 | **sí** |
| `block_increases` | 0,85 | 9,05 % | 13,8 % | 23 | no |
| `halve` | 0,18 | 0,85 % | 15,0 % | 6 | **sí** |
| `off` | 0,85 | 9,05 % | 13,8 % | 0 | no |

Liquidar realiza cada pérdida en el mínimo; la acumulación lleva al HALT
del 15 % y el bot queda fuera para siempre. Los datos son sintéticos y las
caídas, por construcción, cierran cerca del open: el resultado favorece a
`block_increases` por diseño del escenario. **Hay que repetir la comparación
con datos reales** (`run_backtest.py --compare-daily-loss`) antes de vivo.

**Simulación** (`TargetExposureEngine`, opcional, activada por defecto en
`build_target_engine` y `run_backtest.py`):

- HALT: vende todo y nunca más sube la exposición.
- Pérdida diaria: aplica la acción elegida.
- Camino intradía: cada par va linealmente del open al low, y el límite
  dispara donde se cruza. Es pesimista: mínimos simultáneos y chequeo
  continuo, cuando el bot chequea cada hora.
- Los eventos quedan en `risk_events` y se cuentan en `metrics()`
  (`daily_loss_events`, `halted`).

Sin límites, el motor es idéntico al anterior (los tests de exactitud pasan).

**Auditoría de consistencia de la 3.ª ronda:**

- La paridad usa el motor sin kill-switch y compara solo targets antes de
  riesgo: no cambia.
- `block_increases` y `off` coinciden en el backtest, porque el rebalanceo
  ocurre en la apertura, antes del camino intradía. En vivo, `block_increases`
  además frena los reintentos horarios de un rebalanceo fallido ese día.
  Diferencia esperada y documentada.
- `DrawdownMonitor(daily_loss_limit=...)` solo lo construye el bot (dos
  sitios, ambos actualizados); el dashboard y el healthcheck leen el estado
  sin construir el monitor.
- `halve` usa los precios del ciclo en curso (`_last_cycle_prices`), nunca
  atributos privados del executor; sin precio, ese par no se toca.
- `.env.example` documenta el setting con su default (lo verifica el test
  de sincronía), y README y docstrings reflejan el comportamiento nuevo.

Verificación: **764 ok, 1 skip** (red), cobertura 90 %; ruff, format y mypy
limpios.

## Fuentes consultadas

- Código de ccxt 4.5.84 instalado (`.venv/.../ccxt/async_support/binance.py`).
- [Ejemplos de stop-loss en ccxt (Binance)](https://docs.ccxt.com/docs/examples/py/binance-stop-loss-take-profit) y [guía de tipos de orden](https://docs.ccxt.com/blog/order-types-guide).
- [Issue ccxt: unificar stopLossPrice/takeProfitPrice](https://github.com/ccxt/ccxt/issues/14109).
- [Binance Academy: stop-loss con la API](https://www.binance.com/ru/academy/articles/how-to-place-a-stop-loss-order-with-binance-api).
- Límite de 1000 velas por página en `/api/v3/klines` (fuentes secundarias): [rustrade-data](https://docs.rs/rustrade-data/0.5.0/src/rustrade_data/exchange/binance/historical.rs.html).
- Deprecación de la unidad genérica de `timedelta64` en numpy 2.5: [numpy-discussion](https://go.mail-archive.com/numpy-discussion@python.org/msg08568.html), [xarray PR 11340](https://github.com/pydata/xarray/pull/11340.patch).
- Kaminski & Lo, *When Do Stop-Loss Rules Stop Losses?* (Journal of Financial Markets, 2014): [MIT DSpace](https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf).
- Reversión intradía en Bitcoin: [De Nicola, *On the Intraday Behavior of Bitcoin* (Ledger, 2021)](https://ledger.pitt.edu/ojs/ledger/article/view/213); [estudio de shocks intradía (2022)](https://informahealthcare.com/doi/abs/10.1080/15140326.2022.2151253).
