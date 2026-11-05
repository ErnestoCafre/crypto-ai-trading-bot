# Notas para próximas sesiones y pruebas

Información práctica reunida durante la auditoría del 2026-11-05
([informe](2026-11-05-auditoria-sistema.md)). Sirve para retomar pruebas en
otra sesión (local o Claude Code en la nube) sin redescubrir el entorno.

> Regla de la auditoría: **no** ejecutar el bot (`trading-bot`,
> `python -m trading_bot`) ni en paper ni en vivo, y **no** usar claves reales
> de exchange. Todo lo de abajo funciona sin ellas.

---

## 1. Entorno

| Elemento | Valor observado |
|---|---|
| Python | 3.12.3 (`.python-version`) |
| Gestor | `uv` (`uv sync --extra dev`, ~2 min la primera vez) |
| Versiones clave (uv.lock) | ccxt 4.5.84, pandas 2.3.3, numpy 2.5.3, SQLAlchemy 2.1, alembic 1.18 |
| Docker | cliente instalado, **sin daemon** en el contenedor cloud → `docker compose` no funciona |
| PostgreSQL | binarios 16 en `/usr/lib/postgresql/16/bin` (usar `initdb`/`pg_ctl`, ver §3) |
| Red | Proxy de salida: PyPI y GitHub sí; **Binance, Binance US, binance.vision y OKX bloqueados (403)** |

Variable de entorno: `uv` avisa que `UV_NATIVE_TLS` está deprecada; usar
`export UV_SYSTEM_CERTS=1; unset UV_NATIVE_TLS` para silenciarlo.

---

## 2. Pruebas

```bash
uv sync --extra dev
uv run pytest tests -o addopts="" --cov=trading_bot --cov-report=term -rs   # ~2 min
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run mypy src
```

- `tests/integration` **no necesita Postgres**: usa
  `sqlite+aiosqlite:///:memory:`. Solo `TestCCXTIntegration` usa red (Binance)
  y hace *skip* si no hay acceso.
- Resultado de referencia (tras los fixes de la auditoría): 703 passed,
  1 skipped, cobertura 90 %.
- Warnings esperados: `DeprecationWarning` de numpy 2.5 por la unidad genérica
  de `timedelta64` dentro de `pd.Timedelta(...)` (pandas 2.3.3). No es del
  código del proyecto; ver hallazgo M-5 del informe.
- `-p no:cacheprovider` evita escribir `.pytest_cache` si el árbol debe quedar
  limpio.

---

## 3. PostgreSQL local sin Docker

```bash
PG=/usr/lib/postgresql/16/bin
DATA=/tmp/pgaudit/data            # o el scratchpad de la sesión
mkdir -p "$(dirname $DATA)" && chown postgres "$(dirname $DATA)"
# todos los directorios padre deben ser atravesables (chmod o+x) por "postgres"
su postgres -c "$PG/initdb -D $DATA -A trust -U postgres"
su postgres -c "$PG/pg_ctl -D $DATA -o '-p 55432 -k /tmp' -l $DATA/../log start"
psql -h /tmp -p 55432 -U postgres -c "create database trading_bot"
psql -h /tmp -p 55432 -U postgres -c "alter user postgres password 'x'"

export DATABASE_URL=postgresql://postgres:x@127.0.0.1:55432/trading_bot
uv run alembic upgrade head
uv run alembic check                       # debe decir: No new upgrade operations detected
uv run alembic downgrade base && uv run alembic upgrade head
uv run python -I docs/audit/tools/pg_smoke.py   # TradeJournal contra Postgres real
```

- `migrations/env.py` toma la URL de `Settings.DATABASE_URL` (no de
  `alembic.ini`) y la convierte a `postgresql+asyncpg://`.
- `alembic check` imprime avisos "Detected sequence ... assuming SERIAL": son
  normales (columnas `BigInteger` autoincrementales).
- Parar: `su postgres -c "$PG/pg_ctl -D $DATA stop"`.

---

## 4. Backtests sin acceso a exchanges

`scripts/download_historical.py` necesita Binance (bloqueado en la nube) y
además PostgreSQL. Alternativa: datos sintéticos por `--csv`.

```bash
T=docs/audit/tools; D=/tmp/audit-data; mkdir -p $D
uv run python -I $T/synth_ohlcv.py $D/synth_1d.csv 1D 2200 7    # diario, termina ayer
uv run python -I $T/synth_ohlcv.py $D/synth_1h.csv h 6000 11    # horario

# donchian_ensemble (una sola serie: --pair, no --pairs, con --csv)
uv run python scripts/run_backtest.py --strategy-mode donchian_ensemble \
    --csv $D/synth_1d.csv --pair SYN/USDT --holdout-start 2023-06-01
# regime_selector 1h con walk-forward
uv run python scripts/run_backtest.py --csv $D/synth_1h.csv --pair SYN/USDT \
    --timeframe 1h --splits 3
```

- El CSV necesita columnas `timestamp, open, high, low, close, volume`.
- `donchian_ensemble` necesita ≥ 450 velas diarias (360 + 90) antes de
  exponerse; con menos, exposición 0.
- Las métricas sobre datos sintéticos **no** son evidencia de rendimiento.
- Con red disponible: `uv run python scripts/download_historical.py` escribe
  Postgres y la caché parquet `var/cache/<exchange>/<PAIR>/<tf>.parquet`, que
  `run_backtest.py` lee sin `--csv`.

---

## 5. Paridad paper ↔ backtest sin correr el bot

`scripts/paper_parity.py` compara el decision log
(`var/state/donchian_decisions_{mode}.jsonl`) contra `TargetExposureEngine`.
Sin un paper real, `tools/make_decisions.py` genera un log equivalente
calculando `target_exposure()` sobre ventanas de 1000 velas (como el bot):

```bash
R=/tmp/audit-parity; mkdir -p $R && cp $D/synth_1d.csv $R/ && cd $R
uv run --project /ruta/al/repo python -I /ruta/al/repo/docs/audit/tools/make_decisions.py \
    synth_1d.csv . 200 1000        # 200 días, ventana 1000 (probar 450 para ver sensibilidad)
uv run --project /ruta/al/repo python /ruta/al/repo/scripts/paper_parity.py \
    --log ./decisions.jsonl --cache
# esperado: "OK: the run matches the backtest."
```

- `make_decisions.py` escribe la caché parquet **relativa al directorio
  actual** (`var/cache/...`): ejecutarlo fuera del repo.
- El chequeo de continuidad espera una decisión por día hasta *ayer* (UTC):
  los datos deben terminar ayer (lo hace `synth_ohlcv.py`).
- La paridad compara el target **antes de IA**; no cubre kill-switch ni
  mínimos del exchange.

---

## 6. Datos útiles sobre APIs (verificados el 2026-11-05)

- **Binance spot `/api/v3/klines`**: máximo 1000 velas por petición. El
  fetcher pagina (`batch_limit = min(limit, 1000)`) y recorta con `tail(limit)`.
  El bot pide `HISTORY_BARS + 1 = 1001` velas diarias → 2 páginas.
- **Stops nativos (ccxt 4.5.84)**: `create_order(type="market", side="sell",
  params={"stopLossPrice": p})` en spot se mapea a `STOP_LOSS` de Binance
  (ver `create_order_request` en `ccxt/async_support/binance.py`). En
  contratos sería `STOP_MARKET`. Si se actualiza ccxt, revisar ese mapeo.
- **ccxt y `__aenter__`**: si `load_markets()` falla, hay que cerrar el
  exchange a mano (ya corregido en `CCXTFetcher`); si no, aparece
  `Unclosed client session`.
- **numpy ≥ 2.5**: depreca la unidad genérica de `timedelta64`
  (futuro error). pandas 2.3.3 la dispara en `pd.Timedelta(...)`.
  Recomendado `numpy<2.6` hasta que pandas lo resuelva.

---

## 7. Mapa rápido del código auditado

| Tema | Archivo |
|---|---|
| Bucle, ciclo, rebalanceo, stops, estado | `src/trading_bot/core/bot.py` |
| Estrategia Donchian (pura) | `src/trading_bot/core/strategies/donchian_ensemble.py` |
| Backtest target / paridad | `src/trading_bot/backtest/target_engine.py`, `backtest/parity.py` |
| Riesgo | `src/trading_bot/risk/{position_sizer,drawdown,stop_tracker,risk_state}.py` |
| Ejecución | `src/trading_bot/execution/{paper_executor,live_executor}.py` |
| Lock / escritura atómica / decision log | `state_lock.py`, `state_file.py`, `core/decision_log.py` |
| BD / migraciones | `src/trading_bot/db/`, `migrations/versions/001_initial_schema.py` |

Estado persistido (todo bajo `var/`, ignorado por git):
`var/state/risk_state_{mode}.json` (+ `.lock`),
`var/state/donchian_decisions_{mode}.jsonl`, `var/cache/` (parquet).

---

## 8. Pendientes abiertos (del informe)

H-1 cuenta compartida en live · H-2 orden no ejecutada marcada como hecha ·
M-1 liquidación con historia insuficiente · M-2 `filled=None` · M-3 stop no
reevaluado · M-4 kill-switch diario vs rebalanceo · M-5 tope de numpy ·
B-1..B-7 robustez. Detalle y escenarios en el informe.
