# Changelog del validador

Cada resultado de backtest o de entrenamiento lo produce una versión
concreta de la simulación. Si la simulación cambia, un número viejo deja de ser
comparable con uno nuevo, y el número por sí solo no lo dice. Por eso cada
reporte JSON lleva un bloque `provenance`
(`src/trading_bot/backtest/provenance.py`) con `validation_version`, commit,
huella de los datos y los settings de los que depende el resultado.

Reglas:

- Todo cambio que pueda mover un resultado sube `VALIDATION_VERSION` y se
  anota acá con fecha, qué cambió y **en qué dirección** mueve los resultados
  (más estricto, más permisivo o neutro).
- `tests/unit/test_validation_version.py` fija la salida de los dos motores
  sobre datos sintéticos para cada versión. Si cambia la simulación, el test
  falla hasta que se sube la versión y se anota el cambio.
- Los reportes viejos no se reescriben: llevan la versión con la que se
  generaron. Los que no tienen bloque `provenance` son anteriores a v1.

Idea tomada del `JUDGE_CHANGELOG.md` de
[CacheCarti/Crypto-Strategies](https://github.com/CacheCarti/Crypto-Strategies),
donde cada cambio del validador quedó con fecha y dirección.

## v1 — 2026-11-06 (línea base)

Primera versión con sello. Describe lo que simula hoy el código:

- **`BacktestEngine`** (`regime_selector`): la señal sale de la vela cerrada i
  y se ejecuta al open de i+1. El stop ATR×2 se revisa en cada vela posterior
  a la entrada: si la vela abre por debajo del stop (gap), se ejecuta al open;
  si solo lo toca, al precio del stop. Spot long-only. Fee y slippage
  porcentuales por lado. Los indicadores se calculan sobre toda la historia;
  `scripts/window_parity.py` comprueba que coincidan con la ventana de
  `OHLCV_LOOKBACK_BARS` que usa el bot en vivo.
- **`TargetExposureEngine`** (`donchian_ensemble`): el objetivo diario se
  opera al open siguiente. El umbral de rebalanceo es `REBALANCE_THRESHOLD`,
  las ventas van antes que las compras y un día sin vela no liquida el par.
  El kill-switch se simula con un recorrido lineal del open al low y mínimos
  simultáneos (aproximación pesimista).
- **Robustez** (`backtest/robustness.py`): inicios aleatorios con semilla y
  estrés de slippage ×1, ×2 y ×3. El veredicto exige: al menos 60 % de
  ventanas rentables, Sharpe mediano > 0 y Sharpe > 0 con slippage ×2.

Cambios anteriores a la versión con sello, en el historial de git; todos
afectan a resultados sin `provenance`:

- 2026-11 (`4fd7fa4`): se empieza a simular el kill-switch y las políticas de
  pérdida diaria en `donchian_ensemble` (más estricto).
- 2026-11 (`42026ba`): un día sin vela deja de liquidar el par (neutro;
  corrige ventas que el bot no hace).
