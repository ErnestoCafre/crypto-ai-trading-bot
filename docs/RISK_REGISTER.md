# Risk Register — Registro de Riesgos

Riesgos técnicos y operacionales identificados para el bot de trading crypto con AI.

**Escala de probabilidad**: Baja (1) | Media (2) | Alta (3)
**Escala de impacto**: Bajo (1) | Medio (2) | Alto (3) | Crítico (4)
**Prioridad**: Probabilidad x Impacto

---

## Riesgos de Infraestructura y APIs

| ID | Riesgo | Prob. | Impacto | Prio. | Mitigación | Owner |
|----|--------|-------|---------|-------|------------|-------|
| R01 | **Exchange API caída** (Binance, Bybit down o rate limited) | 2 | 4 | 8 | Fallback a segundo exchange via CCXT. Queue de órdenes pendientes. Telegram alerta inmediata. No operar si no hay confirmación de ejecución. | execution/ |
| R02 | **Pérdida de conexión a internet del VPS** | 2 | 4 | 8 | Stop-loss en exchange (server-side, no depende del bot). Telegram heartbeat cada 5 min — si no llega, investigar. Docker auto-restart. | docker-compose.yml |
| R03 | **Base de datos PostgreSQL se llena o corrompe** | 1 | 3 | 3 | Monitoreo de disco. Retention policy: purgar OHLCV cache >1 año. Backups diarios automatizados. SQLite como fallback temporal. | db/ |
| R04 | **CoinGecko / CryptoCompare API cambia o depreca endpoints** | 2 | 2 | 4 | Interfaz abstracta `BaseFetcher` / `BaseNewsFetcher`. Cambiar implementación sin tocar lógica. Tests de integración con health check semanal. | data/ |

---

## Riesgos de Trading y Riesgo Financiero

| ID | Riesgo | Prob. | Impacto | Prio. | Mitigación | Owner |
|----|--------|-------|---------|-------|------------|-------|
| R05 | **Flash crash / liquidation cascade** mueve precio 10-20% en minutos | 2 | 4 | 8 | Stop-loss server-side en exchange. Kill-switch drawdown >15%. Max apalancamiento 2-3x. Daily loss limit 3-5%. | risk/ |
| R06 | **Slippage real muy superior al modelado** en altcoins | 3 | 3 | 9 | Solo operar pares de alta liquidez (BTC, ETH, SOL). Modelar slippage dinámico (no fijo). Limitar tamaño de orden vs profundidad del order book. Estrés de slippage ×2/×3 en `run_backtest.py --robustness`. | risk/ + backtest/ |
| R07 | **Estrategia funciona en backtest pero falla en live** (overfitting) | 3 | 3 | 9 | Walk-forward validation. Monte Carlo 5000 sim. Paper trading mínimo 2-4 semanas antes de live. Out-of-sample testing estricto. Inicios aleatorios (`--robustness`), paridad backtest/ventana en vivo (`scripts/window_parity.py`) y versión del validador en cada reporte (`docs/VALIDATION_CHANGELOG.md`). | backtest/ |
| R08 | **Exchange delista un par que el bot tiene en posición** | 1 | 3 | 3 | Monitorear anuncios del exchange. Solo operar top 20 por market cap. Alert si par va a ser delistado. | data/ |
| R09 | **Funding rate acumula costos no previstos** en futuros perpetuos | 2 | 2 | 4 | Monitorear funding rate antes de abrir posición. Limitar holding time. Incluir funding en backtesting siempre. | risk/ |

---

## Riesgos de AI / Machine Learning

| ID | Riesgo | Prob. | Impacto | Prio. | Mitigación | Owner |
|----|--------|-------|---------|-------|------------|-------|
| R10 | **Modelo DL overfittea** y genera señales inversas en live | 3 | 3 | 9 | Walk-forward validation estricta. Early stopping + regularización. Accuracy direccional >55% como umbral mínimo. Peso máximo de DL: 0.2 (nunca domina la señal). | ai/prediction/ |
| R11 | **LLM alucina** análisis convincente pero incorrecto | 3 | 2 | 6 | LLM nunca es única fuente de decisión (peso 0.2). Requiere consenso 2/3 agentes. Validar contra señales cuantitativas. Log completo para auditoría posterior. | ai/agents/ |
| R12 | **Costo de API LLM se dispara** por bug o loop infinito | 2 | 2 | 4 | Circuit breaker: `LLM_MAX_DAILY_COST_USD=5.0`. Monitoreo de tokens consumidos. Alerta Telegram si >80% del presupuesto diario. | ai/agents/ |
| R13 | **CryptoBERT eliminado de Hugging Face** o modelo corrupto | 1 | 2 | 2 | Descargar y cachear modelo localmente en primera ejecución. Fallback a `ProsusAI/finbert`. Feature flag `USE_SENTIMENT` permite desactivar sin afectar el bot. | ai/sentiment/ |
| R14 | **Data drift**: mercado cambia de régimen y modelos pierden accuracy | 3 | 3 | 9 | Monitoreo continuo de accuracy en producción. Retraining automático cada 14 días. Alerta si accuracy < umbral. Desactivar modelo automáticamente si performance degrada. | ai/prediction/ |
| R15 | **Sentimiento como indicador contrario**: sentimiento extremo positivo indica techo | 2 | 2 | 4 | Implementar lógica de sentimiento contrario cuando score > 0.8 o < -0.8 persistente. Documentar este comportamiento en la estrategia. | ai/sentiment/ |
| R16 | **Latencia de API LLM** (2-10s) causa pérdida de oportunidad de entrada | 2 | 1 | 2 | LLM analiza en background, no en hot path. Si latencia > timeout, operar sin LLM (graceful degradation). Señal técnica siempre tiene prioridad. | ai/agents/ |

---

## Riesgos de Seguridad

| ID | Riesgo | Prob. | Impacto | Prio. | Mitigación | Owner |
|----|--------|-------|---------|-------|------------|-------|
| R17 | **API keys de exchange filtradas** (commit accidental, log) | 1 | 4 | 4 | `.env` en `.gitignore`. pre-commit hook para detectar secrets. API keys con permisos mínimos (solo trading, sin withdraw). IP whitelist en exchange. | config/ |
| R18 | **API key de LLM filtrada** | 1 | 2 | 2 | Mismas medidas que R17. Límite de gasto en la cuenta de Anthropic/OpenAI. | config/ |
| R19 | **Bot comprometido ejecuta órdenes maliciosas** | 1 | 4 | 4 | API keys sin permiso de withdraw. Limits de tamaño de orden. Kill-switch manual via Telegram `/pause`. Logs de todas las órdenes. | execution/ |

---

## Riesgos de Proyecto / Portfolio

| ID | Riesgo | Prob. | Impacto | Prio. | Mitigación | Owner |
|----|--------|-------|---------|-------|------------|-------|
| R20 | **`pandas_ta` archivada** antes de completar el bot (jul 2026) | 3 | 2 | 6 | Usar `pandas-ta-classic` desde el inicio. No depender del paquete original. | core/ |
| R21 | **Scope creep**: añadir features no planificados que retrasan el MVP | 3 | 2 | 6 | Timeline estricto con fases. MVP primero (fases 0-4), AI después (8-11), polish al final (5-7). No agregar features que no estén en ARCHITECTURE.md. | proyecto |
| R22 | **Timeline subestimado** para fases AI | 2 | 2 | 4 | Estimaciones ya incluyen margen. Cada fase AI es independiente y se puede posponer sin afectar el MVP. Feature flags permiten activar/desactivar. | proyecto |

---

## Top 5 Riesgos (ordenados por prioridad)

1. **R06** (Prio 9) — Slippage real superior al modelado
2. **R07** (Prio 9) — Estrategia overfittea en backtest
3. **R10** (Prio 9) — Modelo DL overfittea
4. **R14** (Prio 9) — Data drift en modelos AI
5. **R01/R05** (Prio 8) — Exchange down / Flash crash

Estos 5 riesgos deben tener mitigaciones implementadas **antes de pasar a live trading**.
