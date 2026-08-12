# ADR-006: Arquitectura multi-agente LLM para análisis de mercado

**Estado**: Aceptado
**Fecha**: 2026-03-15

## Contexto

Se quiere integrar un LLM (Claude/GPT) como analista de mercado en el bot de trading. La decisión clave es: un solo agente monolítico vs múltiples agentes especializados.

## Opciones Evaluadas

| Criterio | Agente único (monolítico) | Multi-agente (3 agentes + orquestador) |
|----------|--------------------------|---------------------------------------|
| Prompt complexity | Un prompt muy largo con todo el contexto | Prompts cortos y enfocados por rol |
| Calidad de análisis | Puede perder foco con mucho contexto | Cada agente se especializa |
| Costo por análisis | 1 llamada API (~1000-2000 tokens) | 4 llamadas API (~500-800 tokens c/u) |
| Latencia | ~2-5s | ~3-8s (paralelo) o ~8-20s (secuencial) |
| Robustez | Si falla, no hay análisis | Si un agente falla, los otros 2 operan |
| Consensus / Checks | No hay validación cruzada | Desacuerdos = señal de cautela |

## Decisión

**Usar arquitectura multi-agente** con 3 agentes especializados + orquestador.

## Diseño

1. **Technical Analyst**: recibe indicadores técnicos, produce análisis técnico + recomendación.
2. **Sentiment Analyst**: recibe noticias + score de sentimiento, produce resumen narrativo + sesgo.
3. **Risk Evaluator**: recibe estado del portfolio + drawdown + VaR, evalúa riesgo.
4. **Orchestrator**: recibe las 3 salidas, aplica consenso, produce decisión final.

### Reglas de Consenso
- 3/3 coinciden → confianza alta (sizing 100%)
- 2/3 coinciden → confianza media (sizing 75%)
- 0 acuerdo o empate → SKIP (no operar)

## Razones

- La separación por roles mejora la calidad del output (cada agente tiene un prompt enfocado).
- El consenso actúa como mecanismo de seguridad: si hay desacuerdo, el bot no opera.
- Tolerancia a fallos: si la API falla en una llamada, 2/3 agentes pueden operar.
- Inspirado en TradingAgents framework (2026) que simula roles de un trading desk profesional.

## Consecuencias

- Costo ~2-3x mayor que agente único. Mitigado usando Claude Haiku (~$0.25/M tokens).
- Los 3 agentes se ejecutan en paralelo (asyncio) para minimizar latencia.
- Circuit breaker de costo diario ($5/día default) para prevenir gastos excesivos.
- Toda la capa LLM es opcional (feature flag `USE_LLM_AGENT`). El bot funciona sin ella.
