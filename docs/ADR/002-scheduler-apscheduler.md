# ADR-002: APScheduler en lugar de Celery+Redis

**Estado**: Aceptado
**Fecha**: 2026-03-15

## Contexto

El bot necesita ejecutar tareas periódicas (análisis cada 1-5 min, sentimiento cada 15 min, retraining cada 14 días). Se debía elegir entre Celery+Redis y APScheduler.

## Opciones Evaluadas

| Criterio | APScheduler | Celery + Redis |
|----------|------------|---------------|
| Complejidad de setup | Baja (pip install) | Alta (Redis server + workers) |
| Dependencias externas | Ninguna | Redis |
| Overhead operacional | Mínimo | Alto (mantener Redis + workers) |
| Escalabilidad | Un proceso | Distribuido multi-worker |
| Caso de uso ideal | Scheduling local, baja frecuencia | Workloads distribuidos, alta concurrencia |

## Decisión

**Usar APScheduler** para la orquestación de tareas.

## Razones

- Un bot de trading ejecutando cada 1-5 min en 3 pares no necesita distribución de carga.
- Celery+Redis añade complejidad operacional sin beneficio para este caso de uso.
- APScheduler se integra directamente en el proceso Python, sin servicios externos.
- Menos puntos de fallo en producción (no depende de que Redis esté up).

## Consecuencias

- Si el bot escala a +50 pares con múltiples timeframes simultáneos, re-evaluar Celery.
- El bot corre en un solo proceso. Para paralelismo, usar asyncio o threading interno.
