# ADR-001: uv como gestor de dependencias

**Estado**: Aceptado
**Fecha**: 2026-03-15

## Contexto

Se necesita un gestor de dependencias Python que sea reproducible, rápido y profesional para un proyecto de portfolio.

## Opciones Evaluadas

| Criterio | uv | Poetry | pipenv |
|----------|-----|--------|--------|
| Velocidad de instalación | ~3s (cold) | ~11s (cold) | ~15s+ |
| Lock file generation | ~8s | ~22s | ~30s+ |
| Descargas mensuales PyPI | ~75M | ~66M | ~20M |
| Escrito en | Rust | Python | Python |
| pyproject.toml nativo | Si | Si | No (usa Pipfile) |
| Publicación a PyPI | Basica | Excelente | No |

## Decisión

**Usar `uv`** como gestor principal de dependencias.

## Razones

- 3-4x más rápido que Poetry en todas las operaciones.
- Ha superado a Poetry en adopción (75M vs 66M descargas/mes).
- Compatible con pyproject.toml estándar.
- Para un bot de trading (no una librería publicable), no necesitamos las features de publicación de Poetry.

## Consecuencias

- Los comandos usan `uv run` en lugar de `poetry run`.
- `uv.lock` en lugar de `poetry.lock`.
- Si en el futuro se publica como librería, evaluar migración a Poetry para publicación.
