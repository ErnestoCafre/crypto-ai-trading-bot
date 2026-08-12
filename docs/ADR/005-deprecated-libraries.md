# ADR-005: Reemplazo de librerías deprecadas/en riesgo

**Estado**: Aceptado
**Fecha**: 2026-03-15

## Contexto

Varias librerías populares en el ecosistema de trading Python están deprecadas, sin mantenimiento activo, o en riesgo de ser archivadas. Usar estas librerías crea riesgo de rotura a mediano plazo.

## Librerías Reemplazadas

### 1. pandas_ta → pandas-ta-classic

| | pandas_ta (original) | pandas-ta-classic |
|--|---------------------|-------------------|
| Estado | Riesgo de archivo (jul 2026) | Mantenido (fork comunitario) |
| API | Idéntica | Idéntica |
| Python support | Sin garantía futura | Rolling últimas 5 versiones |

**Razón**: el mantenedor del original anunció que archivará la librería si no recibe financiamiento antes de julio 2026.

### 2. pyfolio / empyrical → quantstats (o forks -reloaded)

| | pyfolio + empyrical | quantstats | pyfolio-reloaded |
|--|--------------------|-----------|--------------------|
| Estado | Deprecados (Quantopian cerró 2020) | Activo | Mantenido (Stefan Jansen) |
| Pandas 3.0 compat | No | Si | Si |
| Funcionalidad | Parcialmente rota | Todo-en-uno | Compatible con API original |

**Razón**: Quantopian cerró en 2020. Múltiples funciones están deprecadas con warnings y algunas están rotas con Pandas 3.0.

### 3. pipenv → uv (ver ADR-001)

**Razón**: pipenv es significativamente más lento y no usa pyproject.toml estándar.

## Decisión

- Usar `pandas-ta-classic` en lugar de `pandas_ta`.
- Usar `quantstats` como solución todo-en-uno para métricas financieras.
- Nunca importar `pyfolio`, `empyrical`, `pandas_ta` (originales) en el proyecto.

## Consecuencias

- Agregar regla en ruff/linting para detectar imports de librerías prohibidas.
- Documentar en CLAUDE.md para que Claude Code nunca sugiera estas librerías.
