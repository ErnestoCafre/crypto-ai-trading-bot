# ADR-004: GRU sobre LSTM para predicción de precio crypto

**Estado**: Aceptado
**Fecha**: 2026-03-15

## Contexto

Se necesita un modelo de deep learning para predicción de retorno/dirección de precio en la siguiente barra. Los candidatos principales son LSTM, GRU y Transformers.

## Opciones Evaluadas

| Criterio | GRU | LSTM | CNN-LSTM | Temporal Fusion Transformer |
|----------|-----|------|---------|---------------------------|
| Performance en crypto | Superior a LSTM (estudios recientes) | Baseline | Superior a ambos | Estado del arte |
| Velocidad de training | Rápido | Medio | Medio | Lento |
| Parámetros | Menos (2 gates vs 3) | Más | Más | Muchos más |
| Riesgo de overfitting | Menor | Mayor | Mayor | Alto |
| Requiere GPU | Recomendada | Recomendada | Recomendada | Necesaria |
| Complejidad de implementación | Baja | Baja | Media | Alta |
| Datos necesarios | Moderados | Moderados | Más | Muchos |

## Decisión

**Usar GRU como modelo principal** de deep learning, con CNN-LSTM como upgrade opcional.

## Razones

- Estudios recientes muestran que GRU supera a LSTM en forecasting de crypto (menor error, menores niveles de overfitting).
- Menos parámetros = entrena más rápido y generaliza mejor con datos limitados.
- 2 gates (reset + update) vs 3 gates de LSTM (input + forget + output) — más eficiente sin sacrificar capacidad.
- CNN-LSTM se reserva como upgrade cuando hay suficientes datos (+2 años de barras horarias).
- Transformers descartados para primera versión: requieren más datos, GPU obligatoria, y complejidad de implementación alta.

## Consecuencias

- Pipeline de predicción: LightGBM (baseline) → GRU (principal) → CNN-LSTM (opcional).
- Retraining cada 14 días con walk-forward validation.
- Si accuracy direccional < 55%, desactivar automáticamente y usar solo LightGBM.
