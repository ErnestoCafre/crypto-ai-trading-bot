# ADR-003: CryptoBERT para análisis de sentimiento

**Estado**: Aceptado
**Fecha**: 2026-03-15

## Contexto

Se necesita un modelo de NLP para analizar sentimiento en noticias y redes sociales crypto. El modelo debe funcionar en CPU (inference), ser gratuito y entender jerga crypto.

## Opciones Evaluadas

| Criterio | CryptoBERT | FinBERT (base) | VADER + TextBlob | LLM API (Claude/GPT) |
|----------|-----------|---------------|-------------------|----------------------|
| Especialización crypto | Alta (fine-tuned) | Media (finanzas general) | Baja (general) | Media-Alta |
| Costo por análisis | Gratis | Gratis | Gratis | ~$0.001-0.01 por texto |
| Latencia (CPU) | ~50-100ms | ~50-100ms | ~1ms | ~1-5s |
| Entiende "moon", "rekt" | Si | No | No | Si |
| Requiere internet | No (modelo local) | No | No | Si |
| Tamaño modelo | ~500MB | ~500MB | ~0MB | N/A |

## Decisión

**Usar CryptoBERT** (`kk08/CryptoBERT` en Hugging Face) como modelo principal de sentimiento, con FinBERT (`ProsusAI/finbert`) como fallback.

## Razones

- Fine-tuned específicamente para textos crypto — entiende jerga del mercado.
- Inference en CPU en ~50-100ms, suficiente para análisis cada 15 min.
- Sin costo por análisis (a diferencia de llamadas API a LLM).
- Modelo se cachea localmente, no depende de internet para inference.

## Consecuencias

- Requiere ~500MB de disco y ~1-2GB de RAM para mantener el modelo cargado.
- Si el modelo se elimina de Hugging Face, fallback a FinBERT (menos preciso en crypto).
- Para sentimiento más sofisticado (resúmenes, contexto), se complementa con LLM en la Capa 2 (Fase 9).
