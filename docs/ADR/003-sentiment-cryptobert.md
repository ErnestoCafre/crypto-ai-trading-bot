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

## Revisión 2026-09-28

**Cambio de modelo por defecto**: `kk08/CryptoBERT` → `ElKulako/cryptobert`.

- `kk08/CryptoBERT` no define `id2label` en su `config.json`: el pipeline
  devuelve `LABEL_0` (negativo) / `LABEL_1` (positivo), que el analizador no
  reconocía, así que el score de sentimiento era **siempre 0.0**. Además está
  sin mantenimiento (última actualización dic-2024, ~950 descargas) y su card
  indica un corpus propio reducido.
- `ElKulako/cryptobert` (RoBERTa, 3 clases `Bearish`/`Neutral`/`Bullish`,
  fine-tuned sobre 2M posts de StockTwits; actualizado may-2025, ~457k
  descargas). Entrenado con secuencias de **128 tokens**: se usa ese
  `max_length`.
- Cada modelo soportado tiene un perfil explícito de etiquetas → polaridad
  (verificado contra su `config.json`); los no perfilados se mapean desde
  `config.id2label` y una etiqueta desconocida genera un warning, nunca un 0
  silencioso.
- El score pasa a ser la polaridad ponderada por probabilidad de todas las
  clases (P(bullish) − P(bearish)), en [-1, 1].
- La inferencia corre en batch y fuera del event loop (`asyncio.to_thread`).

El fallback documentado sigue siendo `ProsusAI/finbert`.
