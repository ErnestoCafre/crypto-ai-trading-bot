# Testing Strategy

*pytest + hypothesis + mocks para APIs externas*

## Principios

1. **Ningún test debe depender de APIs externas** — todo se mockea o usa fixtures.
2. **Los datos financieros se testean con fixtures reales** — no datos inventados.
3. **Los módulos AI se testean por contrato** — verificar inputs/outputs, no la calidad del modelo.
4. **Coverage mínimo: 90%** en `core/`, `risk/`, `execution/`. Coverage de AI puede ser menor (~70%) por la naturaleza probabilística.

---

## Estructura de Tests

```
tests/
├── conftest.py                 # Fixtures globales (OHLCV sample, config mock)
├── fixtures/                   # Datos de prueba estáticos
│   ├── ohlcv_btcusdt_1h.parquet   # 30 días de datos BTC/USDT 1h reales
│   ├── ohlcv_ethusdt_1h.parquet   # 30 días de datos ETH/USDT 1h reales
│   ├── news_sample.json           # 50 noticias crypto reales
│   └── sentiment_sample.json      # Resultados de sentiment precalculados
├── unit/
│   ├── test_strategies.py          # Cada estrategia con datos conocidos
│   ├── test_signal_combiner.py     # Combinaciones de señales + edge cases
│   ├── test_position_sizer.py      # Position sizing + AI confidence adjustment
│   ├── test_stop_loss.py           # Stop-loss dinámico, trailing
│   ├── test_drawdown.py            # Kill-switch, daily loss limit
│   └── test_config.py              # Validación de config Pydantic
├── integration/
│   ├── test_data_pipeline.py       # Fetcher → DB → cache (con DB test)
│   ├── test_execution_paper.py     # Paper executor end-to-end
│   ├── test_backtest_pipeline.py   # Data → Strategy → Backtest → Metrics
│   └── test_notification.py        # Telegram mock
└── ai/
    ├── test_sentiment.py           # CryptoBERT contract tests
    ├── test_prediction.py          # Predictor contract tests
    ├── test_agents.py              # LLM agent contract tests (mocked)
    ├── test_aggregator.py          # Sentiment aggregator lógica
    └── test_orchestrator.py        # Consensus logic
```

---

## Fixtures Globales (`conftest.py`)

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """30 días de datos BTC/USDT 1h reales desde fixtures/."""
    return pd.read_parquet("tests/fixtures/ohlcv_btcusdt_1h.parquet")

@pytest.fixture
def sample_config() -> dict:
    """Config de test con AI deshabilitado por defecto."""
    return {
        "EXCHANGE_ID": "binance",
        "EXCHANGE_SANDBOX": True,
        "TRADING_PAIRS": ["BTC/USDT"],
        "USE_SENTIMENT": False,
        "USE_LLM_AGENT": False,
        "USE_DL_PREDICTION": False,
        "MAX_RISK_PER_TRADE_PCT": 1.0,
        "MAX_DRAWDOWN_PCT": 15.0,
    }

@pytest.fixture
def mock_sentiment_result():
    """Resultado de sentimiento para tests."""
    return SentimentResult(
        score=0.65,
        confidence=0.85,
        label="positive",
        model_used="CryptoBERT",
        analyzed_text="Bitcoin ETF approved...",
    )
```

---

## Estrategia por Módulo

### `core/strategies/` — Unit Tests

Qué testear:
- Señal correcta con datos que producen cruce de EMAs conocido.
- Señal neutral cuando no hay condiciones.
- Integración con `AISignals` opcionales (con y sin).
- Edge cases: DataFrame vacío, datos con NaN, un solo bar.

```python
def test_trend_following_generates_long_on_bullish_cross(sample_ohlcv):
    strategy = TrendFollowingStrategy(fast_period=9, slow_period=21)
    # Insertar datos donde EMA9 cruza por encima de EMA21
    signal = strategy.generate_signal(bullish_ohlcv)
    assert signal is not None
    assert signal.type == SignalType.LONG
    assert 0.0 <= signal.confidence <= 1.0

def test_strategy_works_without_ai_signals(sample_ohlcv):
    strategy = TrendFollowingStrategy()
    signal = strategy.generate_signal(sample_ohlcv, ai_signals=None)
    # No debe fallar sin señales AI
```

### `core/signal_combiner.py` — Unit Tests

Qué testear:
- Pesos suman ~1.0.
- `should_trade=False` cuando confianza < threshold.
- Sizing reducido cuando señales contradictorias.
- Funciona con AI signals parciales (solo sentimiento, sin predicción).
- Funciona sin ninguna señal AI.

### `risk/` — Unit Tests (críticos)

Qué testear:
- Position sizing nunca excede `MAX_POSITION_PCT`.
- Kill-switch activa cuando drawdown > threshold.
- Daily loss limit pausa trading.
- AI confidence adjustment reduce sizing correctamente.
- Edge cases: portfolio value = 0, volatilidad = 0.

```python
def test_kill_switch_activates_on_excessive_drawdown():
    manager = DrawdownManager(max_drawdown_pct=15.0)
    manager.update(peak=10000, current=8400)  # 16% drawdown
    assert manager.should_kill_switch() is True

def test_position_size_adjusted_by_ai_confidence():
    sizer = PositionSizer(max_risk_pct=1.0)
    full = sizer.calculate(..., ai_confidence=0.9)
    reduced = sizer.calculate(..., ai_confidence=0.5)
    assert reduced.quantity < full.quantity
```

### `ai/sentiment/` — Contract Tests

Qué testear:
- Output tiene los campos esperados (`score`, `confidence`, `label`).
- `score` está en rango [-1.0, +1.0].
- `confidence` está en rango [0.0, 1.0].
- `analyze_batch()` retorna misma cantidad de resultados que inputs.
- Graceful failure: texto vacío, texto muy largo, caracteres especiales.

**No testear**: calidad del sentimiento (eso es evaluación del modelo, no test de software).

```python
def test_cryptobert_output_contract():
    analyzer = CryptoBERTAnalyzer()
    result = analyzer.analyze("Bitcoin hits all time high")
    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.label in ("positive", "negative", "neutral")
    assert result.model_used == "CryptoBERT"
```

### `ai/prediction/` — Contract Tests

Qué testear:
- Output tiene campos esperados.
- `predicted_direction` es uno de `["up", "down", "flat"]`.
- `confidence` en rango [0.0, 1.0].
- `train()` retorna métricas esperadas.
- `save()` + `load()` produce misma predicción.

**No testear**: accuracy del modelo (eso se evalúa en backtesting).

### `ai/agents/` — Mocked Tests

Qué testear:
- Cada agente retorna `AgentAnalysis` con campos completos.
- Orquestador aplica regla de consenso correctamente (2/3 coinciden → señal).
- Orquestador retorna SKIP si hay desacuerdo.
- Fallback cuando API del LLM falla (timeout, error).
- Circuit breaker se activa cuando costo diario excede límite.

**Mock**: la llamada al API del LLM se mockea siempre. Nunca llamar al API real en tests.

```python
@pytest.fixture
def mock_llm_response():
    return AgentAnalysis(
        agent_name="technical_analyst",
        recommendation="LONG",
        confidence=0.8,
        reasoning="EMA crossover bullish...",
        key_factors=["EMA9 > EMA21", "RSI < 70"],
        raw_response="...",
        tokens_used=500,
        latency_ms=1200,
    )

def test_orchestrator_consensus_2_of_3(mock_llm_response):
    tech = mock_llm_response  # LONG
    sent = replace(mock_llm_response, agent_name="sentiment", recommendation="LONG")
    risk = replace(mock_llm_response, agent_name="risk", recommendation="NEUTRAL")
    decision = orchestrator.decide([tech, sent, risk])
    assert decision.final_recommendation == "LONG"
    assert decision.consensus is True

def test_orchestrator_skip_on_full_disagreement():
    tech = AgentAnalysis(..., recommendation="LONG")
    sent = AgentAnalysis(..., recommendation="SHORT")
    risk = AgentAnalysis(..., recommendation="NEUTRAL")
    decision = orchestrator.decide([tech, sent, risk])
    assert decision.final_recommendation == "SKIP"
```

### `execution/` — Integration Tests

Qué testear:
- Paper executor simula órdenes correctamente.
- `close_all_positions()` cierra todo (kill-switch).
- Slippage simulado está en rango esperado.
- Fees se calculan correctamente.

**Usar**: exchange sandbox/testnet para integration tests (no producción).

---

## Property-Based Testing (hypothesis)

Usar para validar invariantes:

```python
from hypothesis import given, strategies as st

@given(
    risk_pct=st.floats(min_value=0.001, max_value=0.05),
    portfolio_value=st.floats(min_value=100, max_value=1_000_000),
    volatility=st.floats(min_value=0.001, max_value=0.5),
)
def test_position_size_never_exceeds_max(risk_pct, portfolio_value, volatility):
    sizer = PositionSizer(max_risk_pct=risk_pct, max_position_pct=0.15)
    result = sizer.calculate(
        pair="BTC/USDT", side="long",
        entry_price=65000, portfolio_value=portfolio_value,
        current_volatility=volatility,
    )
    assert result.value_usd <= portfolio_value * 0.15

@given(score=st.floats(min_value=-1.0, max_value=1.0))
def test_sentiment_score_always_in_range(score):
    aggregated = aggregator.normalize(score)
    assert -1.0 <= aggregated <= 1.0
```

---

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: trading_bot_test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run mypy .
      - run: uv run pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v4
```

---

## Qué NO testear

- Calidad de predicciones ML/DL (eso es evaluación de modelo, se mide con backtesting).
- Calidad de sentimiento de CryptoBERT (eso es evaluación del modelo preentrenado).
- Calidad del análisis del LLM (demasiado subjetivo, se evalúa con métricas de trading).
- Conectividad real a exchanges en CI (usar mocks/sandbox).
- Performance/latencia (eso se mide con profiling, no con pytest).
