# API Contracts — Interfaces Abstractas

Define las interfaces que mantienen la Clean Architecture. Cada capa externa (exchanges, modelos AI, brokers) se comunica a través de estas interfaces abstractas. Esto permite cambiar implementaciones sin tocar la lógica de negocio.

---

## 1. Data Layer

### `BaseFetcher` — Fetcher de datos de mercado

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd


class BaseFetcher(ABC):
    """Interfaz para obtener datos OHLCV de cualquier fuente."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        pair: str,               # "BTC/USDT"
        timeframe: str,          # "1h", "4h", "1d"
        since: datetime,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retorna DataFrame con columnas:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Index: DatetimeIndex (UTC)
        """
        ...

    @abstractmethod
    async def fetch_ticker(self, pair: str) -> dict:
        """
        Retorna precio actual:
        {'pair': 'BTC/USDT', 'bid': 65000.0, 'ask': 65010.0,
         'last': 65005.0, 'volume_24h': 12345.67, 'timestamp': datetime}
        """
        ...

    @abstractmethod
    async def get_available_pairs(self) -> list[str]:
        """Retorna lista de pares disponibles."""
        ...
```

**Implementaciones**: `CCXTFetcher`, `CoinGeckoFetcher`, `BinanceFetcher`

---

### `BaseNewsFetcher` — Fetcher de noticias

```python
from dataclasses import dataclass


@dataclass
class NewsArticle:
    title: str
    body: str                    # Texto completo o resumen
    source: str                  # "cryptocompare", "rss", etc.
    url: str
    published_at: datetime
    categories: list[str]        # ["BTC", "regulation", "ETF"]
    related_pairs: list[str]     # ["BTC/USDT", "ETH/USDT"]


class BaseNewsFetcher(ABC):
    """Interfaz para obtener noticias crypto."""

    @abstractmethod
    async def fetch_latest(
        self,
        limit: int = 50,
        categories: Optional[list[str]] = None,
    ) -> list[NewsArticle]:
        """Retorna últimas noticias ordenadas por fecha desc."""
        ...

    @abstractmethod
    async def fetch_by_pair(
        self,
        pair: str,
        since: datetime,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Retorna noticias relacionadas con un par específico."""
        ...
```

**Implementaciones**: `CryptoCompareNewsFetcher`, `RSSFetcher`

---

## 2. Strategies

### `BaseStrategy` — Estrategia de trading

```python
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    NEUTRAL = "neutral"


@dataclass
class Signal:
    type: SignalType
    pair: str
    confidence: float            # 0.0 a 1.0
    strength: float              # -1.0 (fuerte short) a +1.0 (fuerte long)
    source: str                  # "technical", "ai_hybrid", "rl_agent"
    metadata: dict               # Info adicional (indicadores, scores AI, etc.)
    timestamp: datetime


@dataclass
class AISignals:
    """Señales provenientes de módulos AI (opcional)."""
    sentiment_score: Optional[float] = None    # -1.0 a +1.0
    prediction_return: Optional[float] = None  # Retorno predicho
    prediction_confidence: Optional[float] = None
    llm_recommendation: Optional[str] = None   # "LONG", "SHORT", "NEUTRAL"
    llm_confidence: Optional[float] = None


class BaseStrategy(ABC):
    """Interfaz para todas las estrategias de trading."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre único de la estrategia."""
        ...

    @abstractmethod
    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        ai_signals: Optional[AISignals] = None,
    ) -> Optional[Signal]:
        """
        Genera señal de trading basada en datos OHLCV y señales AI opcionales.
        Retorna None si no hay señal.
        """
        ...

    @abstractmethod
    def get_required_lookback(self) -> int:
        """Número mínimo de barras históricas necesarias."""
        ...
```

**Implementaciones**: `TrendFollowingStrategy`, `MeanReversionStrategy`, `LightGBMStrategy`, `AIHybridStrategy`

---

### `SignalCombiner` — Combinador de señales

```python
@dataclass
class CombinedSignal:
    type: SignalType
    pair: str
    combined_confidence: float   # Confianza ponderada final
    technical_signal: Optional[Signal]
    ai_signals: Optional[AISignals]
    weights_used: dict           # {"technical": 0.4, "sentiment": 0.2, ...}
    should_trade: bool           # True si confianza >= threshold
    recommended_sizing_pct: float  # Sizing sugerido (0.25, 0.5, 0.75, 1.0)
    reasoning: str               # Explicación de la decisión
    timestamp: datetime


class SignalCombiner:
    """Combina señales técnicas + AI con pesos configurables."""

    def combine(
        self,
        technical_signal: Optional[Signal],
        ai_signals: Optional[AISignals],
    ) -> CombinedSignal:
        """
        Aplica pesos configurados y reglas de negocio:
        - Si confianza < MIN_CONFIDENCE_THRESHOLD → should_trade = False
        - Si señales contradictorias → reducir sizing
        - Si todos alineados → sizing completo
        """
        ...
```

---

## 3. AI Module

### `BaseSentimentAnalyzer` — Análisis de sentimiento

```python
@dataclass
class SentimentResult:
    score: float                 # -1.0 (muy negativo) a +1.0 (muy positivo)
    confidence: float            # 0.0 a 1.0
    label: str                   # "positive", "negative", "neutral"
    model_used: str              # "CryptoBERT", "FinBERT", "LLM"
    analyzed_text: str           # Texto que se analizó (truncado)


class BaseSentimentAnalyzer(ABC):
    """Interfaz para modelos de análisis de sentimiento."""

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analiza el sentimiento de un texto individual."""
        ...

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analiza sentimiento de múltiples textos (más eficiente)."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Retorna True si el modelo está cargado en memoria."""
        ...
```

**Implementaciones**: `CryptoBERTAnalyzer`, `FinBERTAnalyzer`, `LLMSentimentAnalyzer`

---

### `SentimentAggregator` — Agregador de sentimiento

```python
@dataclass
class AggregatedSentiment:
    pair: str
    overall_score: float         # Score ponderado final (-1.0 a +1.0)
    news_score: float            # Score solo de noticias
    social_score: float          # Score solo de social media
    fear_greed_score: float      # Fear & Greed normalizado a -1/+1
    num_sources_analyzed: int
    time_window: str             # "4h", "24h", "7d"
    timestamp: datetime


class SentimentAggregator:
    """Agrega sentimiento de múltiples fuentes con pesos configurables."""

    def aggregate(
        self,
        pair: str,
        news_results: list[SentimentResult],
        social_results: list[SentimentResult],
        fear_greed_value: Optional[int],  # 0-100
        weights: Optional[dict] = None,   # {"news": 0.4, "social": 0.3, "fear_greed": 0.3}
    ) -> AggregatedSentiment:
        ...
```

---

### `BasePredictor` — Predicción de precio

```python
@dataclass
class Prediction:
    pair: str
    timeframe: str
    predicted_return: float      # Retorno esperado (e.g., 0.02 = +2%)
    predicted_direction: str     # "up", "down", "flat"
    confidence: float            # 0.0 a 1.0
    model_type: str              # "lightgbm", "gru", "cnn_lstm"
    model_version: str
    features_used: list[str]     # Lista de features usados
    timestamp: datetime


class BasePredictor(ABC):
    """Interfaz para modelos de predicción de precio."""

    @abstractmethod
    def predict(
        self,
        ohlcv: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Prediction:
        """Genera predicción para la siguiente barra."""
        ...

    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hyperparameters: Optional[dict] = None,
    ) -> dict:
        """
        Entrena el modelo. Retorna métricas:
        {'accuracy': 0.58, 'sharpe': 1.2, 'mse': 0.001, ...}
        """
        ...

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Carga modelo serializado desde disco."""
        ...

    @abstractmethod
    def save(self, model_path: str) -> None:
        """Serializa modelo a disco."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Retorna True si el modelo está listo para inference."""
        ...
```

**Implementaciones**: `LightGBMPredictor`, `GRUPredictor`, `CNNLSTMPredictor`

---

### `BaseAgent` — Agente LLM

```python
@dataclass
class AgentAnalysis:
    agent_name: str              # "technical_analyst", "sentiment_analyst", "risk_evaluator"
    recommendation: str          # "LONG", "SHORT", "NEUTRAL", "SKIP"
    confidence: float            # 0.0 a 1.0
    reasoning: str               # Chain-of-thought del agente
    key_factors: list[str]       # Factores principales de la decisión
    raw_response: str            # Respuesta raw del LLM
    tokens_used: int
    latency_ms: int


class BaseAgent(ABC):
    """Interfaz para agentes LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def analyze(self, context: dict) -> AgentAnalysis:
        """
        Ejecuta el análisis del agente.
        `context` contiene los datos relevantes para el agente.
        """
        ...
```

**Implementaciones**: `TechnicalAnalystAgent`, `SentimentAnalystAgent`, `RiskEvaluatorAgent`

---

### `Orchestrator` — Orquestador multi-agente

```python
@dataclass
class OrchestratorDecision:
    pair: str
    final_recommendation: str    # "LONG", "SHORT", "NEUTRAL", "SKIP"
    confidence: float
    agent_analyses: list[AgentAnalysis]
    consensus: bool              # True si mayoría de agentes coinciden
    reasoning: str               # Explicación de la decisión final
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: int
    timestamp: datetime


class Orchestrator:
    """Coordina múltiples agentes LLM y toma decisión final."""

    async def analyze(
        self,
        pair: str,
        ohlcv: pd.DataFrame,
        indicators: dict,
        sentiment: Optional[AggregatedSentiment],
        portfolio_state: dict,
    ) -> OrchestratorDecision:
        """
        Ejecuta todos los agentes en paralelo, aplica reglas de consenso.
        Si 2/3 agentes coinciden → señal válida.
        Si desacuerdo → SKIP o reducir sizing.
        """
        ...
```

---

## 4. Risk Management

### `PositionSizer` — Cálculo de tamaño de posición

```python
@dataclass
class PositionSize:
    pair: str
    quantity: float              # Cantidad en base currency
    value_usd: float             # Valor en USD
    risk_pct: float              # % del portfolio en riesgo
    sizing_method: str           # "kelly", "volatility_target", "fixed_pct"
    ai_adjustment: float         # Multiplicador AI (0.25, 0.5, 0.75, 1.0)
    stop_loss_price: float
    stop_loss_pct: float


class PositionSizer:
    """Calcula tamaño de posición con ajuste de confianza AI."""

    def calculate(
        self,
        pair: str,
        side: str,                   # "long" o "short"
        entry_price: float,
        portfolio_value: float,
        current_volatility: float,   # ATR actual
        ai_confidence: Optional[float] = None,  # 0.0 a 1.0
    ) -> PositionSize:
        ...
```

---

## 5. Execution

### `BaseExecutor` — Ejecución de órdenes

```python
@dataclass
class OrderResult:
    order_id: str
    exchange_order_id: Optional[str]
    pair: str
    side: str
    filled_price: float
    filled_quantity: float
    fee: float
    slippage_pct: float
    status: str                  # "filled", "partially_filled", "failed"
    timestamp: datetime


class BaseExecutor(ABC):
    """Interfaz para ejecutar órdenes en exchange."""

    @abstractmethod
    async def place_order(
        self,
        pair: str,
        side: str,               # "buy" o "sell"
        order_type: str,         # "market", "limit"
        quantity: float,
        price: Optional[float] = None,
    ) -> OrderResult:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    async def get_open_orders(self, pair: Optional[str] = None) -> list[dict]:
        ...

    @abstractmethod
    async def get_balance(self) -> dict:
        """Retorna {'USDT': 10000.0, 'BTC': 0.5, ...}"""
        ...

    @abstractmethod
    async def close_all_positions(self) -> list[OrderResult]:
        """Kill-switch: cierra todas las posiciones abiertas."""
        ...
```

**Implementaciones**: `PaperExecutor` (sandbox), `LiveExecutor` (CCXT real)

---

## 6. Notifications

### `BaseNotifier` — Notificaciones

```python
class BaseNotifier(ABC):
    """Interfaz para enviar notificaciones."""

    @abstractmethod
    async def send_alert(self, message: str, level: str = "info") -> bool:
        """level: 'info', 'warning', 'error', 'critical'"""
        ...

    @abstractmethod
    async def send_trade_notification(self, trade: dict) -> bool:
        ...

    @abstractmethod
    async def send_ai_report(self, analysis: dict) -> bool:
        ...
```

**Implementaciones**: `TelegramNotifier`

---

## Flujo de Datos entre Interfaces

```
BaseFetcher.fetch_ohlcv()
    │
    ├──► BaseStrategy.generate_signal(ohlcv, ai_signals?)
    │        │
    │        │   ┌── BaseSentimentAnalyzer.analyze_batch()
    │        │   │      └──► SentimentAggregator.aggregate()
    │        │   │
    │        ├── │── BasePredictor.predict()
    │        │   │
    │        │   └── Orchestrator.analyze()
    │        │          ├── BaseAgent.analyze() × 3
    │        │          └── consensus logic
    │        │
    │        └──► Signal
    │
    ├──► SignalCombiner.combine(signal, ai_signals)
    │        │
    │        └──► CombinedSignal (should_trade, confidence, sizing)
    │
    ├──► PositionSizer.calculate(confidence)
    │        │
    │        └──► PositionSize (quantity, stop_loss)
    │
    └──► BaseExecutor.place_order()
             │
             ├──► OrderResult
             └──► BaseNotifier.send_trade_notification()
```
