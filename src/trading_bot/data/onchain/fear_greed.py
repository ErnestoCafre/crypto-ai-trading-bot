"""Fear & Greed Index fetcher (alternative.me API)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from trading_bot.data._http import create_http_client, retry_with_backoff

BASE_URL = "https://api.alternative.me/fng/"


class FearGreedIndex:
    """Fetches the Crypto Fear & Greed Index (free, no API key)."""

    def __init__(self) -> None:
        self._client = create_http_client()

    async def close(self) -> None:
        await self._client.aclose()

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _get(self, params: dict[str, str | int] | None = None) -> dict:  # type: ignore[type-arg]
        resp = await self._client.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def fetch_latest(self) -> dict[str, object]:
        """Fetch the latest Fear & Greed value."""
        data = await self._get(params={"limit": 1})
        items = data.get("data", [])
        if not items:
            return {}

        item = items[0]
        return {
            "value": int(item["value"]),
            "classification": item["value_classification"],
            "timestamp": datetime.fromtimestamp(int(item["timestamp"]), tz=UTC),
        }

    async def fetch_history(self, days: int = 365) -> pd.DataFrame:
        """Fetch historical Fear & Greed data for backtesting."""
        data = await self._get(params={"limit": days})
        items = data.get("data", [])
        if not items:
            return pd.DataFrame(columns=["value", "classification"])

        rows = []
        for item in items:
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(int(item["timestamp"]), tz=UTC),
                    "value": int(item["value"]),
                    "classification": item["value_classification"],
                }
            )

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        logger.debug("Fear & Greed: fetched {} days of history", len(df))
        return df
