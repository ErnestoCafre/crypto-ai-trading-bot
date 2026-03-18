"""Tests for data.onchain fetchers."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pandas as pd
import pytest


class TestFearGreedIndex:
    """Tests for FearGreedIndex fetcher."""

    def _make_fetcher(self) -> object:
        from data.onchain.fear_greed import FearGreedIndex

        fetcher = FearGreedIndex()
        return fetcher

    @pytest.mark.asyncio
    async def test_fetch_latest(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._get = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "data": [
                    {
                        "value": "72",
                        "value_classification": "Greed",
                        "timestamp": "1704067200",
                    }
                ]
            }
        )

        result = await fetcher.fetch_latest()  # type: ignore[union-attr]

        assert result["value"] == 72
        assert result["classification"] == "Greed"
        assert isinstance(result["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_fetch_latest_empty(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._get = AsyncMock(return_value={"data": []})  # type: ignore[attr-defined]

        result = await fetcher.fetch_latest()  # type: ignore[union-attr]
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_history_returns_dataframe(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._get = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "data": [
                    {
                        "value": "72",
                        "value_classification": "Greed",
                        "timestamp": "1704067200",
                    },
                    {
                        "value": "45",
                        "value_classification": "Fear",
                        "timestamp": "1704153600",
                    },
                    {
                        "value": "55",
                        "value_classification": "Neutral",
                        "timestamp": "1704240000",
                    },
                ]
            }
        )

        df = await fetcher.fetch_history(days=3)  # type: ignore[union-attr]

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "value" in df.columns
        assert "classification" in df.columns
        assert df.index.name == "timestamp"

    @pytest.mark.asyncio
    async def test_fetch_history_empty(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._get = AsyncMock(return_value={"data": []})  # type: ignore[attr-defined]

        df = await fetcher.fetch_history(days=7)  # type: ignore[union-attr]

        assert isinstance(df, pd.DataFrame)
        assert df.empty
