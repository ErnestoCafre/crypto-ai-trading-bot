"""CoinGecko market data fetcher for historical OHLC."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from trading_bot.data._http import create_http_client, retry_with_backoff
from trading_bot.data.market.base import BaseFetcher

BASE_URL = "https://api.coingecko.com/api/v3"

# Pair -> CoinGecko coin ID mapping
_PAIR_TO_COIN: dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "BTC/USD": "bitcoin",
    "ETH/USDT": "ethereum",
    "ETH/USD": "ethereum",
    "SOL/USDT": "solana",
    "SOL/USD": "solana",
    "BNB/USDT": "binancecoin",
    "XRP/USDT": "ripple",
    "ADA/USDT": "cardano",
    "DOGE/USDT": "dogecoin",
    "AVAX/USDT": "avalanche-2",
    "DOT/USDT": "polkadot",
    "MATIC/USDT": "matic-network",
    "LINK/USDT": "chainlink",
}


class CoinGeckoFetcher(BaseFetcher):
    """Fetches OHLC data from CoinGecko (free tier, no API key required)."""

    def __init__(self, api_key: str = "") -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
        self._client = create_http_client(base_url=BASE_URL, headers=headers)
        self._rate_delay = 2.0  # 30 calls/min free tier

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def pair_to_coin_id(pair: str) -> str:
        """Convert trading pair to CoinGecko coin ID."""
        coin_id = _PAIR_TO_COIN.get(pair)
        if coin_id is None:
            raise ValueError(f"Unknown pair {pair}. Add mapping to _PAIR_TO_COIN.")
        return coin_id

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    async def _get(self, path: str, params: dict[str, str | int] | None = None) -> dict:  # type: ignore[type-arg]
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        await asyncio.sleep(self._rate_delay)
        return resp.json()  # type: ignore[no-any-return]

    async def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLC via /coins/{id}/market_chart/range endpoint."""
        coin_id = self.pair_to_coin_id(pair)
        until_dt = until or datetime.now(UTC)
        since_ts = int(since.timestamp())
        until_ts = int(until_dt.timestamp())

        data = await self._get(
            f"/coins/{coin_id}/market_chart/range",
            params={
                "vs_currency": "usd",
                "from": since_ts,
                "to": until_ts,
            },
        )

        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        volumes = data.get("total_volumes", [])
        vol_map = {int(v[0]): v[1] for v in volumes} if volumes else {}

        rows = []
        for ts_ms, price in prices:
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": vol_map.get(int(ts_ms), 0.0),
                }
            )

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        if limit:
            df = df.head(limit)

        logger.debug(
            "CoinGecko: {} rows for {} ({} -> {})",
            len(df),
            pair,
            since,
            until_dt,
        )
        return df

    async def fetch_ticker(self, pair: str) -> dict[str, object]:
        """Fetch current price from /simple/price."""
        coin_id = self.pair_to_coin_id(pair)
        data = await self._get(
            "/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
            },
        )
        coin_data = data.get(coin_id, {})
        price = coin_data.get("usd")
        return {
            "bid": price,
            "ask": price,
            "last": price,
            "volume_24h": coin_data.get("usd_24h_vol"),
        }

    async def get_available_pairs(self) -> list[str]:
        """Return pairs that have a known CoinGecko mapping."""
        return list(_PAIR_TO_COIN.keys())
