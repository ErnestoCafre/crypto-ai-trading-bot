"""Tests for data.market fetchers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# CCXTFetcher
# ---------------------------------------------------------------------------


class TestCCXTFetcher:
    """Tests for CCXTFetcher."""

    def _make_fetcher(self) -> object:
        """Create a CCXTFetcher with a mocked exchange."""
        from trading_bot.data.market.ccxt_fetcher import CCXTFetcher

        with patch("trading_bot.data.market.ccxt_fetcher.ccxt_async") as mock_ccxt:
            mock_exchange = AsyncMock()
            mock_exchange.rateLimit = 1200
            mock_exchange.markets = {"BTC/USDT": {"active": True}}
            mock_ccxt.binance.return_value = mock_exchange
            fetcher = CCXTFetcher(exchange_id="binance", sandbox=False)
        return fetcher

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_converts_to_dataframe(self) -> None:
        fetcher = self._make_fetcher()
        raw = [
            [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0],
            [1704070800000, 42200.0, 42700.0, 42100.0, 42600.0, 120.0],
        ]
        fetcher._exchange.fetch_ohlcv = AsyncMock(return_value=raw)

        df = await fetcher.fetch_ohlcv(
            "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df.index.name == "timestamp"

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_pagination(self) -> None:
        fetcher = self._make_fetcher()
        batch1 = [
            [1704067200000 + i * 3600000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0]
            for i in range(1000)
        ]
        batch2 = [
            [
                1704067200000 + (1000 + i) * 3600000,
                42000.0,
                42500.0,
                41800.0,
                42200.0,
                100.0,
            ]
            for i in range(500)
        ]
        fetcher._exchange.fetch_ohlcv = AsyncMock(side_effect=[batch1, batch2])

        df = await fetcher.fetch_ohlcv(
            "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
        )

        assert len(df) == 1500
        assert fetcher._exchange.fetch_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_respects_until(self) -> None:
        fetcher = self._make_fetcher()
        raw = [
            [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0],
            [1704070800000, 42200.0, 42700.0, 42100.0, 42600.0, 120.0],
            [1704074400000, 42600.0, 43000.0, 42500.0, 42900.0, 110.0],
        ]
        fetcher._exchange.fetch_ohlcv = AsyncMock(return_value=raw)

        until = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
        df = await fetcher.fetch_ohlcv(
            "BTC/USDT",
            "1h",
            since=datetime(2024, 1, 1, tzinfo=UTC),
            until=until,
        )

        # Only first candle should be included (ts < until_ms)
        assert len(df) == 1

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_empty_response(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._exchange.fetch_ohlcv = AsyncMock(return_value=[])

        df = await fetcher.fetch_ohlcv(
            "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
        )

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_rate_limit_retry(self) -> None:
        import ccxt.async_support as ccxt_async

        fetcher = self._make_fetcher()
        raw = [[1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0]]
        fetcher._exchange.fetch_ohlcv = AsyncMock(
            side_effect=[ccxt_async.RateLimitExceeded("rate limit"), raw]
        )

        with patch(
            "trading_bot.data.market.ccxt_fetcher.asyncio.sleep", new_callable=AsyncMock
        ):
            df = await fetcher.fetch_ohlcv(
                "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
            )

        assert len(df) == 1
        assert fetcher._exchange.fetch_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_ticker(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._exchange.fetch_ticker = AsyncMock(
            return_value={
                "bid": 42000.0,
                "ask": 42050.0,
                "last": 42025.0,
                "quoteVolume": 1_000_000.0,
                "timestamp": 1704067200000,
            }
        )

        ticker = await fetcher.fetch_ticker("BTC/USDT")

        assert ticker["bid"] == 42000.0
        assert ticker["ask"] == 42050.0
        assert ticker["last"] == 42025.0
        assert ticker["volume_24h"] == 1_000_000.0

    @pytest.mark.asyncio
    async def test_get_available_pairs(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._exchange.markets = {
            "BTC/USDT": {"active": True},
            "ETH/USDT": {"active": True},
            "OLD/USDT": {"active": False},
        }
        fetcher._exchange.load_markets = AsyncMock()

        pairs = await fetcher.get_available_pairs()

        assert "BTC/USDT" in pairs
        assert "ETH/USDT" in pairs
        assert "OLD/USDT" not in pairs


# ---------------------------------------------------------------------------
# CoinGeckoFetcher
# ---------------------------------------------------------------------------


class TestCoinGeckoFetcher:
    """Tests for CoinGeckoFetcher."""

    def test_pair_to_coin_id_mapping(self) -> None:
        from trading_bot.data.market.coingecko import CoinGeckoFetcher

        assert CoinGeckoFetcher.pair_to_coin_id("BTC/USDT") == "bitcoin"
        assert CoinGeckoFetcher.pair_to_coin_id("ETH/USDT") == "ethereum"
        assert CoinGeckoFetcher.pair_to_coin_id("SOL/USDT") == "solana"

    def test_pair_to_coin_id_unknown_raises(self) -> None:
        from trading_bot.data.market.coingecko import CoinGeckoFetcher

        with pytest.raises(ValueError, match="Unknown pair"):
            CoinGeckoFetcher.pair_to_coin_id("UNKNOWN/USDT")

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_parses_response(self) -> None:
        from trading_bot.data.market.coingecko import CoinGeckoFetcher

        fetcher = CoinGeckoFetcher()
        mock_response = {
            "prices": [
                [1704067200000, 42000.0],
                [1704070800000, 42200.0],
            ],
            "total_volumes": [
                [1704067200000, 1000000.0],
                [1704070800000, 1200000.0],
            ],
        }
        fetcher._get = AsyncMock(return_value=mock_response)

        df = await fetcher.fetch_ohlcv(
            "BTC/USDT",
            "1d",
            since=datetime(2024, 1, 1, tzinfo=UTC),
        )

        assert len(df) == 2
        assert df.iloc[0]["close"] == 42000.0
        assert df.iloc[0]["volume"] == 1000000.0
        await fetcher.close()


# ---------------------------------------------------------------------------
# BinanceFetcher
# ---------------------------------------------------------------------------


class TestBinanceFetcher:
    """Tests for BinanceFetcher — delegates to CCXTFetcher."""

    @pytest.mark.asyncio
    async def test_delegates_to_ccxt(self) -> None:
        from trading_bot.data.market.binance import BinanceFetcher

        with patch("trading_bot.data.market.ccxt_fetcher.ccxt_async") as mock_ccxt:
            mock_exchange = AsyncMock()
            mock_exchange.rateLimit = 1200
            mock_exchange.markets = {"BTC/USDT": {"active": True}}
            mock_ccxt.binance.return_value = mock_exchange
            fetcher = BinanceFetcher(sandbox=False)

        raw = [[1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0]]
        fetcher._fetcher._exchange.fetch_ohlcv = AsyncMock(return_value=raw)

        df = await fetcher.fetch_ohlcv(
            "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
        )

        assert len(df) == 1
        assert fetcher._fetcher._exchange_id == "binance"
