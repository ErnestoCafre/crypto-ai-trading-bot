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


# ---------------------------------------------------------------------------
# CCXTFetcher — bounded retries, timeout, most-recent candles
# ---------------------------------------------------------------------------


def _fetcher(**kwargs: object):
    from trading_bot.data.market.ccxt_fetcher import CCXTFetcher

    with patch("trading_bot.data.market.ccxt_fetcher.ccxt_async") as mock_ccxt:
        import ccxt.async_support as real_ccxt

        # Keep the real exception classes for the except clauses.
        mock_ccxt.NetworkError = real_ccxt.NetworkError
        mock_ccxt.RateLimitExceeded = real_ccxt.RateLimitExceeded
        exchange = AsyncMock()
        exchange.rateLimit = 50
        mock_ccxt.binance.return_value = exchange
        fetcher = CCXTFetcher(exchange_id="binance", sandbox=False, **kwargs)
    return fetcher


def _rows(n: int, start_ms: int = 1704067200000) -> list[list[float]]:
    return [
        [start_ms + i * 3_600_000, 1.0 + i, 2.0 + i, 0.5 + i, 1.5 + i, 10.0]
        for i in range(n)
    ]


class TestFetcherResilience:
    @pytest.mark.asyncio
    async def test_reraises_after_max_retries(self) -> None:
        import ccxt.async_support as ccxt_async

        fetcher = _fetcher(max_retries=3)
        fetcher._exchange.fetch_ohlcv = AsyncMock(
            side_effect=ccxt_async.NetworkError("exchange down")
        )
        sleep = AsyncMock()
        with (
            patch("trading_bot.data.market.ccxt_fetcher.asyncio.sleep", sleep),
            pytest.raises(ccxt_async.NetworkError),
        ):
            await fetcher.fetch_ohlcv(
                "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
            )

        assert fetcher._exchange.fetch_ohlcv.await_count == 4  # 1 + 3 retries
        delays = [c.args[0] for c in sleep.await_args_list]
        assert len(delays) == 3
        # Exponential backoff with jitter: each delay within [cap/2, cap].
        for attempt, delay in enumerate(delays):
            cap = min(30.0, 1.0 * 2**attempt)
            assert cap / 2 <= delay <= cap

    @pytest.mark.asyncio
    async def test_recovers_within_retry_budget(self) -> None:
        import ccxt.async_support as ccxt_async

        fetcher = _fetcher(max_retries=3)
        fetcher._exchange.fetch_ohlcv = AsyncMock(
            side_effect=[ccxt_async.RequestTimeout("slow"), _rows(3)]
        )
        with patch("trading_bot.data.market.ccxt_fetcher.asyncio.sleep", AsyncMock()):
            df = await fetcher.fetch_ohlcv(
                "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
            )
        assert len(df) == 3

    @pytest.mark.asyncio
    async def test_total_timeout(self) -> None:
        import asyncio

        fetcher = _fetcher(timeout_s=0.05)

        async def hang(*args: object, **kwargs: object) -> list[list[float]]:
            await asyncio.sleep(10)
            return []

        fetcher._exchange.fetch_ohlcv = AsyncMock(side_effect=hang)
        with pytest.raises(TimeoutError):
            await fetcher.fetch_ohlcv(
                "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC)
            )

    @pytest.mark.asyncio
    async def test_limit_keeps_most_recent_candles(self) -> None:
        fetcher = _fetcher()
        rows = _rows(8)
        # Pages of `limit` rows: 5 then 3 (short page ends pagination).
        fetcher._exchange.fetch_ohlcv = AsyncMock(side_effect=[rows[:5], rows[5:]])
        with patch("trading_bot.data.market.ccxt_fetcher.asyncio.sleep", AsyncMock()):
            df = await fetcher.fetch_ohlcv(
                "BTC/USDT", "1h", since=datetime(2024, 1, 1, tzinfo=UTC), limit=5
            )
        assert len(df) == 5
        expected_last = pd.Timestamp(rows[-1][0], unit="ms", tz="UTC")
        assert df.index[-1] == expected_last
        assert df.index[0] == pd.Timestamp(rows[3][0], unit="ms", tz="UTC")


class TestCandles:
    @pytest.mark.parametrize(
        ("tf", "seconds"), [("15m", 900), ("1h", 3600), ("4h", 14400), ("1d", 86400)]
    )
    def test_timeframe_to_timedelta(self, tf: str, seconds: int) -> None:
        from trading_bot.data.market.candles import timeframe_to_timedelta

        assert timeframe_to_timedelta(tf).total_seconds() == seconds

    def test_open_candle_is_dropped_but_prices_latest(self) -> None:
        from trading_bot.data.market.candles import split_closed_candles

        index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": [10.0, 11.0, 12.5],
                "volume": 1.0,
            },
            index=index,
        )
        now = datetime(2026, 1, 1, 2, 30, tzinfo=UTC)  # 02:00 candle still open

        closed, price = split_closed_candles(df, "1h", now)

        assert list(closed.index) == list(index[:2])
        assert price == 12.5

    def test_candle_closing_exactly_now_is_closed(self) -> None:
        from trading_bot.data.market.candles import split_closed_candles

        index = pd.date_range("2026-01-01 00:00", periods=2, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=index,
        )
        closed, _ = split_closed_candles(
            df, "15m", datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
        )
        assert len(closed) == 2

    def test_empty_frame(self) -> None:
        from trading_bot.data.market.candles import split_closed_candles

        closed, price = split_closed_candles(pd.DataFrame(), "1h", datetime.now(UTC))
        assert closed.empty and price is None
