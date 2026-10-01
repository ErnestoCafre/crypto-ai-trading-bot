"""Unit tests for the notifications module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from trading_bot.notifications.telegram_notifier import TelegramNotifier


@pytest.fixture
def unconfigured_notifier() -> TelegramNotifier:
    return TelegramNotifier(bot_token="", chat_id="")


@pytest.fixture
def configured_notifier() -> TelegramNotifier:
    return TelegramNotifier(bot_token="test_token_123", chat_id="chat_456")


@pytest.mark.asyncio
async def test_unconfigured_notifier_returns_false(
    unconfigured_notifier: TelegramNotifier,
) -> None:
    assert not unconfigured_notifier.is_configured
    assert await unconfigured_notifier.send_alert("Test alert") is False
    assert (
        await unconfigured_notifier.send_trade_notification(
            {"side": "buy", "pair": "BTC/USDT"}
        )
        is False
    )
    assert await unconfigured_notifier.send_ai_report({"pair": "BTC/USDT"}) is False


@pytest.mark.asyncio
async def test_send_alert_success(configured_notifier: TelegramNotifier) -> None:
    assert configured_notifier.is_configured

    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200

    with patch(
        "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
    ) as mock_post:
        result = await configured_notifier.send_alert(
            "System running normally", level="info"
        )
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "System running normally" in call_kwargs["json"]["text"]
        assert call_kwargs["json"]["chat_id"] == "chat_456"


@pytest.mark.asyncio
async def test_send_trade_notification_success(
    configured_notifier: TelegramNotifier,
) -> None:
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200

    trade_data = {
        "side": "buy",
        "pair": "BTC/USDT",
        "price": 65000.0,
        "quantity": 0.5,
        "strategy": "trend_following",
        "status": "filled",
    }

    with patch(
        "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
    ) as mock_post:
        result = await configured_notifier.send_trade_notification(trade_data)
        assert result is True
        call_kwargs = mock_post.call_args[1]
        text = call_kwargs["json"]["text"]
        assert "BUY BTC/USDT" in text
        assert "65,000.00" in text


@pytest.mark.asyncio
async def test_send_ai_report_success(configured_notifier: TelegramNotifier) -> None:
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200

    report_data = {
        "pair": "ETH/USDT",
        "recommendation": "LONG",
        "confidence": 0.85,
        "reasoning": "Strong momentum confirmed by volume.",
    }

    with patch(
        "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
    ) as mock_post:
        result = await configured_notifier.send_ai_report(report_data)
        assert result is True
        call_kwargs = mock_post.call_args[1]
        text = call_kwargs["json"]["text"]
        assert "ETH/USDT" in text
        assert "85.0%" in text
        assert "Strong momentum" in text


@pytest.mark.asyncio
async def test_send_notification_network_failure(
    configured_notifier: TelegramNotifier,
) -> None:
    with patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = await configured_notifier.send_alert("Test alert")
        assert result is False


@pytest.mark.asyncio
async def test_dynamic_text_is_html_escaped(
    configured_notifier: TelegramNotifier,
) -> None:
    """parse_mode=HTML: a raw < or & would make Telegram reject the message."""
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.status_code = 200
    with patch(
        "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp
    ) as mock_post:
        await configured_notifier.send_alert("drawdown <15% & <class 'X'>")
        await configured_notifier.send_ai_report(
            {"pair": "BTC/USDT", "reasoning": "RSI < 30 & <b>bold</b>"}
        )
    alert, report = (c[1]["json"]["text"] for c in mock_post.call_args_list)
    assert "drawdown &lt;15% &amp; &lt;class 'X'&gt;" in alert
    assert alert.startswith("<b>")  # the notifier's own markup is kept
    assert "RSI &lt; 30 &amp; &lt;b&gt;bold&lt;/b&gt;" in report
