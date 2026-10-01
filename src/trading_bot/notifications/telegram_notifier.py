"""Telegram notifier implementation using Telegram Bot HTTP API."""

from __future__ import annotations

import html
import logging
from typing import Any

import httpx

from trading_bot.config import get_settings
from trading_bot.notifications.base import BaseNotifier

logger = logging.getLogger(__name__)


class TelegramNotifier(BaseNotifier):
    """Sends trading signals, execution alerts, and AI reports to a Telegram chat."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        settings = get_settings()
        self._bot_token = (
            bot_token if bot_token is not None else settings.TELEGRAM_BOT_TOKEN
        )
        self._chat_id = chat_id if chat_id is not None else settings.TELEGRAM_CHAT_ID
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    async def _send_raw_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.is_configured:
            logger.debug(
                "TelegramNotifier: bot_token or chat_id not configured; "
                "skipping notification"
            )
            return False

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    logger.warning(
                        "Telegram notification failed (status %s): %s",
                        response.status_code,
                        response.text,
                    )
                    return False
                return True
        except Exception:
            logger.exception("Failed to send Telegram notification")
            return False

    async def send_alert(self, message: str, level: str = "info") -> bool:
        """Send a formatted system or trading alert."""
        icon_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
            "kill_switch": "🛑",
            "success": "✅",
        }
        icon = icon_map.get(level.lower(), "📢")
        header = f"<b>{icon} Trading Bot Alert [{_esc(level.upper())}]</b>"
        text = f"{header}\n\n{_esc(message)}"
        return await self._send_raw_message(text)

    async def send_trade_notification(self, trade: dict[str, Any]) -> bool:
        """Send a notification for an executed trade or order."""
        side = str(trade.get("side", "")).upper()
        pair = str(trade.get("pair", "UNKNOWN"))
        price = trade.get("price", trade.get("filled_price", 0.0))
        qty = trade.get("quantity", trade.get("filled_quantity", 0.0))
        strategy = trade.get("strategy", "N/A")
        status = trade.get("status", "filled")

        icon = "🟢" if side == "BUY" else "🔴"
        price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else str(price)
        text = (
            f"<b>{icon} Trade Executed: {_esc(side)} {_esc(pair)}</b>\n\n"
            f"• <b>Status:</b> {_esc(status)}\n"
            f"• <b>Quantity:</b> {_esc(qty)}\n"
            f"• <b>Price:</b> {_esc(price_str)}\n"
            f"• <b>Strategy:</b> {_esc(strategy)}"
        )
        return await self._send_raw_message(text)

    async def send_ai_report(self, analysis: dict[str, Any]) -> bool:
        """Send an AI market analysis summary."""
        pair = str(analysis.get("pair", "GLOBAL"))
        recommendation = str(analysis.get("recommendation", "NEUTRAL")).upper()
        confidence = analysis.get("confidence", 0.0)
        reasoning = str(analysis.get("reasoning", "No details available."))

        conf_str = (
            f"{float(confidence) * 100:.1f}%"
            if isinstance(confidence, (int, float))
            else str(confidence)
        )
        text = (
            f"<b>🤖 AI Market Analysis: {_esc(pair)}</b>\n\n"
            f"• <b>Recommendation:</b> {_esc(recommendation)}\n"
            f"• <b>Confidence:</b> {_esc(conf_str)}\n"
            f"• <b>Reasoning:</b> {_esc(reasoning)}"
        )
        return await self._send_raw_message(text)


def _esc(value: object) -> str:
    """Escape for parse_mode=HTML: a stray < or & (exception text, LLM output)
    makes Telegram reject the whole message with 400."""
    return html.escape(str(value), quote=False)
