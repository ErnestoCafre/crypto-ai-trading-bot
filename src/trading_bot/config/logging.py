"""Logging setup with loguru.

Standard-library loggers (``logging.getLogger``) are routed into loguru by
:class:`InterceptHandler`, so their records reach the same sinks.
"""

from __future__ import annotations

import inspect
import logging
import re
import sys
from typing import TYPE_CHECKING

from loguru import logger

from trading_bot.config.settings import get_settings

if TYPE_CHECKING:
    from loguru import Record

# Telegram puts the bot token in the URL path: .../bot<id>:<secret>/method
_TELEGRAM_TOKEN = re.compile(r"bot\d+:[A-Za-z0-9_-]+")
# Third-party DEBUG output is not routed to the sinks: ccxt logs request
# headers (exchange API key) and signed URLs, the LLM SDKs log full request
# options (prompts). Only trading_bot logs at DEBUG.
_THIRD_PARTY_LEVEL = logging.INFO
# Pinned per library so lowering the root level can never expose them:
# httpx/httpcore log full request URLs (Telegram token) at INFO.
_LIBRARY_LEVELS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "ccxt": logging.INFO,
    "anthropic": logging.INFO,
    "openai": logging.INFO,
}


class InterceptHandler(logging.Handler):
    """Forward standard-library log records to loguru (loguru docs recipe)."""

    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller that issued the record, past the logging module.
        frame, depth = inspect.currentframe(), 0
        while frame is not None and (
            depth == 0 or frame.f_code.co_filename == logging.__file__
        ):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def redact_secrets(text: str) -> str:
    return _TELEGRAM_TOKEN.sub("bot<redacted>", text)


def _redact_record(record: Record) -> None:
    record["message"] = redact_secrets(record["message"])


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()
    logger.configure(patcher=_redact_record)
    # diagnose=False: variable values in tracebacks could expose secrets
    # (e.g. the Telegram URL in TelegramNotifier._send_raw_message).
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        diagnose=False,
    )
    logger.add(
        "var/logs/trading_bot_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        compression="gz",
        diagnose=False,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger().setLevel(_THIRD_PARTY_LEVEL)
    logging.getLogger("trading_bot").setLevel(logging.DEBUG)
    for name, level in _LIBRARY_LEVELS.items():
        logging.getLogger(name).setLevel(level)
