"""setup_logging: stdlib records reach loguru sinks, secrets never do."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator

import pytest
from loguru import logger

from trading_bot.config.logging import redact_secrets, setup_logging

_TOKEN = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
_URL = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"


@pytest.fixture()
def captured(tmp_path, monkeypatch) -> Iterator[list[str]]:
    """Run setup_logging (file sink under tmp_path) and capture every record."""
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    names = ("httpx", "httpcore", "ccxt", "anthropic", "openai", "trading_bot")
    quiet = {n: logging.getLogger(n).level for n in names}

    setup_logging()
    lines: list[str] = []
    logger.add(lines.append, level="DEBUG", format="{name} | {message}")
    yield lines

    logger.remove()
    logger.configure(patcher=None)
    logger.add(sys.stderr)
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    for name, level in quiet.items():
        logging.getLogger(name).setLevel(level)


def test_stdlib_records_reach_loguru(captured: list[str]) -> None:
    logging.getLogger("trading_bot.execution.live_executor").error("order rejected")
    assert any("order rejected" in line for line in captured)


def test_stdlib_exception_keeps_traceback(captured: list[str]) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("trading_bot.ai.agents.orchestrator").exception("failed")
    joined = "".join(captured)
    assert "failed" in joined
    assert "RuntimeError: boom" in joined


def test_httpx_request_urls_are_not_logged(captured: list[str]) -> None:
    logging.getLogger("httpx").info('HTTP Request: POST %s "HTTP/1.1 200 OK"', _URL)
    assert not any(_TOKEN in line for line in captured)


def test_exchange_request_headers_are_not_logged(captured: list[str]) -> None:
    ccxt_log = logging.getLogger("ccxt.base.exchange")
    ccxt_log.debug("GET url, Request: %s", {"X-MBX-APIKEY": "exchange-secret"})
    assert not any("exchange-secret" in line for line in captured)


def test_own_debug_records_are_kept(captured: list[str]) -> None:
    logging.getLogger("trading_bot.core.bot").debug("pair skipped")
    assert any("pair skipped" in line for line in captured)


def test_telegram_token_is_redacted_from_messages(captured: list[str]) -> None:
    logging.getLogger("some.library").warning("request to %s failed", _URL)
    logger.error("loguru call with {}", _URL)
    joined = "".join(captured)
    assert _TOKEN not in joined
    assert "bot<redacted>/sendMessage" in joined


def test_redact_secrets_leaves_other_text() -> None:
    assert redact_secrets("robot 42 is fine") == "robot 42 is fine"
