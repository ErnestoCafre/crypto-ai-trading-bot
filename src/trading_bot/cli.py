"""Command-line entry point for the trading bot."""

from __future__ import annotations

import asyncio
import sys

from trading_bot.config import get_settings
from trading_bot.config.logging import setup_logging


def main() -> None:
    setup_logging()

    from loguru import logger

    settings = get_settings()

    logger.info("Trading Bot v0.1.0")
    logger.info("Mode: {}", settings.BOT_MODE)
    logger.info("Exchange: {}", settings.EXCHANGE_ID)
    logger.info("Pairs: {}", settings.TRADING_PAIRS)

    ai_flags = {
        "sentiment": settings.USE_SENTIMENT,
        "llm_agent": settings.USE_LLM_AGENT,
        "dl_prediction": settings.USE_DL_PREDICTION,
        "rl_agent": settings.USE_RL_AGENT,
    }
    active_ai = [k for k, v in ai_flags.items() if v]
    logger.info("AI modules: {}", active_ai if active_ai else "none")

    from trading_bot.core.bot import TradingOrchestrator

    orchestrator = TradingOrchestrator(settings)
    asyncio.run(orchestrator.start())


def run() -> None:
    """Console-script wrapper: exits cleanly on Ctrl-C."""
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
