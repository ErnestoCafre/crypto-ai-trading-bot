"""Tests for AI RL stubs and module structure."""

from __future__ import annotations

from trading_bot.ai.rl.agent import RLAgent
from trading_bot.ai.rl.environment import TradingEnvironment


def test_rl_components_instantiation() -> None:
    agent = RLAgent()
    env = TradingEnvironment()
    assert agent is not None
    assert env is not None
