"""Reward calculator stub."""

from __future__ import annotations


class RewardCalculator:
    def calculate(self, action: object, state: object, next_state: object) -> float:
        raise NotImplementedError
