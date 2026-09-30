"""Crash-safe JSON persistence for risk state (drawdown latch and stops)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from trading_bot.state_file import atomic_write_json


class RiskStateStore:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, Any] | None:
        """Return the saved state, or None if missing or unreadable."""
        if not self._path.exists():
            logger.info("No risk state at {} — starting fresh", self._path)
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning(
                "Risk state at {} is unreadable — starting fresh", self._path
            )
            return None
        if not isinstance(data, dict):
            logger.warning("Risk state at {} is malformed — starting fresh", self._path)
            return None
        return data

    def save(self, data: dict[str, Any]) -> None:
        """Atomically replace the state file (temp file + fsync + os.replace)."""
        atomic_write_json(self._path, data)
