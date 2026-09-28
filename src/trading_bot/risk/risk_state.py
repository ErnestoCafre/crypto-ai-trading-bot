"""Crash-safe JSON persistence for risk state (drawdown latch and stops)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
