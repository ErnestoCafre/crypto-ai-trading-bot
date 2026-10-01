"""Append-only JSONL log of donchian_ensemble rebalance decisions.

One line per pair each time the bot computes a daily target: what the
strategy asked for, what AI and risk let through, and the fill. The paper
parity check (``scripts/paper_parity.py``) replays the same candles through
the backtest engine and compares against it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger


class DecisionLog:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: Mapping[str, Any]) -> None:
        """Write one record as a line and fsync it."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def read_decisions(path: Path | str) -> list[dict[str, Any]]:
    """All records in file order; malformed lines are skipped with a warning."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                logger.warning("{}:{}: malformed decision line skipped", path, number)
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                logger.warning("{}:{}: decision is not an object", path, number)
    return records
