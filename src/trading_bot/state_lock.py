"""Single-instance guard: one bot process per state file.

Two bots on the same risk state would each trade the same daily candle and
overwrite each other's paper account, stops and drawdown latch. The lock is
an OS advisory lock (``flock``; ``msvcrt.locking`` on Windows) on a file
next to the state: the kernel releases it when the process dies, so a crash
never leaves a stale lock. It also holds across containers sharing the
state volume on one host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO


class InstanceLockError(RuntimeError):
    """Another process holds the lock."""


class InstanceLock:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._fh: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> None:
        """Take the lock without waiting; raise InstanceLockError if taken."""
        if self._fh is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = self._path.open("a+", encoding="utf-8")
        try:
            _lock(fh)
        except OSError as exc:
            fh.seek(0)
            owner = fh.read().strip() or "unknown"
            fh.close()
            raise InstanceLockError(
                f"{self._path} is locked by another bot (pid {owner}); "
                "stop it before starting a new one"
            ) from exc
        # Diagnostics only: the lock itself is the kernel's.
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            _unlock(self._fh)
        finally:
            self._fh.close()
            self._fh = None


if sys.platform == "win32":
    import msvcrt

    def _lock(fh: IO[str]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh: IO[str]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh: IO[str]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
