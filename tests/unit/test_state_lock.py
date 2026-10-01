"""Single-instance guard (state_lock.py)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from trading_bot.state_lock import InstanceLock, InstanceLockError

_HOLDER = textwrap.dedent(
    """
    import sys, time
    from trading_bot.state_lock import InstanceLock
    lock = InstanceLock(sys.argv[1])  # keep a reference: closing frees it
    lock.acquire()
    print("locked", flush=True)
    time.sleep(60)
    """
)


@pytest.fixture()
def other_process(tmp_path):
    """A separate process holding the lock, as a second bot would."""
    path = tmp_path / "risk_state_paper.json.lock"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "locked"
    yield path, proc
    proc.kill()
    proc.wait()


def test_second_instance_is_refused(other_process):
    path, proc = other_process
    lock = InstanceLock(path)
    with pytest.raises(InstanceLockError, match=f"pid {proc.pid}"):
        lock.acquire()
    assert not lock.held


def test_lock_is_freed_when_the_holder_dies(other_process):
    path, proc = other_process
    proc.kill()
    proc.wait()
    lock = InstanceLock(path)
    lock.acquire()  # no stale lock after a crash
    assert lock.held
    lock.release()


def test_acquire_release_reacquire(tmp_path):
    path = tmp_path / "nested" / "state.lock"
    first = InstanceLock(path)
    first.acquire()
    first.acquire()  # idempotent
    assert path.read_text(encoding="utf-8").strip() == str(os.getpid())
    with pytest.raises(InstanceLockError):
        InstanceLock(path).acquire()  # separate open file: still refused
    first.release()
    first.release()  # idempotent
    second = InstanceLock(path)
    second.acquire()
    assert second.held
    second.release()
