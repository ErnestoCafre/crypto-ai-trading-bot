"""Crash-safety of the state file and the decision log."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_bot.core.decision_log import DecisionLog, read_decisions
from trading_bot.state_file import atomic_write_json, read_json_dict


class TestDecisionLog:
    def test_appends_one_record_per_line(self, tmp_path: Path) -> None:
        log = DecisionLog(tmp_path / "d.jsonl")
        log.append({"pair": "BTC/USDT", "n": 1})
        log.append({"pair": "BTC/USDT", "n": 2})
        assert [r["n"] for r in read_decisions(log.path)] == [1, 2]

    def test_torn_last_line_does_not_swallow_the_next_record(
        self, tmp_path: Path
    ) -> None:
        # A crash mid-write left a line without its newline (audit B-2).
        path = tmp_path / "d.jsonl"
        path.write_text('{"n": 1}\n{"n": 2, "pa', encoding="utf-8")
        DecisionLog(path).append({"n": 3})
        assert [r["n"] for r in read_decisions(path)] == [1, 3]

    def test_empty_file_gets_no_blank_line(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.touch()
        DecisionLog(path).append({"n": 1})
        assert path.read_text(encoding="utf-8") == '{"n": 1}\n'


class TestAtomicWrite:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "state.json"
        atomic_write_json(path, {"a": 1})
        assert read_json_dict(path) == {"a": 1}
        assert [p.name for p in path.parent.iterdir()] == ["state.json"]

    @pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-only")
    def test_directory_is_fsynced_after_the_rename(self, tmp_path: Path) -> None:
        # Persist the rename, not only the file contents (audit B-4).
        path = tmp_path / "state.json"
        synced: list[int] = []
        real_fsync = os.fsync

        def spy(fd: int) -> None:
            synced.append(os.fstat(fd).st_ino)
            real_fsync(fd)

        with patch("trading_bot.state_file.os.fsync", side_effect=spy):
            atomic_write_json(path, {"a": 1})
        assert tmp_path.stat().st_ino in synced

    def test_failed_write_leaves_the_old_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        atomic_write_json(path, {"a": 1})
        with pytest.raises(TypeError):
            atomic_write_json(path, {"a": object()})
        assert read_json_dict(path) == {"a": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
