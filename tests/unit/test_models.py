"""Tests for core.models domain types."""

from __future__ import annotations

from core.models import Signal, SignalType


def test_signal_type_values() -> None:
    assert SignalType.LONG.value == "long"
    assert SignalType.SHORT.value == "short"


def test_signal_creation(sample_signal: Signal) -> None:
    assert sample_signal.pair == "BTC/USDT"
    assert sample_signal.type == SignalType.LONG
