"""Tests for LabelGenerator."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.ai.prediction.label_generator import LabelGenerator


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000] * 10,
        }
    )


class TestForwardReturn:
    def test_correct_length(self, ohlcv: pd.DataFrame) -> None:
        ret = LabelGenerator.forward_return(ohlcv, horizon=1)
        assert len(ret) == len(ohlcv)
        # Last value should be NaN
        assert pd.isna(ret.iloc[-1])

    def test_values_correct(self, ohlcv: pd.DataFrame) -> None:
        ret = LabelGenerator.forward_return(ohlcv, horizon=1)
        # close[0]=101, close[1]=102 → return = 102/101 - 1
        expected = 102 / 101 - 1
        assert ret.iloc[0] == pytest.approx(expected, abs=1e-6)


class TestDirectionLabel:
    def test_binary_output(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.005, -0.002, 0.0])
        labels = LabelGenerator.direction_label(returns, threshold=0.0)
        assert set(labels.dropna().unique()).issubset({0, 1})

    def test_threshold_applied(self) -> None:
        returns = pd.Series([0.01, 0.005, 0.001, -0.01])
        labels = LabelGenerator.direction_label(returns, threshold=0.005)
        # Only 0.01 > 0.005 → label=1
        assert labels.iloc[0] == 1
        assert labels.iloc[1] == 0  # 0.005 is not > 0.005
        assert labels.iloc[2] == 0


class TestTripleBarrierLabel:
    def test_known_labels(self) -> None:
        """Steadily rising close → profit-take labels."""
        close = [100.0] + [100 + i * 3.0 for i in range(1, 20)]  # +3% per bar
        df = pd.DataFrame(
            {
                "open": close,
                "high": [c + 1 for c in close],
                "low": [c - 1 for c in close],
                "close": close,
                "volume": [1000] * 20,
            }
        )
        labels = LabelGenerator.triple_barrier_label(
            df, profit_take=0.02, stop_loss=0.01, max_holding=5
        )
        assert len(labels) == 20
        # Early bars should hit profit take (price keeps rising)
        assert labels.iloc[0] == 1
