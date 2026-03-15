"""Model trainer stub."""

from __future__ import annotations

import pandas as pd

from ai.prediction.base import BasePredictor


class ModelTrainer:
    def train_and_evaluate(
        self,
        model: BasePredictor,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ) -> dict[str, float]:
        raise NotImplementedError
