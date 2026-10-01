"""scripts/train_models.py and the purged walk-forward validation (synthetic)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.base import model_path
from trading_bot.ai.prediction.data_pipeline import MLDataPipeline
from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.label_generator import LabelGenerator
from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor
from trading_bot.ai.prediction.validation import (
    ValidationReport,
    candidate_columns,
    evaluate_walk_forward,
    serving_parity_columns,
    walk_forward_splits,
)
from trading_bot.data.storage import OHLCVRepository

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_models.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_models", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_models"] = module
    spec.loader.exec_module(module)
    return module


tm = _load_script()


def _ohlcv(n: int, phi: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Hourly bars whose log-returns follow an AR(1) with coefficient ``phi``."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.004, n)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = phi * ret[i - 1] + eps[i]
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.002, n)) * close
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(500.0, 1500.0, n),
        },
        index=idx,
    )


def _cfg(**kw: object) -> object:
    base = {"folds": 3, "test_bars": 200, "trials": 0, "threshold": 0.0}
    return tm.TrainConfig(**{**base, **kw})


def _report(acc: float, base: float) -> ValidationReport:
    return ValidationReport(
        n_folds=1,
        n_test=100,
        accuracy=acc,
        directional_accuracy=acc,
        auc=0.5,
        log_loss=0.69,
        baseline_accuracy=base,
        baseline_directional_accuracy=base,
        baseline_log_loss=0.69,
        positive_rate=0.5,
    )


class TestSplits:
    @pytest.mark.parametrize(("horizon", "embargo"), [(1, 1), (4, 2), (3, 0)])
    def test_training_labels_end_before_the_test_window(
        self, horizon: int, embargo: int
    ) -> None:
        # Gaps in bar_pos mimic rows dropped by NaN handling.
        bar_pos = np.delete(np.arange(3000), [100, 1500, 2600])
        splits = walk_forward_splits(
            bar_pos, horizon, n_folds=4, test_bars=150, embargo=embargo
        )
        assert len(splits) == 4
        for split in splits:
            test_start = bar_pos[split.test].min()
            # The label of the last training/val row is fully known
            # `embargo` bars before the first test bar.
            assert bar_pos[split.train].max() + horizon + embargo < test_start
            assert bar_pos[split.val].max() + horizon + embargo < test_start
            assert bar_pos[split.train].max() + horizon + embargo < (
                bar_pos[split.val].min()
            )
        tests = [bar_pos[s.test] for s in splits]
        for a, b in zip(tests, tests[1:], strict=False):
            assert a.max() < b.min()  # consecutive, non-overlapping
        assert tests[-1].max() == bar_pos[-1]

    def test_too_little_history_raises(self) -> None:
        with pytest.raises(ValueError, match="training rows"):
            walk_forward_splits(np.arange(1000), 1, n_folds=4, test_bars=200)


class TestNoLookahead:
    def test_fold_results_ignore_data_after_its_test_window(self) -> None:
        """Rewriting every bar after fold 0's test window (and the bars its
        labels read) must not change fold 0's out-of-sample score."""
        horizon = 2
        ohlcv = _ohlcv(2200, phi=-0.3, seed=1)
        pipe = MLDataPipeline(FeatureEngine())
        cols = ["RSI_14", "WILLR_14", "CCI_20_0.015", "BBP_20_2.0"]

        def fold0(frame: pd.DataFrame) -> dict[str, float]:
            x, y = pipe.prepare(frame, horizon, 0.0, feature_columns=cols)
            ret = LabelGenerator.forward_return(frame, horizon).loc[x.index]
            pos = frame.index.get_indexer(x.index)
            splits = walk_forward_splits(pos, horizon, n_folds=3, test_bars=200)
            report = evaluate_walk_forward(x, y, ret, splits[:1], {"random_state": 0})
            return report.folds[0] | {"cutoff": float(pos[splits[0].test].max())}

        before = fold0(ohlcv)
        cutoff = int(before["cutoff"])
        tampered = ohlcv.copy()
        rng = np.random.default_rng(99)
        after = slice(cutoff + horizon + 1, None)
        tampered.iloc[after, :4] = tampered.iloc[after, :4].to_numpy() * rng.uniform(
            0.5, 2.0, (len(tampered) - cutoff - horizon - 1, 1)
        )
        assert fold0(tampered) == before


class TestFeatures:
    def test_scale_free_set_has_no_price_levels(self) -> None:
        features = FeatureEngine().build_features(_ohlcv(400))
        cols = candidate_columns(features, "scale_free")
        assert "RSI_14" in cols and "BBP_20_2.0" in cols
        for level in ("close", "EMA_50", "SMA_200", "OBV", "VWAP_D", "ISA_9"):
            assert level not in cols

    def test_parity_drops_path_dependent_columns(self) -> None:
        ohlcv = _ohlcv(1500)
        cols = ["RSI_14", "SMA_50", "OBV", "EMA_200"]
        kept = serving_parity_columns(ohlcv, cols, window=250, samples=8)
        # OBV is a running sum from the window start; a 200-bar EMA has not
        # forgotten its seed after 250 bars. Both differ in live serving.
        assert kept == ["RSI_14", "SMA_50"]


class TestGate:
    def test_below_min_accuracy_fails(self) -> None:
        ok, reason = tm.passes_gate(_report(0.54, 0.50), 0.55)
        assert not ok and "< 55%" in reason

    def test_must_beat_baseline(self) -> None:
        ok, reason = tm.passes_gate(_report(0.56, 0.57), 0.55)
        assert not ok and "baseline" in reason

    def test_passes(self) -> None:
        assert tm.passes_gate(_report(0.60, 0.51), 0.55)[0]


class TestTrainPair:
    def test_random_walk_is_not_published(self) -> None:
        result = tm.train_pair("BTC/USDT", _ohlcv(2200, phi=0.0, seed=3), _cfg())
        assert not result.passed
        assert 0.4 < result.report.directional_accuracy < 0.6

    def test_predictable_series_passes_and_serves(self, tmp_path: Path) -> None:
        ohlcv = _ohlcv(2200, phi=-0.6, seed=4)
        result = tm.train_pair("ETH/USDT", ohlcv, _cfg())
        assert result.passed, result.reason
        assert result.report.directional_accuracy > 0.6
        assert result.report.auc > 0.6
        assert result.report.log_loss < result.report.baseline_log_loss

        tm.publish(result, str(tmp_path), "1h")
        path = model_path(tmp_path, "ETH/USDT", "1h")
        assert path.exists() and Path(f"{path}.sha256").exists()

        # Load it like the bot and predict from a 250-bar live window.
        live = LightGBMPredictor(pair="ETH/USDT", timeframe="1h")
        live.load(str(path))
        window = ohlcv.iloc[-250:]
        pred = live.predict(window, features=FeatureEngine().build_features(window))
        assert pred.features_used == result.features


class TestCli:
    def test_parse_args(self) -> None:
        args = tm.parse_args(
            ["--task", "direction", "--pairs", "BTC/USDT", "SOL/USDT"]
            + ["--timeframe", "1h", "--years", "2"]
            + ["--trials", "5", "--model-dir", "m", "--publish"]
        )
        assert args.pairs == ["BTC/USDT", "SOL/USDT"]
        assert args.years == 2.0 and args.trials == 5 and args.publish
        cfg = tm.config_from_args(args)
        assert cfg.test_bars == 720  # 30 days of 1h bars
        assert cfg.embargo_bars == cfg.horizon

    def test_meta_label_is_the_default_task(self) -> None:
        args = tm.parse_args([])
        assert args.task == "meta_label"
        assert args.years is None and args.test_days == 180.0
        assert args.weights == "clipped"
        assert (args.risk_pct, args.max_position_pct) == (0.20, 25.0)
        assert tm.history_start(args) == tm.META_HISTORY_START

    def test_direction_defaults(self) -> None:
        args = tm.parse_args(["--task", "direction"])
        assert args.years == 2.0 and args.test_days == 30.0

    def test_rejects_non_positive_years(self) -> None:
        with pytest.raises(SystemExit):
            tm.parse_args(["--years", "0"])

    def test_offline_run_publishes_only_passing_pairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        now = pd.Timestamp.now(tz="UTC").floor("h")
        for pair, phi, seed in (("AAA/USDT", -0.6, 5), ("BBB/USDT", 0.0, 6)):
            df = _ohlcv(2200, phi=phi, seed=seed)
            df.index = pd.date_range(end=now, periods=len(df), freq="h", tz="UTC")
            OHLCVRepository.save_parquet(df, "binance", pair, "1h")

        model_dir = tmp_path / "models"
        report = tmp_path / "report.json"
        argv = ["--task", "direction", "--pairs", "AAA/USDT", "BBB/USDT"]
        argv += ["--years", "1", "--trials", "0"]
        argv += ["--folds", "3", "--test-days", "8", "--offline", "--publish"]
        argv += ["--model-dir", str(model_dir), "--report", str(report)]
        tm.main(argv)

        assert model_path(model_dir, "AAA/USDT", "1h").exists()
        assert not model_path(model_dir, "BBB/USDT", "1h").exists()
        payload = json.loads(report.read_text())
        by_pair = {r["pair"]: r for r in payload["results"]}
        assert by_pair["AAA/USDT"]["published_path"]
        assert by_pair["BBB/USDT"]["published_path"] is None
        assert "directional_accuracy" in by_pair["BBB/USDT"]["report"]


class TestEdgeCases:
    def test_no_training_rows_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="training rows"):
            walk_forward_splits(np.arange(300), 1, n_folds=3, test_bars=100)

    def test_parity_drops_columns_it_could_not_compare(self) -> None:
        class WithEmptyColumn(FeatureEngine):
            def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
                out = super().build_features(ohlcv)
                out["ALL_NAN"] = np.nan
                return out

        kept = serving_parity_columns(
            _ohlcv(600), ["RSI_14", "ALL_NAN"], 250, WithEmptyColumn(), samples=5
        )
        assert kept == ["RSI_14"]

    async def test_short_offline_history_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        df = _ohlcv(500)
        df.index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("h"), periods=500, freq="h"
        )
        OHLCVRepository.save_parquet(df, "binance", "AAA/USDT", "1h")
        warnings: list[str] = []
        monkeypatch.setattr(tm.logger, "warning", lambda *a: warnings.append(a[0]))
        since = datetime.now(UTC) - timedelta(days=365)
        out = await tm.load_history("binance", "AAA/USDT", "1h", since, offline=True)
        assert len(out) == 500
        assert warnings and "short" in warnings[0]
