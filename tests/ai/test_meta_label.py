"""Meta-labeling: events, triple-barrier labels, features, model, validation."""

from __future__ import annotations

import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.ai.prediction.base import PredictionUnavailableError, write_artifact
from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor
from trading_bot.ai.prediction.meta_events import (
    average_uniqueness,
    entry_events,
    label_events,
    technical_signals,
)
from trading_bot.ai.prediction.meta_features import PRICE_FEATURES, SIGNAL_FEATURES
from trading_bot.ai.prediction.meta_label import (
    MetaLabelPredictor,
    MetaModel,
    choose_threshold,
    fit_meta,
    prediction_fields,
    sample_weights,
    sequential_return,
)
from trading_bot.ai.prediction.meta_training import (
    MetaConfig,
    build_predictor,
    meta_parity_columns,
    run_meta_label,
)
from trading_bot.ai.prediction.meta_validation import (
    BacktestComparison,
    EventMetrics,
    FoldPrediction,
    build_pair_data,
    cpcv_splits,
    event_metrics,
    meta_gate,
    pool_events,
    walk_forward_time_splits,
)
from trading_bot.backtest.pipeline import build_live_engine


def trending_ohlcv(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Hourly bars alternating up/down trends so trend entries happen."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0015, 0.002], size=n // 150 + 1), 150)[:n]
    ret = drift + rng.normal(0, 0.006, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    wick = np.abs(rng.normal(0, 0.004, n)) * close
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": rng.uniform(500, 1500, n),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return trending_ohlcv()


@pytest.fixture(scope="module")
def signals(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return technical_signals(ohlcv, "BTC/USDT")


def entry_window(
    ohlcv: pd.DataFrame, signals: pd.DataFrame, k: int = -1
) -> pd.DataFrame:
    """The live 250-bar window ending at the ``k``-th entry event."""
    end = int(entry_events(signals)[k])
    return ohlcv.iloc[end - 249 : end + 1]


def non_entry_window(ohlcv: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    events = set(entry_events(signals).tolist())
    end = next(i for i in range(len(ohlcv) - 1, 250, -1) if i not in events)
    return ohlcv.iloc[end - 249 : end + 1]


class TestLabels:
    def test_labels_reproduce_the_backtest_trades(
        self, ohlcv: pd.DataFrame, signals: pd.DataFrame
    ) -> None:
        """Every unfiltered backtest trade is an event labelled with its PnL."""
        events = entry_events(signals)
        labels = label_events(ohlcv, signals, events, max_bars=10_000)
        result = build_live_engine("BTC/USDT").run(ohlcv, "BTC/USDT")
        trades = [t for t in result.trades if t.exit_reason != "end_of_data"]
        assert len(trades) >= 10
        by_entry = labels.set_index("entry")
        for t in trades:
            row = by_entry.loc[ohlcv.index.get_loc(pd.Timestamp(t.entry_time))]
            assert row["ret"] * 100 == pytest.approx(t.pnl_pct, abs=1e-9)
            assert row["exit_reason"] == t.exit_reason
            assert ohlcv.index[row["t1"]] == pd.Timestamp(t.exit_time)

    def _frame(self, opens: list[float], lows: list[float]) -> pd.DataFrame:
        n = len(opens)
        return pd.DataFrame(
            {
                "open": opens,
                "high": [o + 1 for o in opens],
                "low": lows,
                "close": opens,
                "volume": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )

    @staticmethod
    def _signals(types: list[str | None]) -> pd.DataFrame:
        return pd.DataFrame({"type": types})

    def test_stop_wins_over_exit_signal_on_the_same_bar(self) -> None:
        opens = [100.0] * 30
        lows = [99.0] * 30
        lows[22] = 50.0  # stop touched on bar 22, which also signals CLOSE
        df = self._frame(opens, lows)
        types: list[str | None] = [None] * 30
        types[22] = "close"
        out = label_events(df, self._signals(types).set_index(df.index), [20])
        assert out.iloc[0]["exit_reason"] == "stop_loss"
        assert out.iloc[0]["t1"] == 22
        assert out.iloc[0]["label"] == 0

    def test_gap_through_stop_fills_at_open_and_costs_count(self) -> None:
        opens = [100.0] * 30
        opens[25] = 60.0  # opens below any 2xATR stop
        lows = [99.5] * 30
        lows[25] = 59.0
        df = self._frame(opens, lows)
        out = label_events(df, self._signals([None] * 30).set_index(df.index), [20])
        fill = 100.0 * 1.0005
        exit_fill = 60.0 * 0.9995
        expected = (exit_fill - fill - 0.001 * (fill + exit_fill)) / fill
        assert out.iloc[0]["ret"] == pytest.approx(expected)

    def test_flat_trade_is_a_loss_after_costs_and_max_bars_caps(self) -> None:
        df = self._frame([100.0] * 40, [99.9] * 40)
        out = label_events(
            df, self._signals([None] * 40).set_index(df.index), [5], max_bars=10
        )
        row = out.iloc[0]
        assert row["exit_reason"] == "max_bars" and row["t1"] == 15
        assert row["label"] == 0 and row["ret"] < 0

    def test_unfinished_trade_is_dropped(self) -> None:
        df = self._frame([100.0] * 12, [99.9] * 12)
        out = label_events(df, self._signals([None] * 12).set_index(df.index), [5])
        assert out.empty


class TestUniquenessAndWeights:
    def test_average_uniqueness(self) -> None:
        # A spans bars 1-4, B spans 3-4: bars 3-4 are shared.
        u = average_uniqueness(np.array([0, 2]), np.array([4, 4]), 6)
        assert u == pytest.approx([(1 + 1 + 0.5 + 0.5) / 4, 0.5])

    def test_clipped_weights_cap_outliers(self) -> None:
        ret = np.r_[np.full(99, 0.01), 1.0]
        raw = sample_weights(np.ones(100), ret, "raw")
        clipped = sample_weights(np.ones(100), ret, "clipped")
        assert raw[-1] / raw[0] == pytest.approx(100)
        assert clipped[-1] / clipped[0] == pytest.approx(1)
        assert clipped.mean() == pytest.approx(1)


class TestSplits:
    @staticmethod
    def _events(n: int = 400) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(0)
        t0 = pd.Series(pd.date_range("2024-01-01", periods=n, freq="6h", tz="UTC"))
        t1 = t0 + pd.to_timedelta(rng.integers(1, 96, n), unit="h")
        return t0, t1

    def test_walk_forward_purges_by_label_end(self) -> None:
        t0, t1 = self._events()
        embargo = pd.Timedelta(timedelta(hours=96))
        splits = walk_forward_time_splits(
            t0, t1, 3, pd.Timedelta(timedelta(days=15)), embargo
        )
        for split in splits:
            start = split.windows[0][0]
            assert (t1.iloc[split.train] + embargo < start).all()
            assert (t0.iloc[split.test] >= start).all()

    def test_cpcv_train_never_overlaps_test_groups(self) -> None:
        t0, t1 = self._events()
        embargo = pd.Timedelta(timedelta(hours=96))
        splits = cpcv_splits(t0, t1, 6, 2, embargo)
        assert len(splits) == 15
        for split in splits:
            for start, stop in split.windows:
                tr0, tr1 = t0.iloc[split.train], t1.iloc[split.train]
                assert not ((tr0 <= stop) & (tr1 >= start)).any()
                assert not ((tr0 > stop) & (tr0 <= stop + embargo)).any()


class TestThresholdAndContract:
    def test_threshold_accepts_all_when_filter_cannot_help(self) -> None:
        p = np.linspace(0, 1, 100)
        ret = np.full(100, 0.01)  # every trade wins: rejecting any loses PnL
        assert choose_threshold(p, ret) == 0.0

    def test_threshold_cuts_predicted_losers(self) -> None:
        p = np.linspace(0, 1, 100)
        ret = np.where(p > 0.6, 0.02, -0.01)
        assert 0.55 <= choose_threshold(p, ret) <= 0.65

    def test_binary_contract_is_a_hard_filter(self) -> None:
        assert prediction_fields(0.6, 0.5, "binary")[:2] == ("up", 1.0)
        assert prediction_fields(0.49, 0.5, "binary")[:2] == ("down", 1.0)
        with pytest.raises(ValueError):
            prediction_fields(0.5, 0.5, "soft")

    def test_sequential_return_takes_one_position_per_pair(self) -> None:
        # Pair A: events 0..2 overlap (t1=10); only the first is traded, then
        # event at t0=10 opens again. Pair B trades independently.
        t0 = np.array([0, 1, 2, 10, 0])
        t1 = np.array([10, 11, 12, 20, 5])
        ret = np.array([0.01, 0.5, 0.5, 0.02, 0.03])
        groups = np.array(["A", "A", "A", "A", "B"])
        accept = np.ones(5, dtype=bool)
        total = sequential_return(accept, ret, t0, t1, groups)
        assert total == pytest.approx(0.01 + 0.02 + 0.03)
        # Rejecting the first entry lets the next overlapping one in.
        accept[0] = False
        total = sequential_return(accept, ret, t0, t1, groups)
        assert total == pytest.approx(0.5 + 0.03)

    def test_threshold_uses_the_trades_actually_taken(self) -> None:
        """A long chain of overlapping winners counts as one trade: the bot
        holds one position. Unweighted, the chain would dominate."""
        n = 30
        p = np.r_[np.full(n, 0.2), np.full(3, 0.2), np.full(5, 0.8)]
        ret = np.r_[np.full(n, 0.01), np.full(3, -0.02), np.full(5, 0.01)]
        t0 = np.r_[np.arange(n), [100, 200, 300], [400, 500, 600, 700, 800]]
        t1 = np.r_[np.full(n, 50), [150, 250, 350], [450, 550, 650, 750, 850]]
        assert choose_threshold(p, ret) == 0.0  # per event: +0.30 - 0.06
        assert choose_threshold(p, ret, t0, t1) > 0.2  # per trade: 0.01 - 0.06

    def test_binary_is_the_default_contract(self) -> None:
        assert prediction_fields(0.6, 0.5) == prediction_fields(0.6, 0.5, "binary")
        names = [*PRICE_FEATURES, *SIGNAL_FEATURES, "pair_code"]
        model = MetaModel(_ConstantClassifier(0.6, len(names)), (1.0, 0.0), 0.5, names)
        predictor = MetaLabelPredictor("BTC/USDT", "1h", model, {"BTC/USDT": 0})
        assert predictor._contract == "binary"

    @pytest.mark.parametrize(
        ("p", "thr", "direction", "conf"),
        [(0.8, 0.4, "up", 2 / 3), (0.4, 0.4, "up", 0.0), (0.1, 0.4, "down", 0.75)],
    )
    def test_prediction_fields(
        self, p: float, thr: float, direction: str, conf: float
    ) -> None:
        d, c, r = prediction_fields(p, thr, "scaled")
        assert d == direction and c == pytest.approx(conf)
        assert r == pytest.approx(p - thr)


class _ConstantClassifier:
    def __init__(self, p: float, n_features: int) -> None:
        self.p = p
        self.feature_importances_ = np.zeros(n_features)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile([1 - self.p, self.p], (len(x), 1))


def _predictor(
    p: float, threshold: float, pair: str = "BTC/USDT", contract: str = "scaled"
) -> MetaLabelPredictor:
    names = [*PRICE_FEATURES, *SIGNAL_FEATURES, "pair_code"]
    model = MetaModel(_ConstantClassifier(p, len(names)), (1.0, 0.0), threshold, names)
    return MetaLabelPredictor(
        pair, "1h", model, {"BTC/USDT": 0, "ETH/USDT": 1}, contract=contract
    )


class TestArtifact:
    def test_bot_loads_meta_artifact_through_lightgbm_predictor(
        self, tmp_path: Path, ohlcv: pd.DataFrame, signals: pd.DataFrame
    ) -> None:
        path = tmp_path / "BTC_USDT_1h.pkl"
        _predictor(0.30, 0.45).save(str(path))
        assert Path(f"{path}.sha256").exists()

        bot_side = LightGBMPredictor(pair="BTC/USDT", timeframe="1h")
        bot_side.load(str(path))
        window = entry_window(ohlcv, signals)
        pred = bot_side.predict(window, FeatureEngine().build_features(window))
        assert bot_side.is_loaded()
        assert pred.model_type == "lightgbm_meta_label"
        assert pred.predicted_direction == "down"
        assert pred.confidence == pytest.approx((0.45 - 0.30) / 0.45)

    def test_up_when_probability_reaches_threshold(
        self, ohlcv: pd.DataFrame, signals: pd.DataFrame
    ) -> None:
        pred = _predictor(0.70, 0.45).predict(entry_window(ohlcv, signals))
        assert pred.predicted_direction == "up"

    def test_unknown_pair_has_no_prediction(
        self, ohlcv: pd.DataFrame, signals: pd.DataFrame
    ) -> None:
        with pytest.raises(PredictionUnavailableError):
            _predictor(0.7, 0.5, pair="DOGE/USDT").predict(entry_window(ohlcv, signals))

    def test_no_prediction_outside_an_entry_event(
        self, ohlcv: pd.DataFrame, signals: pd.DataFrame
    ) -> None:
        """Trained on entries only: elsewhere the bot gets no ML signal."""
        with pytest.raises(PredictionUnavailableError, match="LONG entry"):
            _predictor(0.9, 0.5).predict(non_entry_window(ohlcv, signals))

    def test_training_a_direction_model_drops_a_loaded_meta_delegate(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "m.pkl"
        _predictor(0.3, 0.5).save(str(path))
        model = LightGBMPredictor()
        model.load(str(path))
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"f": rng.normal(size=300)})
        frame["y"] = (frame["f"] > 0).astype(int)
        model.train(frame.iloc[:200], frame.iloc[200:], {"n_estimators": 20})
        pred = model.predict(frame, frame)
        assert pred.model_type == "lightgbm"

    def test_legacy_direction_artifact_still_loads(self, tmp_path: Path) -> None:
        """A .pkl written before the ``task`` field keeps loading as direction."""
        path = tmp_path / "legacy.pkl"
        legacy = {"model": _ConstantClassifier(0.9, 1), "feature_names": ["RSI_14"]}
        payload = pickle.dumps(legacy)
        path.write_bytes(payload)
        model = LightGBMPredictor()
        model.load(str(path))
        frame = pd.DataFrame({"RSI_14": [55.0]})
        assert model.predict(frame, frame).predicted_direction == "up"

    def test_unknown_task_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.pkl"
        write_artifact(path, {"task": "regression", "model": None})
        with pytest.raises(ValueError, match="Unknown model task"):
            LightGBMPredictor().load(str(path))


class TestFeatureParity:
    def test_meta_features_match_the_live_window(self, ohlcv: pd.DataFrame) -> None:
        data = build_pair_data("BTC/USDT", ohlcv, 0)
        cols = [*PRICE_FEATURES, *SIGNAL_FEATURES]
        assert meta_parity_columns(data, cols, window=250, samples=8) == cols

    def test_live_predictor_sees_the_training_row(self, ohlcv: pd.DataFrame) -> None:
        """predict() on a 250-bar window rebuilds the stored training row."""
        data = build_pair_data("BTC/USDT", ohlcv, 0)
        captured: list[pd.DataFrame] = []

        class Spy(_ConstantClassifier):
            def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
                captured.append(x)
                return super().predict_proba(x)

        predictor = _predictor(0.5, 0.5)
        predictor._model.classifier = Spy(0.5, len(predictor.feature_names))
        ts = data.events["t0"].iloc[len(data.events) // 2]
        end = ohlcv.index.get_loc(ts)
        predictor.predict(ohlcv.iloc[end - 249 : end + 1])
        live = captured[0].iloc[0]
        ref = data.features.loc[ts, predictor.feature_names]
        # EMA50 over 250 bars keeps ~1e-4 of its seed: tiny, but not exact.
        np.testing.assert_allclose(live.to_numpy(float), ref.to_numpy(float), rtol=1e-3)


class TestValidation:
    def _events(self, n: int = 600, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t0 = pd.date_range("2023-01-01", periods=n, freq="12h", tz="UTC")
        signal = rng.normal(size=n)
        ret = 0.01 * signal + rng.normal(0, 0.01, n)
        return pd.DataFrame(
            {
                "f": signal + rng.normal(0, 0.5, n),
                "noise": rng.normal(size=n),
                "ret": ret,
                "label": (ret > 0).astype(int),
                "t0": t0,
                "t1": t0 + pd.Timedelta(timedelta(hours=10)),
                "pair": "BTC/USDT",
                "uniqueness": 1.0,
                "exit_reason": "signal",
                "cluster": np.arange(n),
            }
        )

    def _preds(self, ev: pd.DataFrame, p: np.ndarray, folds: int = 3) -> list:
        rows = np.array_split(np.arange(len(ev)), folds)
        return [
            FoldPrediction(k, r, p[r], 0.5, None)  # type: ignore[arg-type]
            for k, r in enumerate(rows)
        ]

    def test_informative_filter_beats_random(self) -> None:
        ev = self._events()
        p = 1 / (1 + np.exp(-ev["f"].to_numpy()))
        m = event_metrics(ev, self._preds(ev, p), n_boot=300)
        assert m.expectancy_delta > 0 and m.delta_ci_low > 0
        assert m.random_filter_p < 0.05
        assert m.auc > 0.7

    def test_noise_filter_does_not(self) -> None:
        ev = self._events()
        p = 1 / (1 + np.exp(-ev["noise"].to_numpy()))
        m = event_metrics(ev, self._preds(ev, p), n_boot=300)
        assert m.delta_ci_low < 0 < 0.05 < m.random_filter_p

    def test_clustered_noise_is_not_significant(self) -> None:
        """30 overlapping events per trade share one outcome; a filter picking
        whole trades at random must not look significant."""
        rng = np.random.default_rng(5)
        n_trades, size = 60, 30
        ret = np.repeat(rng.normal(0, 0.02, n_trades), size)
        cluster = np.repeat(np.arange(n_trades), size)
        ev = self._events(n_trades * size)
        ev["ret"], ev["label"], ev["cluster"] = ret, (ret > 0).astype(int), cluster
        p = np.repeat(rng.uniform(size=n_trades), size)
        m = event_metrics(ev, self._preds(ev, p), n_boot=500)
        assert m.random_filter_p > 0.05
        assert m.delta_ci_low < 0

    def test_fold_with_nothing_accepted_is_not_positive(self) -> None:
        ev = self._events(300)
        p = np.r_[np.full(100, 0.9), np.full(200, 0.1)]  # folds 2-3 reject all
        ev["ret"] = 0.01
        m = event_metrics(ev, self._preds(ev, p), n_boot=10)
        assert m.folds_positive_expectancy == pytest.approx(1 / 3)

    def test_overlap_clusters(self) -> None:
        from trading_bot.ai.prediction.meta_validation import overlap_clusters

        t0 = np.array([0, 1, 2, 10, 11, 30])
        t1 = np.array([5, 4, 9, 12, 20, 31])
        assert overlap_clusters(t0, t1).tolist() == [0, 0, 0, 1, 1, 2]

    def test_fit_meta_learns_and_calibrates(self) -> None:
        ev = self._events(2000)
        t0 = ev["t0"].to_numpy()
        model = fit_meta(
            ev[["f", "noise"]],
            ev["label"].to_numpy(),
            ev["ret"].to_numpy(),
            np.ones(len(ev)),
            t0,
            ev["t1"].to_numpy(),
            np.timedelta64(10, "h"),
            {"min_child_samples": 20},
        )
        p = model.predict_proba(ev[["f", "noise"]])
        assert 0.0 < model.threshold < 1.0
        assert abs(p.mean() - ev["label"].mean()) < 0.05  # calibrated level

    def test_gate_requires_every_condition(self) -> None:
        good = EventMetrics(
            100, 50, 0.5, 0.4, 0.5, 0.6, 0.2, 0.05, 0.0, 0.01, 0.01, 0.0, 0.5,
            delta_ci_low=0.001, random_filter_p=0.01, folds_positive_expectancy=0.75,
        )  # fmt: skip
        bt = BacktestComparison(80, 50, 0.1, 0.3, 2.0, 3.0, 8.0, 6.0, 0.5, 0.9)
        assert meta_gate(good, good, bt).passed
        worse_dd = BacktestComparison(80, 50, 0.1, 0.3, 2.0, 3.0, 8.0, 9.0, 0.5, 0.9)
        assert not meta_gate(good, good, worse_dd).passed
        lucky = EventMetrics(**{**good.__dict__, "random_filter_p": 0.2})
        assert not meta_gate(lucky, good, bt).passed
        few = BacktestComparison(80, 20, 0.1, 0.3, 2.0, 3.0, 8.0, 6.0, 0.5, 0.9)
        assert not meta_gate(good, good, few).passed


class TestNoLookahead:
    def test_fold_model_ignores_data_after_its_window(self) -> None:
        """Rewriting prices after a walk-forward test window (plus the label
        horizon) must not change that window's predictions."""
        ohlcv = trending_ohlcv(3600, seed=3)
        cut = ohlcv.index[2400]
        tampered = ohlcv.copy()
        after = ohlcv.index > cut + pd.Timedelta(timedelta(hours=200))
        rng = np.random.default_rng(9)
        tampered.loc[after, ["open", "high", "low", "close"]] *= rng.uniform(
            0.7, 1.3, (int(after.sum()), 1)
        )

        def first_fold(frame: pd.DataFrame) -> np.ndarray:
            ev = pool_events([build_pair_data("BTC/USDT", frame, 0)])
            feats = [*PRICE_FEATURES, *SIGNAL_FEATURES, "pair_code"]
            splits = walk_forward_time_splits(
                ev["t0"],
                ev["t1"],
                1,
                pd.Timedelta(timedelta(days=15)),
                pd.Timedelta(timedelta(hours=96)),
                end=cut,
            )
            from trading_bot.ai.prediction.meta_validation import predict_splits

            (fp,) = predict_splits(
                ev, feats, splits, pd.Timedelta(timedelta(hours=96)), "clipped",
                {"min_child_samples": 10},
            )  # fmt: skip
            return np.c_[ev["t0"].iloc[fp.rows].astype("int64"), fp.p]

        np.testing.assert_array_equal(first_fold(ohlcv), first_fold(tampered))


class TestEndToEnd:
    def test_small_run_produces_reports_and_a_servable_model(
        self, tmp_path: Path
    ) -> None:
        histories = {
            "BTC/USDT": trending_ohlcv(3600, seed=1),
            "ETH/USDT": trending_ohlcv(3600, seed=2),
        }
        cfg = MetaConfig(
            folds=2,
            test_days=20,
            cpcv_groups=4,
            cpcv_test_groups=1,
            n_boot=50,
            parity_samples=4,
            compare_weight_modes=(),
        )
        from trading_bot.config.settings import get_settings

        run = run_meta_label(histories, cfg, get_settings())
        assert set(run.events_per_pair) == {"BTC/USDT", "ETH/USDT"}
        assert len(run.primary.pairs) == 2 and len(run.primary.cpcv_delta) == 4
        assert run.final_model is not None
        for rep in run.primary.pairs:
            assert set(rep.backtests) == {"scaled", "binary"}
            assert rep.backtest is rep.backtests["binary"]  # the default contract
            assert rep.backtest.trades_base > 0
            assert rep.gate is not None

        predictor = build_predictor(
            run, "ETH/USDT", "1h", {"trained_at": "t"}, contract="binary"
        )
        path = tmp_path / "ETH_USDT_1h.pkl"
        predictor.save(str(path))
        bot_side = LightGBMPredictor(pair="ETH/USDT", timeframe="1h")
        bot_side.load(str(path))
        eth = histories["ETH/USDT"]
        eth_signals = technical_signals(eth, "ETH/USDT")
        pred = bot_side.predict(entry_window(eth, eth_signals))
        assert pred.predicted_direction in {"up", "down"}
        assert pred.confidence == 1.0  # binary contract survives the artifact
        assert run.as_dict()["threshold"] == run.final_model.threshold
