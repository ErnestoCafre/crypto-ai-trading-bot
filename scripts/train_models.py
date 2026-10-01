"""Train, validate and (optionally) publish the per-pair ML models.

``--task meta_label`` (default): a meta-label model on the technical LONG
entries (see ``ai/prediction/meta_label.py``). One model is trained on the
pooled events of every pair, validated with purged walk-forward + CPCV on
calendar time, and compared with the live pipeline in the backtest. It is
published to a pair's ``{model_dir}/{BTC_USDT}_{tf}.pkl`` only when the
filter beats a random filter and the unfiltered pipeline (``meta_gate``).

``--task direction``: the original next-bar direction classifier, kept for
reference. It was discarded: out of sample its directional accuracy is
~50%, and its label (return > ML_LABEL_THRESHOLD) disagrees with the bot's
reading of p > 0.5 as "up". Its gate is ADR-004's 55% directional accuracy.

Usage:
    uv run python scripts/train_models.py --pairs BTC/USDT ETH/USDT SOL/USDT \\
        --timeframe 1h --publish
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from trading_bot.ai.prediction.base import model_path
from trading_bot.ai.prediction.data_pipeline import MLDataPipeline
from trading_bot.ai.prediction.feature_engine import FeatureEngine
from trading_bot.ai.prediction.label_generator import LabelGenerator
from trading_bot.ai.prediction.lightgbm_model import LightGBMPredictor
from trading_bot.ai.prediction.meta_training import (
    MetaConfig,
    MetaRun,
    build_predictor,
    run_meta_label,
)
from trading_bot.ai.prediction.trainer import ModelTrainer
from trading_bot.ai.prediction.validation import (
    ValidationReport,
    candidate_columns,
    evaluate_walk_forward,
    fit_model,
    purged_holdout,
    serving_parity_columns,
    tune_params,
    walk_forward_splits,
)
from trading_bot.config import get_settings
from trading_bot.config.logging import setup_logging
from trading_bot.data.market.candles import split_closed_candles
from trading_bot.data.market.ccxt_fetcher import CCXTFetcher
from trading_bot.data.storage import OHLCVRepository

# Binance spot history starts in 2017.
META_HISTORY_START = datetime(2017, 1, 1, tzinfo=UTC)
_BAR = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


@dataclass
class TrainConfig:
    timeframe: str = "1h"
    horizon: int = 1
    threshold: float = 0.001
    embargo: int | None = None
    folds: int = 6
    test_bars: int = 720
    trials: int = 0
    feature_set: str = "scale_free"
    serving_window: int = 250
    min_accuracy: float = 0.55
    seed: int = 42

    @property
    def embargo_bars(self) -> int:
        return self.horizon if self.embargo is None else self.embargo


@dataclass
class PairResult:
    pair: str
    rows: int
    data_start: str
    data_end: str
    features: list[str]
    params: dict[str, Any]
    report: ValidationReport
    passed: bool
    reason: str
    model: LightGBMPredictor | None = field(default=None, repr=False)
    published_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k not in ("model",)}
        out["report"] = self.report.as_dict()
        return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", nargs="+", default=list(s.TRADING_PAIRS))
    p.add_argument("--timeframe", default=s.TRADING_TIMEFRAME, choices=sorted(_BAR))
    p.add_argument("--task", choices=("meta_label", "direction"), default="meta_label")
    p.add_argument(
        "--years",
        type=float,
        default=None,
        help="history to train on (default: meta_label all since 2017, direction 2)",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=s.OPTUNA_ML_TRIALS,
        help="direction: Optuna trials (meta_label uses fixed small trees)",
    )
    p.add_argument("--model-dir", default=s.ML_MODEL_DIR)
    p.add_argument(
        "--publish",
        action="store_true",
        help="save and publish the models that pass the gate (default: report only)",
    )
    p.add_argument("--folds", type=int, default=6)
    p.add_argument(
        "--test-days",
        type=float,
        default=None,
        help="length of each test window (default: meta_label 180, direction 30)",
    )
    p.add_argument(
        "--max-bars", type=int, default=96, help="meta_label: label horizon cap"
    )
    p.add_argument(
        "--weights",
        choices=("clipped", "raw", "uniqueness"),
        default="clipped",
        help="meta_label sample weights (clipped: uniqueness x |ret| capped at p95)",
    )
    p.add_argument(
        "--contract",
        choices=("scaled", "binary"),
        default="scaled",
        help="meta_label prediction contract published (both are backtested)",
    )
    p.add_argument(
        "--risk-pct",
        type=float,
        default=0.20,
        help="meta_label backtest MAX_RISK_PER_TRADE_PCT",
    )
    p.add_argument(
        "--max-position-pct",
        type=float,
        default=25.0,
        help="meta_label backtest MAX_POSITION_PCT",
    )
    p.add_argument(
        "--feature-set",
        choices=("scale_free", "all"),
        default="scale_free",
        help="direction only",
    )
    p.add_argument(
        "--min-accuracy", type=float, default=0.55, help="direction only (ADR-004)"
    )
    p.add_argument("--exchange", default=s.EXCHANGE_ID)
    p.add_argument("--offline", action="store_true", help="use the parquet cache only")
    p.add_argument("--report", default=None, help="JSON report path")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.test_days is None:
        args.test_days = 180.0 if args.task == "meta_label" else 30.0
    if args.years is None and args.task == "direction":
        args.years = 2.0
    if (args.years is not None and args.years <= 0) or args.folds <= 0:
        p.error("--years and --folds must be positive")
    if args.test_days <= 0 or args.max_bars <= 0:
        p.error("--test-days and --max-bars must be positive")
    return args


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    s = get_settings()
    return TrainConfig(
        timeframe=args.timeframe,
        horizon=s.ML_LABEL_HORIZON,
        threshold=s.ML_LABEL_THRESHOLD,
        folds=args.folds,
        test_bars=int(args.test_days * 1440 / _BAR[args.timeframe]),
        trials=args.trials,
        feature_set=args.feature_set,
        serving_window=s.OHLCV_LOOKBACK_BARS,
        min_accuracy=args.min_accuracy,
        seed=args.seed,
    )


def passes_gate(report: ValidationReport, min_accuracy: float) -> tuple[bool, str]:
    """ADR-004 gate: OOS directional accuracy >= min and above the baseline."""
    acc = report.directional_accuracy
    base = report.baseline_directional_accuracy
    if math.isnan(acc) or acc < min_accuracy:
        return False, f"directional accuracy {acc:.2%} < {min_accuracy:.0%}"
    if acc <= base:
        return False, f"directional accuracy {acc:.2%} <= baseline {base:.2%}"
    return True, f"directional accuracy {acc:.2%} >= {min_accuracy:.0%}"


def train_pair(pair: str, ohlcv: pd.DataFrame, cfg: TrainConfig) -> PairResult:
    """Validate a model for one pair and fit the final one on all the data."""
    fe = FeatureEngine()
    ohlcv = ohlcv.sort_index()
    candidates = candidate_columns(fe.build_features(ohlcv.iloc[:300]), cfg.feature_set)
    columns = serving_parity_columns(ohlcv, candidates, cfg.serving_window, fe)
    if not columns:
        raise ValueError(f"{pair}: no feature survives the parity check")

    x, y = MLDataPipeline(fe).prepare(
        ohlcv, cfg.horizon, cfg.threshold, feature_columns=columns
    )
    forward_ret = LabelGenerator.forward_return(ohlcv, cfg.horizon).loc[x.index]
    bar_pos = ohlcv.index.get_indexer(x.index)

    splits = walk_forward_splits(
        bar_pos, cfg.horizon, cfg.folds, cfg.test_bars, cfg.embargo_bars
    )
    base_params: dict[str, Any] = {"random_state": cfg.seed}
    # Tune only on rows that end before the first test window (purged).
    pre_test = np.flatnonzero(
        bar_pos + cfg.horizon + cfg.embargo_bars < bar_pos[splits[0].test[0]]
    )
    params = tune_params(
        x,
        y,
        bar_pos,
        pre_test,
        cfg.horizon,
        cfg.embargo_bars,
        cfg.trials,
        base_params,
        cfg.seed,
    )
    report = evaluate_walk_forward(x, y, forward_ret, splits, params)
    passed, reason = passes_gate(report, cfg.min_accuracy)

    fit_rows, val_rows = purged_holdout(
        bar_pos, np.arange(len(x)), cfg.horizon, cfg.embargo_bars, 0.15
    )
    model = fit_model(x, y, fit_rows, val_rows, params, pair, cfg.timeframe)
    return PairResult(
        pair=pair,
        rows=len(x),
        data_start=str(x.index[0]),
        data_end=str(x.index[-1]),
        features=columns,
        params=params,
        report=report,
        passed=passed,
        reason=reason,
        model=model,
    )


def publish(result: PairResult, model_dir: str, timeframe: str) -> str:
    assert result.model is not None
    r = result.report
    metrics = {
        "oos_directional_accuracy": r.directional_accuracy,
        "oos_accuracy": r.accuracy,
        "oos_auc": r.auc,
        "oos_log_loss": r.log_loss,
        "baseline_directional_accuracy": r.baseline_directional_accuracy,
    }
    return ModelTrainer(model_dir).save_model_with_metadata(
        result.model, metrics, result.pair, timeframe, publish=True
    )


async def load_history(
    exchange: str,
    pair: str,
    timeframe: str,
    since: datetime,
    offline: bool,
    warn_short: bool = True,
) -> pd.DataFrame:
    """Closed candles since ``since``: parquet cache, topped up from the exchange."""
    cached = OHLCVRepository.load_parquet(exchange, pair, timeframe)
    have = cached is not None and not cached.empty
    if not offline:
        start = since
        if have and cached is not None and cached.index[0] <= since:
            start = cached.index[-1].to_pydatetime() + timedelta(milliseconds=1)
        # Historical data always comes from the real exchange, never sandbox.
        fetcher = CCXTFetcher(exchange_id=exchange, sandbox=False, timeout_s=None)
        async with fetcher:
            fresh = await fetcher.fetch_ohlcv(pair, timeframe, since=start)
        fresh, _ = split_closed_candles(fresh, timeframe, datetime.now(UTC))
        if not fresh.empty:
            OHLCVRepository.save_parquet(fresh, exchange, pair, timeframe)
        cached = OHLCVRepository.load_parquet(exchange, pair, timeframe)
    if cached is None or cached.empty:
        raise ValueError(f"no cached history for {pair} {timeframe} on {exchange}")
    history = cached[cached.index >= since][["open", "high", "low", "close", "volume"]]
    if history.empty:
        raise ValueError(f"no history since {since} for {pair} {timeframe}")
    if warn_short and history.index[0] - pd.Timestamp(since) > timedelta(days=7):
        logger.warning(
            "{}: history starts {} — {} days short of the requested window",
            pair,
            history.index[0],
            (history.index[0] - pd.Timestamp(since)).days,
        )
    return history


def format_table(results: Sequence[PairResult]) -> str:
    head = (
        "| Par | Filas | OOS (barras) | Acc. direccional | Baseline dir. | "
        "Accuracy | Baseline acc. | AUC | Log-loss | Baseline LL | Publicado |"
    )
    lines = [head, "|" + "---|" * 11]
    for res in results:
        r = res.report
        lines.append(
            f"| {res.pair} | {res.rows} | {r.n_test} | "
            f"{r.directional_accuracy:.2%} | {r.baseline_directional_accuracy:.2%} | "
            f"{r.accuracy:.2%} | {r.baseline_accuracy:.2%} | {r.auc:.3f} | "
            f"{r.log_loss:.4f} | {r.baseline_log_loss:.4f} | "
            f"{'sí' if res.published_path else 'no'} |"
        )
    return "\n".join(lines)


def history_start(args: argparse.Namespace) -> datetime:
    if args.years is None:
        return META_HISTORY_START
    return datetime.now(UTC) - timedelta(days=365 * args.years)


async def run(args: argparse.Namespace) -> Any:
    if args.task == "meta_label":
        return await run_meta(args)
    return await run_direction(args)


async def run_meta(args: argparse.Namespace) -> MetaRun:
    s = get_settings()
    settings = s.model_copy(
        update={
            "MAX_RISK_PER_TRADE_PCT": args.risk_pct,
            "MAX_POSITION_PCT": args.max_position_pct,
        }
    )
    cfg = MetaConfig(
        timeframe=args.timeframe,
        max_bars=args.max_bars,
        folds=args.folds,
        test_days=args.test_days,
        weight_mode=args.weights,
        contract=args.contract,
        compare_weight_modes=tuple(m for m in ("clipped", "raw") if m != args.weights),
        serving_window=s.OHLCV_LOOKBACK_BARS,
        seed=args.seed,
    )
    since = history_start(args)
    histories = {}
    for pair in args.pairs:
        ohlcv = await load_history(
            args.exchange,
            pair,
            cfg.timeframe,
            since,
            args.offline,
            warn_short=args.years is not None,
        )
        logger.info(
            "{}: {} bars {} → {}", pair, len(ohlcv), ohlcv.index[0], ohlcv.index[-1]
        )
        histories[pair] = ohlcv

    result = run_meta_label(histories, cfg, settings)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    published: dict[str, str] = {}
    for rep in result.primary.pairs:
        path = model_path(args.model_dir, rep.pair, cfg.timeframe)
        assert rep.gate is not None
        if rep.gate.passed and args.publish:
            predictor = build_predictor(
                result,
                rep.pair,
                cfg.timeframe,
                {
                    "trained_at": stamp,
                    "pairs": result.pairs,
                    "weight_mode": cfg.weight_mode,
                    "max_bars": cfg.max_bars,
                    "contract": cfg.contract,
                    "oos_backtest": asdict(rep.backtest) if rep.backtest else None,
                },
                cfg.contract,
            )
            predictor.save(str(path))
            published[rep.pair] = str(path)
        elif not rep.gate.passed and path.exists():
            logger.warning(
                "{}: failed the gate but {} is still published — remove it "
                "to leave the pair without prediction",
                rep.pair,
                path,
            )
        logger.info(
            "{}: {} — {}",
            rep.pair,
            "PUBLISHED" if rep.pair in published else "not published",
            "; ".join(rep.gate.reasons),
        )

    report_path = Path(
        args.report or Path(args.model_dir) / f"meta_label_report_{stamp}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "meta_label",
        "config": {**asdict(cfg), "embargo": str(cfg.embargo)},
        "history_start": str(since),
        "settings": {
            "MAX_RISK_PER_TRADE_PCT": args.risk_pct,
            "MAX_POSITION_PCT": args.max_position_pct,
        },
        "exchange": args.exchange,
        "published": published,
        **result.as_dict(),
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str))
    print(format_meta_table(result, published, cfg.contract))
    print(f"\nReport: {report_path}")
    return result


def format_meta_table(
    result: MetaRun, published: dict[str, str], contract: str = "scaled"
) -> str:
    lines = []
    for mode in result.modes:
        pm = mode.pooled
        lines.append(
            f"\nPesos={mode.weight_mode}: OOS {pm.n} eventos, acepta "
            f"{pm.acceptance_rate:.0%}, precisión {pm.precision:.1%} "
            f"(base {pm.base_rate:.1%}), AUC {pm.auc:.3f}, Brier {pm.brier:.4f}, "
            f"ECE {pm.ece:.3f}, expectativa por evento {pm.expectancy_accepted:+.3%}"
            f" vs {pm.expectancy_all:+.3%} (Δ {pm.expectancy_delta:+.3%}, "
            f"IC5% {pm.delta_ci_low:+.3%}), p filtro aleatorio "
            f"{pm.random_filter_p:.3f}, CPCV Δ>0 en {mode.cpcv_positive:.0%}"
        )
        lines.append(
            "| Par | Eventos | Acepta | Precisión | AUC | Brier | Exp. evento "
            "filtro/todos | Folds + |"
        )
        lines.append("|" + "---|" * 8)
        for rep in mode.pairs:
            m = rep.metrics
            lines.append(
                f"| {rep.pair} | {rep.events} | {m.acceptance_rate:.0%} | "
                f"{m.precision:.1%} | {m.auc:.3f} | {m.brier:.4f} | "
                f"{m.expectancy_accepted:+.3%} / {m.expectancy_all:+.3%} | "
                f"{m.folds_positive_expectancy:.0%} |"
            )
    lines.append(
        "\n| Par | Contrato | Trades sin/con | Exp. trade sin/con | Ret. sin/con | "
        "Max DD sin/con | Sharpe sin/con | Publicado |"
    )
    lines.append("|" + "---|" * 8)
    for rep in result.primary.pairs:
        for name, b in rep.backtests.items():
            pub = "sí" if rep.pair in published and name == contract else "no"
            lines.append(
                f"| {rep.pair} | {name} | {b.trades_base} / {b.trades_ml} | "
                f"{b.expectancy_base:+.3f}% / {b.expectancy_ml:+.3f}% | "
                f"{b.return_base:+.2f}% / {b.return_ml:+.2f}% | "
                f"{b.max_dd_base:.2f}% / {b.max_dd_ml:.2f}% | "
                f"{b.sharpe_base:.2f} / {b.sharpe_ml:.2f} | {pub} |"
            )
    return "\n".join(lines)


async def run_direction(args: argparse.Namespace) -> list[PairResult]:
    cfg = config_from_args(args)
    since = history_start(args)
    results: list[PairResult] = []
    for pair in args.pairs:
        ohlcv = await load_history(
            args.exchange, pair, cfg.timeframe, since, args.offline
        )
        logger.info(
            "{}: {} bars {} → {}", pair, len(ohlcv), ohlcv.index[0], ohlcv.index[-1]
        )
        result = train_pair(pair, ohlcv, cfg)
        if result.passed and args.publish:
            result.published_path = publish(result, args.model_dir, cfg.timeframe)
        elif not result.passed:
            stale = model_path(args.model_dir, pair, cfg.timeframe)
            if stale.exists():
                logger.warning(
                    "{}: failed the gate but {} is still published — remove it "
                    "to leave the pair without prediction",
                    pair,
                    stale,
                )
        logger.info(
            "{}: {} — {}",
            pair,
            "PUBLISHED" if result.published_path else "not published",
            result.reason,
        )
        results.append(result)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = Path(
        args.report or Path(args.model_dir) / f"training_report_{stamp}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {**cfg.__dict__, "embargo": cfg.embargo_bars, "years": args.years},
        "exchange": args.exchange,
        "results": [r.as_dict() for r in results],
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str))
    print(format_table(results))
    print(f"\nReport: {report_path}")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
