"""CRC Calibration Improvement Experiment.

Investigates why CRC under-covers (PICP < 0.90) in the benchmark and validates
six complementary strategies:

Conditions
----------
baseline          : CRC, train_tail, uniform,        calib_size=3, alpha=0.10  (benchmark config)
A1_test_prefix    : CRC, test_prefix, uniform,        calib_size=1, alpha=0.10  (in-distribution calib)
A2_test_prefix_2m : CRC, test_prefix, uniform,        calib_size=2, alpha=0.10  (2 months in-distribution calib)
A3_test_prefix_3m   : CRC, test_prefix, uniform,        calib_size=3, test_size=4, alpha=0.10  (3m calib, 1m eval)
A4_test_prefix_3m3m : CRC, test_prefix, uniform,        calib_size=3, test_size=6, alpha=0.10  (3m calib + 3m eval)
B_temporal          : CRC, train_tail,  temporal λ=1.0, calib_size=3, alpha=0.10  (recency bias)
B_temporal_strong : CRC, train_tail,  temporal λ=2.0, calib_size=3, alpha=0.10  (strong recency bias)
C_large_calib     : CRC, train_tail,  uniform,        calib_size=6, alpha=0.10  (more calib diversity)
D_adaptive_alpha  : CRC, train_tail,  uniform,        calib_size=3, alpha=0.07  (inflated alpha for shift)
E_season_matched  : CRC, season_matched, rolling train=12m, calib_size=3, alpha=0.10 (held-out same-season calib)
F1_prefix2_alpha08: CRC, test_prefix, calib_size=2, test_size=4, alpha=0.08 (in-dist calib + mild inflation)
F2_prefix3_alpha08: CRC, test_prefix, calib_size=3, test_size=6, alpha=0.08 (in-dist 3m calib + mild inflation)

Root cause hypothesis
---------------------
Rolling-origin folds shift from autumn (calibration tail) to winter (test window).
The seasonal distribution shift means autumn calibration scores underestimate
winter uncertainty, producing a conservative q_hat that still misses winter peaks.

The most effective remedy is to calibrate on data drawn from the same distribution
as the test window.  Using ``calib_source="test_prefix"`` with increasing calib_size
(1 → 3 months) sacrifices some evaluation months but guarantees exchangeability.

There was also a bug in the sign-res + temporal path: ``calculate_conformal_quantile``
used ``self.alpha`` instead of ``self.alpha / 2`` for each tail, producing ~80 % coverage
instead of 90 %. This is now fixed in the library.

Usage::

    python experiment/ch6_conformal/exploratory/crc_calib_improvement.py
    python experiment/ch6_conformal/exploratory/crc_calib_improvement.py --dataset PV-MLVS-UK
    python experiment/ch6_conformal/exploratory/crc_calib_improvement.py \
        --condition baseline A1_test_prefix A2_test_prefix_2m A3_test_prefix_3m
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path

import pandas as pd
from twiga.core.config import ConformalConfig, ExperimentConfig
from twiga.core.utils import LogContext, configure, get_logger, log_section
from twiga.models.nn import MLPGAMCRCConfig
from twiga.utils.run_forecast import (
    PipelineConfig,
    build_data_config,
    build_forecaster,
    load_and_preprocess,
    setup_environment,
)

from twiga_thesis.paths import thesis_data_dir

log = get_logger(__name__)

_EXPERIMENT_PREFIX = "calib"
_EXPERIMENT_NAME = "CRC Calib Improvement"
_REFERENCE_CONDITION = {
    "test_prefix": "baseline",
    "temporal": "baseline",
    "calib_size": "baseline",
    "alpha": "baseline",
    "season": "baseline",
    "final": "baseline",
}

DEFAULT_ALPHA = 0.10

ABLATION_BACKBONE_TEMPLATE = "../experiments/mlpgamcrc_ablation/backbone_hpo/{dataset_name}/mlpgam_best_params.json"
BENCHMARK_BACKBONE_TEMPLATE = "../experiments/mlpgamcrc_benchmark/{dataset_name}/hpo/backbone_best_params.json"

CRC_CONTROLLED_FIELDS: frozenset[str] = frozenset(
    {"two_stage", "stage1_epochs", "stage1_epochs_frac", "sigma_loss_fn", "monitor", "lambda_lasso"}
)
NON_TRANSFERABLE_FIELDS: frozenset[str] = frozenset({"opt_lr", "opt_weight_decay", "sch_warmup_epochs", "sch_eta_min"})

DEFAULT_CRC_OVERRIDES: dict = {
    "two_stage": True,
    "sigma_loss_fn": "hybrid_sqrt",
    "monitor": "sigma_loss",
    "lambda_lasso": 1e-6,
}


# ---------------------------------------------------------------------------
# Dataset registry (mirrors mlpgamcrc_benchmark.py)
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str
    train_start: str
    train_end: str
    period: str = "30min"
    input_window_size: int = 96
    forecast_horizon: int = 48
    window_stride: int = 48
    out_activation: str = "Identity"


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "MLVS-PT": DatasetConfig(name="MLVS-PT", train_start="2019-03", train_end="2022-01"),
    "PV-MLVS-UK": DatasetConfig(name="PV-MLVS-UK", train_start="2018-01", train_end="2020-07"),
    "ALBANIA": DatasetConfig(name="ALBANIA", train_start="2016-01", train_end="2019-12"),  # data ends 2019-12-31
    "Building-Load": DatasetConfig(name="Building-Load", train_start="2019-01", train_end="2022-01"),
    "Building-PV": DatasetConfig(name="Building-PV", train_start="2019-01", train_end="2022-01"),
}


# ---------------------------------------------------------------------------
# Calibration conditions
# ---------------------------------------------------------------------------


@dataclass
class CalibCondition:
    """One CRC calibration variant to evaluate.

    Attributes:
        name: Short identifier used in filenames and log messages.
        description: Human-readable explanation.
        calib_source: Where to carve calibration data from.
        calib_size: Calibration window length in months.
        test_size: Total test window in months (including any test_prefix calib months).
        window: CV window strategy ('expanding' or 'rolling').
        train_size: Training window size in months; used only when window='rolling'.
        calib_method: Quantile estimation ('uniform' or 'temporal').
        lambda_: Exponential decay rate for temporal weighting (ignored if uniform).
        score_type: CRC score type ('res' or 'sign-res').
        alpha: Conformal significance level. Default 0.10 targets 90 % coverage.
        stage1_epochs_frac: Fraction of total epochs for CRC stage-1.
    """

    name: str
    description: str
    calib_source: str = "train_tail"
    calib_size: int = 3
    test_size: int = 4
    window: str = "expanding"
    train_size: int = 1
    calib_method: str = "uniform"
    lambda_: float = 1.0
    score_type: str = "sign-res"
    alpha: float = DEFAULT_ALPHA
    stage1_epochs_frac: float = 0.30


CALIB_CONDITIONS: list[CalibCondition] = [
    CalibCondition(
        name="baseline",
        description="Benchmark config: train_tail, uniform, calib_size=3, alpha=0.10",
        calib_source="train_tail",
        calib_size=3,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="A1_test_prefix",
        description="In-distribution calib: test_prefix, uniform, calib_size=1, test_size=4",
        calib_source="test_prefix",
        calib_size=1,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="A2_test_prefix_2m",
        description="In-distribution calib: test_prefix, uniform, calib_size=2, test_size=4",
        calib_source="test_prefix",
        calib_size=2,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="A3_test_prefix_3m",
        description="In-distribution calib: test_prefix, uniform, calib_size=3, test_size=4",
        calib_source="test_prefix",
        calib_size=3,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="A4_test_prefix_3m3m",
        description="In-distribution calib: test_prefix, uniform, calib_size=3, test_size=6 (3m calib + 3m eval)",
        calib_source="test_prefix",
        calib_size=3,
        test_size=6,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="B_temporal",
        description="Recency bias: train_tail, temporal (lambda=1.0), calib_size=3",
        calib_source="train_tail",
        calib_size=3,
        test_size=4,
        calib_method="temporal",
        lambda_=1.0,
        score_type="sign-res",
    ),
    CalibCondition(
        name="B_temporal_strong",
        description="Strong recency bias: train_tail, temporal (lambda=2.0), calib_size=3",
        calib_source="train_tail",
        calib_size=3,
        test_size=4,
        calib_method="temporal",
        lambda_=2.0,
        score_type="sign-res",
    ),
    CalibCondition(
        name="C_large_calib",
        description="More calib diversity: train_tail, uniform, calib_size=6",
        calib_source="train_tail",
        calib_size=6,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="C_4m",
        description="Calib size sweep: train_tail, uniform, calib_size=4",
        calib_source="train_tail",
        calib_size=4,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="C_9m",
        description="Calib size sweep: train_tail, uniform, calib_size=9",
        calib_source="train_tail",
        calib_size=9,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="C_12m",
        description="Calib size sweep: train_tail, uniform, calib_size=12",
        calib_source="train_tail",
        calib_size=12,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="D_adaptive_alpha",
        description="Inflated alpha to correct seasonal shift: train_tail, uniform, alpha=0.07",
        calib_source="train_tail",
        calib_size=3,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
        alpha=0.07,
    ),
    CalibCondition(
        name="E_season_matched",
        description=(
            "Season-matched calib from pre-training history: rolling train_size=12, calib_size=3. "
            "Selects same-calendar-month rows from before the training window — held-out and in-season."
        ),
        calib_source="season_matched",
        calib_size=3,
        test_size=4,
        window="rolling",
        train_size=12,
        calib_method="uniform",
        score_type="sign-res",
    ),
    CalibCondition(
        name="F1_prefix2_alpha08",
        description="In-dist calib + mild inflation: test_prefix, calib_size=2, test_size=4, alpha=0.08",
        calib_source="test_prefix",
        calib_size=2,
        test_size=4,
        calib_method="uniform",
        score_type="sign-res",
        alpha=0.08,
    ),
    CalibCondition(
        name="F2_prefix3_alpha08",
        description="In-dist calib + mild inflation: test_prefix, calib_size=3, test_size=6, alpha=0.08",
        calib_source="test_prefix",
        calib_size=3,
        test_size=6,
        calib_method="uniform",
        score_type="sign-res",
        alpha=0.08,
    ),
]

_CONDITION_MAP: dict[str, CalibCondition] = {c.name: c for c in CALIB_CONDITIONS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _condition_group(name: str) -> str:
    if name == "baseline":
        return "baseline"
    if name.startswith("A"):
        return "test_prefix"
    if name.startswith("B"):
        return "temporal"
    if name.startswith("C"):
        return "calib_size"
    if name.startswith("D"):
        return "alpha"
    if name.startswith("E"):
        return "season"
    if name.startswith("F"):
        return "final"
    return "other"


def _load_backbone_params(dataset_name: str) -> dict:
    """Load backbone HPO params, preferring ablation study output."""
    for template in [ABLATION_BACKBONE_TEMPLATE, BENCHMARK_BACKBONE_TEMPLATE]:
        path = Path(template.format(dataset_name=dataset_name))
        if path.exists():
            params = json.loads(path.read_text())
            log.info("Loaded backbone HPO params from %s  (n=%d keys)", path, len(params))
            return params
    log.warning("No backbone HPO params found for '%s' — using defaults", dataset_name)
    return {}


def _build_model_config(condition: CalibCondition, data_config, backbone_params: dict, cfg: PipelineConfig):
    """Build MLPGAMCRCConfig with backbone HPO params and condition-specific overrides."""
    model_config = MLPGAMCRCConfig.from_data_config(data_config)

    for key, value in backbone_params.items():
        if key in CRC_CONTROLLED_FIELDS or key in NON_TRANSFERABLE_FIELDS:
            continue
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    if hasattr(model_config, "out_activation_function"):
        model_config.out_activation_function = cfg.out_activation
    if hasattr(model_config, "max_epochs"):
        model_config.max_epochs = cfg.epochs
    if hasattr(model_config, "rich_progress_bar"):
        model_config.rich_progress_bar = False
    if hasattr(model_config, "resume_training"):
        model_config.resume_training = False

    for key, value in DEFAULT_CRC_OVERRIDES.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    if condition.stage1_epochs_frac is not None and hasattr(model_config, "stage1_epochs"):
        model_config.stage1_epochs = int(condition.stage1_epochs_frac * cfg.epochs)

    return model_config


def _build_experiment_config(
    condition: CalibCondition, cfg: PipelineConfig, output_dir: Path, num_splits: int
) -> ExperimentConfig:
    return ExperimentConfig(
        split_freq=cfg.split_freq,
        test_size=condition.test_size,
        train_size=condition.train_size,
        stride=1,
        val_size=2,
        calib_size=condition.calib_size,
        calib_source=condition.calib_source,
        num_splits=num_splits,
        window=condition.window,
        project_name=f"crc_calib/{cfg.dataset_name}/{condition.name}",
        root_dir=str(output_dir / condition.name),
        metrics=cfg.metrics,
    )


def _conformal_config(condition: CalibCondition) -> ConformalConfig:
    return ConformalConfig(
        method="residual-fitting",
        score_type=condition.score_type,
        alpha=condition.alpha,
        calib_method=condition.calib_method,
        lambda_=condition.lambda_,
    )


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

POINT_METRIC_COLS = ["mae", "rmse", "smape", "corr"]
INTERVAL_METRIC_COLS = ["cwe", "picp", "nmpi", "winkler_score", "ace"]
INTERVAL_COL_RENAME = {"cwe": "CWE", "picp": "PICP", "nmpi": "NMPI", "winkler_score": "WS", "ace": "ACE"}


def _log_fold_summary(metrics: pd.DataFrame, condition_name: str) -> None:
    """Log mean metrics across all folds for a completed condition."""
    numeric = metrics.select_dtypes("number")
    means = {col: f"{numeric[col].mean():.4f}" for col in numeric.columns if col not in ("Folds",)}
    log.info("Done  condition=%s  %s", condition_name, "  ".join(f"{k}={v}" for k, v in means.items()))


def run_condition(
    condition: CalibCondition,
    data: pd.DataFrame,
    data_config,
    backbone_params: dict,
    cfg: PipelineConfig,
    output_dir: Path,
    num_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run one calibration condition and return ``(metrics_df, interval_df)``.

    ``interval_df`` is ``None`` when no interval columns are present in the
    backtesting output.  Both DataFrames carry a ``condition`` column for
    downstream aggregation.

    Args:
        condition: Calibration condition to evaluate.
        data: Full dataset.
        data_config: Populated data pipeline config.
        backbone_params: Best backbone HPO params.
        cfg: Shared pipeline configuration.
        output_dir: Root output directory.
        num_splits: Maximum number of CV folds.

    Returns:
        ``(metrics_df, interval_df)`` where ``interval_df`` may be ``None``.
    """
    log_section(log, f"Condition: {condition.name}")
    log.info("%s", condition.description)

    model_config = _build_model_config(condition, data_config, backbone_params, cfg)
    exp_config = _build_experiment_config(condition, cfg, output_dir, num_splits)
    conf_config = _conformal_config(condition)

    forecaster = build_forecaster(data_config, [model_config], exp_config)
    forecaster.conformal_params = conf_config
    forecaster.interval_alpha = condition.alpha

    with LogContext(log, model=condition.name):
        preds, metrics = forecaster.backtesting(data, verbose=True, eval_interval=True)

    metrics["condition"] = condition.name
    metrics["group"] = _condition_group(condition.name)
    preds["condition"] = condition.name

    cond_dir = output_dir / condition.name
    cond_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(cond_dir / "metrics.csv", index=False)
    preds.to_parquet(cond_dir / "predictions.parquet", index=False)

    avail_iv = [c for c in INTERVAL_METRIC_COLS if c in metrics.columns]
    interval_metrics: pd.DataFrame | None = None
    if avail_iv:
        interval_metrics = metrics.copy()
        mean_iv = metrics[avail_iv].mean().to_dict()
        with (cond_dir / "interval_metrics_mean.json").open("w") as f:
            json.dump(mean_iv, f, indent=2)

    del forecaster
    gc.collect()
    _log_fold_summary(metrics, condition.name)
    return metrics, interval_metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _add_delta(summary: pd.DataFrame, col: str, ref: str, lower_is_better: bool) -> None:
    """Add a Δ-vs-baseline column in-place.

    Positive Δ always means the condition is better than the reference.

    Args:
        summary: Index-by-condition summary DataFrame.
        col: Column to compute delta for (e.g. ``"CWE"`` or ``"PICP"``).
        ref: Reference condition name (baseline).
        lower_is_better: Sign convention.
    """
    if col not in summary.columns:
        return
    cond_vals = (
        summary.index.get_level_values("condition") if isinstance(summary.index, pd.MultiIndex) else summary.index
    )
    if ref not in cond_vals:
        return
    sign = -1.0 if lower_is_better else 1.0
    ref_key = (_condition_group(ref), ref) if isinstance(summary.index, pd.MultiIndex) else ref
    ref_val = summary.loc[ref_key, col]
    if ref_val == 0:
        return
    summary[f"Δ{col} vs baseline (%, +ve=better)"] = (sign * (summary[col] - ref_val) / abs(ref_val) * 100).round(2)


def summarise_calib(
    all_point: list[pd.DataFrame],
    all_interval: list[pd.DataFrame],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-condition metrics into ranked summary tables.

    Point metrics are sorted ascending by MAE.
    Interval metrics are sorted descending by CWE.
    A Δ vs baseline column is added for MAE and CWE.

    Args:
        all_point: Per-condition point metric DataFrames (each tagged with a
            ``condition`` column).
        all_interval: Per-condition interval metric DataFrames.
        output_dir: Root output directory for CSV export.

    Returns:
        ``(point_summary, interval_summary)`` DataFrames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not all_point:
        log.warning("summarise_calib called with no completed conditions")
        return pd.DataFrame(), pd.DataFrame()

    # ── Point summary ─────────────────────────────────────────────────────
    point_df = pd.concat(all_point, ignore_index=True)
    point_df.to_csv(results_dir / "calib_point_full.csv", index=False)

    avail_pt = [c for c in POINT_METRIC_COLS if c in point_df.columns]
    pt_mean = point_df.groupby(["group", "condition"])[avail_pt].mean().round(4).rename(columns=str.upper)
    pt_std = (
        point_df.groupby(["group", "condition"])[avail_pt]
        .std()
        .round(4)
        .rename(columns={c: f"{c.upper()}_std" for c in avail_pt})
    )
    point_summary = pd.concat([pt_mean, pt_std], axis=1)
    if "MAE" in point_summary.columns:
        point_summary = point_summary.sort_values("MAE")
        _add_delta(point_summary, "MAE", "baseline", lower_is_better=True)

    point_summary.reset_index().to_csv(results_dir / "calib_point_summary.csv", index=False)
    log.info("Point summary saved  path=%s\n%s", results_dir / "calib_point_summary.csv", point_summary.to_string())

    # ── Interval summary ──────────────────────────────────────────────────
    if not all_interval:
        log.warning("No interval metrics collected — interval summary skipped")
        return point_summary, pd.DataFrame()

    interval_df = pd.concat(all_interval, ignore_index=True)
    interval_df.to_csv(results_dir / "calib_interval_full.csv", index=False)

    avail_iv = [c for c in INTERVAL_METRIC_COLS if c in interval_df.columns]
    iv_mean = interval_df.groupby(["group", "condition"])[avail_iv].mean().round(4).rename(columns=INTERVAL_COL_RENAME)
    iv_std = (
        interval_df.groupby(["group", "condition"])[avail_iv]
        .std()
        .round(4)
        .rename(columns={c: f"{INTERVAL_COL_RENAME.get(c, c.upper())}_std" for c in avail_iv})
    )
    interval_summary = pd.concat([iv_mean, iv_std], axis=1)
    if "CWE" in interval_summary.columns:
        interval_summary = interval_summary.sort_values("CWE", ascending=False)
        _add_delta(interval_summary, "CWE", "baseline", lower_is_better=False)
    if "PICP" in interval_summary.columns:
        _add_delta(interval_summary, "PICP", "baseline", lower_is_better=False)

    interval_summary.reset_index().to_csv(results_dir / "calib_interval_summary.csv", index=False)
    log.info(
        "Interval summary saved  path=%s\n%s",
        results_dir / "calib_interval_summary.csv",
        interval_summary.to_string(),
    )

    cond_index = interval_summary.index.get_level_values("condition")
    for cond in CALIB_CONDITIONS:
        if cond.name in cond_index.values:
            target = 1 - cond.alpha
            grp = _condition_group(cond.name)
            achieved = (
                interval_summary.loc[(grp, cond.name), "PICP"] if "PICP" in interval_summary.columns else float("nan")
            )
            log.info(
                "Coverage  condition=%-24s  target=%.2f  achieved=%.4f  %s",
                cond.name,
                target,
                achieved,
                "OK" if achieved >= target - 0.01 else "MISS",
            )

    _save_plots(results_dir)
    return point_summary, interval_summary


def _save_plots(root: Path) -> None:
    try:
        from twiga.core.plot.evaluation import save_ablation_plots  # noqa: PLC0415

        save_ablation_plots(root, prefix=_EXPERIMENT_PREFIX, name=_EXPERIMENT_NAME, reference=_REFERENCE_CONDITION)
        log.info("Ablation figures saved  dir=%s", root.parent / "figures")
    except Exception:
        log.warning("Figure export skipped (non-critical)", exc_info=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_for_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    ds = DATASET_CONFIGS[dataset_name]
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        dataset_name=ds.name,
        data_path=str(thesis_data_dir()),
        train_start=ds.train_start,
        train_end=ds.train_end,
        period=ds.period,
        input_window_size=ds.input_window_size,
        forecast_horizon=ds.forecast_horizon,
        window_stride=ds.window_stride,
        out_activation=ds.out_activation,
        epochs=args.epochs,
        split_freq="months",
        metrics=["mae", "rmse", "picp", "nmpi", "cwe", "winkler_score", "ace"],
        log_file=str(output_dir / "logs" / "experiment.log"),
    )

    configure(level=cfg.log_level, log_file=cfg.log_file)
    setup_environment(cfg)

    with LogContext(log, experiment=f"CRC Calib — {dataset_name}"):
        (
            data,
            target_series,
            calendar_variables,
            exogenous_features,
            lags,
            latitude,
            longitude,
            lookback,
        ) = load_and_preprocess(cfg)

        data_config = build_data_config(
            cfg,
            target_series,
            lags,
            calendar_variables,
            exogenous_features,
            latitude,
            longitude,
            lookback_window_size=lookback,
        )

        backbone_params = _load_backbone_params(ds.name)

        conditions_to_run = [_CONDITION_MAP[n] for n in args.condition] if args.condition else CALIB_CONDITIONS

        all_point: list[pd.DataFrame] = []
        all_interval: list[pd.DataFrame] = []
        for condition in conditions_to_run:
            metrics_df, interval_df = run_condition(
                condition=condition,
                data=data,
                data_config=data_config,
                backbone_params=backbone_params,
                cfg=cfg,
                output_dir=output_dir,
                num_splits=args.num_splits,
            )
            all_point.append(metrics_df)
            if interval_df is not None:
                all_interval.append(interval_df)

        point_summary, interval_summary = summarise_calib(all_point, all_interval, output_dir)
        log.info("Results saved to %s", output_dir)

        print("\n=== Point Summary ===")
        print(point_summary.to_string())
        print("\n=== Interval Summary ===")
        print(interval_summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="CRC calibration improvement experiment")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["MLVS-PT"],
        choices=[*DATASET_CONFIGS, "all"],
        help="Dataset(s) to evaluate, or 'all' for every dataset in DATASET_CONFIGS (default: MLVS-PT)",
    )
    parser.add_argument(
        "--condition",
        nargs="+",
        choices=list(_CONDITION_MAP),
        default=None,
        help="Condition(s) to run (default: all four)",
    )
    parser.add_argument("--num-splits", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs per fold")
    parser.add_argument("--output-dir", default="../experiments/crc_calib_improvement")
    args = parser.parse_args()

    dataset_names = list(DATASET_CONFIGS) if "all" in args.dataset else args.dataset
    for dataset_name in dataset_names:
        run_for_dataset(args, dataset_name)


if __name__ == "__main__":
    main()
