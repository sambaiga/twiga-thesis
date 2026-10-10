"""Chapter 4 RQ4 analysis — accuracy, significance, cost, and seasonal breakdown.

Reads `benchmark.py`'s own saved CSVs (`results/metrics/backtest_metrics.csv`,
`results/predictions/backtest_predictions.csv`, and
`results/metrics/backtest_train_times.csv`) and reuses the shared statistics
layer (`twiga_thesis.analysis.metric_table`/`dm_and_mcs`) that every other
chapter's engine-based scripts already use, rather than reimplementing
aggregation or significance testing here.

`benchmark.py` is a hand-rolled script, not built on `twiga.experiment`'s
`ExperimentSpec`/`Condition`/`ExperimentEngine` pattern, so its result frames
use `Model`/`Folds` columns rather than the engine's `condition`/`group`/
`dataset` schema `twiga_thesis.analysis.fold_level` expects. This module
bridges that gap locally (`_as_fold_frame`) instead of reshaping
`benchmark.py` itself onto the engine pattern, which is explicitly out of
scope for this pass.

Covers (`Results-analysis.ipynb` cells 78+, confirmed as Chapter 4's own):
per-model fold-then-mean accuracy, DM/MCS significance, inference time, the
accuracy-vs-inference-time trade-off, and the seasonal NRMSE breakdown RQ4
itself reports (`subfig:benchmark_PT`/`benchmark_UK`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from twiga.core.metrics.point import nrmse

from twiga_thesis.analysis import dm_and_mcs, metric_table

# Local, proportionate metric-list constant (ch6_conformal/shared/metrics.py's naming convention,
# this chapter's own list, not imported from chapter 6/7): the same five point metrics
# benchmark.py itself already requests from TwigaForecaster.backtesting().
POINT_METRICS = ["rmse", "mae", "corr", "wmape", "smape"]
LOWER_BETTER = {"rmse", "mae", "wmape", "smape"}
HIGHER_BETTER = {"corr"}

# Meteorological-calendar seasons (Northern-hemisphere convention, matching MLVS-PT's Portugal
# site and the paper's own seasonal breakdown): month -> season label.
_SEASON_BY_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def load_benchmark_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load `benchmark.py`'s own saved metrics, predictions, and train-time CSVs.

    Args:
        output_dir: `benchmark.py`'s `OUTPUT_DIR` (the run this analysis reports on).

    Returns:
        A tuple `(metrics, predictions, train_times)`.

    Raises:
        FileNotFoundError: If `benchmark.py` has not been run from this `output_dir` yet.
    """
    metrics_path = output_dir / "results" / "metrics" / "backtest_metrics.csv"
    predictions_path = output_dir / "results" / "predictions" / "backtest_predictions.csv"
    train_times_path = output_dir / "results" / "metrics" / "backtest_train_times.csv"
    for path in (metrics_path, predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run benchmark.py first.")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    train_times = pd.read_csv(train_times_path) if train_times_path.exists() else pd.DataFrame()
    return metrics, predictions, train_times


def _as_fold_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """`benchmark.py`'s `Model`/`Folds` metrics frame, renamed to the shared layer's `condition` column."""
    return metrics.rename(columns={"Model": "condition"})


def accuracy_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-model fold-then-mean accuracy across `POINT_METRICS`, ranked by weighted Borda score.

    Args:
        metrics: `benchmark.py`'s own per-fold metrics frame (`Model`, `Folds`, point-metric columns).

    Returns:
        One row per model: means, std, per-metric mean rank, and aggregate ranks
        (`twiga_thesis.analysis.metric_table`).
    """
    available = [metric for metric in POINT_METRICS if metric in metrics.columns]
    return metric_table(_as_fold_frame(metrics), available, LOWER_BETTER, HIGHER_BETTER, target_values={})


def significance_table(metrics: pd.DataFrame, metric: str = "rmse", alpha: float = 0.05):
    """Diebold-Mariano/MCS significance on one point-accuracy metric across models.

    Args:
        metrics: `benchmark.py`'s own per-fold metrics frame.
        metric: Loss column to test (`"rmse"` by default; RQ4 reports on NRMSE/RMSE, not CWE).
        alpha: Significance level for both the DM table and the MCS.

    Returns:
        `(dm_table, p_matrix, model_confidence_set)`, as `twiga_thesis.analysis.dm_and_mcs`.
    """
    return dm_and_mcs(_as_fold_frame(metrics), alpha=alpha, metric=metric, fold_column="Folds")


def inference_time_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std inference wall-time per model, from the `inference-time` column `TwigaForecaster` already logs.

    `TwigaForecaster.backtesting()` calls `evaluate_point_forecast()`, which (via
    `BaseForecaster._evaluate_core`) sets `inference-time` once per model per fold — the forecast
    wall-clock for that fold's whole test set; this is the only per-fold cost figure the engine
    persists (training wall-time is not: see `benchmark.measure_train_times`'s own docstring), so
    it is read directly rather than re-timed here.

    Args:
        metrics: `benchmark.py`'s own per-fold metrics frame (must include `inference-time`).

    Returns:
        Columns `Model`, `inference_time_mean_s`, `inference_time_std_s`.

    Raises:
        KeyError: If `metrics` has no `inference-time` column.
    """
    if "inference-time" not in metrics.columns:
        raise KeyError("metrics is missing 'inference-time'; re-run benchmark.py to regenerate it")
    summary = metrics.groupby("Model")["inference-time"].agg(["mean", "std"]).reset_index()
    return summary.rename(columns={"mean": "inference_time_mean_s", "std": "inference_time_std_s"})


def accuracy_vs_cost_table(
    metrics: pd.DataFrame, train_times: pd.DataFrame, accuracy_metric: str = "rmse"
) -> pd.DataFrame:
    """One row per model: mean accuracy, mean inference time, and train time, for the cost/accuracy trade-off figure.

    Args:
        metrics: `benchmark.py`'s own per-fold metrics frame.
        train_times: `capture_train_times()`'s output (`Model`, `train_time_s`); may be empty if
            `benchmark.py` was run before that column was added, in which case `train_time_s` is
            all-`NaN` rather than raising.
        accuracy_metric: Point-accuracy column to report (`"rmse"` by default).

    Returns:
        Columns `Model`, `accuracy_metric`'s mean, `inference_time_mean_s`, `train_time_s`.
    """
    accuracy = metrics.groupby("Model")[accuracy_metric].mean().reset_index()
    inference = inference_time_table(metrics)[["Model", "inference_time_mean_s"]]
    table = accuracy.merge(inference, on="Model", how="left")
    if train_times.empty:
        table["train_time_s"] = np.nan
    else:
        table = table.merge(train_times, on="Model", how="left")
    return table


def seasonal_nrmse_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """NRMSE per model, per meteorological season (`subfig:benchmark_PT`/`benchmark_UK`'s own breakdown).

    No existing `twiga`/`twiga_thesis` helper stratifies a point metric by calendar season (`get_interval_metrics`'s
    own stratification is for interval metrics, by lead time or an explicit array, not a month-derived season label),
    so this groups by `_SEASON_BY_MONTH` and calls `twiga.core.metrics.point.nrmse` within each group directly.

    Args:
        predictions: `benchmark.py`'s own predictions frame (`Model`, `timestamp`, `Actual`, `forecast`).

    Returns:
        Columns `Model`, `season`, `nrmse`, `n`.

    Raises:
        KeyError: If `predictions` is missing `timestamp`, `Actual`, or `forecast`.
    """
    required = {"timestamp", "Actual", "forecast", "Model"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions is missing columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["season"] = frame["timestamp"].dt.month.map(_SEASON_BY_MONTH)

    rows = []
    for (model, season), group in frame.groupby(["Model", "season"]):
        rows.append(
            {
                "Model": model,
                "season": season,
                "nrmse": float(nrmse(group["Actual"].to_numpy(), group["forecast"].to_numpy(), axis=None)),
                "n": len(group),
            }
        )
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    table = pd.DataFrame(rows)
    table["season"] = pd.Categorical(table["season"], categories=season_order, ordered=True)
    return table.sort_values(["Model", "season"]).reset_index(drop=True)
