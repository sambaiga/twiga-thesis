r"""Good/bad-day forecast panels for Chapter 7's architectural evaluation and probabilistic benchmark.

The Chapter 6 convention (``ch7_multitarget/shared/ch6_forecast_examples.py``): one shared best
and one shared worst calendar day, picked once from the mean daily NRMSE pooled over every condition
that shares the (date, fold), so every panel in a figure shows the same day. Panels use
``experiment/common/forecast_panels.draw`` (observed hollow markers, dashed median, interval band).

Reads each condition's saved ``predictions.parquet``/``metrics.parquet`` under
``<output-root>/<experiment>/<dataset>/<condition>/``; nothing is retrained. ``predictions.parquet`` repeats
every timestamp many times (2,304x on MLVS-PT), so the day is read with a row-group filter and de-duplicated.

Usage::

    uv run python experiment/ch6_conformal/shared/forecast_examples.py \
        --experiment probabilistic_benchmark --dataset MLVS-UK
    uv run python experiment/ch6_conformal/shared/forecast_examples.py \
        --experiment architectural_evaluation --dataset all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[1]))  # experiment/ (thesis_figures, common)
sys.path.insert(0, str(script_dir.parent))  # chapter directory (shared.*)
sys.path.insert(0, str(script_dir.parents[1] / "common"))

from experiment.common.comparison_plots import head_label  # noqa: E402
from forecast_panels import draw  # noqa: E402
from shared.ch7_analyse_results import EXPERIMENTS  # noqa: E402
from shared.protocol import CHAPTER_DATASETS, DEFAULT_OUTPUT_ROOT, experiment_output_dir  # noqa: E402
from thesis_figures.style import configure, panel, tex_safe  # noqa: E402
from twiga.core.utils import slugify  # noqa: E402

_FAN_LEVELS = np.array([0.05, 0.5, 0.95])


def condition_dir(experiment_dir: Path, dataset: str, condition: str) -> Path | None:
    """A condition's artifact directory: the latest ExperimentEngine run first, then the earlier flat layout.

    Engine layout: ``<experiment_dir>/<spec slug>/runs/<run id>/conditions/<dataset slug>/<group>/<condition>``.
    Earlier layout: ``<experiment_dir>/<dataset>/<condition>``, under whichever dataset-directory casing exists.
    """
    pattern = f"*/runs/*/conditions/{slugify(dataset)}/*/{condition}/metrics.parquet"
    engine_runs = sorted(experiment_dir.glob(pattern), reverse=True)
    if engine_runs:
        return engine_runs[0].parent
    for name in (dataset.lower(), dataset):
        path = experiment_dir / name / condition
        if (path / "metrics.parquet").is_file():
            return path
    return None


def daily_error(metrics: pd.DataFrame) -> pd.Series:
    """Mean NRMSE per (calendar date, fold)."""
    frame = metrics[["timestamp", "Folds", "nrmse"]].copy()
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    return frame.groupby(["date", "Folds"])["nrmse"].mean()


def best_interval_day(metrics: pd.DataFrame, candidates: pd.Index) -> tuple[pd.Timestamp, int]:
    """Highest daily CWE among ``candidates`` (date, fold), ties broken by the lowest daily Winkler score.

    Daily CWE saturates at 1 on many well-covered days, so the Winkler score picks the sharpest of them. The
    ``cwe_stratified`` column is a fold-level value repeated on every day and cannot rank days.
    """
    frame = metrics[["timestamp", "Folds", "cwe", "winkler_score"]].copy()
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    daily = frame.groupby(["date", "Folds"])[["cwe", "winkler_score"]].mean().reindex(candidates).dropna()
    return daily.sort_values(["cwe", "winkler_score"], ascending=[False, True]).index[0]


def pick_days(daily: dict[str, pd.Series]) -> tuple[tuple[pd.Timestamp, int], tuple[pd.Timestamp, int]]:
    """Shared best/worst (date, fold): mean daily NRMSE over conditions, restricted to keys every condition has."""
    pooled = pd.concat(daily, axis=1).dropna()
    if pooled.empty:
        raise RuntimeError("no (date, fold) pair is shared by every requested condition")
    mean_error = pooled.mean(axis=1)
    return mean_error.idxmin(), mean_error.idxmax()


def day_frame(path: Path, date: pd.Timestamp, fold: int) -> pd.DataFrame:
    """One condition's forecast for one day, read through a row-group filter and de-duplicated."""
    day = pd.read_parquet(
        path / "predictions.parquet",
        filters=[("Folds", "==", int(fold)), ("timestamp", ">=", date), ("timestamp", "<", date + pd.Timedelta("1D"))],
    )
    if "lower" in day.columns:  # engine runs keep point and interval predictions; plot the interval rows
        day = day.dropna(subset=["lower", "upper"])
    return day.drop_duplicates("timestamp").sort_values("timestamp")


def day_metric(metrics: pd.DataFrame, date: pd.Timestamp, fold: int, column: str) -> float:
    mask = (metrics["Folds"] == fold) & (pd.to_datetime(metrics["timestamp"]).dt.normalize() == date)
    return float(metrics.loc[mask, column].mean()) if mask.any() else float("nan")


def draw_panel(ax, day: pd.DataFrame, label: str) -> None:
    """One condition's panel; the condition name is the column label, per-day metrics go to the summary CSV."""
    q = np.stack([day["lower"].to_numpy(), day["forecast"].to_numpy(), day["upper"].to_numpy()])
    draw(ax, day["timestamp"].to_numpy(), day["Actual"].to_numpy(), q, _FAN_LEVELS, t_i=0)
    ax.set_title(tex_safe(label), fontsize=9)
    ax.set_ylabel(tex_safe(str(day["target"].iloc[0])), fontsize=8)
    ax.tick_params(labelsize=7)


def run(experiment: str, dataset: str, output_root: Path, conditions: list[str] | None, out_dir: Path) -> pd.DataFrame:
    experiment_dir = experiment_output_dir(experiment, output_root)
    conditions = conditions or EXPERIMENTS[experiment]["interval_conditions"]
    paths = {c: p for c in conditions if (p := condition_dir(experiment_dir, dataset, c)) is not None}
    missing = [c for c in conditions if c not in paths]
    if missing:
        print(f"[{dataset}] no completed backtest for {missing} (skipped)")
    if len(paths) < 2:
        print(f"[skip] {experiment}/{dataset}: fewer than two conditions available")
        return pd.DataFrame()

    metrics = {c: pd.read_parquet(p / "metrics.parquet") for c, p in paths.items()}
    good, bad = pick_days({c: daily_error(m) for c, m in metrics.items()})
    print(f"[{dataset}] best day {good[0]:%Y-%m-%d} fold {good[1]}; worst day {bad[0]:%Y-%m-%d} fold {bad[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, (date, fold) in (("good", good), ("bad", bad)):
        ncol = min(len(paths), 3)
        nrow = -(-len(paths) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=panel(ncol, aspect=0.75), squeeze=False)
        flat = axes.flatten()
        for ax, (condition, path) in zip(flat, paths.items(), strict=False):
            m = metrics[condition]
            stats = {k: day_metric(m, date, fold, k) for k in ("nrmse", "winkler_score", "picp")}
            rows.append({"day": label, "date": date.date(), "fold": fold, "condition": condition, **stats})
            draw_panel(ax, day_frame(path, date, fold), head_label(condition))
        for ax in flat[len(paths) :]:
            ax.axis("off")
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{dataset}_{experiment}_{label}_day.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{dataset}_{experiment}_forecast_examples.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", nargs="+", default=["all"], choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--dataset", nargs="+", default=["all"], choices=[*CHAPTER_DATASETS, "all"])
    ap.add_argument("--conditions", nargs="+", default=None, help="defaults to the experiment's interval conditions")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    configure(latex=not args.no_latex)
    out_dir = args.output_root / "analysis" / "figures" / "forecast_examples"
    for experiment in list(EXPERIMENTS) if "all" in args.experiment else args.experiment:
        for dataset in CHAPTER_DATASETS if "all" in args.dataset else args.dataset:
            run(experiment, dataset, args.output_root, args.conditions, out_dir)


if __name__ == "__main__":
    main()
