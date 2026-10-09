r"""Analyze completed Chapter 5 runs and produce thesis-ready result figures.

For each dataset and experiment group, this script selects the latest run
covering that group, reports coverage and missing conditions, aggregates
metrics by fold, computes metric rankings and significance results, and saves
CSV tables and figures under ``results/analysis``. Generic aggregation,
significance, and comparison figures are provided by ``experiment.common``;
Chapter 5 retains its run selection, metric configuration, and timing
diagnostic.

Figures use the thesis style and include a multi-metric forest overview,
per-metric condition bars, a binary adjusted-p-value matrix, best/worst-day
comparison, and cross-dataset metric comparisons. The computation-time plot
is specific to the benchmark groups.

Usage:

    uv run python experiment/chapter-5-fpqr/analyse_results.py --group all
    uv run python experiment/chapter-5-fpqr/analyse_results.py --group benchmark benchmark_nhits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from experiment.common.comparison_plots import (
    bootstrap_ci as shared_bootstrap_ci,
    display_metric as shared_display_metric,
    display_name,
    head_label as shared_head_label,
    plot_dm_significance_matrix as shared_plot_dm_significance_matrix,
    plot_forest_grid,
    plot_good_bad_day as shared_plot_good_bad_day,
    plot_metric_bar_grid,
    plot_metric_bars,
    plot_normalized_comparison as shared_plot_normalized_comparison,
    plot_primary_metric_comparison as shared_plot_primary_metric_comparison,
    save_metric_figures,
)
from experiment.common.result_analysis import (
    dm_and_mcs as shared_dm_and_mcs,
    fold_level as aggregate_fold_level,
    kupiec_table as shared_kupiec_table,
    metric_table as build_metric_table,
)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from twiga.experiment.run_catalog import (
    discover_runs as _discover_experiment_runs,
    latest_run_for_group as _latest_experiment_run,
    load_group_csv,
)

script_dir = Path(__file__).resolve().parent
RUNS_DIR = script_dir / "results" / "chapter_5_fpqr" / "runs"
OUT_DIR = script_dir / "results" / "analysis"
FIG_DIR = OUT_DIR / "figures"

sys.path.insert(0, str(script_dir))
from fpqr_experiment import SPEC  # noqa: E402

sys.path.insert(0, str(script_dir.parent))
from thesis_figures import (
    style as S,  # noqa: E402
    tokens as T,  # noqa: E402
)

METRIC_COLS = ["nrmse", "crps", "winkler_score", "cwe_stratified", "picp", "nmpi", "calibration_error"]
PRIMARY_METRIC_COLS = ["nrmse", "crps", "winkler_score", "cwe_stratified"]
LOWER_BETTER = {"nrmse", "crps", "winkler_score", "nmpi", "calibration_error"}
HIGHER_BETTER = {"cwe_stratified"}
NOMINAL_PICP = 0.90  # Chapter 5's stated nominal coverage level, 1 - alpha = 0.90.

ALL_GROUPS = list(SPEC.reference_conditions)
TIMING_GROUPS = {"benchmark", "benchmark_nhits"}

# ── Display names: thesis acronym casing (Thesis-MSc-Acronyms.tex), title case otherwise ──
METRIC_DISPLAY = {
    "nrmse": "NRMSE",
    "crps": "CRPS",
    "winkler_score": "WS",
    "cwe_stratified": "CWE (Stratified)",
    "picp": "PICP",
    "nmpi": "NMPI",
    "calibration_error": "Calibration Error",
}


def display_metric(metric: str) -> str:
    """Metric column name -> thesis-cased display label, falling back to ``display_name``."""
    return shared_display_metric(metric, METRIC_DISPLAY)


# Condition-name prefixes that name the backbone a benchmark/benchmark_nhits
# figure is already scoped to (the figure's own title/caption states the
# backbone), so repeating it on every bar/tick label is redundant.
BACKBONE_PREFIXES = ("nhits_",)


def head_label(condition: str) -> str:
    """Format a head label without repeating the benchmark's backbone prefix."""
    return shared_head_label(condition, backbone_prefixes=BACKBONE_PREFIXES)


# ── Run discovery ────────────────────────────────────────────────────────────


def discover_runs() -> list[dict]:
    """Every run under ``RUNS_DIR``, read from its own ``meta.json``."""
    return _discover_experiment_runs(RUNS_DIR)


def latest_run_for_group(runs: list[dict], group: str) -> dict | None:
    """The most recently timestamped run (of ``runs``) that covers ``group``, or ``None``."""
    return _latest_experiment_run(runs, group)


def load_csv(groups: list[str], csv_name: str) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Concatenate one CSV (``ch5_fpqr_full.csv`` or ``..._representative.csv``) per group.

    Returns the concatenated frame and a coverage log of
    ``(group, run_dir_name, run_timestamp)`` for groups that were found, so
    callers can print which run each group came from before trusting any
    number derived from it.
    """
    full, coverage, missing = load_group_csv(RUNS_DIR, groups, csv_name)
    for issue in missing:
        print(f"[missing] {issue}")
    return full, coverage


def print_coverage(full: pd.DataFrame, coverage: list[tuple[str, str, str]]) -> None:
    """Print which run each requested group was read from, and its per-condition row counts."""
    print("\n=== Coverage: which run each group came from ===")
    for group, run_name, timestamp in coverage:
        print(f"  {group:20s} <- runs/{run_name}  ({timestamp})")

    print("\n=== Condition row counts (0 rows means never completed) ===")
    if full.empty:
        print("  (nothing loaded)")
        return
    counts = full.groupby(["dataset", "group", "condition"]).size()
    print(counts.to_string())


# ── Fold-level aggregation and ranking ──────────────────────────────────────


def fold_level(full: pd.DataFrame, dataset: str, group: str, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """Return one mean metric row per condition and fold for this dataset/group."""
    metric_cols = METRIC_COLS if metric_cols is None else metric_cols
    return aggregate_fold_level(full, dataset, group, metric_cols=metric_cols)


def metric_table(
    fold_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    lower_better: set[str] | None = None,
    higher_better: set[str] | None = None,
) -> pd.DataFrame:
    """Build Chapter 5's configured fold summary and diversity-weighted ranking."""
    metric_cols = METRIC_COLS if metric_cols is None else metric_cols
    lower_better = LOWER_BETTER if lower_better is None else lower_better
    higher_better = HIGHER_BETTER if higher_better is None else higher_better
    return build_metric_table(
        fold_df,
        metric_cols=metric_cols,
        lower_better=lower_better,
        higher_better=higher_better,
        target_values={"picp": NOMINAL_PICP},
    )


# ── Significance ─────────────────────────────────────────────────────────────


def dm_and_mcs(
    fold_df: pd.DataFrame, alpha: float = 0.05, metric: str = "winkler_score"
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Return pairwise adjusted significance, its matrix, and the MCS for a fold loss."""
    return shared_dm_and_mcs(fold_df, alpha=alpha, metric=metric)


def kupiec_table(full: pd.DataFrame, dataset: str, group: str, nominal_alpha: float = 1 - NOMINAL_PICP) -> pd.DataFrame:
    """Return approximate unconditional-coverage tests from fold PICP values."""
    return shared_kupiec_table(full, dataset, group, nominal_alpha=nominal_alpha)


# ── Figures ──────────────────────────────────────────────────────────────────


def plot_forest(
    fold_df: pd.DataFrame, dataset: str, group: str, reference: str, metric_cols: list[str] | None = None
) -> plt.Figure:
    """Bootstrap-CI forest plot, one row per condition per metric, against ``reference``.

    The reference condition is drawn in the palette's primary teal; every
    other condition in muted grey, so the comparison the plot exists to make
    -- everything else against the reference -- is the one colour marks,
    rather than every condition competing for attention equally.

    ``metric_cols`` defaults to this chapter's own ``METRIC_COLS`` -- pass a
    different list to reuse this function for a different chapter's metric
    set.
    """
    """Create the shared multi-metric forest overview for this chapter's metrics."""
    metric_cols = METRIC_COLS if metric_cols is None else metric_cols
    return plot_forest_grid(
        fold_df,
        dataset,
        group,
        reference,
        metric_cols,
        metric_display=METRIC_DISPLAY,
        backbone_prefixes=BACKBONE_PREFIXES,
    )


def plot_dm_significance_matrix(
    p_matrix: pd.DataFrame, dataset: str, group: str, alpha: float = 0.05
) -> plt.Figure | None:
    """Create the shared binary adjusted-p-value matrix."""
    return shared_plot_dm_significance_matrix(p_matrix, dataset, group, alpha=alpha)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, rng=None) -> tuple[float, float]:
    """Return the shared percentile-bootstrap confidence interval for a mean."""
    return shared_bootstrap_ci(values, n_boot=n_boot, ci=ci, rng=rng)


def fold_start_timestamps(run_dir: Path, dataset: str, group: str, condition: str) -> list[int]:
    """Unix start-timestamps embedded in each fold's lightning_logs directory name."""
    base = run_dir / "conditions" / dataset / group / condition / "logs" / condition
    if not base.exists():
        return []
    timestamps = []
    for log_dir in base.rglob("lightning_logs/*"):
        if not log_dir.is_dir():
            continue
        try:
            timestamps.append(int(log_dir.name.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return sorted(timestamps)


def plot_computation_time(
    full: pd.DataFrame, coverage: list[tuple[str, str, str]], dataset: str, group: str
) -> plt.Figure | None:
    """Inference time (from the CSV) and approximate training time (from log-dir timestamps)."""
    run_name = next((name for entry_group, name, _ in coverage if entry_group == group), None)
    if run_name is None:
        return None
    run_dir = RUNS_DIR / run_name

    group_rows = full[(full["dataset"] == dataset) & (full["group"] == group)]
    if group_rows.empty:
        return None

    inference = group_rows.groupby(["condition", "Folds"])["inference-time"].mean().groupby("condition").mean()

    training_rows = []
    for condition in group_rows["condition"].unique():
        timestamps = fold_start_timestamps(run_dir, dataset, group, condition)
        if len(timestamps) < 2:
            continue
        gaps = np.diff(timestamps)
        training_rows.append(
            {"condition": condition, "mean": gaps.mean(), "std": gaps.std(ddof=1) if len(gaps) > 1 else 0.0}
        )
    training = pd.DataFrame(training_rows).set_index("condition") if training_rows else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(8, 0.4 * len(inference) + 1.5))
    order = inference.sort_values().index
    axes[0].barh(range(len(order)), [inference[condition] for condition in order], color=T.INFO)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([S.tex_safe(head_label(condition)) for condition in order])
    axes[0].set_xlabel("Inference Time (s)")
    S.format_axes(axes[0])

    if not training.empty:
        training_order = [condition for condition in order if condition in training.index]
        axes[1].barh(
            range(len(training_order)),
            [training.loc[condition, "mean"] for condition in training_order],
            xerr=[training.loc[condition, "std"] for condition in training_order],
            color=T.HIGHLIGHT,
        )
        axes[1].set_yticks(range(len(training_order)))
        axes[1].set_yticklabels([S.tex_safe(head_label(condition)) for condition in training_order])
        axes[1].set_xlabel("Approx. Training Time per Fold (s)")
        S.format_axes(axes[1])
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No Timing Logs Found", ha="center")

    fig.tight_layout()
    return fig


def plot_good_bad_day(representative: pd.DataFrame, dataset: str, group: str) -> plt.Figure | None:
    """Create the shared best/worst-day comparison for Chapter 5 NRMSE."""
    return shared_plot_good_bad_day(representative, dataset, group, metric="nrmse")


def plot_normalized_comparison(
    full: pd.DataFrame, group: str, metrics: list[tuple[str, bool]] | None = None
) -> plt.Figure | None:
    """Create Chapter 5's shared normalized cross-dataset metric comparison."""
    metric_directions = metrics or [("winkler_score", True), ("crps", True), ("cwe_stratified", False)]
    return shared_plot_normalized_comparison(
        full,
        group,
        metrics=metric_directions,
        legend_title="Prob-Head:",
        metric_display=METRIC_DISPLAY,
    )


def plot_primary_metric_comparison(
    full: pd.DataFrame, group: str, primary_metric_cols: list[str] | None = None
) -> plt.Figure | None:
    """Create the shared raw-metric comparison across Chapter 5 datasets."""
    metrics = PRIMARY_METRIC_COLS if primary_metric_cols is None else primary_metric_cols
    return shared_plot_primary_metric_comparison(
        full,
        group,
        metrics,
        metric_display=METRIC_DISPLAY,
        log_metrics={"winkler_score"},
        legend_title="Prob-Head:",
        backbone_prefixes=BACKBONE_PREFIXES,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def analyse_group(
    full: pd.DataFrame,
    representative: pd.DataFrame,
    coverage: list[tuple[str, str, str]],
    dataset: str,
    group: str,
) -> None:
    """Print and save the metric table, DM/MCS significance, Kupiec test, and figures for one (dataset, group)."""
    fold_df = fold_level(full, dataset, group)
    if fold_df.empty:
        print(f"[skip] {dataset} / {group}: no rows")
        return
    if fold_df["condition"].nunique() < 2:
        print(f"[skip] {dataset} / {group}: only one condition, nothing to compare")
        return

    print(f"\n=== {dataset} / {group} ===")
    table = metric_table(fold_df)
    print(table.round(4).to_string(index=False))
    table.to_csv(OUT_DIR / f"{dataset}_{group}_metrics.csv", index=False)

    dm_table, p_matrix, mcs = dm_and_mcs(fold_df)
    print(dm_table.to_string(index=False))
    print(f"MCS (alpha=0.05): {mcs.included}")
    dm_table.to_csv(OUT_DIR / f"{dataset}_{group}_significance.csv", index=False)

    kupiec = kupiec_table(full, dataset, group)
    print(f"\nKupiec POF (unconditional coverage, alpha={1 - NOMINAL_PICP:.2f}):")
    print(kupiec.round(4).to_string(index=False))
    kupiec.to_csv(OUT_DIR / f"{dataset}_{group}_kupiec.csv", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reference = SPEC.reference_conditions.get(group, fold_df["condition"].iloc[0])
    fig = plot_forest_grid(
        fold_df,
        dataset,
        group,
        reference,
        METRIC_COLS,
        metric_display=METRIC_DISPLAY,
        backbone_prefixes=BACKBONE_PREFIXES,
    )
    fig.savefig(FIG_DIR / f"{dataset}_{group}_forest.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metric_figures = plot_metric_bars(
        fold_df,
        dataset,
        group,
        reference,
        METRIC_COLS,
        metric_display=METRIC_DISPLAY,
        lower_better=LOWER_BETTER,
        higher_better=HIGHER_BETTER,
        target_values={"picp": NOMINAL_PICP},
        backbone_prefixes=BACKBONE_PREFIXES,
    )
    save_metric_figures(metric_figures, FIG_DIR, f"{dataset}_{group}", kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        dataset,
        group,
        reference,
        METRIC_COLS,
        metric_display=METRIC_DISPLAY,
        lower_better=LOWER_BETTER,
        higher_better=HIGHER_BETTER,
        target_values={"picp": NOMINAL_PICP},
        backbone_prefixes=BACKBONE_PREFIXES,
    )
    if metric_grid is not None:
        metric_grid.savefig(FIG_DIR / f"{dataset}_{group}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    fig = shared_plot_dm_significance_matrix(p_matrix, dataset, group)
    if fig is not None:
        fig.savefig(FIG_DIR / f"{dataset}_{group}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if group in TIMING_GROUPS:
        fig = plot_computation_time(full, coverage, dataset, group)
        if fig is not None:
            fig.savefig(FIG_DIR / f"{dataset}_{group}_computation_time.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

    fig = plot_good_bad_day(representative, dataset, group)
    if fig is not None:
        fig.savefig(FIG_DIR / f"{dataset}_{group}_good_bad_day.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  figures written under {FIG_DIR}")


def main() -> None:
    """Parse CLI arguments, load every requested group/dataset, and run the full analysis."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", default=["all"], choices=[*ALL_GROUPS, "all"])
    parser.add_argument("--dataset", nargs="+", default=["all"], choices=["MLVS-PT", "MLVS-UK", "all"])
    parser.add_argument(
        "--no-latex", action="store_true", help="fall back to matplotlib's own text renderer (no LaTeX toolchain)"
    )
    args = parser.parse_args()

    S.configure(latex=not args.no_latex)

    groups = ALL_GROUPS if "all" in args.group else args.group
    datasets = ["MLVS-PT", "MLVS-UK"] if "all" in args.dataset else args.dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full, coverage = load_csv(groups, "ch5_fpqr_full.csv")
    representative, _ = load_csv(groups, "ch5_fpqr_representative.csv")
    print_coverage(full, coverage)

    if full.empty:
        print("Nothing to analyse.")
        return

    for dataset in datasets:
        for group in groups:
            analyse_group(full, representative, coverage, dataset, group)

    # Cross-dataset comparison figures: one per group (not per dataset), only
    # meaningful when both datasets were requested.
    if set(datasets) == {"MLVS-PT", "MLVS-UK"}:
        for group in groups:
            fig = plot_primary_metric_comparison(full, group)
            if fig is not None:
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                out = FIG_DIR / f"{group}_primary_metrics.pdf"
                fig.savefig(out, dpi=480, bbox_inches="tight")
                plt.close(fig)
                print(f"wrote {out}")

            if group not in TIMING_GROUPS:
                continue
            fig = plot_normalized_comparison(full, group)
            if fig is not None:
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                out = FIG_DIR / f"{group}_normalised_comparison.pdf"
                fig.savefig(out, dpi=480, bbox_inches="tight")
                plt.close(fig)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
