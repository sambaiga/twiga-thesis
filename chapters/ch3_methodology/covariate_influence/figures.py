"""Chapter 3, RQ2: four-condition NRMSE/CWE bars for the covariate ablation.

Run:  uv run python -c "from chapters.ch3_methodology.covariate_influence.figures import render_all; render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from chapters.ch3_methodology.covariate_influence.analysis import LOWER_BETTER, METRIC_COLS, load_results
from chapters.ch3_methodology.covariate_influence.experiment import DATASETS
from twiga_thesis.analysis import fold_level
from twiga_thesis.plotting import comparison_plots

OUT = Path(__file__).resolve().parent


def covariate_bars(results, dataset: str) -> dict[str, plt.Figure]:
    """Grouped bar of NRMSE/CWE/PICP/nMPI, one bar per covariate condition."""
    folds = fold_level(results, dataset, "covariate_influence", METRIC_COLS)
    return comparison_plots.plot_metric_bars(
        folds,
        dataset=dataset,
        group="covariate_influence",
        reference="ghi_temp_time",
        metric_cols=METRIC_COLS,
        lower_better=LOWER_BETTER,
    )


def render_all(out: Path | None = None) -> None:
    """Build and save the bar figures for every dataset present in the results."""
    out_dir = out or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results()
    for dataset in DATASETS:
        if dataset not in results["dataset"].unique():
            continue
        figures = covariate_bars(results, dataset)
        comparison_plots.save_metric_figures(figures, out_dir, stem=f"covariate_influence_{dataset}", kind="bar")


if __name__ == "__main__":
    render_all()
