"""Chapter 3: figures for the protocol comparison (single split vs expanding-window backtest).

Two figures, each built with one of the thesis's two established conventions (never raw matplotlib/seaborn
defaults): a model x protocol grouped bar (``twiga_thesis.plotting.comparison_plots``) and a test-time-vs-NRMSE
efficiency scatter (``twiga_thesis.plotting.line_style``).

Run:  uv run python -c "from chapters.ch3_methodology.best_practices.figures import render_all; render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chapters.ch3_methodology.best_practices.conditions import POINT_METRICS
from chapters.ch3_methodology.best_practices.protocol import DEFAULT_OUTPUT_ROOT, experiment_output_dir
from twiga_thesis.plotting import comparison_plots, line_style, style, tokens

OUT = Path(__file__).resolve().parent


def _combined_folds(output_root: Path | None = None) -> pd.DataFrame:
    """Both protocols' results CSVs concatenated, with ``condition`` relabelled ``"<condition> (<protocol>)"``."""
    root = output_root or DEFAULT_OUTPUT_ROOT
    frames = []
    for protocol in ("single_split", "expanding"):
        path = experiment_output_dir(protocol, root) / f"results_{protocol}.csv"
        frame = pd.read_csv(path)
        frame["condition"] = frame["condition"] + f" ({protocol})"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def model_by_protocol_bars(fold_frame: pd.DataFrame, dataset: str) -> dict[str, plt.Figure]:
    """Grouped bar of each point metric, one bar per (model, protocol) combination."""
    return comparison_plots.plot_metric_bars(
        fold_frame,
        dataset=dataset,
        group="baselines",
        reference="seasonal_naive (expanding)",
        metric_cols=POINT_METRICS,
        lower_better={"nrmse", "mae", "mape", "smape"},
    )


def test_time_vs_nrmse(fold_frame: pd.DataFrame) -> plt.Figure:
    """Scatter of per-condition mean inference time against mean NRMSE, one marker per (model, protocol)."""
    line_style.configure(latex=False)
    fig, ax = line_style.new_panel()
    summary = fold_frame.groupby("condition")[["inference_time", "nrmse"]].mean().reset_index()
    colors = comparison_plots.thesis_palette(len(summary))
    for (_, row), color in zip(summary.iterrows(), colors, strict=True):
        ax.scatter(row["inference_time"], row["nrmse"], color=color, s=18, label=row["condition"])
    ax.set_xlabel("Mean inference time (s)")
    ax.set_ylabel("Mean NRMSE")
    line_style.set_y_axis(ax, summary["nrmse"].to_numpy(), from_zero=True)
    line_style.legend_below(ax, ncol=2)
    style.format_axes(ax)
    return fig


def render_all(out: Path | None = None, output_root: Path | None = None) -> None:
    """Build and save both figures for every dataset in the combined results."""
    out_dir = out or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = _combined_folds(output_root)
    for dataset in fold_frame["dataset"].unique():
        scoped = fold_frame[fold_frame["dataset"] == dataset]
        bars = model_by_protocol_bars(scoped, dataset)
        comparison_plots.save_metric_figures(bars, out_dir, stem=f"best_practices_{dataset}", kind="bar")
        if "inference_time" in scoped.columns:
            scatter_fig = test_time_vs_nrmse(scoped)
            scatter_fig.savefig(out_dir / f"best_practices_{dataset}_efficiency_scatter.pdf", bbox_inches="tight")
            plt.close(scatter_fig)


if __name__ == "__main__":
    render_all()
