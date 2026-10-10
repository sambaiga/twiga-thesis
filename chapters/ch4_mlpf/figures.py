"""Chapter 4 RQ4 figures — accuracy-vs-inference-time trade-off and seasonal NRMSE breakdown.

Built on the two established thesis figure conventions
(`twiga_thesis.plotting.line_style` for the cost/accuracy scatter and seasonal
line plot, `twiga_thesis.plotting.tokens` for the palette) rather than ad-hoc
matplotlib, matching every other chapter's figure scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from twiga_thesis.plotting import line_style, tokens


def plot_accuracy_vs_inference_time(table: pd.DataFrame, accuracy_metric: str = "rmse") -> plt.Figure:
    """Scatter of inference time against point-forecast accuracy, one marker per model.

    The real version of the illustrative schematic already in this chapter's
    own README (`subfig` the paper reports qualitatively as "MLPF's inference
    time comparable to S-Naive" and "least training time").

    Args:
        table: `analysis.accuracy_vs_cost_table`'s output (`Model`,
            `accuracy_metric`, `inference_time_mean_s`, `train_time_s`).
        accuracy_metric: Column name of the accuracy metric plotted on the y-axis.

    Returns:
        The figure; save with `line_style.save`.
    """
    line_style.configure(latex=False)
    fig, ax = line_style.new_panel()
    colors = tokens.PALETTE
    for i, row in table.reset_index(drop=True).iterrows():
        color = colors[i % len(colors)]
        ax.plot(
            row["inference_time_mean_s"],
            row[accuracy_metric],
            "o",
            color=color,
            mec=color,
            mfc=color,
            ms=5,
            label=str(row["Model"]),
        )
    ax.set_xlabel("Inference time (s, mean across folds)")
    ax.set_ylabel(accuracy_metric.upper())
    line_style.set_y_axis(ax, table[accuracy_metric], from_zero=True)
    line_style.finish(ax)
    line_style.legend_below(ax, ncol=3)
    return fig


def plot_seasonal_nrmse(seasonal_table: pd.DataFrame) -> plt.Figure:
    """NRMSE per season, one line per model (`subfig:benchmark_PT`/`benchmark_UK`'s own breakdown).

    Args:
        seasonal_table: `analysis.seasonal_nrmse_table`'s output (`Model`,
            `season`, `nrmse`), `season` already ordered Winter-Spring-Summer-Autumn.

    Returns:
        The figure; save with `line_style.save`.
    """
    line_style.configure(latex=False)
    fig, ax = line_style.new_panel()
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    colors = tokens.PALETTE
    for i, (model, group) in enumerate(seasonal_table.groupby("Model")):
        ordered = group.set_index("season").reindex(season_order)
        color = colors[i % len(colors)]
        ax.plot(season_order, ordered["nrmse"].to_numpy(), "-o", color=color, lw=line_style.LINE_WIDTH, label=model)
    ax.set_xlabel("Season")
    ax.set_ylabel("NRMSE")
    line_style.set_y_axis(ax, seasonal_table["nrmse"], from_zero=True)
    line_style.finish(ax)
    line_style.legend_below(ax, ncol=3)
    return fig


def save_all(
    accuracy_vs_cost: pd.DataFrame,
    seasonal_table: pd.DataFrame,
    figures_dir: Path,
    accuracy_metric: str = "rmse",
) -> None:
    """Build and save both figures to `figures_dir` (`<stem>.pdf` and `.png`).

    Args:
        accuracy_vs_cost: `analysis.accuracy_vs_cost_table`'s output.
        seasonal_table: `analysis.seasonal_nrmse_table`'s output.
        figures_dir: Directory to save into (created if missing).
        accuracy_metric: Accuracy column plotted against inference time.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    line_style.save(
        plot_accuracy_vs_inference_time(accuracy_vs_cost, accuracy_metric),
        figures_dir / "accuracy_vs_inference_time",
    )
    line_style.save(plot_seasonal_nrmse(seasonal_table), figures_dir / "seasonal_nrmse")
