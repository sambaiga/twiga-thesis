r"""Forest and interaction figures for two Experiment-1 findings.

The dominant training-schedule effect, and the schedule x conformal-score
additivity check, are the two that most need visual, not just prose,
support.

`fig_ch7_calibration_timing.pdf` reports the training-schedule effect as a
bar chart of percentage deltas with no uncertainty shown -- fine for a group
where the effect is real and large, but it asserts precision the underlying
data can support directly. `fig_ch7_training_schedule_forest.pdf` gives the
same comparison as a bootstrap-CI forest plot, so the reader sees the
interval as well as the point estimate, following the same convention
`chapter-5-fpqr/analyse_results.py`'s own `plot_forest` uses (reference
condition in the palette's primary teal, everything else in muted grey).

The schedule x conformal-score interaction (`training_x_conformal`,
`chap:7-expriment-protol`'s promised additivity check) has never had a
figure at all -- only the ANOVA's F-statistic and p-value in prose
(`design_ablation_significance.py`'s `_factorial_interaction`). An
interaction plot is the standard way to show this directly: parallel lines
across training schedules visually confirm additivity (MLVS-PT, this
group's ANOVA finds no significant interaction); crossing or diverging
lines show the interaction the ANOVA finds significant on SPS-UK.

Run:  uv run python experiment/ch6_conformal/design_ablation/design_ablation_figures.py \
        --run-dir <mlpgam_design_ablation.py's output dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiment/ (thesis_figures, common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chapter directory (shared.*)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from design_ablation_significance import _fold_level  # noqa: E402
from thesis_figures import (
    style as S,  # noqa: E402
    tokens as T,  # noqa: E402
)
from twiga.core.plot import bootstrap_ci  # noqa: E402

DATASETS = ["MLVS-PT", "MLVS-UK"]
TRAINING_MODE_REF = "two_stage_batch"
TRAINING_MODE_ORDER = ["joint", "two_stage_ep_05", "two_stage_ep_10", "two_stage_ep_20", "two_stage_batch"]
TRAINING_MODE_LABELS = {
    "joint": "Joint",
    "two_stage_ep_05": "Two-stage (5%)",
    "two_stage_ep_10": "Two-stage (10%)",
    "two_stage_ep_20": "Two-stage (20%)",
    "two_stage_batch": "Per-batch\n(proposed)",
}
SCORE_LABELS = {"res": "Absolute residual", "sign": "Signed residual", "scp": "Split conformal"}
SCORE_COLORS = {"res": T.PRIMARY, "sign": T.HIGHLIGHT, "scp": T.TEXT_MUTED}

# training_x_conformal's own condition names abbreviate the schedule
# differently from training_mode's (`ep_05` not `two_stage_ep_05`) --
# FACTORIAL_TRAINING_MODES in design_ablation/conditions.py, not TRAINING_MODE_ORDER.
INTERACTION_MODE_ORDER = ["joint", "ep_05", "ep_10", "ep_20", "two_stage_batch"]
INTERACTION_MODE_LABELS = {
    "joint": TRAINING_MODE_LABELS["joint"],
    "ep_05": TRAINING_MODE_LABELS["two_stage_ep_05"],
    "ep_10": TRAINING_MODE_LABELS["two_stage_ep_10"],
    "ep_20": TRAINING_MODE_LABELS["two_stage_ep_20"],
    "two_stage_batch": TRAINING_MODE_LABELS["two_stage_batch"],
}


def forest_figure(full: pd.DataFrame) -> plt.Figure:
    """Bootstrap-CI forest plot, training_mode's Winkler score vs. two_stage_batch, one panel per dataset."""
    rng = np.random.default_rng(42)
    with S.style():
        fig, axes = plt.subplots(1, 2, figsize=S.panel(1.7, aspect=0.6), sharex=False)
        for ax, dataset in zip(axes, DATASETS, strict=True):
            fold_df = _fold_level(full, dataset, "training_mode")
            rows = []
            for cond in TRAINING_MODE_ORDER:
                vals = fold_df.loc[fold_df["condition"] == cond, "winkler_score"].to_numpy()
                lo, hi = bootstrap_ci(vals, rng=rng)
                rows.append({"condition": cond, "mean": vals.mean(), "lo": lo, "hi": hi})
            data = pd.DataFrame(rows)
            ref_mean = data.loc[data["condition"] == TRAINING_MODE_REF, "mean"].iloc[0]
            ax.axvline(ref_mean, color=T.TEXT_MUTED, lw=0.7, ls="--", zorder=0)

            for y, (_, row) in enumerate(data.iterrows()):
                color = T.PRIMARY if row["condition"] == TRAINING_MODE_REF else T.TEXT_MUTED
                ax.errorbar(
                    row["mean"],
                    y,
                    xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                    fmt="o",
                    color=color,
                    ms=6,
                )
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([TRAINING_MODE_LABELS[c] for c in data["condition"]])
            ax.set_xlabel(r"Winkler score (bootstrap 95\% CI)")
            ax.set_title(dataset)
            S.format_axes(ax)
    return fig


def interaction_figure(full: pd.DataFrame) -> plt.Figure:
    """Interaction plot: mean Winkler score per training schedule, one line per conformal score."""
    with S.style():
        fig, axes = plt.subplots(1, 2, figsize=S.panel(1.7, aspect=0.6), sharex=True)
        for ax, dataset in zip(axes, DATASETS, strict=True):
            fold_df = _fold_level(full, dataset, "training_x_conformal")
            parts = fold_df["condition"].str.split("_x_", n=1, expand=True)
            fold_df = fold_df.assign(training_mode=parts[0], conformal_score=parts[1])

            for score, color in SCORE_COLORS.items():
                means = [
                    fold_df.loc[
                        (fold_df["training_mode"] == tm) & (fold_df["conformal_score"] == score), "winkler_score"
                    ].mean()
                    for tm in INTERACTION_MODE_ORDER
                ]
                ax.plot(range(len(INTERACTION_MODE_ORDER)), means, marker="o", color=color, label=SCORE_LABELS[score])
            ax.set_xticks(range(len(INTERACTION_MODE_ORDER)))
            ax.set_xticklabels([INTERACTION_MODE_LABELS[c] for c in INTERACTION_MODE_ORDER], fontsize=8)
            ax.set_ylabel("Mean Winkler score")
            ax.set_title(dataset)
            S.format_axes(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    return fig


FIGURES = {
    "fig_ch7_training_schedule_forest.pdf": forest_figure,
    "fig_ch7_training_x_conformal_interaction.pdf": interaction_figure,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--latex", action="store_true", default=True)
    args = parser.parse_args()

    full = pd.read_csv(args.run_dir / "results" / "ch7_design_ablation_full.csv")
    S.configure(latex=args.latex)

    here = Path(__file__).resolve().parent
    out = here / "figures"
    out.mkdir(parents=True, exist_ok=True)
    images = here.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"

    for name, fn in FIGURES.items():
        fig = fn(full)
        fig.savefig(out / name, bbox_inches="tight")
        fig.savefig(out / name.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        if images.exists():
            fig.savefig(images / name, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    main()
