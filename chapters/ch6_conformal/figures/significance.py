r"""Chapter 7 statistical-significance figures, in Chapter 4's plot style.

Chapter 4's `run_statistical_test` (`PhD-Thesis-Anthony/src/utils/significant_test.py`)
renders a pairwise Diebold-Mariano test as a binarised black/white `imshow`
grid: a cell is dark if the p-value clears alpha (row model significantly
worse than column model), light otherwise, with a diagonal marker and a
colorbar. This module reproduces that same visual family for Chapter 7's
design-ablation groups (`best_config`, `gating`, `training_x_conformal`),
styled with the thesis palette rather than Chapter 4's raw grey, and Chapter
7's own already-established statistic: `twiga.core.metrics.stats`'s
Diebold-Mariano implementation on per-fold Winkler score, Benjamini-Hochberg
corrected across each group's pairwise comparisons (Chapter 4's own test has
no multiple-comparison correction; Chapter 7's design-ablation groups run up
to fifteen conditions, where that correction matters).

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from twiga_thesis.plotting import ch7_significance as F; F.render_all()"
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from twiga.core.metrics.stats import benjamini_hochberg, diebold_mariano_from_losses

from twiga_thesis.plotting import style as S, tokens as T

from .design_ablation import DATASETS, RECORD

ALPHA = 0.05

# (group name, condition display order)
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("best_config", ("best_config", "best_config_v1", "best_config_v2")),
    ("gating", ("gate_default", "gate_none", "gate_unconstrained")),
)


def _fold_level(dataset_key: str, group: str) -> pd.DataFrame:
    """One row per (condition, Folds): the correct resampling unit for a DM test."""
    df = pd.read_csv(RECORD)
    sub = df[(df["dataset"] == dataset_key) & (df["group"] == group)]
    return sub.groupby(["condition", "Folds"], as_index=False)["winkler_score"].mean()


def _p_value_matrix(fold_df: pd.DataFrame, conditions: list[str]) -> np.ndarray:
    """Symmetric BH-adjusted p-value matrix over Winkler score, NaN on the diagonal."""
    pivot = fold_df.pivot(index="Folds", columns="condition", values="winkler_score").dropna()
    pivot = pivot[[c for c in conditions if c in pivot.columns]]
    present = list(pivot.columns)
    n = len(present)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    raw_p = [
        float(
            np.mean(
                diebold_mariano_from_losses(
                    pivot[present[i]].to_numpy(), pivot[present[j]].to_numpy(), version="multivariate"
                )
            )
        )
        for i, j in pairs
    ]
    _rejected, adjusted = benjamini_hochberg(np.array(raw_p), alpha=ALPHA) if raw_p else (np.array([]), np.array([]))

    matrix = np.full((len(conditions), len(conditions)), np.nan)
    label_idx = {c: k for k, c in enumerate(conditions)}
    for (i, j), p_adj in zip(pairs, adjusted, strict=True):
        a, b = label_idx[present[i]], label_idx[present[j]]
        matrix[a, b] = p_adj
        matrix[b, a] = p_adj
    return matrix


def _significance_panel(ax: plt.Axes, matrix: np.ndarray, labels: list[str]) -> None:
    """One dataset's binarised significance grid, Chapter 4's visual recipe."""
    n = len(labels)
    binarised = np.where(np.isnan(matrix), np.nan, np.where(matrix < ALPHA, 1.0, 0.0))

    cmap = ListedColormap(["#ffffff", T.DANGER])
    im = ax.imshow(binarised, cmap=cmap, vmin=0, vmax=1)
    ax.plot(range(n), range(n), "x", color=T.TEXT_MUTED, markersize=6)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    return im


def significance_figure(group: str, conditions: tuple[str, ...]) -> plt.Figure:
    """Two-panel (MLVS-PT / SPS-UK) significance grid for one ablation group."""
    labels = list(conditions)
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTWIDTH_IN, 0.55 * S.TEXTWIDTH_IN), gridspec_kw={"wspace": 0.35})
    im = None
    for ax, (dataset_key, dataset_label) in zip(axes, DATASETS.items(), strict=True):
        fold_df = _fold_level(dataset_key, group)
        present = [c for c in labels if c in fold_df["condition"].unique()]
        matrix = _p_value_matrix(fold_df, present)
        im = _significance_panel(ax, matrix, present)
        ax.set_title(dataset_label, fontsize=9.5, pad=6)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, ticks=[0, 1])
        cb.ax.set_yticklabels(["Not sig.", f"Sig. ($p_\\mathrm{{adj}}<{ALPHA}$)"], fontsize=7)
    return fig


FIGURES = {
    f"fig_ch7_{group}_significance.pdf": (lambda g=group, c=conds: significance_figure(g, c)) for group, conds in GROUPS
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every significance figure to ``out`` and to the Chapter 7 image folder."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    images = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
    for name, fn in FIGURES.items():
        with S.style(latex=latex):
            fig = fn()
        fig.savefig(out / name)
        if images.exists():
            fig.savefig(images / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
