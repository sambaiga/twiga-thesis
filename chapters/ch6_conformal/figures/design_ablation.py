r"""Chapter 7 design-evaluation figure: Winkler-score deltas by dataset.

Reads the per-fold record behind ``ch7_design_ablation_summary.csv``
(one row per target per fold per condition) and, for every design group,
computes each non-reference condition's percentage change in Winkler score
relative to that group's own reference condition, separately per dataset.
Positive is better throughout (a lower Winkler score than the reference).

Two datasets are plotted side by side rather than pooled, since several
groups disagree in direction between MLVS-PT and SPS-UK (most visibly
``conf_scp`` and the sigma-loss choice); a pooled bar would hide exactly the
comparison the chapter needs to make.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import ch7_design_ablation as F; F.render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twiga_thesis.plotting import style as S, tokens as T

# The path this constant previously pointed at
# (ch6_conformal/ch6_conformal-results/design_ablation/...) has
# never existed on disk -- fixed to the stable copy the design-ablation run
# is kept at after each re-run, matching what mlpgam_design_ablation.py's own
# meta.json records as its spec (`"MLPGAM Design Evaluation (Chapter 7)"`).
RECORD = Path(__file__).parents[1] / "ch6_conformal" / "results" / "ch7_design_ablation_full.csv"

DATASETS = {"MLVS-PT": "MLVS-PT", "MLVS-UK": "SPS-UK"}  # code key -> thesis acronym

# The training-schedule/calibration-timing effect is an order of magnitude
# larger than every other group (up to -186% vs at most +34% elsewhere), so it
# is plotted on its own figure at its own scale; sharing an axis with the
# second-order groups would compress them into illegibility (confirmed by
# rendering both on one axis first).
DOMINANT_GROUP: tuple[str, str, tuple[tuple[str, str], ...]] = (
    "Training schedule",
    "two_stage_batch",
    (
        ("two_stage_ep_20", "Two-stage, 20\\% Stage 1"),
        ("two_stage_ep_10", "Two-stage, 10\\% Stage 1"),
        ("two_stage_ep_05", "Two-stage, 5\\% Stage 1"),
        ("joint", "Joint optimisation"),
    ),
)

# (group, reference condition, [(condition, label), ...] in display order, top to bottom)
SECOND_ORDER_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Scale-loss function",
        "sigma_hybrid_sqrt",
        (
            ("sigma_mse", "MSE"),
            ("sigma_hybrid", "Hybrid MSE--MAE"),
            ("sigma_laplace", "Laplace NLL"),
            ("sigma_gaussian", "Gaussian NLL"),
            ("sigma_log", "Log-space MSE"),
        ),
    ),
    (
        "Checkpoint monitor",
        "monitor_sigma",
        (
            ("monitor_mae", "Point MAE"),
            ("monitor_loss", "Combined loss"),
        ),
    ),
    (
        "Conformal method",
        "conf_90_res",
        (
            ("conf_90_sign", "Signed residual (CRC)"),
            ("conf_scp", "Split conformal"),
        ),
    ),
    (
        "Feature gating",
        "gate_default",
        (
            ("gate_unconstrained", "Gate, no penalty"),
            ("gate_none", "No gate"),
        ),
    ),
    (
        "Target coverage",
        "alpha_90_sign",
        (("alpha_95_sign", r"$\alpha=0.05$ (95\% target)"),),
    ),
)


def _per_fold_winkler() -> pd.DataFrame:
    """One row per (dataset, fold, condition), averaged over target/timestamp."""
    df = pd.read_csv(RECORD)
    return df.groupby(["dataset", "Folds", "condition", "group"])["winkler_score"].mean().reset_index()


def _deltas(groups: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]) -> dict[tuple[str, str], float]:
    """Percentage Winkler-score improvement over each group's reference, per dataset.

    Keyed by ``(dataset_label, "group: condition_label")``, matching ``groups``'s
    row order, so the figure's bars and labels never depend on a join
    surviving group-by reordering.
    """
    per_fold = _per_fold_winkler()
    out: dict[tuple[str, str], float] = {}
    for group_label, ref, conditions in groups:
        for dataset_key, dataset_label in DATASETS.items():
            sub = per_fold[per_fold.dataset == dataset_key]
            ref_ws = sub[sub.condition == ref].winkler_score.mean()
            for cond, cond_label in conditions:
                ws = sub[sub.condition == cond].winkler_score.mean()
                out[(dataset_label, f"{group_label}: {cond_label}")] = 100.0 * (ref_ws - ws) / ref_ws
    return out


def _diverging_bars(groups: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...], label_width: float) -> plt.Figure:
    """Shared plotting body for both figures: horizontal diverging bars per dataset."""
    deltas = _deltas(groups)
    display_labels = [f"{g}: {cl}" for g, ref, conds in groups for _, cl in conds]
    y = np.arange(len(display_labels))[::-1]

    height = max(1.1, 0.34 * len(display_labels))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(S.TEXTWIDTH_IN, height + 0.55),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "left": label_width, "top": height / (height + 0.55)},
    )
    bar_h = 0.6
    for ax, (_dataset_key, dataset_label) in zip(axes, DATASETS.items(), strict=True):
        values = [deltas[(dataset_label, label)] for label in display_labels]
        colors = [T.PRIMARY if v >= 0 else T.DANGER for v in values]
        ax.barh(y, values, height=bar_h, color=colors, edgecolor="none")
        ax.axvline(0, color=T.TEXT_MUTED, lw=0.8, zorder=0)
        ax.set_xlabel(r"$\Delta$ Winkler score (\%)")
        ax.set_title(dataset_label, fontsize=10, pad=6)
        S.format_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=False)

    axes[0].tick_params(axis="y", labelleft=True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(display_labels, fontsize=8)
    return fig


def calibration_timing_summary() -> plt.Figure:
    """The dominant lever: training schedule / calibration timing, its own scale."""
    return _diverging_bars((DOMINANT_GROUP,), label_width=0.30)


def design_ablation_summary() -> plt.Figure:
    """The second-order design choices, at a scale where they stay legible."""
    return _diverging_bars(SECOND_ORDER_GROUPS, label_width=0.30)


FIGURES = {
    "fig_ch7_calibration_timing.pdf": calibration_timing_summary,
    "fig_ch7_design_ablation_summary.pdf": design_ablation_summary,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every figure to ``out`` and to the Chapter 7 image folder."""
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
