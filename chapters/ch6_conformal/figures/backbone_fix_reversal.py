r"""Chapter 7: what the backbone-HPO fix changed in the checkpoint-monitor and feature-gating groups.

``experiment/ch6_conformal/README.md`` documents a critical bug found
2026-09-18: both datasets' backbone HPO used ``num_trials=10 <
n_startup_trials=20``, so every trial fell inside the TPE sampler's pure-random
startup phase and the two datasets ended up with byte-identical "tuned"
backbones despite training on genuinely different data. Fixed
(``--num-trials`` default raised to 50, ``early_stop_patience=20`` to stop the
scheduler-patience race), and the full design ablation re-run under the
corrected backbone.

Most groups held their pre-fix direction (see the README's post-fix findings
table). Two did not: the checkpoint-monitor group's ``monitor_sigma`` reference
was previously the best-performing checkpoint criterion on both datasets
(directionally, never significant) and is now the *worst* on both; the
feature-gating group's ``gate_unconstrained`` condition beat the default on
MLVS-PT before the fix and now loses to it. Neither reversal clears BH
correction in either run -- the DM/MCS verdict for both groups was "not
significant" before the fix and remains so after -- so this is a change in
which noisy direction the point estimate happens to point, not a new
significant finding. It is still worth showing rather than only stating in
prose, because the reported percentages in an earlier chapter draft would
otherwise read as directionally confirmed when they are two-sided noise.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from twiga_thesis.plotting import ch7_backbone_fix_reversal as F; F.render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twiga_thesis.plotting import style as S, tokens as T

OLD_RECORD = (
    Path(__file__).parents[1]
    / "ch6_conformal"
    / "results"
    / "ch6_conformal"
    / "design_ablation"
    / "mlpgam_design_evaluation__chapter_7"
    / "runs"
    / "20260913_083254_08cf63"
    / "results"
    / "ch7_design_ablation_full.csv"
)
NEW_RECORD = Path(__file__).parents[1] / "ch6_conformal" / "results" / "ch7_design_ablation_full.csv"

DATASETS = {"MLVS-PT": "MLVS-PT", "MLVS-UK": "SPS-UK"}

# (group label, reference condition, [(condition, display label), ...])
GROUPS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Checkpoint monitor\n(ref.\\ monitor\\_sigma)",
        "monitor_sigma",
        (
            ("monitor_loss", "Combined loss"),
            ("monitor_mae", "Point MAE"),
        ),
    ),
    (
        "Feature gating\n(ref.\\ gate\\_default)",
        "gate_default",
        (
            ("gate_unconstrained", "Gate, no penalty"),
            ("gate_none", "No gate"),
        ),
    ),
)


def _per_fold_winkler(record: Path) -> pd.DataFrame:
    df = pd.read_csv(record)
    return df.groupby(["dataset", "Folds", "condition", "group"])["winkler_score"].mean().reset_index()


def _deltas(per_fold: pd.DataFrame) -> dict[tuple[str, str], float]:
    out = {}
    for group_label, ref, conditions in GROUPS:
        for dataset_key, dataset_label in DATASETS.items():
            sub = per_fold[per_fold.dataset == dataset_key]
            ref_ws = sub[sub.condition == ref].winkler_score.mean()
            for cond, cond_label in conditions:
                ws = sub[sub.condition == cond].winkler_score.mean()
                out[(dataset_label, f"{group_label}: {cond_label}")] = 100.0 * (ref_ws - ws) / ref_ws
    return out


def backbone_fix_reversal() -> plt.Figure:
    r"""Slope plot: pre-fix vs.\ post-fix Winkler-score delta, one line per condition."""
    old_deltas = _deltas(_per_fold_winkler(OLD_RECORD))
    new_deltas = _deltas(_per_fold_winkler(NEW_RECORD))

    labels = [f"{g}: {cl}" for g, ref, conds in GROUPS for _, cl in conds]

    with S.style():
        fig, axes = plt.subplots(1, 2, figsize=S.panel(1.7, aspect=0.75), sharey=True)
        for ax, (_dataset_key, dataset_label) in zip(axes, DATASETS.items(), strict=True):
            rows = []
            for label in labels:
                before = old_deltas[(dataset_label, label)]
                after = new_deltas[(dataset_label, label)]
                reversed_sign = (before > 0) != (after > 0)
                color = T.DANGER if reversed_sign else T.TEXT_MUTED
                ax.plot([0, 1], [before, after], "-o", color=color, ms=5, lw=1.6, zorder=2 if reversed_sign else 1)
                rows.append({"label": label, "y": after, "color": color})

            # Spread label y-positions apart when points land close together,
            # so nearby annotations (e.g. two conditions within ~1 point of
            # each other) don't overlap.
            rows.sort(key=lambda r: r["y"])
            min_gap = 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if ax.get_ylim()[1] != ax.get_ylim()[0] else 1.0
            for i in range(1, len(rows)):
                if rows[i]["y"] - rows[i - 1]["y"] < min_gap:
                    rows[i]["y"] = rows[i - 1]["y"] + min_gap

            for row in rows:
                ax.annotate(
                    row["label"],
                    xy=(1.04, row["y"]),
                    xycoords=("axes fraction", "data"),
                    va="center",
                    fontsize=7,
                    color=row["color"],
                )
            ax.axhline(0, color=T.TEXT_MUTED, lw=0.7, ls="--", zorder=0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pre-fix\nbackbone", "Post-fix\nbackbone"])
            ax.set_xlim(-0.15, 1.9)
            ax.set_ylabel(r"$\Delta$ Winkler score vs.\ reference (\%)")
            ax.set_title(dataset_label)
            S.format_axes(ax)
        fig.suptitle(
            r"Backbone-HPO fix reverses two directional (non-significant) findings",
            y=1.03,
        )
    return fig


FIGURES = {
    "fig_ch7_backbone_fix_reversal.pdf": backbone_fix_reversal,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    images = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
    for name, fn in FIGURES.items():
        with S.style(latex=latex):
            fig = fn()
        fig.savefig(out / name, bbox_inches="tight")
        if images.exists():
            fig.savefig(images / name, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
