r"""Training-dynamics figure for the Chapter 7 training-schedule finding.

The design ablation's headline result (`training_mode`, `two_stage_batch`
beating `joint` optimisation by 53.5%/186.4% Winkler score on MLVS-PT/SPS-UK,
the largest effect in the whole design evaluation -- see this directory's
README and `Chapter-7.tex`'s `chap7:result-design-eval`) is reported purely
as an endpoint metric. Every condition's training run already logs
per-epoch validation curves to TensorBoard (`design_ablation/conditions.py`'s Lightning
trainer), one `events.out.tfevents` file per fold under each condition's
`logs/` tree, but nothing reads them: the chapter states the effect size
without ever showing the training dynamics that produce it.

This script extracts `val_mae` (point-accuracy) and `val_nll` (the
combined-likelihood objective driving calibration) per epoch, across all ten
MLVS-PT folds, for `joint` vs. `two_stage_batch`, and plots the two side by
side. `val_sigma_loss` -- the more directly calibration-relevant curve --
is only logged by `two_stage_batch`, since alternating training separates
the scale head's loss from the point head's; `joint` optimises a single
combined objective and never isolates it, which is itself part of the
mechanism story, not a data-availability gap to work around. `val_nll` is
logged by both and is used as the common calibration-relevant curve instead.

Run:  uv run python experiment/ch6_conformal/design_ablation/training_schedule_mechanism_figure.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiment/ (twiga_thesis.plotting, common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chapter directory (shared.*)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twiga_thesis.plotting import (
    style as S,  # noqa: E402
    tokens as T,  # noqa: E402
)

RUN_DIR = (
    Path(__file__).resolve().parent
    / "results/ch6_conformal/design_ablation/mlpgam_design_evaluation__chapter_7"
    / "runs/20260913_083254_08cf63"
)
CONDITION_COLORS = {"joint": T.TEXT_MUTED, "two_stage_batch": T.PRIMARY}
CONDITION_LABELS = {"joint": "Joint optimisation", "two_stage_batch": "Per-batch alternation (proposed)"}


def _fold_dirs(dataset: str, condition: str) -> list[Path]:
    base = (
        RUN_DIR
        / "conditions"
        / dataset
        / "training_mode"
        / condition
        / "logs"
        / condition
        / "mlpgamcrc"
        / "lightning_logs"
    )
    folds = sorted(base.glob("backtesting_fold_*"), key=lambda p: int(re.search(r"fold_(\d+)", p.name).group(1)))
    return list(folds)


def _epoch_series(fold_dir: Path, tag: str) -> np.ndarray:
    """One value per epoch for *tag*, in epoch order (positional -- Lightning logs once per validation epoch)."""
    ea = EventAccumulator(str(fold_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([])
    return np.array([e.value for e in ea.Scalars(tag)])


def _collect(dataset: str, condition: str, tag: str) -> pd.DataFrame:
    """Long-form (epoch, fold, value) for *tag* across every fold of *condition*."""
    rows = []
    for fold_dir in _fold_dirs(dataset, condition):
        fold = int(re.search(r"fold_(\d+)", fold_dir.name).group(1))
        values = _epoch_series(fold_dir, tag)
        rows.extend({"fold": fold, "epoch": e, "value": v} for e, v in enumerate(values))
    return pd.DataFrame(rows)


def _band_plot(ax: plt.Axes, dataset: str, tag: str, ylabel: str) -> None:
    """Mean curve + IQR band per condition, only over epochs where at least 5 of the 10 folds are still running."""
    for condition, color in CONDITION_COLORS.items():
        df = _collect(dataset, condition, tag)
        if df.empty:
            continue
        agg = df.groupby("epoch")["value"].agg(
            mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="count"
        )
        agg = agg[agg["n"] >= 5]
        ax.plot(agg.index, agg["mean"], color=color, lw=1.6, label=CONDITION_LABELS[condition], zorder=3)
        ax.fill_between(agg.index, agg["q25"], agg["q75"], color=color, alpha=0.18, lw=0, zorder=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    S.format_axes(ax)


def mechanism_figure(dataset: str) -> plt.Figure:
    with S.style():
        fig, axes = plt.subplots(1, 2, figsize=S.panel(2.0, aspect=0.42))
        _band_plot(axes[0], dataset, "val_mae", "Validation MAE")
        _band_plot(axes[1], dataset, "val_nll", "Validation NLL")
        axes[0].set_title("Point accuracy", fontsize=10)
        axes[1].set_title("Calibration objective", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    return fig


FIGURES = {
    "fig_ch7_training_schedule_mechanism_pt.pdf": lambda: mechanism_figure("MLVS-PT"),
    "fig_ch7_training_schedule_mechanism_uk.pdf": lambda: mechanism_figure("MLVS-UK"),
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "figures"
    out.mkdir(parents=True, exist_ok=True)
    images = here.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name, bbox_inches="tight")
        fig.savefig(out / name.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        if images.exists():
            fig.savefig(images / name, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
