r"""CV split-scheme figure for Chapter 7, in the thesis's own production style.

Renders the same information as `ExperimentEngine`'s lets-plot sanity plot
(``twiga.experiment.engine._save_split_scheme_plot``) --- one Gantt row per
fold, coloured by Train / Val / Calib / Test --- but through
:mod:`twiga_thesis.plotting.style`, so it sits in the document as a vector PDF with
the same serif type, palette and spine treatment as every other Chapter 7
figure rather than the lets-plot PNG the engine emits for a quick sanity
check during a run.

The fold boundaries are read directly from a real
:class:`~twiga.core.backtester.TimeBasedCV` built with the same
``cv_train_size=12, cv_test_size=4, cv_val_size=2, cv_calib_size=3,
cv_stride=1, cv_folds=10`` protocol `design_ablation/conditions.py`'s ``ABLATION_SPEC``
uses, run against the real MLVS-PT / MLVS-UK series --- not a schematic.

Run:  uv run python experiment/ch6_conformal/design_ablation/split_scheme_figure.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiment/ (twiga_thesis.plotting, common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chapter directory (shared.*)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twiga.core.backtester import TimeBasedCV  # noqa: E402
from twiga.utils.run_forecast import PipelineConfig, load_and_preprocess  # noqa: E402

from chapters.ch6_conformal.design_ablation.conditions import ABLATION_SPEC
from chapters.ch6_conformal.shared.protocol import DATASETS  # noqa: E402
from twiga_thesis.plotting import style as S, tokens as T  # noqa: E402

SEGMENT_COLORS = {
    "Train": T.PRIMARY,  # teal
    "Val": T.INFO,  # cyan
    "Calib": T.HIGHLIGHT,  # orange
    "Test": T.TEXT_MUTED,  # dark grey
}
BAR_HEIGHT = 0.6


def _build_cv(dataset_key: str) -> tuple[TimeBasedCV, pd.DataFrame]:
    """Load *dataset_key*'s real series and fit a CV matching ABLATION_SPEC's protocol."""
    overrides = DATASETS[dataset_key]
    cfg = PipelineConfig(output_dir="experiment/ch6_conformal/figures/_tmp", **overrides)
    data, *_ = load_and_preprocess(cfg)

    cv = TimeBasedCV(
        split_freq="months",
        train_size=ABLATION_SPEC.cv_train_size,
        test_size=ABLATION_SPEC.cv_test_size,
        val_size=ABLATION_SPEC.cv_val_size,
        calib_size=ABLATION_SPEC.cv_calib_size,
        stride=ABLATION_SPEC.cv_stride,
        num_splits=ABLATION_SPEC.cv_folds,
        window=ABLATION_SPEC.cv_window,
        date_column="timestamp",
    )
    cv.set_split_scheme(data["timestamp"])
    return cv, data


def split_scheme(dataset_key: str) -> plt.Figure:
    """Gantt-style CV split scheme for *dataset_key*, one row per fold."""
    cv, _ = _build_cv(dataset_key)

    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(1.0, aspect=0.5))

        for fold_idx, scheme in cv._split_scheme.items():
            y = fold_idx + 1
            for segment, period_key in (
                ("Train", "train_period"),
                ("Val", "val_period"),
                ("Calib", "calib_period"),
                ("Test", "test_period"),
            ):
                period = scheme.get(period_key)
                if period is None:
                    continue
                start, end = period
                width = (end - start).total_seconds() / 86400  # days, matplotlib date units
                ax.barh(
                    y,
                    width,
                    left=start,
                    height=BAR_HEIGHT,
                    color=SEGMENT_COLORS[segment],
                    edgecolor="none",
                    zorder=3,
                )

        n_folds = len(cv._split_scheme)
        ax.set_ylim(0.5, n_folds + 0.5)
        ax.set_yticks(range(1, n_folds + 1))
        ax.set_ylabel("Fold")
        ax.set_xlabel("Date")
        S.format_axes(ax)

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEGMENT_COLORS.values()]
        ax.legend(
            handles,
            SEGMENT_COLORS.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=4,
            frameon=False,
            handlelength=1.2,
            columnspacing=1.4,
        )
        fig.autofmt_xdate(rotation=30, ha="right")
    return fig


FIGURES = {
    "fig_ch7_split_scheme_mlvs_pt.pdf": lambda: split_scheme("MLVS-PT"),
    "fig_ch7_split_scheme_mlvs_uk.pdf": lambda: split_scheme("MLVS-UK"),
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every split-scheme figure to ``out`` and to Images/Chapter-7."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "figures"
    out.mkdir(parents=True, exist_ok=True)
    images = here.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name)
        fig.savefig(out / name.replace(".pdf", ".png"), dpi=200)
        if images.exists():
            fig.savefig(images / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    print("rendering chapter 7 split-scheme figures:")
    render_all()
