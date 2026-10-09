r"""CV split-scheme figure for Chapter 6, in the thesis's own production style.

Same pattern as ``experiment/ch6_conformal/split_scheme_figure.py``:
one Gantt row per fold, coloured by Train / Val / Calib / Test, through
:mod:`twiga_thesis.plotting.style` so it sits in the document as a vector PDF with
the same serif type, palette and spine treatment as every other chapter's
figures, rather than the lets-plot PNG ``ExperimentEngine`` emits for a
quick sanity check during a run.

DSS-EU's panel is built from ``fpqr_gating_ablation.py``'s own spec (Experiment
1's protocol, unrelated to the backbone/conformal study below). MLVS-PT-3PHASE,
DSS-UK, and DSS-US are built from ``backbone_conformal_study.py``'s specs --
Experiment 2's protocol -- since each dataset there carries its own
``cv_split_freq``, ``cv_stride``, and fold count.

Run:  uv run python experiment/ch7_multitarget/split_scheme_figure.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_2_backbone_conformal"))

from twiga.core.backtester import TimeBasedCV  # noqa: E402
from twiga.core.utils import get_logger  # noqa: E402
from twiga.utils.run_forecast import PipelineConfig, load_and_preprocess  # noqa: E402

from chapters.ch7_multitarget.experiment_2_backbone_conformal.backbone_conformal_study import (  # noqa: E402
    DSS_UK as BACKBONE_DSS_UK,  # noqa: E402
    DSS_US,
    MLVS_PT_3PHASE,
    SPEC_DSS_UK as BACKBONE_SPEC_DSS_UK,  # noqa: E402
    SPEC_DSS_US,
    SPEC_MLVS_PT_3PHASE,
)
from chapters.ch7_multitarget.experiment_2_backbone_conformal.fpqr_gating_ablation import (  # noqa: E402
    DSS_EU_REFERENCE,
    SPEC_DSS_EU,
)
from twiga_thesis.plotting import (
    style as S,  # noqa: E402
    tokens as T,  # noqa: E402
)

log = get_logger(__name__)

SEGMENT_COLORS = {
    "Train": T.PRIMARY,
    "Val": T.INFO,
    "Calib": T.HIGHLIGHT,
    "Test": T.TEXT_MUTED,
}
BAR_HEIGHT = 0.6
SPECS = {
    DSS_EU_REFERENCE: SPEC_DSS_EU,
    MLVS_PT_3PHASE: SPEC_MLVS_PT_3PHASE,
    BACKBONE_DSS_UK: BACKBONE_SPEC_DSS_UK,
    DSS_US: SPEC_DSS_US,
}


def _build_cv(dataset_key: str) -> tuple[TimeBasedCV, pd.DataFrame]:
    """Load *dataset_key*'s real series and fit a CV matching its own spec's protocol."""
    spec = SPECS[dataset_key]
    overrides = spec.datasets[dataset_key]
    cfg = PipelineConfig(output_dir="experiment/ch7_multitarget/figures/_tmp", **overrides)
    data, *_ = load_and_preprocess(cfg)

    cv = TimeBasedCV(
        split_freq=spec.cv_split_freq,
        train_size=spec.cv_train_size,
        test_size=spec.cv_test_size,
        val_size=spec.cv_val_size,
        calib_size=spec.cv_calib_size,
        stride=spec.cv_stride,
        num_splits=spec.cv_folds,
        window=spec.cv_window,
        date_column="timestamp",
    )
    cv.set_split_scheme(data["timestamp"])
    return cv, data


def split_scheme(dataset_key: str, label: str) -> plt.Figure:
    """Gantt-style CV split scheme for *dataset_key*, one row per fold."""
    cv, _ = _build_cv(dataset_key)

    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(1.0, aspect=0.5))

        # enumerate() over sorted keys, not `fold_idx + 1` from the raw dict
        # key: `_split_scheme`'s keys are candidate split indices from
        # *before* TimeBasedCV drops any whose train/test mask came out
        # empty, so they are not guaranteed dense from 0 -- confirmed on
        # DSS-EU, where "Requested 2 splits but generated 1" left a single
        # surviving fold keyed `1`, not `0`, landing the bar at y=2 while
        # ylim was set from n_folds=1 assuming a dense 0..n-1 range: outside
        # the axes, invisible, with no error.
        for y, (_, scheme) in enumerate(sorted(cv._split_scheme.items()), start=1):
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
        ax.set_ylim(0.5, max(n_folds, 1) + 0.5)
        ax.set_yticks(range(1, n_folds + 1))
        ax.set_ylabel("Fold")
        ax.set_xlabel("Date")
        S.format_axes(ax)

        unit = {"months": "mo", "weeks": "wk"}.get(cv.split_freq, cv.split_freq)
        fig.suptitle(f"CV split scheme --- {label} (stride={cv.stride}{unit}, {n_folds} folds)", fontsize=11, y=1.06)
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
    "fig_ch6_split_scheme_dss_eu.pdf": lambda: split_scheme(DSS_EU_REFERENCE, "DSS-EU"),
    "fig_ch6_split_scheme_mlvs_pt_3phase.pdf": lambda: split_scheme(MLVS_PT_3PHASE, "MLVS-PT-3PHASE"),
    "fig_ch6_split_scheme_dss_uk.pdf": lambda: split_scheme(BACKBONE_DSS_UK, "DSS-UK"),
    "fig_ch6_split_scheme_dss_us.pdf": lambda: split_scheme(DSS_US, "DSS-US"),
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every split-scheme figure to *out* and to Images/Chapter-6, where present."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    images = here.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
    for name, fn in FIGURES.items():
        try:
            fig = fn()
        except FileNotFoundError as exc:
            log.warning("  [skip] %s: %s", name, exc)
            continue
        fig.savefig(out / name, bbox_inches="tight")
        fig.savefig(out / name.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        if images.exists():
            fig.savefig(images / name, bbox_inches="tight")
        plt.close(fig)
        log.info("  wrote %s", name)


if __name__ == "__main__":
    render_all()
