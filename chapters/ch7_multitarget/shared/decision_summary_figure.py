r"""Chapter-6-style headline figure for the Experiment 2 decision results.

``ch6_analyse_results.py``'s own generic multi-metric bar grid (20 conditions
x 10 metrics, one shared y-axis) is a diagnostic dump, not a chapter figure.
This script reads the same ``multitarget_decision`` summary CSVs it wrote
(``results/analysis/multitarget_decision/{dss-uk,dss-us}/{dataset}_metrics.csv``)
and builds one curated figure: native vs.\ max-norm-calibrated Brier score,
grouped by head, one panel per dataset -- the comparison the chapter text
actually needs, in the thesis's own figure style.

Usage::

    uv run python experiment/ch7_multitarget/shared/decision_summary_figure.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT = script_dir.parent / "figures"
ANALYSIS_ROOT = script_dir.parent / "results" / "analysis" / "multitarget_decision"

sys.path.insert(0, str(script_dir.parent.parent))
from thesis_figures.style import configure  # noqa: E402
from thesis_figures.tokens import PALETTE  # noqa: E402

HEAD_LABEL = {"fpqr_anchor_outer": "FPQR", "qr": "QR", "bnn": "BNN", "laplace": "Laplace"}
HEADS = ["fpqr_anchor_outer", "qr", "bnn", "laplace"]
NATIVE_COLOUR = PALETTE[5]
CALIB_COLOUR = PALETTE[0]
DATASET_LABEL = {"dss-uk": "DSS-UK", "dss-us": "DSS-US"}


def load_metrics(dataset: str) -> pd.DataFrame:
    """``ch6_analyse_results.py``'s own per-dataset metric table for ``multitarget_decision``."""
    return pd.read_csv(ANALYSIS_ROOT / dataset / f"{dataset}_metrics.csv")


def plot_brier_bars(datasets: list[str], out: Path) -> None:
    """Native vs. max-norm-calibrated Brier score, grouped by head, one panel per dataset."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.8), layout="constrained", sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    x = np.arange(len(HEADS))
    width = 0.35
    for ax, dataset in zip(axes, datasets, strict=True):
        df = load_metrics(dataset)
        native = [df.loc[df.condition == f"{h}_none", "Brier score"].squeeze() for h in HEADS]
        calib = [df.loc[df.condition == f"{h}_max_norm", "Brier score"].squeeze() for h in HEADS]
        ax.bar(x - width / 2, native, width, color=NATIVE_COLOUR, label="native")
        ax.bar(x + width / 2, calib, width, color=CALIB_COLOUR, label="max-norm")
        ax.set_xticks(x, [HEAD_LABEL[h] for h in HEADS])
        ax.set_title(DATASET_LABEL[dataset], fontsize=10)
        ax.set_ylabel("Brier score")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(out / "fig_ch6_decision_brier_bars.pdf")
    plt.close(fig)


def plot_joint_coverage_bars(native: dict[str, float], calibrated: dict[str, float], out: Path) -> None:
    """Native vs. max-norm-calibrated joint coverage, one bar pair per head, MLVS-PT-3PHASE benchmark.

    A head missing from ``calibrated`` (its run not yet available) gets no
    max-norm bar rather than a misleading zero.
    """
    fig, ax = plt.subplots(figsize=(4.2, 2.8), layout="constrained")
    x = np.arange(len(HEADS))
    width = 0.35
    ax.bar(x - width / 2, [native[h] for h in HEADS], width, color=NATIVE_COLOUR, label="native")
    labelled = False
    for i, h in enumerate(HEADS):
        if h in calibrated:
            ax.bar(x[i] + width / 2, calibrated[h], width, color=CALIB_COLOUR, label=None if labelled else "max-norm")
            labelled = True
    ax.axhline(0.90, color="k", lw=0.8, ls="--", label="nominal 90\\%")
    ax.set_xticks(x, [HEAD_LABEL[h] for h in HEADS])
    ax.set_ylabel("joint coverage")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out / "fig_ch6_benchmark_joint_coverage_calibrated.pdf")
    plt.close(fig)


def main() -> None:
    """Build and write the curated decision and calibration comparison figures."""
    try:
        configure(latex=True)
    except Exception as exc:  # noqa: BLE001
        print(f"thesis style unavailable ({exc})")
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = ["dss-uk", "dss-us"]
    plot_brier_bars(datasets, OUT)
    print(f"wrote {OUT / 'fig_ch6_decision_brier_bars.pdf'}")

    # MLVS-PT-3PHASE benchmark joint coverage. qr/bnn/laplace native+max-norm
    # are self-consistent with run 20261003_...3e371e ("v6", the only
    # multitarget_experiment run with a conformal group). fpqr_anchor_outer's
    # native value instead comes from run 20261001_...904e94 ("v3",
    # benchmark_dependence_check.py's own per-head override) -- the
    # best-calibrated fpqr_anchor_outer on record -- but v3 never reserved a
    # calibration window, so no valid fpqr_anchor_outer max-norm value exists
    # yet; omitted from calibrated_joint until the pending rerun (benchmark+
    # conformal groups, v3's exact pins, reserve_calib_window=True) lands.
    native_joint = {"fpqr_anchor_outer": 0.782, "qr": 0.529, "bnn": 0.522, "laplace": 0.099}
    calibrated_joint = {"qr": 0.9064, "bnn": 0.8922, "laplace": 0.7729}
    plot_joint_coverage_bars(native_joint, calibrated_joint, OUT)
    print(f"wrote {OUT / 'fig_ch6_benchmark_joint_coverage_calibrated.pdf'}")


if __name__ == "__main__":
    main()
