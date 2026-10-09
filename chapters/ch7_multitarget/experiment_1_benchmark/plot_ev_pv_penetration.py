r"""Experiment 1 penetration-sweep figures for Chapter 6, from the existing results file.

Reads ``results_ev_pv_rate.xlsx`` directly rather than rerunning DSS-EU: the
file already contains the full penetration sweep (13 (ev-rate, pv-rate)
combinations x 3 targets x 5 folds x 56 daily rows), predating the current
``twiga.experiment`` pipeline. The sweep is two one-axis-at-a-time studies
sharing a (0, 0) baseline -- PV varied with ev-rate=0, EV varied with
pv-rate=0 -- plus four coupled points where ev-rate equals pv-rate; this
script plots the two one-axis sweeps, matching the chapter text's own PV-only
and EV-only framing.

Requires ``openpyxl`` to read the xlsx, which is not a project dependency
(this is the only script that needs it):

    uv run --with openpyxl python experiment/ch7_multitarget/plot_ev_pv_penetration.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twiga.core.utils import get_logger  # noqa: E402

from twiga_thesis.plotting import (  # noqa: E402
    style as S,
    tokens as T,
)

log = get_logger(__name__)

RESULTS_XLSX = Path(__file__).resolve().parent / "results_ev_pv_rate.xlsx"

TARGETS = {"P(kW)": "p", "Q(kvar)": "q", "V(V)": "v"}
TARGET_LABELS = {"P(kW)": "Active power", "Q(kvar)": "Reactive power", "V(V)": "Voltage"}
METRICS = {
    "nrmse": ("NRMSE", T.PRIMARY),
    "nmpi": ("NMPI", T.INFO),
    "pic": ("PICP", T.SUCCESS),
    "ws": ("WS", T.HIGHLIGHT),
}
SWEEP_COLORS = {"pv-rate": T.PRIMARY, "ev-rate": T.DANGER}


def _load() -> pd.DataFrame:
    """Load the penetration-sweep results, aggregated to one row per (target, ev-rate, pv-rate)."""
    df = pd.read_excel(RESULTS_XLSX, sheet_name="results_ev_pv_rate")
    metric_cols = list(METRICS)
    return df.groupby(["Target", "ev-rate", "pv-rate"], as_index=False)[metric_cols].mean()


def _sweep(df: pd.DataFrame, target: str, sweep_axis: str, metric: str) -> tuple[pd.Series, pd.Series]:
    """One-axis sweep (the other axis held at 0) for *target*, sorted by the swept rate."""
    other_axis = "pv-rate" if sweep_axis == "ev-rate" else "ev-rate"
    sub = df[(df["Target"] == target) & (df[other_axis] == 0)].sort_values(sweep_axis)
    return sub[sweep_axis], sub[metric]


def penetration_panel(df: pd.DataFrame, target: str, metric: str) -> plt.Figure:
    """Single-panel PV/EV penetration sweep for one target and one metric."""
    label, _color = METRICS[metric]
    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.85))
        y_max = 0.0
        for sweep_axis, sweep_label in (("pv-rate", "PV"), ("ev-rate", "EV")):
            x, y = _sweep(df, target, sweep_axis, metric)
            y_max = max(y_max, float(y.max()))
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=SWEEP_COLORS[sweep_axis],
                label=sweep_label,
                zorder=3,
            )
        ax.set_xlabel("Penetration (\\%)")
        ax.set_ylabel(label)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlim(-4, 84)
        # A true-zero baseline keeps the PV/EV gap honest instead of a
        # tight autoscaled range exaggerating it; PICP is a coverage rate
        # (already a fraction, autoscale reads fine on its own axis).
        if metric != "pic":
            ax.set_ylim(0, y_max * 1.12)
        S.format_axes(ax)
        ax.legend(frameon=False, loc="best", handlelength=1.4, fontsize=8)
        fig.tight_layout(pad=0.3)
    return fig


def _figure_name(target: str, metric: str) -> str:
    return f"fig_ch6_penetration_{TARGETS[target]}_{metric}.pdf"


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every penetration-sweep panel to *out* and to Images/Chapter-6, where present."""
    S.configure(latex=latex)
    df = _load()
    here = Path(__file__).resolve().parent
    out = out or here.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    images = here.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
    for target in TARGETS:
        for metric in METRICS:
            name = _figure_name(target, metric)
            fig = penetration_panel(df, target, metric)
            fig.savefig(out / name, bbox_inches="tight")
            fig.savefig(out / name.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
            if images.exists():
                fig.savefig(images / name, bbox_inches="tight")
            plt.close(fig)
            log.info("  wrote %s", name)


if __name__ == "__main__":
    render_all()
