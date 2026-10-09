r"""Chapter 6 Section 6.3: what it costs to assume the targets are independent.

Both panels read ``dependence_cost.csv``, produced by
``experiment/ch7_multitarget/dependence_cost.py``. One row is one scenario
cell: a feeder and penetration pair for the European record, a transformer and
penetration pair for the North American one.

The second panel exists because the first argument a reader will raise is that
Pearson correlation measures linear association only. It does, which is the
point: distance correlation is zero if and only if two variables are
independent, so the gap between the two measures is dependence that a linear
summary cannot represent.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import ch6_dependence as D; D.render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twiga_thesis.plotting import style as S, tokens as T

RECORD = Path(__file__).parents[1] / "ch7_multitarget" / "dependence_cost.csv"

# Only the two records the chapter reports; the synthesised-Q variant is a
# control for the power-factor assumption and is discussed in the text instead.
SERIES = (
    ("b1_measured", "European substation", T.PRIMARY, "o"),
    ("b2", "US pole transformer", T.DANGER, "s"),
)


def _record() -> pd.DataFrame:
    return pd.read_csv(RECORD)


def exceedance_bias() -> plt.Figure:
    """Violation probability under permuted margins against the joint record."""
    d = _record()
    d = d[d["thermal_joint"] > 0]

    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.92))
    lo, hi = 2e-4, 0.6
    ax.plot([lo, hi], [lo, hi], "-", color=T.TEXT_MUTED, lw=1.0, zorder=1)
    ax.fill_between([lo, hi], [lo, hi], lo, color=T.DANGER, alpha=0.07, lw=0, zorder=0)
    for key, label, colour, marker in SERIES:
        g = d[d["record"] == key]
        ax.plot(
            g["thermal_joint"],
            g["thermal_indep"],
            marker,
            ms=3.6,
            mfc="none",
            mec=colour,
            mew=1.0,
            ls="none",
            label=label,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Joint record")
    ax.set_ylabel("Independent margins")
    ax.annotate(
        "risk understated",
        xy=(0.5, 0.16),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color=T.DANGER,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def dependence_measures() -> plt.Figure:
    r"""What linear correlation misses, against a measure that misses nothing."""
    d = _record()
    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.92))

    # Pearson passing near zero is the interesting region: it reports
    # near-independence exactly where the relationship stops being monotone.
    ax.axvspan(0, 0.3, color=T.DANGER, alpha=0.07, lw=0, zorder=0)
    ax.plot([0, 1], [0, 1], "-", color=T.TEXT_MUTED, lw=1.0, zorder=1)

    for key, label, colour, marker in SERIES:
        g = d[d["record"] == key]
        ax.plot(
            g["r_pq"].abs(),
            g["dcor_pq"],
            marker,
            ms=3.6,
            mfc="none",
            mec=colour,
            mew=1.0,
            ls="none",
            label=label,
            zorder=3,
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$|\mathrm{corr}(P, Q)|$")
    ax.set_ylabel("Distance correlation")
    ax.annotate(
        "Pearson near zero",
        xy=(0.15, 0.97),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        rotation=90,
        fontsize=7.5,
        color=T.DANGER,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return fig


FIGURES = {
    "fig_ch6_exceedance_bias.pdf": exceedance_bias,
    "fig_ch6_dependence_measures.pdf": dependence_measures,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render both panels to ``out`` and to the Chapter 6 image folder."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    thesis = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name)
        if thesis.exists():
            fig.savefig(thesis / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
