r"""Chapter 6 case-study figures: forecast quality against DER penetration.

The record is the EU-DS backtest reported in the CIRED 2024 paper, one row per
forecast day per fold, read from ``results_ev_pv_rate.xlsx``. The penetration
grid is not a full factorial: it is three one-dimensional sweeps that share the
zero-DER origin, so each panel shows three curves rather than a surface.

Two metrics are plotted. ``nrmse`` carries point accuracy and ``ws`` the
normalised Winkler score, which is proper and therefore the one comparative
conclusions rest on. Both are normalised by the same ``target_range`` for all
three targets, so curves may be compared within a target across penetration
but never across targets; see the caption in the chapter.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import ch6_metrics as M; M.render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from twiga_thesis.plotting import style as S, tokens as T

RECORD = Path(__file__).parents[1] / "ch7_multitarget" / "results_ev_pv_rate.xlsx"
TARGETS = {"P": "P(kW)", "Q": "Q(kvar)", "V": "V(V)"}
METRICS = {"nrmse": "NRMSE", "ws": "Winkler score"}

# The three sweeps, each keyed by how a row's (ev, pv) pair is selected.
SWEEPS = (
    ("PV only", lambda d: d["ev-rate"] == 0, T.PRIMARY, "o", "-"),
    ("EV only", lambda d: d["pv-rate"] == 0, T.DANGER, "s", "-"),
    ("EV and PV", lambda d: d["ev-rate"] == d["pv-rate"], T.TEXT_MUTED, "^", "--"),
)


def _record() -> pd.DataFrame:
    """The per-day metrics, with the penetration level of each sweep resolved."""
    d = pd.read_excel(RECORD, sheet_name="results_ev_pv_rate")
    d["level"] = d[["ev-rate", "pv-rate"]].max(axis=1)
    return d


def _panel(metric: str, target: str) -> plt.Figure:
    d = _record()
    d = d[d["Target"] == TARGETS[target]]

    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.86))
    for label, select, colour, marker, dash in SWEEPS:
        s = d[select(d)].groupby("level")[metric].mean()
        ax.plot(s.index, s.to_numpy(), dash, color=colour, marker=marker, ms=3.4, lw=1.3, label=label)

    ax.set_xlabel(r"Penetration (\%)")
    ax.set_ylabel(METRICS[metric])
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_ylim(bottom=0)
    if target == "P":
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    return fig


FIGURES = {f"fig_ch6_{m}_{t}.pdf": (lambda m=m, t=t: _panel(m, t)) for m in METRICS for t in TARGETS}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every panel to ``out`` and to the Chapter 6 image folder."""
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
