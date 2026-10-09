r"""Lets-plot figures for the thesis.

Kept separate from :mod:`data_figures` because lets-plot does not typeset through
LaTeX. Text in these figures is set by lets-plot's own renderer, so it will not
match the document's serif in the way the matplotlib figures do. That is a
deliberate trade accepted only where the grammar-of-graphics construction earns
it; everything else stays in matplotlib.

lets-plot is already a twiga dependency and exports vector PDF, so nothing new is
introduced.
"""

from __future__ import annotations

from pathlib import Path

import lets_plot as lp
import pandas as pd

from twiga_thesis.plotting import tokens

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "chapter-3-cwe-validation"
THESIS_IMG = HERE.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-3"

DISPLAY = {"MLVS-PT": "MLVS-PT", "MLVS-UK": "SPS-UK"}
ORDER = ["CWE marg.", "CWC", "CRPS", "WS", "CWE strat."]


def _margins() -> pd.DataFrame:
    """Per-fold margin favouring the conditional forecast, tidied for plotting."""
    m = pd.read_csv(RESULTS / "s3524_margins.csv")
    m["dataset"] = m["dataset"].map(DISPLAY)
    return m


def margin_strip_lp() -> lp.core.PlotSpec:
    r"""The per-fold margin strip, drawn with lets-plot and the twiga theme.

    A lets-plot rendering of the same figure produced by
    ``data_figures.margin_strip``. It uses ``twiga_theme`` for layout and the
    thesis palette for colour, so the library's house style is applied without
    its Wong-based palette. Text is set by lets-plot rather than by LaTeX, so the
    type will not match the matplotlib figures elsewhere in the document.
    """
    from twiga.core.plot import twiga_theme

    m = _margins()
    return (
        lp.ggplot(m, lp.aes(x="margin", y="metric"))
        + lp.geom_vline(xintercept=0, color=tokens.TEXT_MUTED, linetype="dashed", size=0.5)
        + lp.geom_point(
            lp.aes(color="dataset", shape="dataset"),
            size=3.2,
            alpha=0.9,
            position=lp.position_jitter(height=0.14, width=0, seed=0),
        )
        + lp.scale_x_continuous(
            name="margin favouring the conditional forecast (%)",
            trans="symlog",
            breaks=[-10, -1, 0, 1, 10, 100],
            labels=["-10", "-1", "0", "1", "10", "100"],
        )
        + lp.scale_y_discrete(name="", limits=ORDER)
        + lp.scale_color_manual(name="", values=[tokens.PRIMARY, tokens.DANGER])
        + lp.scale_shape_manual(name="", values=[21, 22])
        + twiga_theme(grid=False, legend_pos="bottom", legend_direction="horizontal", font_size=11)
        + lp.ggsize(760, 330)
    )


FIGURES = {"fig_margin_strip_lp.pdf": margin_strip_lp}


def render_all(out: Path | None = None) -> None:
    """Render every lets-plot figure to ``out`` and to the thesis image folder."""
    out = out or HERE / "output"
    out.mkdir(exist_ok=True)
    for name, fn in FIGURES.items():
        spec = fn()
        lp.ggsave(spec, name, path=str(out), w=6.3, h=2.8, unit="in", dpi=300)
        if THESIS_IMG.exists():
            lp.ggsave(spec, name, path=str(THESIS_IMG), w=6.3, h=2.8, unit="in", dpi=300)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
