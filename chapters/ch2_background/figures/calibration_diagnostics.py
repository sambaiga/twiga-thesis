r"""Chapter 2 illustration: calibration over the whole predictive distribution (reliability diagram and PIT).

Five panels in the thesis line-plot style (:mod:`line_style`), composed as a 2x3 grid of subfigures in LaTeX: how
the PIT is computed and the reliability diagram at 0.48 of the text width, then three PIT histograms at 0.31. The
three forecasters share the conditional mean and differ only in dispersion (``illustrations._forecast_family``:
0.6x, 1x and 1.7x the true spread), so the panels isolate calibration. Each forecaster keeps one colour in every
panel, following :mod:`tokens`: teal for the calibrated case, orange for the over-confident failure, cyan for the
conservative one.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from thesis_figures import ch2_calibration_diagnostics as C; C.render()"
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from scipy import stats

from twiga_thesis.plotting import line_style as LS, tokens as T
from twiga_thesis.plotting.illustrations import _forecast_family

OUT = Path(__file__).resolve().parent / "output"
THESIS = Path(__file__).resolve().parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-2"
NOTE_SIZE = 8
SERIES = {  # family key: (label, colour)
    "over-confident": ("Over-confident", T.DANGER),
    "calibrated": ("Calibrated", T.PRIMARY),
    "conservative": ("Conservative", T.WARNING),
}


ROW = (0.31, 0.9)  # each of the three panels: fraction of the text width, aspect


def construction_panel():
    """How the PIT is computed: an outcome mapped through the predictive CDF to u = F(y)."""
    fig, ax = LS.new_panel(*ROW)
    y = np.linspace(-3, 3, 300)
    ax.plot(y, stats.norm.cdf(y), color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH * 1.3)
    outcome = 0.7
    u = float(stats.norm.cdf(outcome))
    ax.plot([outcome, outcome], [0, u], color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH)
    ax.plot([y[0], outcome], [u, u], color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH)
    ax.plot([outcome], [0], "o", color=LS.OBSERVED_COLOR, ms=4, clip_on=False, zorder=5)
    ax.plot([y[0]], [u], "o", color=LS.OBSERVED_COLOR, ms=4, clip_on=False, zorder=5)
    # PIT values of outcomes from a calibrated forecast: evenly spread over [0, 1].
    rug = stats.norm.cdf(np.random.default_rng(4).standard_normal(40))
    ax.plot(np.full(rug.size, y[0] + 0.12), rug, "_", color=LS.OBSERVED_COLOR, ms=5, mew=0.6, alpha=0.7)
    ax.annotate(
        r"$u = F(y)$",
        xy=(y[0] + 0.25, u),
        xytext=(-2.1, 0.9),
        fontsize=NOTE_SIZE,
        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": T.TEXT_MUTED},
    )
    LS.finish(ax)
    ax.set_xlim(y[0], y[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xticks([outcome])
    ax.set_xticklabels([r"$y$"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Net load")
    ax.set_ylabel(r"Predictive CDF $F$")
    return fig


def reliability_panel():
    """Empirical against nominal coverage of central intervals, for the three forecasters, labelled directly."""
    family = _forecast_family()
    levels = np.linspace(0.0, 1.0, 101)
    fig, ax = LS.new_panel(*ROW)
    ax.plot([0, 1], [0, 1], color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH, zorder=1)
    for key, (_, color) in SERIES.items():
        y, mu, sigma = family[key]
        empirical = [float(np.mean(np.abs(y - mu) <= stats.norm.ppf(0.5 + level / 2) * sigma)) for level in levels]
        ax.plot(levels, empirical, color=color, lw=LS.LINE_WIDTH * 1.3, zorder=3)
    labels = {
        "conservative": (0.03, 0.9, "left"),
        "over-confident": (0.97, 0.12, "right"),
    }
    for key, (x, yy, ha) in labels.items():
        ax.text(x, yy, SERIES[key][0], fontsize=NOTE_SIZE, color=SERIES[key][1], ha=ha, va="center")
    # The calibrated curve lies on the diagonal: label it along the line, rotated with it.
    ax.text(
        0.42,
        0.46,
        SERIES["calibrated"][0],
        fontsize=NOTE_SIZE,
        color=SERIES["calibrated"][1],
        rotation=45,
        rotation_mode="anchor",
        transform_rotates_text=True,
        ha="left",
        va="bottom",
    )
    LS.finish(ax)
    ticks = [0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    return fig


def pit_panel():
    """PIT histograms of the three forecasters as outlines on one axis, against the uniform density."""
    fig, ax = LS.new_panel(*ROW)
    ax.axhline(1.0, color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH, zorder=1)
    for key, (_, color) in SERIES.items():
        y, mu, sigma = _forecast_family()[key]
        pit = stats.norm.cdf((y - mu) / sigma)
        ax.hist(pit, bins=20, range=(0, 1), density=True, histtype="step", color=color, lw=LS.LINE_WIDTH * 1.3)
    notes = {
        "over-confident": (0.07, 3.25, "left"),
        "conservative": (0.5, 1.9, "center"),
        "calibrated": (0.5, 0.68, "center"),
    }
    for key, (x, yy, ha) in notes.items():
        ax.text(x, yy, SERIES[key][0], fontsize=NOTE_SIZE, color=SERIES[key][1], ha=ha, va="bottom")
    LS.finish(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    return fig


def render(copy_to_thesis: bool = False) -> list[Path]:
    """Write the three panels to ``output/`` (and, if asked, the Chapter 2 image folder); return the PDF paths."""
    LS.configure()
    OUT.mkdir(exist_ok=True)
    panels = {
        "fig_pit_construction": construction_panel(),
        "fig_reliability": reliability_panel(),
        "fig_pit": pit_panel(),
    }
    paths = []
    for stem, fig in panels.items():
        LS.save(fig, OUT / stem)
        if copy_to_thesis and THESIS.exists():
            shutil.copy(OUT / f"{stem}.pdf", THESIS / f"{stem}.pdf")
        paths.append(OUT / f"{stem}.pdf")
    return paths
