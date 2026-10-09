r"""Chapter 2 illustration: validity and efficiency of a 90% prediction interval.

Three panels at 0.31 of the text width, composed with subfigures in LaTeX, in the thesis line-plot style
(:mod:`line_style`). Every panel shows the same day and the same conditionally correct forecast; only the interval
width changes (a multiple of the calibrated half-width), so the panels differ in nothing but validity and width:

* ``fig_interval_overconfident``: half-width x0.45, coverage well below the nominal 0.90;
* ``fig_interval_calibrated``: half-width x1.0, coverage at the nominal rate with the least width that achieves it;
* ``fig_interval_conservative``: half-width x2.2, every outcome covered at the cost of width.

Covered outcomes are small hollow markers and missed outcomes larger filled markers, both orange; the PICP is
estimated over 120 days of the same process, not only the day shown. Replaces the ``interval_*`` panels of
:mod:`illustrations`, which rendered to the Chapter 3 image folder.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from twiga_thesis.plotting import ch2_interval_validity as V; V.render()"
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from twiga_thesis.plotting import line_style as LS
from twiga_thesis.plotting.illustrations import NOMINAL, Z, _many_days

OUT = Path(__file__).resolve().parent / "output"
THESIS = Path(__file__).resolve().parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-2"
FRACTION, ASPECT = 0.31, 0.86
NOTE_SIZE = 8
SCALES = {"overconfident": 0.45, "calibrated": 1.0, "conservative": 2.2}
MISSED = {"ms": 3.2, "mew": 0.0}


def _data() -> dict[str, np.ndarray]:
    """120 days of the synthetic PV-dominated net-load process; the first day is drawn."""
    return _many_days(n_days=120, seed=5)


def _limits(d: dict[str, np.ndarray]) -> tuple[float, float]:
    """One y-range for the three panels: the widest interval and every outcome of the drawn day."""
    day = slice(0, 48)
    half = Z * d["sigma"][day] * max(SCALES.values())
    lo = min((d["mu"][day] - half).min(), d["y"][day].min())
    hi = max((d["mu"][day] + half).max(), d["y"][day].max())
    pad = 0.06 * (hi - lo)
    # Extra room below the data for the coverage note, so it never overlaps the band.
    return lo - 0.32 * (hi - lo), hi + pad


def interval_panel(scale: float, d: dict[str, np.ndarray], limits: tuple[float, float]):
    """One day of a forecast whose 90% half-width is ``scale`` times the calibrated one."""
    mu, sigma, y = d["mu"], d["sigma"], d["y"]
    half = Z * sigma * scale
    picp = float(np.mean(np.abs(y - mu) <= half))
    day = slice(0, 48)
    hours = d["hod"][day] * 24
    inside = np.abs(y - mu)[day] <= half[day]

    fig, ax = LS.new_panel(FRACTION, ASPECT)
    ax.fill_between(hours, (mu - half)[day], (mu + half)[day], color=LS.BAND_COLOR, lw=0)
    ax.plot(hours, mu[day], color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
    ax.plot(hours[inside], y[day][inside], "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=3, **LS.MARKER)
    ax.plot(hours[~inside], y[day][~inside], "o", color=LS.OBSERVED_COLOR, zorder=4, **MISSED)
    ax.text(
        0.97,
        0.04,
        rf"PICP $= {picp:.2f}$" "\n" rf"nominal $= {NOMINAL:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=NOTE_SIZE,
        linespacing=1.3,
    )
    LS.finish(ax)
    LS.hour_axis(ax)
    ax.set_xticks([0, 12, 24])
    ax.set_xticklabels(["00", "12", "24"])
    ax.set_ylim(*limits)
    ax.set_yticks([])
    ax.set_ylabel("Net load")
    return fig


def render() -> list[Path]:
    """Write the three panels to ``output/`` and the Chapter 2 image folder; return the output PDF paths."""
    LS.configure()
    OUT.mkdir(exist_ok=True)
    d = _data()
    limits = _limits(d)
    paths = []
    for name, scale in SCALES.items():
        fig = interval_panel(scale, d, limits)
        stem = f"fig_interval_{name}"
        LS.save(fig, OUT / stem)
        if THESIS.exists():
            shutil.copy(OUT / f"{stem}.pdf", THESIS / f"{stem}.pdf")
        paths.append(OUT / f"{stem}.pdf")
    return paths
