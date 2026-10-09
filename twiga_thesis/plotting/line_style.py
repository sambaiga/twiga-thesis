r"""Compact, publication-ready line-plot style for the thesis line plots (used by Chapter 7).

One definition for every line plot (forecast examples, coverage and width profiles, sweeps, branch terms), on top of
the thesis style (``thesis_figures.style``: LaTeX serif,
no grid, left and bottom spines only, thesis palette):

* panel ``PANEL`` (0.40 of the text width, aspect 0.74), with the thesis theme's font family and 10 pt axis labels
  and ticks, and an 8 pt legend;
* lines at ``LINE_WIDTH`` (0.85 pt), interval bands with a ``BAND_EDGE`` (0.2 pt) edge, small hollow observation
  markers;
* y-axis from :func:`set_y_axis`: bottom = (min - eps) rounded down to the data's round unit, top = (max + eps)
  rounded up to the tick step, ticks at about four round steps;
* the legend in one row below the axes (:func:`legend_below`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from . import style as S, tokens as T

PANEL = (0.40, 0.74)  # (fraction of \textwidth, height / width); 0.30 where three panels share a row
# One colour per role, the same in every line plot of the thesis.
FORECAST_COLOR = T.PRIMARY  # model forecast (solid line)
OBSERVED_COLOR = T.HIGHLIGHT  # observations (small hollow markers, or a solid line for an aggregate)
BAND_COLOR = T.PRIMARY_TINT  # prediction interval fill, edged in FORECAST_COLOR
REFERENCE_COLOR = T.TEXT_MUTED  # nominal level, zero, record mean: dotted
SEPARATOR_COLOR = T.GRID  # structural divider between groups of points
LINE_WIDTH = 0.85
BAND_EDGE = 0.2
MARKER = {"ms": 1.8, "mew": 0.5}
EPS_FRAC = 0.04
# Line weights and tick geometry; the font family and the axis-label and tick sizes are the thesis theme's own
# (``style.configure``: twiga's matplotlib theme plus ``THESIS_RC``, 10 pt, as in Chapter 6). Only the legend is
# smaller (8 pt), so a legend with several entries fits the 0.40-width panel in one or two rows.
LEGEND_FONTSIZE = 8
COMPACT_RC = {
    "legend.fontsize": LEGEND_FONTSIZE,
    "axes.linewidth": 0.5,
    "lines.linewidth": LINE_WIDTH,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}


def configure(latex: bool = True) -> None:
    """The thesis style with the compact line-plot sizes on top."""
    S.configure(latex=latex)
    plt.rcParams.update(COMPACT_RC)


def new_panel(fraction: float = PANEL[0], aspect: float = PANEL[1]) -> tuple[plt.Figure, plt.Axes]:
    """A single-axes figure authored at its final size (``fraction`` of the text width)."""
    fig, ax = plt.subplots(figsize=S.panel(fraction, aspect))
    return fig, ax


def reference_line(ax: plt.Axes, y: float, label: str | None = None) -> None:
    """A horizontal reference (nominal level, zero, record mean) in the uniform dotted style."""
    ax.axhline(y, color=REFERENCE_COLOR, ls=":", lw=LINE_WIDTH, label=label, zorder=1)


def set_y_axis(
    ax: plt.Axes,
    *arrays,
    eps_frac: float = EPS_FRAC,
    nbins: int = 4,
    floor: float | None = None,
    cap: float | None = None,
    unit_interval: bool = False,
    from_zero: bool = False,
) -> None:
    """Y-limits and ticks from the plotted data.

    With ``lo = min - eps`` and ``hi = max + eps`` (``eps = eps_frac`` of the data range), the tick step is a round
    step giving about ``nbins`` intervals; the top is ``hi`` rounded up to the step and the bottom ``lo`` rounded down
    to half the step, and ticks sit at the multiples of the step inside the axis, e.g. data 14.6 to 71.4 -> axis 10 to
    80, ticks 20, 40, 60, 80.

    ``unit_interval`` fixes the axis to [0, 1] (ticks 0, 0.25, ..., 1) for a rate bounded by 0 and 1 (coverage,
    NMPI, CWE); ``from_zero`` starts the axis at 0 for a non-negative quantity whose natural floor is zero (an error
    or a width). ``floor``/``cap`` bound the limits otherwise.
    """
    if unit_interval:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        return
    if from_zero:
        arrays = (*arrays, [0.0])
        floor = 0.0
    lo, hi = S.pad_limits(*arrays, frac=eps_frac)
    ticks = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    step = ticks[1] - ticks[0]
    bottom, top = np.floor(lo / (step / 2)) * (step / 2), np.ceil(hi / step) * step
    if floor is not None:
        bottom = max(bottom, floor)
    if cap is not None:
        top = min(top, cap)
    ax.set_ylim(bottom, top)
    ax.set_yticks(np.arange(np.ceil(bottom / step - 1e-9) * step, top + step / 2, step))


LEGEND_GAP_PT = 3.0  # space between the x-axis label and the legend below it


def legend_below(ax: plt.Axes, handles=None, labels=None, ncol: int = 3, offset: float | None = None) -> None:
    """One-row legend centred below the axes; :func:`save` places it ``LEGEND_GAP_PT`` below the x-axis label.

    ``offset`` is accepted for backward compatibility and ignored: the position is measured after layout, so the
    legend never overlaps the x-axis label whatever the panel size.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    legend._thesis_below = True  # marks the legend for placement in save()


def _place_legends_below(fig: plt.Figure) -> None:
    """Anchor every :func:`legend_below` legend just under its axes' x-axis label, in as many columns as fit.

    The number of columns is reduced until the legend is no wider than the figure, so the theme's font size never
    forces the axes to shrink; the legend is then placed ``LEGEND_GAP_PT`` below the x-axis label and tick labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_width = fig.get_figwidth() * fig.dpi
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None or not getattr(legend, "_thesis_below", False):
            continue
        handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
        ncol = legend._ncols
        while ncol > 1 and legend.get_window_extent(renderer).width > 0.98 * fig_width:
            ncol -= 1
            legend.remove()
            legend_below(ax, handles, labels, ncol=ncol)
            legend = ax.get_legend()
            fig.canvas.draw()
        boxes = [ax.xaxis.label.get_window_extent(renderer), ax.xaxis.get_tightbbox(renderer)]
        bottom = min(box.y0 for box in boxes if box is not None)
        gap = LEGEND_GAP_PT * fig.dpi / 72.0
        y = ax.transAxes.inverted().transform((0, bottom - gap))[1]
        legend.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)


def finish(ax: plt.Axes) -> None:
    """Spines and ticks in the thesis house style."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out")
    ax.minorticks_off()


def hour_axis(ax: plt.Axes) -> None:
    """Lead time in hours (0 to 24) shown as time of day; every forecast window starts at 00:00."""
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.set_xlim(0, 24)
    ax.set_xlabel("Hour of day")


def time_axis(ax: plt.Axes, t: pd.Series) -> None:
    """A one-day datetime axis labelled every six hours."""
    ax.set_xlim(t.iloc[0], t.iloc[-1] + pd.Timedelta(minutes=30))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlabel("Hour of day")


def naive_times(day: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(day["timestamp"])
    return t.dt.tz_convert(None) if t.dt.tz is not None else t


def forecast_panel(day: pd.DataFrame, ylabel: str, interval_label: str = r"90\% interval") -> plt.Figure:
    """One day's observations, point forecast and interval: band with a thin edge, forecast line, hollow markers."""
    fig, ax = new_panel()
    t = naive_times(day)
    ax.fill_between(t, day["lower"], day["upper"], color=BAND_COLOR, lw=0, label=interval_label)
    ax.plot(t, day["lower"], color=FORECAST_COLOR, lw=BAND_EDGE)
    ax.plot(t, day["upper"], color=FORECAST_COLOR, lw=BAND_EDGE)
    ax.plot(t, day["forecast"], color=FORECAST_COLOR, lw=LINE_WIDTH, label="Forecast")
    ax.plot(t, day["Actual"], "o", mec=OBSERVED_COLOR, mfc="white", label="Observed", zorder=3, **MARKER)
    time_axis(ax, t)
    set_y_axis(ax, day["Actual"], day["lower"], day["upper"])
    ax.set_ylabel(ylabel)
    finish(ax)
    legend_below(ax)
    return fig


def save(fig: plt.Figure, path: Path) -> None:
    """PDF (the thesis format) and a 600 dpi PNG preview, tightly cropped, with legends placed below the axes."""
    fig.tight_layout(pad=0.2)
    _place_legends_below(fig)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
