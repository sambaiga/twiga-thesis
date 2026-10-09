r"""Shared forecast-panel drawing, extracted verbatim from Chapter 6's ``draw``.

``chapter-6-multitarget/forecast_examples.py`` is the sole source of this
style: observed series in orange hollow markers, predictive median dashed,
and the prediction interval(s) drawn as a translucent fan, one 3-hourly
x-axis convention throughout the thesis. This module extracts exactly its
:func:`draw`/``_format_axes``/``place_legend_below``/``_fmt2`` functions,
unchanged, so other chapters reuse the identical implementation instead of
each keeping its own copy. Chapter 6 itself is updated to import from here
rather than own the code twice.

:func:`draw` takes a stacked quantile array (``q``, ``levels``) rather than a
plain ``lower``/``upper`` pair, because that is what Chapter 6's FPQR/
Gaussian heads produce (a fan of several nested intervals, not one). A
caller with only a single prediction interval and a separate point forecast
-- e.g. a conformal method's ``forecast``/``lower``/``upper`` columns --
still uses this unchanged: stack ``[lower, forecast, upper]`` against
``levels=[alpha/2, 0.5, 1-alpha/2]``, and ``draw`` resolves the median to
the middle level (nearest 0.5) and draws the single band from the outer two,
exactly reproducing the same visual convention for one interval instead of a
fan.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np

from twiga_thesis.plotting import tokens as T

SPINE = "gray"
LEGEND_GAP_PT = 4.0
# Bound to the thesis palette (thesis_figures.tokens), matching
# forecast_examples.py's own OBSERVED = HIGHLIGHT / PREDICTED = PRIMARY
# convention -- draw() previously hardcoded matplotlib's stock tab10 hex
# values (#1f77b4/#ff7f0e), which happen to read as "blue-ish + orange" but
# are neither the thesis's actual teal/orange nor tied to the same
# semantic tokens every other thesis figure uses. Found 2026-09-27 comparing
# a forecast panel directly against a production thesis figure.
PREDICTED_COLOR = T.PRIMARY
OBSERVED_COLOR = T.HIGHLIGHT
BAND_COLOR = T.PRIMARY_TINT


def _fmt2(v: float) -> str:
    """Two significant decimals, in scientific notation when the value is small.

    A quantity reported alongside a panel can span several orders of
    magnitude across a chapter's targets (e.g. tens of kW for active power
    against thousandths of a pu for voltage); a plain ``.2f`` would print
    the small one as ``0.00``.
    """
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2e}"


def place_legend_below(fig, ax, **kwargs):
    """Put the legend under the x label, measuring where the label actually is.

    A fixed ``bbox_to_anchor`` cannot do this. The offset that clears the label
    depends on the axes height in pixels, which varies with the figure size and
    with what ``tight_layout`` did, so a constant that works at one panel width
    silently overlaps at another. Here the label's box is measured after the
    layout has run and the legend is anchored just below it.

    Returns:
        The legend, already attached to ``ax``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_box = ax.xaxis.get_label().get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    gap_px = LEGEND_GAP_PT * fig.dpi / 72.0
    # Axes coordinates put 0 at the axes bottom and 1 at its top, so the label
    # sits at a negative fraction of the axes height.
    drop = (label_box.y0 - gap_px - axes_box.y0) / axes_box.height
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, drop), borderaxespad=0.0, **kwargs)
    fig.canvas.draw()
    return legend


def draw(ax, t_axis, y, q, levels, t_i, ylim=None, lw=1.6):
    """One panel: observed series, predictive median and the quantile fan.

    Sampled paths are deliberately not drawn here. A head that models
    dependence across targets within a lead time and not across lead times
    produces consecutive steps of a draw that are independent, so joining
    them into a line is an artefact of the sampler rather than a trajectory
    the model considers plausible.

    No in-figure title -- a caption (or the surrounding grid's own row/column
    labelling) carries that, per the thesis's own convention (matching
    :func:`~significance_plots.plot_dm_significance_matrix`): a banner title
    duplicates what the caption already states and, built from raw condition
    names, is also where an unescaped em-dash has previously slipped through
    and rendered as a missing-glyph box under ``text.usetex``.

    ``lw`` is the line width of the predictive median (default 1.6; Chapter 7 passes 1.0).

    ``ylim``, when given, should already be epsilon-padded from every real
    value the panel plots (see callers) rather than left to matplotlib's own
    ``autoscale`` margin, so every panel in a chapter uses the same tight,
    deliberate convention instead of an implicit default.
    """
    col = levels[:, t_i] if levels.ndim == 2 else levels
    order = np.argsort(col)
    lv, qq = col[order], q[order]

    # Nested bands from the outside in, each at one low alpha, so opacity
    # accumulates toward the median and the fan reads as a density. Disjoint
    # bands between consecutive levels give a visibly stepped edge wherever
    # the levels are few and unevenly spaced, which an FPQR head produces by
    # construction.
    band = None
    for k in range(len(lv) // 2):
        band = ax.fill_between(t_axis, qq[k], qq[-1 - k], color=BAND_COLOR, alpha=0.55, lw=0)

    med = qq[int(np.abs(lv - 0.5).argmin())]
    ax.plot(t_axis, med, "--", c=PREDICTED_COLOR, lw=lw, alpha=0.9, label="Predicted Median")
    # Hollow markers, so a dense 48-step day does not read as a solid bar;
    # "o" (not the original "."), sized to stay legible at full text width,
    # not just the half/third-width subfigure this style was first tuned for.
    ax.plot(t_axis, y, "o", ms=4.5, mew=1.1, mec=OBSERVED_COLOR, mfc="none", label="Observed")
    if band is not None:
        band.set_label("Predictive Interval")

    ax.autoscale(tight=True)
    # Days are selected to start at midnight, so the axis runs 00 to 23. Every
    # third hour rather than every hour: 24 labels collide at panel width.
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_xlabel("Hour of Day")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(False)
    _format_axes(ax)


def _format_axes(ax):
    """Spines and ticks in the thesis house style."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SPINE)
        ax.spines[spine].set_linewidth(0.5)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(direction="out", color=SPINE)
    return ax
