r"""Matplotlib style for thesis figures.

Applies the SciencePlots ``science`` style, then the thesis overrides on top:
serif type set through LaTeX, no grid anywhere, a colour-blind-safe cycle, and
thin axes. Figures are authored one panel at a time and saved as separate PDFs,
to be composed into sub-figures in LaTeX --- that keeps each panel individually
referenceable (``\\cref{fig:x}{\\subref{sfig:y}}``) and lets the document, rather
than matplotlib, control the layout.

``text.usetex`` is on by default, so the type in a figure matches the type in
the surrounding text. That requires a working LaTeX installation on the path;
call ``configure(latex=False)`` to fall back to matplotlib's own renderer, which
is useful in CI or on a machine without TeX.

Because LaTeX is doing the typesetting, every string reaching matplotlib must be
LaTeX-safe: no unescaped ``%``, ``&``, ``_``, ``#``, and no non-ASCII characters
such as an em-dash. Use :func:`tex_safe` on anything interpolated from data.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  -- registers the "science" style on import

from .tokens import (  # noqa: F401
    BAD,
    DANGER,
    GOOD,
    GRID,
    HIGHLIGHT,
    INFO,
    PALETTE as THESIS_PALETTE,
    PRIMARY,
    SUCCESS,
    TEXT_DARK as TEXT,
    TEXT_MUTED,
    TRUTH,
    WARNING,
)

ACCENT = INFO
WARM = WARNING
MUTED = SUCCESS
NEUTRAL = TEXT_MUTED
ALERT = DANGER
SPINE_COLOR = "gray"
# ── figure geometry ───────────────────────────────────────────────────────────
# \the\textwidth of the IST thesis class is 455.24pt = 6.30in.
TEXTWIDTH_IN = 6.30
GOLDEN = 0.618


def golden_height(width: float = TEXTWIDTH_IN) -> float:
    """Height giving a golden-ratio aspect for a figure of the given width."""
    return width * GOLDEN


def panel(fraction: float = 0.48, aspect: float = GOLDEN) -> tuple[float, float]:
    r"""Size for one panel that will occupy ``fraction`` of the text width.

    Authoring each panel at the size it will finally occupy keeps type at a
    consistent size across a composed figure; scaling panels in LaTeX does not.

    Args:
        fraction: Share of ``\\textwidth`` the panel will occupy. ``0.48`` for
            two side by side, ``0.31`` for three, ``1.0`` for a full-width panel.
        aspect: Height as a multiple of width.

    Returns:
        A ``(width, height)`` pair in inches, for ``figsize``.
    """
    w = TEXTWIDTH_IN * fraction
    return (w, w * aspect)


def pad_limits(*value_arrays, frac: float = 0.04, min_pad: float = 1e-9) -> tuple[float, float] | None:
    """``(min, max)`` across every array in ``value_arrays``, padded by ``frac`` of the range.

    The thesis convention for an axis limit is a small, deliberate epsilon
    margin around the real data rather than matplotlib's own ``margins``
    default (a different, non-configurable-per-figure fraction) or an
    untouched ``autoscale(tight=True)`` (no margin at all, so a marker at the
    exact extreme sits on the spine). Pass every array the axis will plot
    (e.g. both the observed series and the predicted quantile fan) so the
    padding is computed from everything the panel actually shows, not just
    one of them.

    Returns ``None`` -- so a caller falls back to matplotlib's own autoscale
    -- when there is nothing to pad (all arrays empty) or any value is not a
    finite real number (NaN/inf), since padding around a non-finite value
    would silently produce a nonsensical range.
    """
    arrays = [np.asarray(a, dtype=float).ravel() for a in value_arrays if np.size(a)]
    if not arrays:
        return None
    values = np.concatenate(arrays)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    lo, hi = float(values.min()), float(values.max())
    pad = frac * (hi - lo) if hi > lo else max(abs(hi), 1.0) * frac
    pad = max(pad, min_pad)
    return lo - pad, hi + pad


# ── the style ─────────────────────────────────────────────────────────────────
# The thesis style is twiga's own matplotlib style with document-specific
# overrides, rather than a second implementation of it. twiga already applies
# SciencePlots, resolves the LaTeX toolchain and defaults to this palette; only
# the typography and the spine treatment differ, because these figures target
# vector PDF at \textwidth rather than a notebook.
THESIS_RC: dict[str, Any] = {
    # Serif through LaTeX, in the document's face rather than Computer Modern.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    # Axis labels smaller than the body size: panels are authored at their final
    # width, so a label at the body size reads oversized once placed. Fixed at
    # 10pt regardless of `configure`'s `base_size` argument, so a caller that
    # forgets to pass `base_size=10` still gets the document's actual type size
    # here rather than base_size's default of 12.
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # Spines: twiga's style already draws only the left and bottom, so nothing is
    # needed here. A thinner rule suits a figure placed at a third of \textwidth.
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    # Explicit rather than inherited from configure_matlib_style's own default:
    # same ten colours, same order, so the cycle is pinned here even if twiga's
    # own default ever changes.
    "axes.prop_cycle": cycler(color=THESIS_PALETTE),
}


def configure(latex: bool = True, base_size: float = 12.0) -> None:
    """Apply the thesis style globally.

    Delegates to :func:`twiga.core.plot.matplot_theme.configure_matlib_style` and
    layers :data:`THESIS_RC` on top, so the palette and the SciencePlots base stay
    in one place.

    Args:
        latex: Typeset text through LaTeX. twiga falls back automatically when no
            LaTeX toolchain is on the path.
        base_size: Body font size. The axis labels and ticks are set relative to
            it in :data:`THESIS_RC`.
    """
    from twiga.core.plot.matplot_theme import configure_matlib_style

    configure_matlib_style(
        latex=latex,
        font_size=base_size,
        fig_width=TEXTWIDTH_IN,
        fig_height=golden_height(),
        dpi=300,
        custom_rc=THESIS_RC,
    )


@contextmanager
def style(latex: bool = True, **overrides: Any):
    """Apply the thesis style for the duration of a block, then restore."""
    original = mpl.rcParams.copy()
    try:
        configure(latex=latex)
        mpl.rcParams.update(overrides)
        yield
    finally:
        mpl.rcParams.update(original)


@contextmanager
def sketch(latex: bool = True, scale: float = 1.2, length: float = 130, randomness: float = 3, **overrides: Any):
    """The thesis style with hand-drawn lines, for concept illustrations.

    For figures that argue a point rather than report a result --- a schematic,
    not a measurement --- a sketched line signals "illustrative" the way a
    result plot's clean vector line signals "measured". Only the paths (lines,
    patches, spines) are sketched, via matplotlib's own ``path.sketch``; text
    stays in the thesis's LaTeX serif, so labels remain legible and consistent
    with the surrounding document rather than switching to a comic font as
    ``plt.xkcd()`` would.

    Args:
        latex: As in :func:`style`.
        scale: Wobble amplitude in points.
        length: Wavelength of the wobble in points; larger wanders more slowly.
        randomness: Jitter added to the wavelength, as a percentage.
        **overrides: Passed through to :func:`style`.
    """
    with style(latex=latex, **overrides):
        original_sketch = mpl.rcParams["path.sketch"]
        mpl.rcParams["path.sketch"] = (scale, length, randomness)
        try:
            yield
        finally:
            mpl.rcParams["path.sketch"] = original_sketch


# ── LaTeX-safe text ───────────────────────────────────────────────────────────
_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "—": "---",
    "–": "--",
    "×": r"$\times$",
    "≈": r"$\approx$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "ρ": r"$\rho$",
    "σ": r"$\sigma$",
    "α": r"$\alpha$",
}


def tex_safe(text: str) -> str:
    r"""Escape a string for LaTeX, for anything interpolated from data.

    A no-op when ``text.usetex`` is off: the escape sequences this produces
    (e.g. ``\_`` for a literal underscore) are only meaningful to a real TeX
    engine. Found 2026-09-26 rendering as a literal backslash-underscore in
    every in-axes label built this way once a figure was drawn with
    ``configure(latex=False)`` -- exactly the "corrupted symbol" a plain
    ``mathtext`` renderer produces for a raw escape sequence it does not
    itself interpret, not the LaTeX-safe text the caller expected. Do not
    apply this to strings that deliberately contain maths, since it would
    escape those too.
    """
    if not mpl.rcParams["text.usetex"]:
        return text
    out = []
    for ch in text:
        out.append(_ESCAPES.get(ch, ch))
    return "".join(out)


# ── drawing helpers ───────────────────────────────────────────────────────────
def annotate_optimum(ax: plt.Axes, x: float, y: float, label: str, color: str = PRIMARY) -> None:
    """Mark a curve's optimum with a dot and a short label."""
    ax.plot([x], [y], "o", color=color, markersize=5, zorder=5, clip_on=False)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=mpl.rcParams["font.size"] - 3,
        color=color,
    )


def format_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)
    return ax


def despine(ax: plt.Axes, left: bool = False, bottom: bool = False) -> None:
    """Hide additional spines beyond those the style already removes.

    Minor ticks are disabled along with major ones: SciencePlots enables minor
    ticks per side, so hiding a spine without them leaves a row of small marks
    along an edge that is no longer drawn.
    """
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, which="both")
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, which="both")
