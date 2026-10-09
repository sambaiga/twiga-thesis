"""Bootstrap Icons, rendered as matplotlib text glyphs.

Bootstrap Icons (https://icons.getbootstrap.com/, MIT licensed) is this
project's own site's icon set already (see the `bi bi-*` classes in
`_quarto.yml`). The vendored font here (`assets/bootstrap-icons.ttf`, see
`assets/BOOTSTRAP_ICONS_NOTICE.txt`) is the same one Quarto bundles into
`_book/site_libs/bootstrap/` at render time, converted once from `.woff` to
`.ttf`: matplotlib's bundled FreeType cannot open `.woff` directly
(`FT_Open_Face ... broken table`), only `.ttf`/`.otf`. Using it needs no new
runtime dependency - matplotlib is already required, and `fontTools` (used
only for that one conversion, not at import time here) was already a
transitive dependency.

Importing this module registers the font with matplotlib's font manager as
a side effect, the same pattern `ark.plot.matplot_theme.resolve_font` uses
for body fonts.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

_FONT_PATH = Path(__file__).parent / "assets" / "bootstrap-icons.ttf"
_FONT_FAMILY = "bootstrap-icons"

# The font is git-ignored (`*.ttf`), so a fresh checkout may lack it; register it only when present so importing
# this module (and the experiment modules that use it) never fails, e.g. at test collection in CI.
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))

# Bootstrap Icons v1.13.1 codepoints (icons.getbootstrap.com). Only the
# icons an actual diagram has needed so far; add more by looking up the
# codepoint on that site and copying it in here.
ICONS: dict[str, str] = {
    "house-fill": "",
    "sun-fill": "",
    "lightning-charge-fill": "",
    "hdd-network-fill": "",
    "record-circle-fill": "",
    "building-fill": "",
    "ev-front-fill": "",
    "arrow-down-up": "",
    "speedometer2": "",
    "scissors": "",
    "dice-5-fill": "",
    "check-circle-fill": "",
    "x-circle-fill": "",
}


def icon_font(size: float = 24) -> FontProperties:
    """A `FontProperties` for drawing a Bootstrap Icon glyph via `ax.text`.

    Args:
        size: Glyph size in points, same units as any other matplotlib
            font size.

    Returns:
        A `FontProperties` set to the vendored Bootstrap Icons font.

    Examples:
        >>> ax.text(0, 0, ICONS["house-fill"], fontproperties=icon_font(28), color=INFO, ha="center", va="center")
    """
    return FontProperties(family=_FONT_FAMILY, size=size)


def icon_patch(
    ax: Axes,
    name: str,
    xy: tuple[float, float],
    height: float,
    color: str,
    zorder: int = 5,
) -> PathPatch:
    """Draw a Bootstrap Icon as a filled path rather than as text.

    `ax.text` routes through LaTeX when `text.usetex` is on, and LaTeX cannot
    typeset the private-use codepoints an icon font lives in. Converting the
    glyph to a path sidesteps the text pipeline entirely, so the same icon
    renders both in a notebook and in a figure built for the thesis. The glyph
    is placed in data coordinates and scaled to `height` data units, which
    assumes the axes has an equal aspect ratio.

    Args:
        ax: Axes to draw on.
        name: Key into `ICONS`.
        xy: Centre of the glyph, in data coordinates.
        height: Glyph height, in data units.
        color: Fill colour.
        zorder: Drawing order.

    Returns:
        The `PathPatch` that was added to the axes.
    """
    path = TextPath((0, 0), ICONS[name], size=1.0, prop=icon_font(1.0))
    box = path.get_extents()
    transform = (
        Affine2D()
        .translate(-box.x0 - box.width / 2, -box.y0 - box.height / 2)
        .scale(height / box.height)
        .translate(*xy)
    )
    patch = PathPatch(path, transform=transform + ax.transData, facecolor=color, edgecolor="none", zorder=zorder)
    ax.add_patch(patch)
    return patch
