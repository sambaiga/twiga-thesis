"""Design tokens for thesis figures, in the thesis palette.

This module keeps the semantic names used by :mod:`diagrams` --- ``PRIMARY``,
``INFO``, ``WARNING`` and so on --- but binds them to the thesis palette rather
than to the palette of the project the diagram code was vendored from. Vendored
diagrams therefore adopt the thesis identity without being edited.

The palette is the single source of truth. Figures and diagrams should refer to
the semantic names, not to palette positions, so that re-binding a role changes
every figure at once.
"""

from __future__ import annotations

# ── the thesis palette ────────────────────────────────────────────────────────
PALETTE: list[str] = [
    "#107591",  # 0  deep teal
    "#00c0bf",  # 1  bright cyan
    "#f69a48",  # 2  orange
    "#fdcd49",  # 3  yellow
    "#8da798",  # 4  sage
    "#a19368",  # 5  khaki
    "#525252",  # 6  dark grey
    "#a6761d",  # 7  bronze
    "#7035b7",  # 8  purple
    "#cf166e",  # 9  magenta
]


def tint(hex_color: str, amount: float = 0.85) -> str:
    """Blend ``hex_color`` towards white; ``amount`` is the fraction of white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * amount)  # noqa: E731
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


# ── semantic roles, bound to the palette ──────────────────────────────────────
# Thesis figures are coded with the FIRST THREE palette entries only, so that a
# colour means the same thing in every figure:
#
#     PALETTE[0]  #107591  teal    the correct, reference or primary thing
#     PALETTE[1]  #00c0bf  cyan    a second series, or a second failure mode
#     PALETTE[2]  #f69a48  orange  the thing the reader should notice
#
# Grey (TEXT_MUTED) is reserved for reference lines and de-emphasised points and
# is not a data colour. Rebind here rather than in any individual figure.
PRIMARY = PALETTE[0]  # the subject of the figure; the calibrated or correct case
INFO = PALETTE[1]  # a supporting series
WARNING = PALETTE[1]  # a case that needs distinguishing but is not the headline
DANGER = PALETTE[2]  # the failure the figure is about
SUCCESS = PALETTE[0]  # a quantity behaving correctly (same role as PRIMARY)
HIGHLIGHT = PALETTE[2]

TEXT_MUTED = PALETTE[6]  # de-emphasised labels, reference lines, background points
SECONDARY = PALETTE[5]

# Extra accents, used ONLY by the vendored diagrams, which encode more distinct
# concepts than three colours can carry. Thesis illustrations must not use these.
ENERGY_ACCENT = PALETTE[7]
AI_ACCENT = PALETTE[8]

INFO_TINT = tint(INFO)
WARNING_TINT = tint(WARNING)
PRIMARY_TINT = tint(PRIMARY)
DANGER_TINT = tint(DANGER)
SUCCESS_TINT = tint(SUCCESS)

# ── figure-specific roles used by the thesis illustrations ────────────────────
TRUTH = TEXT_MUTED  # a reference line marking a known-correct value
GOOD = PRIMARY  # a method that behaves as intended
BAD = DANGER  # a method that does not

# ── grayscale and text ────────────────────────────────────────────────────────
WHITE = "#FFFFFF"
GRAY_100 = "#F8F9FA"
GRAY_300 = "#DEE2E6"
GRAY_400 = "#CED4DA"
GRAY_600 = "#6B7280"
GRAY_700 = "#495057"
GRAY_900 = "#212529"
BLACK = "#000000"

TEXT_DARK = "#222222"
TEXT_LIGHT = WHITE
GRID = "#DDDDDD"


def matrix_cmap(name: str = "thesis_matrix"):
    """Light-to-dark grey, as the thesis significance matrices use.

    For an unsigned square matrix on ``[0, 1]``: a distance correlation, a
    p-value, a test outcome. Grey carries magnitude without implying a
    direction, and the thesis already draws its Diebold-Mariano matrices this
    way, so a reader meets one convention for square matrices rather than two.

    Not for a signed quantity. A correlation runs from -1 to +1 and grey cannot
    separate the ends: use :func:`diverging_cmap` there.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, [GRAY_100, GRAY_400, GRAY_700])


def diverging_cmap(name: str = "thesis_div", low: str = PRIMARY, high: str = HIGHLIGHT):
    """A ``low``-white-``high`` diverging colormap on the thesis palette.

    For quantities with a meaningful zero and a sign, correlation above all.
    The two ends are the palette's teal and orange: they are the only pair in
    the palette that separates by hue rather than by lightness, so the sign
    survives a greyscale print and a red-green colour deficiency.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, [low, tint(low, 0.55), WHITE, tint(high, 0.55), high])


def sequential_cmap(name: str = "thesis_seq", color: str = PRIMARY):
    """A white-to-``color`` sequential colormap, so heatmaps stay in palette."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, [WHITE, tint(color, 0.55), color])
