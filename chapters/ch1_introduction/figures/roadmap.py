"""Thesis roadmap for Chapter 1: research questions, the chapters that answer them, and how they connect.

The layout follows the original roadmap: thesis stages in a left column joined by downward arrows,
the chapters that address each stage on the right, and a "Research questions" band spanning RQ1 to
RQ4, with RQ2 and RQ3 grouped as single-target model development. Colour is limited to the
thesis's three data colours, applied to the research questions; chapters and the labelled links
between questions are grey. Type matches the chapter figures: Computer Modern through LaTeX,
regular weight, no italics.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from thesis_figures import ch1_roadmap as R; R.render()"
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt

from twiga_thesis.plotting import style as S, tokens as T

OUT = Path(__file__).resolve().parent / "output"

# The thesis's three data colours: teal for RQ1, cyan for the RQ2/RQ3 group (single-target model
# development), orange for RQ4. Framing, conclusion, chapter boxes and links are neutral grey. Dark
# text on cyan and orange, whose contrast with white is below 3:1.
COLOR = {
    "intro": T.TEXT_MUTED,
    "rq1": T.PALETTE[0],
    "rq2": T.PALETTE[1],
    "rq3": T.PALETTE[1],
    "rq4": T.PALETTE[2],
    "end": T.TEXT_MUTED,
}
TEXT_ON = {"rq2": T.TEXT_DARK, "rq3": T.TEXT_DARK, "rq4": T.TEXT_DARK}
LINK = T.GRAY_600

STAGES = {
    "intro": "Introduction and framing",
    "rq1": "RQ1: Setting the stage",
    "rq2": "RQ2: Lightweight forecasting",
    "rq3": "RQ3: Uncertainty estimation\nand calibration",
    "rq4": "RQ4: Multi-target forecasting",
    "end": "Conclusion",
}

# Chapters per stage: (number, title, principal output). Numbers follow the reading order.
CHAPTERS = {
    "intro": [
        (1, "Introduction", "Motivation, research questions, contributions"),
        (2, "Background and Related Works", "Literature review, datasets, simulated records"),
    ],
    "rq1": [(3, "Methodological Foundations for Net-Load Forecasting", "Evaluation protocol, feature fusion, CWE")],
    "rq2": [(4, "Lightweight DNNs for Net-Load Forecasting", "MLPF architecture")],
    "rq3": [
        (5, "Fully-Parameterised Quantile Regression", "FPQR: learned quantile fractions and values"),
        (6, "Distribution-Free Coverage Guarantees", "MLPGAM with adaptive conformal calibration"),
    ],
    "rq4": [(7, "Probabilistic Forecasting of Multiple LV Variables", "Forecast types and empirical copula")],
    "end": [(8, "Conclusion and Future Work Directions", "Findings, limitations, future research")],
}

TITLE_SIZE, NOTE_SIZE, STAGE_SIZE, LINK_SIZE = 8.0, 7.2, 8.4, 7.2

# Geometry in axes units (the axes span the whole figure).
WIDTH, HEIGHT = 10.0, 8.85
STAGE_X, STAGE_W, STAGE_H = 1.62, 2.76, 0.66
CHAP_X, CHAP_W, CHAP_H = 4.84, 5.14, 0.66
STAGE_Y = {"intro": 7.65, "rq1": 6.3, "rq2": 4.95, "rq3": 3.3, "rq4": 1.75, "end": 0.35}
CHAP_GAP = 0.76
SPINE_RQ1, SPINE_RQ2 = 0.95, 1.38


def _box(ax, x, y, w, h, *, face, edge, lw=1.0, ls="-", z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=face, ec=edge, lw=lw, ls=ls, zorder=z
        )
    )


def _arrow(ax, start, end, color=T.GRAY_700, lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=3
        )
    )


def _line(ax, xs, ys, color, lw=0.9):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=3)


def _mid(key: str) -> float:
    return STAGE_Y[key] + STAGE_H / 2


def _stage(ax, key):
    y = STAGE_Y[key]
    _box(ax, STAGE_X, y, STAGE_W, STAGE_H, face=COLOR[key], edge="none", z=4)
    ax.text(
        STAGE_X + STAGE_W / 2,
        y + STAGE_H / 2,
        STAGES[key],
        ha="center",
        va="center",
        color=TEXT_ON.get(key, T.WHITE),
        fontsize=STAGE_SIZE,
        linespacing=1.15,
        zorder=5,
    )


def _chapters(ax, key):
    """Chapter boxes for one stage, joined to it by an arrow or a fork."""
    chapters = CHAPTERS[key]
    centre = _mid(key)
    ys = [centre] if len(chapters) == 1 else [centre + CHAP_GAP / 2, centre - CHAP_GAP / 2]
    for (number, title, note), yc in zip(chapters, ys, strict=True):
        _box(ax, CHAP_X, yc - CHAP_H / 2, CHAP_W, CHAP_H, face=T.WHITE, edge=T.GRAY_400, lw=0.8)
        ax.text(CHAP_X + 0.3, yc, str(number), ha="center", va="center", fontsize=TITLE_SIZE, color=T.TEXT_MUTED)
        _line(ax, [CHAP_X + 0.52] * 2, [yc - CHAP_H / 2 + 0.1, yc + CHAP_H / 2 - 0.1], T.GRAY_400, lw=0.6)
        ax.text(CHAP_X + 0.62, yc + 0.13, title, ha="left", va="center", fontsize=TITLE_SIZE, color=T.TEXT_DARK)
        ax.text(CHAP_X + 0.62, yc - 0.15, note, ha="left", va="center", fontsize=NOTE_SIZE, color=T.TEXT_MUTED)
    right = STAGE_X + STAGE_W
    if len(ys) == 1:
        _arrow(ax, (right, centre), (CHAP_X, centre))
    else:
        fork = right + 0.3
        _line(ax, [right, fork], [centre, centre], T.GRAY_700)
        _line(ax, [fork, fork], [min(ys), max(ys)], T.GRAY_700)
        for yc in ys:
            _arrow(ax, (fork, yc), (CHAP_X, yc))


def _link_label(ax, x, y, text, color, rotation=0, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=LINK_SIZE, color=color, rotation=rotation, zorder=5)


def roadmap() -> plt.Figure:
    """The thesis roadmap figure."""
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * HEIGHT / WIDTH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    # Research-question band, spanning RQ1 to RQ4.
    top, bottom = STAGE_Y["rq1"] + STAGE_H + 0.15, STAGE_Y["rq4"] - 0.15
    ax.add_patch(Rectangle((0.05, bottom), 0.4, top - bottom, fc=T.GRAY_100, ec="none", zorder=1))
    ax.text(0.25, (top + bottom) / 2, "Research questions", rotation=90, ha="center", va="center", fontsize=STAGE_SIZE)

    # RQ2/RQ3 group: single-target model development.
    gx, gy = STAGE_X - 0.15, STAGE_Y["rq3"] - 0.15
    gw, gh = STAGE_W + 0.3, STAGE_Y["rq2"] + STAGE_H + 0.15 - gy
    _box(ax, gx, gy, gw, gh, face=T.GRAY_100, edge=T.GRAY_600, lw=0.8, ls=(0, (3, 2)), z=1)

    for key in STAGES:
        _stage(ax, key)
        _chapters(ax, key)

    # Thesis flow: framing leads to RQ1, RQ4 to the conclusion.
    cx = STAGE_X + STAGE_W / 2
    _arrow(ax, (cx, STAGE_Y["intro"]), (cx, STAGE_Y["rq1"] + STAGE_H))
    _arrow(ax, (cx, STAGE_Y["rq4"]), (cx, STAGE_Y["end"] + STAGE_H))

    # RQ1 -> RQ2, RQ3, RQ4: the evaluation protocol and metrics are used throughout.
    c1 = LINK
    _line(ax, [STAGE_X, SPINE_RQ1], [_mid("rq1")] * 2, c1)
    _line(ax, [SPINE_RQ1, SPINE_RQ1], [_mid("rq4") + 0.12, _mid("rq1")], c1)
    for key in ("rq2", "rq3", "rq4"):
        _arrow(ax, (SPINE_RQ1, _mid(key) + 0.12), (STAGE_X, _mid(key) + 0.12), c1)
    _link_label(ax, SPINE_RQ1 - 0.13, (_mid("rq1") + _mid("rq4")) / 2, "Evaluation protocol, CWE", c1, 90, "center")

    # RQ2 -> RQ4: the MLPF backbone of the multi-target models.
    c2 = LINK
    _line(ax, [gx, SPINE_RQ2], [_mid("rq2") - 0.12] * 2, c2)
    _line(ax, [SPINE_RQ2, SPINE_RQ2], [_mid("rq4") - 0.12, _mid("rq2") - 0.12], c2)
    _arrow(ax, (SPINE_RQ2, _mid("rq4") - 0.12), (STAGE_X, _mid("rq4") - 0.12), c2)
    _link_label(ax, SPINE_RQ2 - 0.13, (_mid("rq2") + _mid("rq3")) / 2, "MLPF backbone", c2, 90, "center")

    # RQ2 <-> RQ3 inside the group: the backbone down, MLPGAM back up.
    gap_top, gap_bottom = STAGE_Y["rq2"], STAGE_Y["rq3"] + STAGE_H
    gap = gap_top - gap_bottom
    _arrow(ax, (cx - 0.95, gap_top), (cx - 0.95, gap_bottom), c2)
    _link_label(ax, cx - 0.87, gap_bottom + 0.7 * gap, "MLPF backbone", c2)
    c3 = LINK
    _arrow(ax, (cx + 0.95, gap_bottom), (cx + 0.95, gap_top), c3)
    _link_label(ax, cx + 0.87, gap_bottom + 0.3 * gap, "MLPGAM", c3, ha="right")

    # RQ3 -> RQ4: the probabilistic heads and their calibration.
    _arrow(ax, (cx, gy), (cx, STAGE_Y["rq4"] + STAGE_H), c3)
    _link_label(ax, cx + 0.08, (gy + STAGE_Y["rq4"] + STAGE_H) / 2, "FPQR and probabilistic heads", c3)
    return fig


# The thesis body is set in Helvetica (istulthesis option ``Helvetica``: \rmdefault = phv), so the roadmap, which is
# all text, uses the same face rather than the Computer Modern of the data figures. It is rendered without LaTeX
# because the roadmap holds no mathematics; TrueType embedding keeps the text selectable in the PDF.
THESIS_FONT = ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]


def render(show: bool = False, font: str = "thesis") -> Path:
    """Write the roadmap to ``output/fig_ch1_roadmap.pdf`` (and a PNG preview) and return its path.

    Args:
        show: Also display the figure.
        font: ``"thesis"`` sets the text in Helvetica, the thesis body font; ``"latex"`` in Computer Modern
            through LaTeX, as the data figures are.
    """
    if font == "latex":
        S.configure()
    else:
        S.configure(latex=False)
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": THESIS_FONT,
                "pdf.fonttype": 42,
            }
        )
    OUT.mkdir(exist_ok=True)
    fig = roadmap()
    path = OUT / "fig_ch1_roadmap.pdf"
    fig.savefig(path, pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=200, pad_inches=0.02)
    if show:
        plt.show()
    plt.close(fig)
    return path
