r"""Chapter 6 illustration: why a constraint on apparent power needs the joint distribution of P and Q.

Two panels in the thesis line-plot style (:mod:`line_style`), composed with subfigures in LaTeX. Both use the same
marginal distributions of active and reactive power (:mod:`illustrations`' ``_PQ_MU``/``_PQ_SD``); the first joins
them independently, the second with a positive correlation. The region beyond the transformer rating, where the
apparent power exceeds :math:`S^{\mathrm{rated}}`, is shaded, and the probability that the forecast places there is
computed by simulation, so the two panels differ only in the dependence.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from thesis_figures import ch6_illustrations as I; I.render()"
"""

from __future__ import annotations

from pathlib import Path
import shutil

from matplotlib.patches import Ellipse, Polygon
import matplotlib.pyplot as plt
import numpy as np

from twiga_thesis.plotting import line_style as LS, style as S, tokens as T
from twiga_thesis.plotting.illustrations import _PQ_MU, _PQ_SD, _S_RATED, _pq_exceedance

OUT = Path(__file__).resolve().parent / "output"
THESIS = Path(__file__).resolve().parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
NOTE_SIZE = 8
X_MAX, Y_MAX = 170.0, 120.0
CONTOURS = ((2.448, 0.55, 0.15), (1.177, 1.0, 0.35))  # (Mahalanobis radius, line alpha, fill alpha): 95%, 50%


def _margin(ax, i: int, horizontal: bool) -> None:
    """The marginal density of P (top) or Q (right), identical in both panels."""
    grid = np.linspace(0.0, X_MAX if horizontal else Y_MAX, 300)
    dens = np.exp(-0.5 * ((grid - _PQ_MU[i]) / _PQ_SD[i]) ** 2)
    if horizontal:
        ax.fill_between(grid, 0, dens, color=LS.BAND_COLOR, lw=0)
        ax.plot(grid, dens, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
        ax.set_ylim(0, 1.15)
    else:
        ax.fill_betweenx(grid, 0, dens, color=LS.BAND_COLOR, lw=0)
        ax.plot(dens, grid, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
        ax.set_xlim(0, 1.15)
    ax.axis("off")


def constraint_panel(rho: float):
    """Joint density of P and Q with fixed margins and correlation ``rho``, against the apparent-power rating."""
    # Size the figure to the equal-aspect plot, so the marginal strips sit flush against it.
    width = S.TEXTWIDTH_IN * 0.46
    left, right, label_h, top_pad = 0.17, 0.99, 0.47, 0.04
    main_h = width * (right - left) * (6 / 7) * Y_MAX / X_MAX
    height = main_h * 7 / 6 + label_h + top_pad
    fig = plt.figure(figsize=(width, height))
    fig.set_layout_engine("none")  # equal-aspect axes beside marginal strips: laid out by hand
    fig.subplots_adjust(left=left, right=right, bottom=label_h / height, top=1 - top_pad / height)
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 6), wspace=0.03, hspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    _margin(fig.add_subplot(gs[0, 0], sharex=ax), 0, horizontal=True)
    _margin(fig.add_subplot(gs[1, 1], sharey=ax), 1, horizontal=False)

    # Region beyond the rating: everything in the axes outside the quarter circle S = S_rated.
    theta = np.linspace(0, np.pi / 2, 300)
    arc = np.column_stack([_S_RATED * np.cos(theta), _S_RATED * np.sin(theta)])
    arc = arc[arc[:, 1] <= Y_MAX]
    beyond = np.vstack([arc, [[0, Y_MAX], [X_MAX, Y_MAX], [X_MAX, 0], [_S_RATED, 0]]])
    ax.add_patch(Polygon(beyond, closed=True, fc=T.tint(T.HIGHLIGHT, 0.82), ec="none", zorder=1))
    ax.plot(arc[:, 0], arc[:, 1], color=T.HIGHLIGHT, lw=LS.LINE_WIDTH * 1.6, zorder=4)

    cov = np.diag(_PQ_SD) @ np.array([[1.0, rho], [rho, 1.0]]) @ np.diag(_PQ_SD)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for radius, line_alpha, fill_alpha in CONTOURS:
        width, height = 2 * radius * np.sqrt(vals[-1]), 2 * radius * np.sqrt(vals[0])
        ax.add_patch(Ellipse(_PQ_MU, width, height, angle=angle, fc=T.PRIMARY, alpha=fill_alpha, ec="none", zorder=2))
        ax.add_patch(
            Ellipse(
                _PQ_MU,
                width,
                height,
                angle=angle,
                fc="none",
                ec=T.PRIMARY,
                alpha=line_alpha,
                lw=LS.LINE_WIDTH,
                zorder=3,
            )
        )
    ax.plot(*_PQ_MU, "o", ms=2.5, color=T.PRIMARY, zorder=5)

    # Rating label in the free corner left of where the arc enters; probability in the shaded corner.
    ax.text(4, Y_MAX - 22, r"$S^{\mathrm{rated}} = 130$ kVA", fontsize=NOTE_SIZE, ha="left", va="top", zorder=6)
    ax.text(
        X_MAX - 4,
        Y_MAX - 5,
        rf"$\Pr(S > S^{{\mathrm{{rated}}}}) = {100 * _pq_exceedance(rho):.1f}\%$",
        fontsize=NOTE_SIZE,
        ha="right",
        va="top",
        zorder=6,
    )

    LS.finish(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel(r"Active power $P$ (kW)")
    ax.set_ylabel(r"Reactive power $Q$ (kvar)")
    return fig


FIGURES = {
    "fig_ch6_constraint_independent": lambda: constraint_panel(0.0),
    "fig_ch6_constraint_joint": lambda: constraint_panel(0.8),
}


def render(copy_to_thesis: bool = False) -> list[Path]:
    """Write both panels to ``output/`` (PDF and PNG); copy the PDFs to the Chapter 6 folder if asked."""
    LS.configure()
    OUT.mkdir(exist_ok=True)
    paths = []
    for stem, build in FIGURES.items():
        fig = build()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=600, pad_inches=0.02)
        plt.close(fig)
        if copy_to_thesis and THESIS.exists():
            shutil.copy(OUT / f"{stem}.pdf", THESIS / f"{stem}.pdf")
        paths.append(OUT / f"{stem}.pdf")
    return paths
