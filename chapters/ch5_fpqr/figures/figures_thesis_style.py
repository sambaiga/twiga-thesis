r"""Chapter 5 figures in the thesis line-plot style.

Restyled versions of the :mod:`fpqr_figures` panels used in Chapter 5, with the same synthetic conditional
distributions and the same saved worked example, so every value is unchanged. The style is that of
:mod:`line_style`: Computer Modern through LaTeX, teal for the model or the preferred case, orange for the case the
reader should notice, dotted grey references, sentence-case labels, legends below the axes, no titles.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from twiga_thesis.plotting import ch5_figures as F; F.render()"
"""

from __future__ import annotations

from pathlib import Path
import shutil

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from twiga_thesis.plotting import line_style as LS, style as S, tokens as T

from .fpqr_figures_shared import (
    _TAU_NPY,
    MIDDAY,
    NIGHT,
    UNIFORM,
    _density_at,
    _mixture,
    _optimal_grid_at,
    _quantile_at,
    _step_density,
)

OUT = Path(__file__).resolve().parent / "output"
THESIS = Path(__file__).resolve().parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-5"
NOTE_SIZE = 8
ROW = (0.31, 0.82)
DIVERGING = LinearSegmentedColormap.from_list("thesis_diverging", [T.PRIMARY, T.WHITE, T.HIGHLIGHT])


def day_fan():
    """A day of forecasts: nine quantile trajectories whose spread and shape change with the sun."""
    fine = np.linspace(0, 24, 241)
    curves = np.array([[_quantile_at(h, np.array([t]))[0] for h in fine] for t in UNIFORM])
    fig, ax = LS.new_panel(*ROW)
    for i in range(len(UNIFORM) // 2):
        ax.fill_between(fine, curves[i], curves[-1 - i], color=T.PRIMARY, alpha=0.10, lw=0)
    for i in range(len(UNIFORM)):
        ax.plot(fine, curves[i], color=T.PRIMARY, lw=0.5)
    ax.plot(fine, curves[len(UNIFORM) // 2], color=T.PRIMARY, lw=LS.LINE_WIDTH * 1.6)
    top = curves.max() + 0.12
    for h, label in ((NIGHT, "Night"), (MIDDAY, "Midday")):
        ax.axvline(h, color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH, zorder=1)
        ax.text(h + 0.5, top, label, ha="left", va="bottom", fontsize=NOTE_SIZE, color=T.TEXT_MUTED)
    LS.finish(ax)
    LS.hour_axis(ax)
    ax.set_ylim(curves.min() - 0.1, top + 0.3)
    ax.set_yticks([0, 1])
    ax.set_ylabel("Net load")
    handles = [
        Line2D([], [], color=T.PRIMARY, lw=LS.LINE_WIDTH * 1.6, label="Median"),
        Line2D([], [], color=T.PRIMARY, lw=0.5, label="Quantiles"),
    ]
    LS.legend_below(ax, handles, [h.get_label() for h in handles], ncol=2)
    return fig


def _reconstruction(h: float):
    """Density implied by nine uniform and nine adaptive quantiles, against the true density at hour ``h``."""
    mix = _mixture(h)
    lo = min(m - 4.2 * s for _, m, s in mix)
    hi = max(m + 4.2 * s for _, m, s in mix)
    x = np.linspace(lo, hi, 900)
    fig, ax = LS.new_panel(*ROW)
    ax.plot(x, _density_at(h, x), color=T.TEXT_MUTED, lw=LS.LINE_WIDTH * 1.3, label="True", zorder=4)
    for grid, colour, name in ((UNIFORM, T.HIGHLIGHT, "Uniform"), (_optimal_grid_at(h), T.PRIMARY, "Adaptive")):
        quantile = _quantile_at(h, grid)
        xs, ys = _step_density(grid, quantile)
        ax.step(xs, ys, where="post", color=colour, lw=LS.LINE_WIDTH * 1.2, label=name, zorder=3)
        ax.plot(quantile, np.zeros_like(quantile), "|", color=colour, ms=6, mew=0.9, zorder=5, clip_on=False)
    LS.finish(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.set_xlabel("Net load")
    ax.set_ylabel("Density")
    # Compact spacing keeps the three entries on one row, level with the fan panel's legend.
    legend = ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        frameon=False,
        handlelength=1.0,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    legend._thesis_below = True
    return fig


def reconstruction_night():
    """Night: a narrow unimodal distribution, reconstructed equally well by both grids."""
    return _reconstruction(NIGHT)


def reconstruction_midday():
    """Midday: a bimodal distribution, whose modes only the adaptive grid separates."""
    return _reconstruction(MIDDAY)


def crossing():
    """Without the non-crossing penalty, adjacent quantile estimates can invert."""
    tau = _optimal_grid_at(MIDDAY)
    base = _quantile_at(MIDDAY, tau)
    span = base.max() - base.min()
    loose = base + 0.16 * span * np.array([0.9, -1.0, 1.1, -1.0, 0.9, -1.0, 1.2, -0.9, 0.7])
    bad = np.nonzero(np.diff(loose) < 0)[0]
    fig, ax = LS.new_panel(0.62, 0.48)
    for i in bad:
        ax.plot(
            tau[i : i + 2], loose[i : i + 2], color=T.tint(T.HIGHLIGHT, 0.6), lw=5, solid_capstyle="round", zorder=1
        )
    ax.plot(
        tau,
        base,
        "o-",
        color=T.PRIMARY,
        ms=3.5,
        lw=LS.LINE_WIDTH * 1.2,
        zorder=3,
        label=r"With $\mathcal{L}_{\mathrm{nc}}$",
    )
    ax.plot(
        tau,
        loose,
        "o--",
        color=T.HIGHLIGHT,
        ms=3.5,
        lw=LS.LINE_WIDTH * 1.2,
        zorder=2,
        label=r"Without $\mathcal{L}_{\mathrm{nc}}$",
    )
    ax.text(
        0.02,
        0.95,
        f"{len(bad)} inverted pairs (shaded)",
        transform=ax.transAxes,
        fontsize=NOTE_SIZE,
        color=T.TEXT_MUTED,
        va="top",
    )
    LS.finish(ax)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.margins(y=0.15)
    ax.set_xlabel(r"Quantile level $\tau$")
    ax.set_ylabel("Net load")
    LS.legend_below(ax, ncol=2)
    return fig


def pipeline():
    """The forward pass through the FPQR head for one forecast on MLVS-PT, with the tensor at each stage."""
    d = np.load(_TAU_NPY / "s563_worked_example.npz")
    tau, quant, actual = d["tau"], d["quant"], d["actual"]
    hist, lead = d["hist"], int(d["lead"])
    n_slots, horizon = tau.shape
    hours = 1.0 + 0.5 * np.arange(horizon)
    label_size, tick_size = 8, 7

    fig = plt.figure(figsize=(S.TEXTWIDTH_IN * 0.85, S.TEXTWIDTH_IN * 0.95))
    gs = fig.add_gridspec(5, 2, width_ratios=[0.32, 0.68], hspace=0.85, wspace=0.28)
    rows = [
        ("Input", r"$\phi(g_h)$", "encode history", r"$(B,\,z)$", T.TEXT_MUTED),
        ("FPN", r"$\hat{\tau}_{\theta}$", "softmax, cumulative sum", r"$(B,\,N,\,H)$", T.PRIMARY),
        ("TauEmb", r"$\Phi(\hat{\tau})$", "cosine basis", r"$(B,\,N,\,H,\,z)$", T.PRIMARY),
        ("QVN", r"$Q_{\theta}(\hat{\tau})$", "gate, decode", r"$(B,\,N,\,H,\,D)$", T.PRIMARY),
        ("CDF", r"$\hat{y}$, PI", "sort, interpolate", r"$(B,\,H,\,D)$", T.HIGHLIGHT),
    ]
    for i, (stage, emits, op, shape, colour) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.58), 0.5, 0.36, fc=T.tint(colour, 0.85), ec=colour, lw=0.6))
        ax.text(0.27, 0.76, stage, ha="center", va="center", fontsize=label_size)
        ax.text(0.58, 0.76, emits, ha="left", va="center", fontsize=label_size)
        ax.text(0.02, 0.38, op, ha="left", va="center", fontsize=tick_size, color=T.TEXT_MUTED)
        ax.text(0.02, 0.14, shape, ha="left", va="center", fontsize=tick_size, color=T.TEXT_MUTED)
        if i < len(rows) - 1:
            ax.annotate(
                "",
                xy=(0.27, -0.3),
                xytext=(0.27, 0.0),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": T.GRAY_600, "lw": 0.7},
            )

    def tidy(ax, xlabel, ylabel):
        LS.finish(ax)
        ax.tick_params(labelsize=tick_size)
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=1)
        ax.set_ylabel(ylabel, fontsize=label_size)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(-len(hist), 0) * 0.5, hist, color=T.TEXT_MUTED, lw=LS.LINE_WIDTH)
    ax.axvline(0, color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH)
    tidy(ax, "Hours before forecast", "kW")

    ax = fig.add_subplot(gs[1, 1])
    for n in range(n_slots):
        ax.plot(hours, tau[n], color=T.PRIMARY, lw=0.6)
    ax.axvline(hours[lead], color=T.HIGHLIGHT, ls=":", lw=LS.LINE_WIDTH)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xticks([0, 6, 12, 18, 24])
    tidy(ax, "Hour of day", r"$\hat{\tau}$")

    ax = fig.add_subplot(gs[2, 1])
    basis = np.cos(np.pi * np.arange(1, 33)[None, :] * tau[:, lead][:, None])
    im = ax.imshow(basis, aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_yticks([0, n_slots - 1])
    ax.set_yticklabels([1, n_slots])
    tidy(ax, r"Cosine index $k$", "Slot")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cbar.ax.tick_params(labelsize=tick_size, length=2)
    cbar.outline.set_visible(False)

    ax = fig.add_subplot(gs[3, 1])
    for n in range(n_slots // 2):
        ax.fill_between(hours, quant[n], quant[-1 - n], color=T.PRIMARY, alpha=0.12, lw=0)
    for n in range(n_slots):
        ax.plot(hours, quant[n], color=T.PRIMARY, lw=0.5)
    ax.plot(hours, actual, color=T.HIGHLIGHT, lw=LS.LINE_WIDTH * 1.2)
    ax.axvline(hours[lead], color=T.HIGHLIGHT, ls=":", lw=LS.LINE_WIDTH)
    ax.set_xticks([0, 6, 12, 18, 24])
    tidy(ax, "Hour of day", "kW")

    ax = fig.add_subplot(gs[4, 1])
    t, q = tau[:, lead], np.sort(quant[:, lead])
    y = float(actual[lead])
    for level in (0.05, 0.95):
        ax.axvline(level, color=LS.REFERENCE_COLOR, ls=":", lw=0.6, zorder=1)
    ax.plot(t, q, "-", color=T.PRIMARY, lw=LS.LINE_WIDTH, zorder=3)
    ax.plot([0, t[0]], [q[0], q[0]], "--", color=T.PRIMARY, lw=LS.LINE_WIDTH, zorder=3)
    ax.plot([t[-1], 1], [q[-1], q[-1]], "--", color=T.PRIMARY, lw=LS.LINE_WIDTH, zorder=3)
    ax.plot(t, q, "o", color=T.PRIMARY, ms=3.2, zorder=5)
    pit = float(np.interp(y, q, t))
    ax.axhline(y, color=T.HIGHLIGHT, lw=LS.LINE_WIDTH, zorder=4)
    ax.plot([pit], [y], "o", color=T.HIGHLIGHT, ms=4, zorder=6)
    ax.annotate(
        rf"$\tau = {pit:.2f}$", xy=(pit, y), xytext=(4, -9), textcoords="offset points", fontsize=tick_size, ha="left"
    )
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    lo_y, hi_y = q.min(), q.max()
    ax.set_ylim(lo_y - 0.12 * (hi_y - lo_y), hi_y + 0.12 * (hi_y - lo_y))
    tidy(ax, r"Quantile level $\tau$", "kW")
    handles = [
        Line2D([], [], color=T.PRIMARY, marker="o", ms=3.2, lw=LS.LINE_WIDTH, label="Learned quantiles"),
        Line2D([], [], color=T.PRIMARY, ls="--", lw=LS.LINE_WIDTH, label="Flat tails"),
        Line2D([], [], color=T.HIGHLIGHT, lw=LS.LINE_WIDTH * 1.2, label="Observed"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=tick_size, bbox_to_anchor=(0.62, 0.0)
    )
    return fig


FIGURES = {
    "day_fan": ("fig_fpqr_day_fan", day_fan),
    "recon_night": ("fig_fpqr_recon_night", reconstruction_night),
    "recon_midday": ("fig_fpqr_recon_midday", reconstruction_midday),
    "pipeline": ("fig_fpqr_pipeline", pipeline),
    "crossing": ("fig_fpqr_crossing", crossing),
}


def render(names: list[str] | None = None, copy_to_thesis: bool = False) -> list[Path]:
    """Render the named figures (all by default) to ``output/``; copy to the Chapter 5 folder if asked."""
    LS.configure()
    OUT.mkdir(exist_ok=True)
    paths = []
    for name in names or list(FIGURES):
        stem, build = FIGURES[name]
        fig = build()
        if name == "pipeline":  # a multi-axes figure: save without the single-panel legend placement
            for ext in ("pdf", "png"):
                fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
        else:
            LS.save(fig, OUT / stem)
        if copy_to_thesis and THESIS.exists():
            shutil.copy(OUT / f"{stem}.pdf", THESIS / f"{stem}.pdf")
        paths.append(OUT / f"{stem}.pdf")
    return paths
