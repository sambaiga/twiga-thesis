r"""Illustrations for the two joint methods of Chapter 6.

Drawn rather than measured. Each panel isolates one idea that the surrounding
equations state precisely but do not make visible:

``elliptical_*``
    Why the choice between a Gaussian and a Laplace joint density matters when
    the constraint is an exceedance. Both are elliptical and share the same
    covariance here, so any difference is the radial part of the density alone.
``covariance_block``
    What is and is not modelled: one covariance across targets per lead time,
    rather than a covariance over every lead time and target at once.
``tau_allocation`` and ``target_conditioning``
    Where a shared grid spends its slots against where conditioned grids do, and
    the architecture that produces the second from one network rather than three.
``copula_*``
    How a copula supplies dependence without touching the margins, and what that
    changes for a constraint. The last two panels share their margins exactly and
    differ only in whether the copula is applied.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from twiga_thesis.plotting import ch6_methods as M; M.render_all()"
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
from twiga.distributions.nn.parametric import MultivariateLaplace

from twiga_thesis.plotting import style as S, tokens as T

# The same small secondary substation as the illustration in Section 6.1: mean
# active and reactive power, their spreads, and the nameplate. Sharing the
# numbers means the two figures can be read against each other.
_MU = np.array([78.0, 42.0])
_SD = np.array([20.0, 14.0])
_RHO_E = 0.6
_RATED_KVA = 130.0
_COV = np.diag(_SD) @ np.array([[1.0, _RHO_E], [_RHO_E, 1.0]]) @ np.diag(_SD)
_TRIL = torch.tensor(np.linalg.cholesky(_COV), dtype=torch.float32)
_LOC = torch.tensor(_MU, dtype=torch.float32)
_STYLE = {"Gaussian": (T.TEXT_MUTED, "--"), "Laplace": (T.PRIMARY, "-")}


def _families() -> dict[str, torch.distributions.Distribution]:
    return {
        "Gaussian": torch.distributions.MultivariateNormal(_LOC, scale_tril=_TRIL),
        "Laplace": MultivariateLaplace(_LOC, _TRIL),
    }


def elliptical_tail() -> plt.Figure:
    """Marginal densities on a log scale, where the tails separate visibly.

    At equal probability the two families' contours are almost indistinguishable,
    because the difference is in how quickly the density decays rather than in
    the shape of any one contour. A logarithmic axis is what makes it legible.
    """
    grid = torch.linspace(0.0, 200.0, 500)
    pts = torch.stack([grid, torch.full_like(grid, float(_MU[1]))], dim=-1)

    fig, ax = plt.subplots(figsize=S.panel(0.42, aspect=0.86))
    for name, dist in _families().items():
        colour, dash = _STYLE[name]
        dens = dist.log_prob(pts).exp().numpy()
        ax.semilogy(grid.numpy(), dens / dens.max(), dash, color=colour, lw=1.4, label=name)
    ax.set_ylim(1e-6, 3)
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Active power $P$ (kW)")
    ax.set_ylabel("Density, normalised")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    return fig


def elliptical_exceedance() -> plt.Figure:
    r"""How the tail choice changes the probability the constraint asks about.

    Both families carry the same covariance, so any difference here comes from
    the radial part of the density alone. The nameplate is marked because that
    is the rating at which the operator's question is actually posed.
    """
    torch.manual_seed(0)
    ratings = np.linspace(90.0, 220.0, 45)
    fig, ax = plt.subplots(figsize=S.panel(0.42, aspect=0.86))
    at_rated = {}
    for name, dist in _families().items():
        colour, dash = _STYLE[name]
        s_mag = dist.sample((400_000,)).norm(dim=-1).numpy()
        ax.semilogy(ratings, [(s_mag > r).mean() for r in ratings], dash, color=colour, lw=1.4, label=name)
        at_rated[name] = (s_mag > _RATED_KVA).mean()

    ax.axvline(_RATED_KVA, color=T.DANGER, lw=1.0, zorder=1)
    ax.annotate(
        rf"$S^{{\mathrm{{rated}}}}$" "\n" rf"${at_rated['Laplace'] / at_rated['Gaussian']:.1f}\times$",
        xy=(_RATED_KVA, 3e-4),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color=T.DANGER,
        va="bottom",
    )
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(90, 220)
    ax.set_xlabel(r"Rating $S^{\mathrm{rated}}$ (kVA)")
    ax.set_ylabel(r"$\Pr(S > S^{\mathrm{rated}})$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return fig


def covariance_block() -> plt.Figure:
    """Which entries of the covariance the head predicts, and which it does not."""
    horizon, targets = 4, 3
    size = horizon * targets
    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.94))

    ax.add_patch(Rectangle((0, 0), size, size, facecolor=T.TEXT_MUTED, alpha=0.12, lw=0))
    for h in range(horizon):
        o = h * targets
        ax.add_patch(Rectangle((o, size - o - targets), targets, targets, facecolor=T.PRIMARY, alpha=0.55, lw=0))
    for k in range(size + 1):
        lw = 0.9 if k % targets == 0 else 0.25
        ax.plot([k, k], [0, size], color="white", lw=lw)
        ax.plot([0, size], [k, k], color="white", lw=lw)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([h * targets + targets / 2 for h in range(horizon)])
    ax.set_xticklabels([str(h + 1) for h in range(horizon)], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Lead time")
    ax.annotate(
        "modelled",
        xy=(1.5, size - 1.5),
        xytext=(6.5, size - 1.0),
        fontsize=8,
        color=T.PRIMARY,
        arrowprops={"arrowstyle": "-", "color": T.PRIMARY, "lw": 0.8},
    )
    ax.annotate("not modelled", xy=(size / 2, 1.2), fontsize=8, color=T.TEXT_MUTED, ha="center")
    ax.set_title("Covariance structure", fontsize=9, pad=4)
    return fig


# ── quantile grids ────────────────────────────────────────────────────────────
# Chapter 5 plots the learned fractions in tau space, and the multi-target
# question is the same question asked of three targets at once: does one set of
# levels serve all of them, or does each need its own?
_TAU_LO, _TAU_HI = 0.02, 0.98
_N_SLOT = 9
_ROWS = (
    ("$P$", 1.6),  # skewed once generation is present: levels pushed outward
    ("$Q$", 0.0),  # broad and near symmetric
    ("$V$", -0.5),  # tight, so useful levels sit close to the centre
)


def _shared_levels() -> np.ndarray:
    """One grid for every target: a compromise, so it spreads to cover all three."""
    return np.linspace(_TAU_LO, _TAU_HI, _N_SLOT)


def _target_levels(name: str) -> np.ndarray:
    """A grid conditioned on the target, concentrated where that target needs it.

    Active power is skewed once generation is present, so its levels crowd the
    upper tail. Reactive power is broad and near symmetric, so a near-uniform
    grid serves it. Voltage occupies a narrow band, so its levels concentrate
    near the centre and the outermost slots are wasted on it.
    """
    u = np.linspace(0.0, 1.0, _N_SLOT)
    warp = {
        "$P$": u**2.2,  # crowd the upper tail
        "$Q$": u,  # near uniform
        "$V$": 0.5 + 0.5 * np.sign(u - 0.5) * np.abs(2 * u - 1) ** 1.9,  # crowd the centre
    }[name]
    return _TAU_LO + (_TAU_HI - _TAU_LO) * warp


def tau_allocation() -> plt.Figure:
    r"""Where each grid spends its slots, in :math:`\tau` space.

    The shared grid is one column of levels applied to every target, so it is
    drawn as vertical rules crossing all three rows. The conditioned grids move
    independently, which is the whole of the difference between
    ``eqn:ch6-shared-grid`` and ``eqn:ch6-per-target-grid``.
    """
    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.72))
    shared = _shared_levels()

    # The clamp keeps fractions inside (0, 1); a grid under pressure piles up here.
    for lo, hi in ((0.0, _TAU_LO), (_TAU_HI, 1.0)):
        ax.axvspan(lo, hi, color=T.DANGER, alpha=0.10, lw=0, zorder=0)
    for v in shared:
        ax.axvline(v, color=T.TEXT_MUTED, lw=0.8, ls="--", alpha=0.85, zorder=1)

    for row, name in enumerate(_ROWS):
        y = len(_ROWS) - 1 - row
        ax.plot(_target_levels(name), np.full(_N_SLOT, y), "|", ms=15, mew=2.0, color=T.PRIMARY, zorder=3)
        ax.text(-0.035, y, name, ha="right", va="center", fontsize=9)

    ax.set_yticks([])
    ax.set_ylim(-0.6, len(_ROWS) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(r"quantile fraction $\hat{\tau}$")
    ax.plot([], [], "--", color=T.TEXT_MUTED, lw=0.8, label="shared")
    ax.plot([], [], "|", ms=9, mew=1.6, color=T.PRIMARY, ls="none", label="per-target")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.annotate("clamp", xy=(0.005, -0.45), fontsize=7, color=T.DANGER, ha="left", va="center")
    return fig


def target_conditioning() -> plt.Figure:
    """How three grids come from one network: one embedding added per target."""
    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.72))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, colour, alpha=0.85, fontsize=8):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, alpha=alpha, lw=0, zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="white", zorder=3)

    box(0.2, 2.4, 1.7, 1.2, r"context" "\n" r"$\mathbf{z}$", T.SECONDARY)
    for i, (name, y) in enumerate(zip(("$P$", "$Q$", "$V$"), (4.2, 2.4, 0.6), strict=False)):
        # The embedding is what makes the grid target-specific; everything
        # downstream is shared, which is the point of the construction.
        box(2.6, y, 1.5, 1.2, rf"$+\,\mathbf{{e}}_{{{name.strip('$')}}}$", T.PRIMARY, alpha=0.55 + 0.15 * i)
        ax.annotate(
            "", xy=(2.6, y + 0.6), xytext=(1.9, 3.0), arrowprops={"arrowstyle": "->", "color": T.TEXT_MUTED, "lw": 0.9}
        )
        ax.annotate(
            "",
            xy=(5.0, y + 0.6),
            xytext=(4.1, y + 0.6),
            arrowprops={"arrowstyle": "->", "color": T.TEXT_MUTED, "lw": 0.9},
        )
        ax.annotate(
            "", xy=(7.9, y + 0.6), xytext=(6.5, 3.0), arrowprops={"arrowstyle": "->", "color": T.TEXT_MUTED, "lw": 0.9}
        )
        ax.text(
            9.6,
            y + 0.6,
            rf"$\hat{{\tau}}^{{({name.strip('$')})}}$",
            ha="right",
            va="center",
            fontsize=9,
            color=T.PRIMARY,
        )
    box(5.0, 2.4, 1.5, 1.2, r"FPN" "\n" r"$\phi$", T.INFO)
    ax.text(5.75, 4.5, "one network,\nreused across targets", ha="center", va="bottom", fontsize=7, color=T.TEXT_MUTED)
    return fig


# ── copula construction ───────────────────────────────────────────────────────
# Built with the head the chapter describes, so the figure is produced by the
# implementation it illustrates rather than by a separate mock-up.
_RHO = 0.8
_N_DRAW = 2500
_N_SLOT_C = 121
_RATED = 4.4


def _fpqr_margins() -> tuple[torch.Tensor, torch.Tensor]:
    """Two learned quantile functions: active and reactive power.

    The constraint drawn on these panels is apparent power, which is a functional
    of $P$ and $Q$, so those are the two margins the illustration needs. Their
    spreads are comparable, which is what lets the dependence move $S$ at all.
    """
    from scipy.stats import skewnorm

    levels = np.linspace(0.5 / _N_SLOT_C, 1 - 0.5 / _N_SLOT_C, _N_SLOT_C)
    p = skewnorm.ppf(levels, 1.1) + 1.9
    v = skewnorm.ppf(levels, -0.4, scale=0.85) + 1.6
    values = torch.tensor(np.stack([p, v], axis=-1), dtype=torch.float32)
    values = values.view(1, _N_SLOT_C, 1, 2)
    taus = torch.tensor(levels, dtype=torch.float32).view(1, _N_SLOT_C, 1)
    return values, taus


def _draw(rho: float) -> np.ndarray:
    """Samples from the copula head at a chosen correlation, through the margins."""
    from twiga.distributions.nn.copula import GaussianCopula

    torch.manual_seed(0)
    values, taus = _fpqr_margins()
    copula = GaussianCopula(num_targets=2, hidden_size=4, forecast_horizon=1)
    with torch.no_grad():
        # The head is zero-initialised at independence; setting the off-diagonal
        # bias is how this illustration asks it for a given correlation.
        copula.offdiag_layer.bias.fill_(float(np.arctanh(rho)) if rho else 0.0)
        tril = copula(torch.zeros(1, 4))
        draws = copula.sample(tril, values, taus, _N_DRAW)
    return draws[:, 0, 0, :].numpy()


def _arc_panel(rho: float, title: str) -> plt.Figure:
    xy = _draw(rho)
    over = np.hypot(xy[:, 0], xy[:, 1]) > _RATED

    fig = plt.figure(figsize=S.panel(0.31, aspect=1.0))
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 6), wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(_RATED * np.cos(th), _RATED * np.sin(th), color=T.DANGER, lw=1.3, zorder=4)
    ax.plot(xy[~over, 0], xy[~over, 1], "o", ms=1.5, color=T.PRIMARY, alpha=0.35, mew=0, zorder=2)
    ax.plot(xy[over, 0], xy[over, 1], "o", ms=2.0, color=T.DANGER, alpha=0.65, mew=0, zorder=3)
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 4.4)
    ax.set_xlabel("$P$")
    ax.set_ylabel("$Q$")
    ax.text(
        0.96,
        0.94,
        rf"${100 * over.mean():.1f}\%$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=T.DANGER,
    )
    # The margins are identical between panels by construction; showing them
    # is what makes that claim checkable rather than asserted.
    axt.hist(xy[:, 0], bins=50, color=T.PRIMARY_TINT, lw=0)
    axr.hist(xy[:, 1], bins=50, orientation="horizontal", color=T.PRIMARY_TINT, lw=0)
    for a in (axt, axr):
        a.axis("off")
    axt.set_title(title, fontsize=9, pad=4)
    return fig


def copula_uniform() -> plt.Figure:
    """The copula itself: dependence with the margins stripped away."""
    from twiga.distributions.nn.copula import GaussianCopula

    torch.manual_seed(0)
    copula = GaussianCopula(num_targets=2, hidden_size=4, forecast_horizon=1)
    with torch.no_grad():
        copula.offdiag_layer.bias.fill_(float(np.arctanh(_RHO)))
        u = copula.sample_uniform(copula(torch.zeros(1, 4)), _N_DRAW)[:, 0, 0, :].numpy()

    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=1.0))
    ax.plot(u[:, 0], u[:, 1], "o", ms=1.5, color=T.PRIMARY, alpha=0.35, mew=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")
    ax.set_title("Copula", fontsize=9, pad=4)
    return fig


def copula_independent() -> plt.Figure:
    """The learned margins sampled independently."""
    return _arc_panel(0.0, "Independent")


def copula_coupled() -> plt.Figure:
    """The same margins sampled through the copula."""
    return _arc_panel(_RHO, "Coupled")


FIGURES = {
    "fig_ch6_elliptical_tail.pdf": elliptical_tail,
    "fig_ch6_elliptical_exceedance.pdf": elliptical_exceedance,
    "fig_ch6_covariance_block.pdf": covariance_block,
    "fig_ch6_tau_allocation.pdf": tau_allocation,
    "fig_ch6_target_conditioning.pdf": target_conditioning,
    "fig_ch6_copula_uniform.pdf": copula_uniform,
    "fig_ch6_copula_independent.pdf": copula_independent,
    "fig_ch6_copula_coupled.pdf": copula_coupled,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every method illustration to ``out`` and to the Chapter 6 folder."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    thesis = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name)
        if thesis.exists():
            fig.savefig(thesis / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
