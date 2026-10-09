r"""Concept illustrations for the FPQR chapter (Chapter 5).

The figures are built around one observation, which is also the argument for
using FPQR on net load rather than on a generic forecasting target: **the shape
of the conditional distribution changes across the day**. At night, with no
photovoltaic generation, net load is narrowly and symmetrically distributed. At
midday under variable cloud it is wide and bimodal, because the day resolves
either to clear sky or to overcast and the two give very different net loads.

A fixed quantile grid has to serve both regimes with the same nine points. An
adaptive grid does not. Every panel here develops that one idea:

``day_fan``
    What the model produces, and that the forecast *is* the quantile set.
``reconstruction``
    What the choice of grid costs, in distribution space rather than in
    :math:`\tau` space a uniform grid smears the two midday modes together.
``grid_measured_*``
    That the required placement changes hour by hour, and that the model
    learns it: the proposal grid before and after training on MLVS-PT.
``routing``
    How the two networks divide the work, and why that is not circular.
``crossing``
    What the non-crossing penalty prevents.

All panels are SYNTHETIC illustrations, not results; the captions say so.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import fpqr_figures as F; F.render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from twiga_thesis.plotting import style as S, tokens as T

N_Q = 9  # twiga's default n_quantiles
HOURS = np.arange(24)


# ── a day of net load whose conditional shape changes with the sun ────────────
def _pv(h: np.ndarray | float) -> np.ndarray:
    """Clear-sky photovoltaic factor, zero outside daylight."""
    return np.clip(np.sin(np.pi * (np.asarray(h, dtype=float) - 6.0) / 12.0), 0.0, None)


def _demand(h: np.ndarray | float) -> np.ndarray:
    """Underlying demand: a morning shoulder and a sharper evening peak."""
    h = np.asarray(h, dtype=float)
    return 1.05 + 0.22 * np.sin(2 * np.pi * (h - 9) / 24) + 0.42 * np.exp(-0.5 * ((h - 19.5) / 2.1) ** 2)


def _mixture(h: float) -> list[tuple[float, float, float]]:
    r"""Conditional distribution of net load at hour ``h``, as (w, mean, sd).

    Outside daylight it is a single narrow Gaussian. Within daylight it splits
    into a clear-sky component and an overcast component whose separation grows
    with the photovoltaic factor which is what makes the midday conditional
    distribution bimodal and the night-time one not.
    """
    p = float(_pv(h))
    base = float(_demand(h))
    if p < 1e-6:
        return [(1.0, base, 0.085)]
    cap = 1.35
    clear = base - cap * p
    overcast = base - 0.22 * cap * p
    return [(0.55, clear, 0.055 + 0.05 * p), (0.45, overcast, 0.075 + 0.09 * p)]


def _quantile_at(h: float, tau: np.ndarray) -> np.ndarray:
    """Quantile function of the hour-``h`` conditional, by numerical inversion."""
    mix = _mixture(h)
    lo = min(m - 5 * s for _, m, s in mix)
    hi = max(m + 5 * s for _, m, s in mix)
    x = np.linspace(lo, hi, 6000)
    cdf = sum(w * norm.cdf(x, m, s) for w, m, s in mix)
    return np.interp(tau, cdf, x)


def _density_at(h: float, x: np.ndarray) -> np.ndarray:
    return sum(w * norm.pdf(x, m, s) for w, m, s in _mixture(h))


def _optimal_grid_at(h: float, n_q: int = N_Q) -> np.ndarray:
    r"""Grid with :math:`\Delta\tau_n \propto |Q''(\tau_n)|^{-1/3}`.

    Widths proportional to :math:`|Q''|^{-1/3}` is the same statement as point
    density proportional to :math:`|Q''|^{1/3}`, so the grid follows by inverse
    transform of that density.
    """
    tau = np.linspace(0.004, 0.996, 3000)
    quantile = _quantile_at(h, tau)
    d2 = np.abs(np.gradient(np.gradient(quantile, tau), tau))
    w = np.clip(d2, np.percentile(d2, 8), None) ** (1.0 / 3.0)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp((np.arange(n_q) + 0.5) / n_q, cdf, tau)


UNIFORM = (np.arange(N_Q) + 0.5) / N_Q
MIDDAY, NIGHT = 12.0, 2.0


# ── 1. what the model produces ────────────────────────────────────────────────
def day_fan() -> plt.Figure:
    """A day of forecasts. The quantile set is not a summary of the forecast it is the forecast."""
    fine = np.linspace(0, 23.6, 240)
    fig, ax = plt.subplots(figsize=S.panel(0.72, 0.52))

    curves = np.array([[_quantile_at(h, np.array([t]))[0] for h in fine] for t in UNIFORM])
    for i in range(len(UNIFORM) // 2):
        ax.fill_between(fine, curves[i], curves[-1 - i], color=T.PRIMARY, alpha=0.10, lw=0)
    for i in range(len(UNIFORM)):
        ax.plot(fine, curves[i], color=T.PRIMARY, lw=0.9, alpha=0.95)
    ax.plot(fine, curves[len(UNIFORM) // 2], color=T.PRIMARY, lw=2.0)

    top = curves.max() + 0.16
    for h, lab in ((NIGHT, "night"), (MIDDAY, "midday")):
        ax.axvline(h, color=T.DANGER, lw=1.0, ls="--", alpha=0.9, zorder=1)
        ax.text(h, top, lab, color=T.DANGER, ha="center", va="bottom", fontsize=plt.rcParams["font.size"] - 4)
    ax.set_ylim(curves.min() - 0.12, top + 0.16)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("net load")
    ax.set_xlim(0, 23.6)
    ax.set_xticks([0, 6, 12, 18])
    fig.tight_layout()
    return fig


# ── 2. the same nine points, spent two ways ───────────────────────────────────
def _step_density(tau: np.ndarray, quantile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant density implied by a quantile set: dtau / dQ."""
    dens = np.diff(tau) / np.maximum(np.diff(quantile), 1e-9)
    return quantile, np.concatenate([dens, dens[-1:]])


def _reconstruction_panel(h: float, title: str) -> plt.Figure:
    mix = _mixture(h)
    lo = min(m - 4.2 * s for _, m, s in mix)
    hi = max(m + 4.2 * s for _, m, s in mix)
    x = np.linspace(lo, hi, 900)

    fig, ax = plt.subplots(figsize=S.panel(0.48, 0.86))
    ax.plot(x, _density_at(h, x), color=T.TEXT_MUTED, lw=1.4, label="true density", zorder=4)
    for grid, colour, name in ((UNIFORM, T.DANGER, "uniform"), (_optimal_grid_at(h), T.PRIMARY, "adaptive")):
        quantile = _quantile_at(h, grid)
        xs, ys = _step_density(grid, quantile)
        ax.step(xs, ys, where="post", color=colour, lw=1.2, label=f"{name} grid", zorder=3)
        ax.plot(quantile, np.zeros_like(quantile), "|", color=colour, ms=9, mew=1.4, zorder=5, clip_on=False)
    ax.set_xlabel("net load")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=plt.rcParams["font.size"] - 3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.34)
    ax.legend(loc="upper right", fontsize=plt.rcParams["font.size"] - 5)
    fig.tight_layout()
    return fig


def reconstruction_midday() -> plt.Figure:
    """Midday: the uniform grid smears the clear/overcast modes into one lump."""
    return _reconstruction_panel(MIDDAY, "Midday bimodal")


def reconstruction_night() -> plt.Figure:
    """Night: no photovoltaic generation, so both grids do equally well."""
    return _reconstruction_panel(NIGHT, "Night unimodal")


# ── 3. the grid has to breathe across the day ─────────────────────────────────
# ── 4. how the work divides ───────────────────────────────────────────────────
def routing() -> plt.Figure:
    """Which loss reaches which network, and where the stop-gradient sits."""
    fig, ax = plt.subplots(figsize=S.panel(1.0, 0.42))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fs = plt.rcParams["font.size"]

    def box(x, y, w, h, label, colour):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=colour, alpha=0.12, edgecolor=colour, lw=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs - 3)

    def arrow(x0, y0, x1, y1, colour, dashed=False, lw=1.0):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "lw": lw,
                "linestyle": "--" if dashed else "-",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    box(0.2, 1.55, 1.5, 0.9, r"$\phi(gh)$", T.TEXT_MUTED)
    box(2.6, 2.45, 1.7, 0.9, "FPN", T.PRIMARY)
    box(2.6, 0.65, 1.7, 0.9, "QVN", T.PRIMARY)
    box(5.5, 2.45, 1.5, 0.9, r"$\hat{\tau}_{\theta}$", T.TEXT_MUTED)
    box(5.5, 0.65, 1.9, 0.9, r"$Q_{\theta}(\hat{\tau}_{\theta})$", T.TEXT_MUTED)

    arrow(1.7, 2.15, 2.6, 2.85, T.TEXT_MUTED)
    arrow(1.7, 1.85, 2.6, 1.15, T.TEXT_MUTED)
    arrow(4.3, 2.90, 5.5, 2.90, T.TEXT_MUTED)
    arrow(4.3, 1.10, 5.5, 1.10, T.TEXT_MUTED)
    arrow(6.25, 2.45, 4.62, 1.55, T.TEXT_MUTED)

    mx, my, px, py = 5.43, 2.00, 0.442, -0.816
    ax.plot(
        [mx - 0.38 * px, mx + 0.38 * px],
        [my - 0.38 * py, my + 0.38 * py],
        color=T.DANGER,
        lw=2.6,
        solid_capstyle="round",
        zorder=6,
    )
    ax.text(mx - 0.92, my + 0.16, r"$\mathrm{sg}(\cdot)$", color=T.DANGER, fontsize=fs - 3, ha="center", va="center")

    arrow(9.15, 2.90, 7.10, 2.90, T.PRIMARY, dashed=True, lw=1.2)
    ax.text(
        8.10,
        3.28,
        r"$\mathcal{L}_{W_1} + \mathcal{L}_{\mathcal{horizon}}$",
        fontsize=fs - 3,
        ha="center",
        color=T.PRIMARY,
    )
    arrow(9.15, 1.10, 7.50, 1.10, T.DANGER, dashed=True, lw=1.2)
    ax.text(
        8.30,
        1.48,
        r"$\mathcal{L}_{\mathrm{pb}} + \mathcal{L}_{\mathrm{nc}}$",
        fontsize=fs - 3,
        ha="center",
        color=T.DANGER,
    )
    ax.text(5.0, 0.10, r"solid: forward pass \quad dashed: gradient", fontsize=fs - 4, ha="center", color=T.TEXT_MUTED)
    fig.tight_layout()
    return fig


# ── 5. what the non-crossing penalty prevents ─────────────────────────────────
def crossing_example() -> plt.Figure:
    """Without the penalty the quantile set need not increase with tau."""
    tau = _optimal_grid_at(MIDDAY)
    base = _quantile_at(MIDDAY, tau)
    span = base.max() - base.min()
    kick = 0.16 * span * np.array([0.9, -1.0, 1.1, -1.0, 0.9, -1.0, 1.2, -0.9, 0.7])
    loose = base + kick

    fig, ax = plt.subplots(figsize=S.panel(0.48, 0.60))
    bad = np.nonzero(np.diff(loose) < 0)[0]
    for i in bad:
        ax.plot(tau[i : i + 2], loose[i : i + 2], color=T.DANGER, lw=5.0, alpha=0.28, solid_capstyle="round", zorder=1)
    ax.plot(tau, base, "o-", color=T.PRIMARY, ms=4.5, lw=1.2, zorder=3, label=r"with $\mathcal{L}_{\mathrm{nc}}$")
    ax.plot(tau, loose, "o--", color=T.DANGER, ms=4.5, lw=1.2, zorder=2, label=r"without $\mathcal{L}_{\mathrm{nc}}$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("net load")
    ax.set_title(f"{len(bad)} inverted pairs", fontsize=plt.rcParams["font.size"] - 3)
    ax.legend(loc="upper left", fontsize=plt.rcParams["font.size"] - 5)
    ax.margins(y=0.20)
    fig.tight_layout()
    return fig


# ── measured: the grid before and after training on net-load ──────────────────
_TAU_CSV = Path(__file__).resolve().parents[1] / "chapter-5-fpqr" / "s563_learned_tau_MLVS-PT.csv"


def _measured_panel(state: str, title: str) -> plt.Figure:
    """One panel of the measured grid, ordered by clock hour."""
    import pandas as pd

    d = pd.read_csv(_TAU_CSV)
    g = d[d.state == state].pivot_table(index="slot", columns="hour", values="tau")
    hours = g.columns.to_numpy()
    fig, ax = plt.subplots(figsize=S.panel(0.48, 0.74))
    ax.fill_between(hours, 0, 1, where=_pv(hours) > 0, color=T.HIGHLIGHT, alpha=0.13, lw=0, zorder=0)
    for n in range(g.shape[0]):
        ax.axhline(UNIFORM[n], color=T.DANGER, lw=0.7, ls="--", alpha=0.5, zorder=1)
        ax.plot(hours, g.iloc[n].to_numpy(), color=T.PRIMARY, lw=1.3, zorder=3)
    rng = float((g.max(axis=1) - g.min(axis=1)).mean())
    ax.set_xlabel("hour of day")
    ax.set_ylabel(r"$\hat{\tau}_{\theta}$")
    ax.set_title(rf"{title} mean range {rng:.3f}", fontsize=plt.rcParams["font.size"] - 3)
    ax.set_xlim(0, 23.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 6, 12, 18])
    fig.tight_layout()
    return fig


def grid_measured_untrained() -> plt.Figure:
    """At initialisation the proposal emits a uniform grid, by construction."""
    return _measured_panel("untrained", "At initialisation")


def grid_measured_trained() -> plt.Figure:
    """After training on MLVS-PT the grid varies with the hour of day."""
    return _measured_panel("trained", "After training")


# ── intermediate outputs: where the adaptivity lives, on both datasets ────────
_TAU_NPY = Path(__file__).resolve().parents[1] / "chapter-5-fpqr"
_DATASETS = ("MLVS-PT", "MLVS-UK")


def tau_dispersion() -> plt.Figure:
    r"""Does the grid adapt per forecast, or only across the horizon?

    Per slot, the spread of $\hat{\tau}$ across forecast steps divided by its
    spread across forecast days. Above one means lead time dominates.
    """
    import pandas as pd

    d = pd.read_csv(_TAU_NPY / "s563_intermediate_both.csv")
    fig, ax = plt.subplots(figsize=S.panel(0.48, 0.74))
    for ds, colour, mk in zip(_DATASETS, (T.PRIMARY, T.DANGER), ("o-", "s--"), strict=False):
        g = d[d.dataset == ds]
        ax.plot(g.slot, g.sd_steps / g.sd_windows, mk, color=colour, ms=4.5, lw=1.2, label=ds)
    ax.axhline(1.0, color=T.TEXT_MUTED, lw=0.9, ls=":")
    ax.text(9.0, 1.06, "equal", fontsize=plt.rcParams["font.size"] - 5, color=T.TEXT_MUTED, ha="right")
    ax.set_xlabel("quantile slot")
    ax.set_ylabel("dispersion ratio")
    ax.set_xticks(range(1, 10))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=plt.rcParams["font.size"] - 5)
    fig.tight_layout()
    return fig


def tau_confound() -> plt.Figure:
    r"""Association between the learned grid and the day's irradiance.

    The raw correlation is reported against the correlation after the slow
    seasonal trend is partialled out. On MLVS-UK the raw figure is almost
    entirely a trend artefact.
    """
    import pandas as pd

    d = pd.read_csv(_TAU_NPY / "s563_intermediate_both.csv")
    fig, ax = plt.subplots(figsize=S.panel(0.48, 0.74))
    x = np.arange(len(_DATASETS))
    raw = [d[d.dataset == ds].raw_corr_ghi.abs().max() for ds in _DATASETS]
    par = [d[d.dataset == ds].partial_corr_ghi.abs().max() for ds in _DATASETS]
    ax.bar(x - 0.19, raw, width=0.38, color=T.DANGER, label="raw")
    ax.bar(x + 0.19, par, width=0.38, color=T.PRIMARY, label="after removing trend")
    for i, (a, b) in enumerate(zip(raw, par, strict=False)):
        ax.text(i - 0.19, a + 0.02, f"{a:.2f}", ha="center", fontsize=plt.rcParams["font.size"] - 5)
        ax.text(i + 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=plt.rcParams["font.size"] - 5)
    ax.set_xticks(x)
    ax.set_xticklabels(_DATASETS)
    ax.set_ylabel(r"$|r|$ with mean irradiance")
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper left", fontsize=plt.rcParams["font.size"] - 5)
    fig.tight_layout()
    return fig


def pipeline() -> plt.Figure:
    r"""The forward pass through the FPQR head, with the real signal at each stage.

    Left column: what the stage does and the shape it emits. Right column: the
    actual tensor for one forecast day, so the reader sees that $\hat{\tau}$ is
    something the model produced rather than a setting. A dashed line marks one
    lead time and is carried through the FPN, QVN and CDF rows.
    """
    d = np.load(_TAU_NPY / "s563_worked_example.npz")
    tau, quant, actual = d["tau"], d["quant"], d["actual"]  # (n_slots,horizon), (n_slots,horizon), (horizon,)
    hist, lead = d["hist"], int(d["lead"])
    n_slots, horizon = tau.shape
    hours = 1.0 + 0.5 * np.arange(horizon)  # origins are 01:00
    fs = plt.rcParams["font.size"]

    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, S.TEXTWIDTH_IN * 0.98))
    gs = fig.add_gridspec(5, 2, width_ratios=[0.34, 0.66], hspace=0.70, wspace=0.26)

    rows = [
        ("input", r"$\phi(gh)$", "encode history", r"$(B,\,z)$", T.TEXT_MUTED),
        ("FPN", r"$\hat{\tau}_{\theta}$", "softmax, cumsum", r"$(B,\,n_slots,\,horizon)$", T.PRIMARY),
        ("TauEmb", r"$\Phi(\hat{\tau})$", "cosine basis", r"$(B,\,n_slots,\,horizon,\,z)$", T.PRIMARY),
        ("QVN", r"$Q_{\theta}(\hat{\tau})$", "gate, decode", r"$(B,\,n_slots,\,horizon,\,D)$", T.PRIMARY),
        ("CDF", r"$\hat{y}$, PI", "sort, interpolate", r"$(B,\,horizon,\,D)$", T.HIGHLIGHT),
    ]
    for i, (stage, emits, op, shape, colour) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.58), 0.52, 0.34, facecolor=colour, alpha=0.13, edgecolor=colour, lw=1.0))
        ax.text(0.28, 0.75, stage, ha="center", va="center", fontsize=fs - 3)
        ax.text(0.60, 0.75, emits, ha="left", va="center", fontsize=fs - 3, color=colour)
        ax.text(0.02, 0.40, op, ha="left", va="center", fontsize=fs - 5, color=T.TEXT_MUTED)
        ax.text(0.02, 0.18, shape, ha="left", va="center", fontsize=fs - 5, color=colour)
        if i < len(rows) - 1:
            ax.annotate(
                "",
                xy=(0.28, -0.22),
                xytext=(0.28, 0.12),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": T.TEXT_MUTED, "lw": 0.9},
            )

    def tidy(ax, xlab=None):
        ax.tick_params(labelsize=fs - 6)
        if xlab:
            ax.set_xlabel(xlab, fontsize=fs - 5, labelpad=1)

    # 1. what goes in
    ax = fig.add_subplot(gs[0, 1])
    hh = np.arange(-len(hist), 0) * 0.5
    ax.plot(hh, hist, color=T.TEXT_MUTED, lw=0.9)
    ax.set_ylabel("kW", fontsize=fs - 5)
    ax.axvline(0, color=T.PRIMARY, lw=1.0, ls="--")
    ax.text(0.6, 0.88, "history", transform=ax.transAxes, fontsize=fs - 6, color=T.TEXT_MUTED)
    tidy(ax, "hours before forecast")

    # 2. the grid the model proposes
    ax = fig.add_subplot(gs[1, 1])
    for n in range(n_slots):
        ax.plot(hours, tau[n], color=T.PRIMARY, lw=0.9)
    ax.axvline(hours[lead], color=T.DANGER, lw=1.0, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$\hat{\tau}$", fontsize=fs - 5)
    tidy(ax, "hour of day")

    # 3. tau rendered as a basis, at the marked lead time
    ax = fig.add_subplot(gs[2, 1])
    kernel = 32
    basis = np.cos(np.pi * np.arange(1, kernel + 1)[None, :] * tau[:, lead][:, None])
    im = ax.imshow(basis, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_ylabel("slot", fontsize=fs - 5)
    ax.set_yticks([0, n_slots - 1])
    ax.set_yticklabels([1, n_slots])
    tidy(ax, "cosine index $k$")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.045).ax.tick_params(labelsize=fs - 7)

    # 4. the values at those levels
    ax = fig.add_subplot(gs[3, 1])
    for n in range(n_slots // 2):
        ax.fill_between(hours, quant[n], quant[-1 - n], color=T.PRIMARY, alpha=0.16, lw=0)
    for n in range(n_slots):
        ax.plot(hours, quant[n], color=T.PRIMARY, lw=0.6)
    ax.plot(hours, actual, color=T.DANGER, lw=1.2, label="observed")
    ax.axvline(hours[lead], color=T.DANGER, lw=1.0, ls="--")
    ax.set_ylabel("kW", fontsize=fs - 5)
    ax.legend(loc="upper left", fontsize=fs - 6, handlelength=1.2)
    tidy(ax, "hour of day")

    # 5. the distribution read off at that lead time
    ax = fig.add_subplot(gs[4, 1])
    t, q = tau[:, lead], np.sort(quant[:, lead])
    y = float(actual[lead])
    # what the model emitted, and what is filled in between
    ax.plot(t, q, "-", color=T.PRIMARY, lw=1.1, zorder=3, label="interpolated")
    ax.plot([0, t[0]], [q[0], q[0]], "--", color=T.PRIMARY, lw=1.1, zorder=3)
    ax.plot([t[-1], 1], [q[-1], q[-1]], "--", color=T.PRIMARY, lw=1.1, zorder=3, label="flat tails")
    ax.plot(t, q, "o", color=T.PRIMARY, ms=4.5, zorder=5)
    # the nominal levels fall outside what the model was asked for
    for lv in (0.05, 0.95):
        ax.axvline(lv, color=T.TEXT_MUTED, lw=0.7, ls=":", zorder=1)
    ax.text(0.05, 0.04, r" $5\%$", transform=ax.get_xaxis_transform(), fontsize=fs - 7, color=T.TEXT_MUTED, ha="left")
    ax.text(0.95, 0.04, r"$95\%$ ", transform=ax.get_xaxis_transform(), fontsize=fs - 7, color=T.TEXT_MUTED, ha="right")
    # where the outcome fell in the predicted distribution
    pit = float(np.interp(y, q, t))
    ax.axhline(y, color=T.DANGER, lw=1.1, zorder=4)
    ax.plot([pit], [y], "o", color=T.DANGER, ms=6, zorder=6)
    ax.text(0.012, y, "observed", va="bottom", ha="left", fontsize=fs - 6, color=T.DANGER)
    ax.annotate(
        rf"$\tau={pit:.2f}$",
        xy=(pit, y),
        xytext=(0, -12),
        textcoords="offset points",
        fontsize=fs - 6,
        color=T.DANGER,
        ha="center",
    )
    med = q[np.argmin(np.abs(t - 0.5))]
    ax.plot([0.5], [med], "s", color=T.PRIMARY, ms=5, zorder=6)
    ax.annotate(
        "median",
        xy=(0.5, med),
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=fs - 6,
        color=T.PRIMARY,
        ha="center",
    )
    ax.set_xlim(0, 1)
    lo_y, hi_y = q.min(), q.max()
    ax.set_ylim(lo_y - 0.10 * (hi_y - lo_y), hi_y + 0.34 * (hi_y - lo_y))
    ax.set_ylabel("kW", fontsize=fs - 5)
    ax.legend(loc="lower right", fontsize=fs - 7, handlelength=1.3, ncol=1, borderpad=0.25)
    tidy(ax, r"$\tau$")
    return fig


def tau_perturbation() -> plt.Figure:
    r"""What changes the learned grid, and what does not.

    Left: how far the grid departs from uniform across the day. Right: how
    strongly it tracks the day's irradiance once the seasonal trend is removed.
    The baseline bar carries the spread over three seeds.
    """
    import pandas as pd

    d = pd.read_csv(_TAU_NPY / "s564_tau_matrix.csv")
    arms = ["baseline", "n_slots=5", "n_slots=21", "no entropy", "horizon proposal"]
    lab = ["baseline", "$n_slots=5$", "$n_slots=21$", "no entropy", "horizon\nproposal"]
    fig, axes = plt.subplots(1, 2, figsize=S.panel(1.0, 0.34))
    x = np.arange(len(arms))
    for ax, col, ylab in (
        (axes[0], "across_day_range", "across-day range"),
        (axes[1], "detrended_corr_ghi", r"detrended $|r|$ with irradiance"),
    ):
        for k, (ds, colour) in enumerate(zip(_DATASETS, (T.PRIMARY, T.DANGER), strict=False)):
            g = d[d.dataset == ds]
            vals = [g[g.arm == a][col].mean() for a in arms]
            errs = [g[g.arm == a][col].std() if a == "baseline" else 0.0 for a in arms]
            ax.bar(
                x + (k - 0.5) * 0.38,
                vals,
                width=0.38,
                yerr=errs,
                capsize=2.5,
                color=colour,
                label=ds,
                error_kw={"lw": 0.9},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=plt.rcParams["font.size"] - 5)
        ax.set_ylabel(ylab, fontsize=plt.rcParams["font.size"] - 4)
    axes[1].axhline(0.0, color=T.TEXT_MUTED, lw=0.6)
    axes[0].legend(loc="upper left", fontsize=plt.rcParams["font.size"] - 5)
    fig.tight_layout()
    return fig


FIGURES = {
    "fig_fpqr_day_fan.pdf": day_fan,
    "fig_fpqr_recon_midday.pdf": reconstruction_midday,
    "fig_fpqr_recon_night.pdf": reconstruction_night,
    "fig_fpqr_routing.pdf": routing,
    "fig_fpqr_pipeline.pdf": pipeline,
    "fig_fpqr_crossing.pdf": crossing_example,
    "fig_fpqr_measured_untrained.pdf": grid_measured_untrained,
    "fig_fpqr_measured_trained.pdf": grid_measured_trained,
    "fig_fpqr_tau_dispersion.pdf": tau_dispersion,
    "fig_fpqr_tau_confound.pdf": tau_confound,
    "fig_fpqr_tau_perturbation.pdf": tau_perturbation,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every FPQR illustration to ``out`` and to the Chapter 5 images."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    thesis = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-5"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name)
        if thesis.exists():
            fig.savefig(thesis / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    print("rendering FPQR illustrations:")
    render_all()
