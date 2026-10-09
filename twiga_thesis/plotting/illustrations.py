r"""Concept illustrations written for this thesis.

Each function draws **one** panel and is saved as its own PDF, to be composed
into a sub-figure grid in LaTeX. That keeps every panel individually
referenceable and leaves layout to the document rather than to matplotlib.

Notation follows the thesis: :math:`\gamma_\alpha` is the miscoverage ratio and
:math:`\gamma_w` the width ratio.

Groups:

``cwe_*``
    How the metric is built. Section 3.5, alongside the definition.
``cond_*``
    Why marginal coverage is not enough. Chapter 2 Section 2.3, where
    calibration is defined; referenced again in Section 3.5.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import illustrations as I; I.render_all()"
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style as S, tokens as T
from .icons import icon_patch

ALPHA = 0.10
NOMINAL = 1 - ALPHA
Z = 1.645
# Bonferroni-corrected per-axis z for the 2-target schematic (z_P, z_Q) shared
# by every conformal_* panel below: each axis calibrated independently at
# 1 - ALPHA/D (D=2), so the union bound gives the same nominal joint coverage
# as pooling, at the cost of a wider per-axis interval than Z's own 90%.
# norm.ppf(1 - (ALPHA / 2) / 2) = norm.ppf(0.975) = 1.95996...
Z_BONFERRONI = 1.960


# ── shared synthetic data ─────────────────────────────────────────────────────
def _kernel(x: np.ndarray | float) -> np.ndarray:
    r"""f(x) = 2x/(1+x^2)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 2 * x / (1 + x**2), 0.0)


def _components(hod: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Demand, PV generation and the clear-sky factor, for one day of the profile."""
    demand = 3.0 + 0.9 * np.exp(-(((hod - 0.33) / 0.06) ** 2)) + 1.3 * np.exp(-(((hod - 0.79) / 0.07) ** 2))
    clear = np.clip(np.sin(np.pi * (hod - 0.25) / 0.5), 0, None) ** 1.4
    return demand, 4.2 * clear * 0.85, clear


def _profile(hod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conditional mean and standard deviation of a PV-dominated net-load day."""
    demand, _, clear = _components(hod)
    # These coefficients must match make_synthetic_netload() in
    # experiment/chapter-3-cwe-validation/cwe_validation.py: the figures quote
    # numbers that also appear in the text, and the two must not drift.
    return demand - 4.2 * clear * 0.85, 0.16 + 0.42 * clear


def _one_day(seed: int = 1) -> dict[str, np.ndarray]:
    """One representative half-hourly day."""
    rng = np.random.default_rng(seed)
    hod = np.arange(48) / 48.0
    mu, sigma = _profile(hod)
    return {"hour": hod * 24, "mu": mu, "sigma": sigma, "y": mu + rng.normal(0, sigma)}


def _many_days(n_days: int = 400, seed: int = 0) -> dict[str, np.ndarray]:
    """Enough days for coverage to be estimated hour by hour."""
    rng = np.random.default_rng(seed)
    hod = np.tile(np.arange(48) / 48.0, n_days)
    mu, sigma = _profile(hod)
    return {"hod": hod, "mu": mu, "sigma": sigma, "y": mu + rng.normal(0, sigma)}


# ── the construction of the metric ────────────────────────────────────────────
def cwe_measured() -> plt.Figure:
    r"""What the metric measures on an interval forecast."""
    d = _one_day()
    h, mu, sig, y = d["hour"], d["mu"], d["sigma"], d["y"]
    lo, up = mu - Z * sig, mu + Z * sig

    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.88))
    ax.fill_between(h, lo, up, color=T.PRIMARY_TINT, zorder=1)
    ax.plot(h, mu, color=T.PRIMARY, zorder=3)
    inside = (y >= lo) & (y <= up)
    ax.plot(h[inside], y[inside], "o", ms=2.0, color=T.TEXT_MUTED, zorder=4)
    ax.plot(h[~inside], y[~inside], "o", ms=3.6, color=T.DANGER, zorder=5)

    if (~inside).any():
        k = int(np.flatnonzero(~inside)[np.argmax(np.abs(y[~inside] - mu[~inside]))])
        ax.annotate(
            r"$\gamma_\alpha$",
            xy=(h[k], y[k]),
            xytext=(-12, 14),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=13,
            color=T.DANGER,
            arrowprops={"arrowstyle": "-", "color": T.DANGER, "lw": 0.6, "alpha": 0.6},
        )
    imax = int(np.argmax(up - lo))
    ax.annotate(
        "",
        xy=(h[imax], up[imax]),
        xytext=(h[imax], lo[imax]),
        arrowprops={"arrowstyle": "<->", "color": T.WARNING, "lw": 1.1},
    )
    ax.annotate(
        r"$\gamma_w$",
        xy=(h[imax], mu[imax]),
        xytext=(9, 0),
        textcoords="offset points",
        fontsize=13,
        color=T.WARNING,
        ha="left",
        va="center",
    )
    ax.set_xlabel("hour")
    ax.set_ylabel("net-load")
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.margins(y=0.14)
    return fig


def cwe_kernel() -> plt.Figure:
    r"""The shared kernel and its reciprocal symmetry."""
    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.88))
    x = np.linspace(0.02, 4.0, 400)
    ax.plot(x, _kernel(x), color=T.PRIMARY, lw=1.4)
    ax.axvline(1.0, color=T.TRUTH, ls="--", lw=0.8)
    for v, lab in ((0.5, r"$\frac{1}{2}$"), (2.0, "2")):
        ax.plot([v], [_kernel(v)], "o", color=T.WARNING, ms=4, zorder=5)
        ax.annotate(
            lab,
            xy=(v, _kernel(v)),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color=T.WARNING,
        )
    ax.annotate(r"$f(x)=f(1/x)$", xy=(1.45, 0.16), fontsize=11, color=T.WARNING)
    ax.plot([1.0], [1.0], "o", color=T.PRIMARY, ms=5, zorder=5)
    ax.set_xlabel(r"ratio $x$")
    ax.set_ylabel(r"$f(x)=2x/(1+x^{2})$")
    ax.set_ylim(0, 1.14)
    ax.set_xlim(0, 4)
    return fig


def cwe_surface() -> plt.Figure:
    r"""The score as a function of the two ratios."""
    fig, ax = plt.subplots(figsize=S.panel(0.34, aspect=0.82))
    g = np.linspace(0.05, 3.0, 240)
    g_a, g_w = np.meshgrid(g, g)
    fa, fw = _kernel(g_a), _kernel(g_w)
    cwe = np.where(fa + fw > 0, 2 * fa * fw / (fa + fw), 0.0)
    cs = ax.contourf(g_a, g_w, cwe, levels=10, cmap=T.sequential_cmap())
    ax.contour(g_a, g_w, cwe, levels=[0.5, 0.8, 0.95], colors=[T.TEXT_MUTED], linewidths=0.5, alpha=0.8)
    ax.plot([1], [1], "*", color=T.DANGER, ms=12, zorder=6)
    ax.set_xlabel(r"$\gamma_\alpha$")
    ax.set_ylabel(r"$\gamma_w$")
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03, label="CWE")
    return fig


# ── marginal versus conditional coverage ──────────────────────────────────────
def _half_width(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, constant: bool) -> np.ndarray:
    """Interval half-width: conditional, or one constant value matched to nominal."""
    if constant:
        return np.full_like(y, float(np.quantile(np.abs(y - mu), NOMINAL)))
    return Z * sigma


def _cond_forecast(constant: bool) -> plt.Figure:
    """One day, with either a conditional or a constant interval width."""
    d = _many_days()
    hod, mu, sigma, y = d["hod"], d["mu"], d["sigma"], d["y"]
    half = _half_width(y, mu, sigma, constant)
    color = T.WARNING if constant else T.PRIMARY
    band = T.WARNING_TINT if constant else T.PRIMARY_TINT

    day = slice(0, 48)
    hours = hod[day] * 24
    fig, ax = plt.subplots(figsize=S.panel(0.48, aspect=0.60))
    ax.fill_between(hours, (mu - half)[day], (mu + half)[day], color=band, zorder=1)
    ax.plot(hours, mu[day], color=color, zorder=3)
    ax.plot(hours, y[day], "o", ms=2.0, color=T.TEXT_MUTED, zorder=4)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("net-load")
    ax.set_xticks([0, 6, 12, 18, 24])
    return fig


def _cond_coverage(constant: bool) -> plt.Figure:
    """Coverage by hour for the same two forecasts."""
    d = _many_days()
    hod, mu, sigma, y = d["hod"], d["mu"], d["sigma"], d["y"]
    half = _half_width(y, mu, sigma, constant)
    color = T.WARNING if constant else T.PRIMARY

    hh = (hod * 48).astype(int) // 2
    cov = [float(np.mean(np.abs(y - mu)[hh == k] <= half[hh == k])) for k in range(24)]
    marginal = float(np.mean(np.abs(y - mu) <= half))

    fig, ax = plt.subplots(figsize=S.panel(0.48, aspect=0.50))
    ax.axhline(NOMINAL, color=T.TRUTH, ls="--", lw=0.8, zorder=2)
    ax.plot(np.arange(24) + 0.5, cov, "o-", color=color, ms=2.8, zorder=3)
    ax.set_ylim(0.55, 1.04)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("hour of day")
    ax.set_ylabel("coverage")
    ax.annotate(
        f"marginal {marginal:.3f}",
        xy=(0.5, 0.08),
        xycoords="axes fraction",
        ha="center",
        fontsize=12,
        color=T.TEXT_MUTED,
    )
    if not constant:
        ax.annotate("nominal", xy=(0.7, NOMINAL), xytext=(0, 6), textcoords="offset points", fontsize=11, color=T.TRUTH)
    return fig


def cond_forecast_calibrated() -> plt.Figure:
    """A conditionally calibrated interval."""
    return _cond_forecast(constant=False)


def cond_forecast_constant() -> plt.Figure:
    """A single constant interval width."""
    return _cond_forecast(constant=True)


def cond_coverage_calibrated() -> plt.Figure:
    """Coverage by hour, conditionally calibrated."""
    return _cond_coverage(constant=False)


def cond_coverage_constant() -> plt.Figure:
    """Coverage by hour, constant width."""
    return _cond_coverage(constant=True)


# ── validity and efficiency (replaces Figure 2.4) ─────────────────────────────
def _interval_case(scale: float) -> plt.Figure:
    """One interval forecast at a given width multiple, with its coverage shown.

    ``scale`` multiplies a conditionally correct interval: unity is calibrated,
    below unity over-confident, above unity conservative.
    """
    d = _many_days(n_days=120, seed=5)
    hod, mu, sigma, y = d["hod"], d["mu"], d["sigma"], d["y"]
    half = Z * sigma * scale
    picp = float(np.mean(np.abs(y - mu) <= half))

    calibrated = abs(picp - NOMINAL) < 0.02
    color = T.PRIMARY if calibrated else T.DANGER
    band = T.PRIMARY_TINT if calibrated else T.DANGER_TINT

    day = slice(0, 48)
    hours = hod[day] * 24
    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.86))
    ax.fill_between(hours, (mu - half)[day], (mu + half)[day], color=band, zorder=1)
    ax.plot(hours, mu[day], color=color, zorder=3)
    inside = np.abs(y - mu)[day] <= half[day]
    ax.plot(hours[inside], y[day][inside], "o", ms=1.8, color=T.TEXT_MUTED, zorder=4)
    ax.plot(hours[~inside], y[day][~inside], "o", ms=3.4, color=T.DANGER, zorder=5)
    ax.annotate(
        rf"$\mathrm{{PICP}} = {picp:.2f}$",
        xy=(0.5, 0.03),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        color=color,
    )
    ax.set_xlabel("hour")
    ax.set_ylabel("net-load")
    ax.set_xticks([0, 12, 24])
    ax.set_ylim(-7.5, 8.5)
    return fig


def interval_overconfident() -> plt.Figure:
    """Too narrow: the interval misses more often than the nominal rate allows."""
    return _interval_case(0.45)


def interval_calibrated() -> plt.Figure:
    """Calibrated: coverage sits at the nominal rate, and the interval is as narrow as that allows."""
    return _interval_case(1.0)


def interval_conservative() -> plt.Figure:
    """Too wide: coverage exceeds the nominal rate, bought with width the decision-maker pays for."""
    return _interval_case(2.2)


# ── reliability and PIT (replaces Figure 2.5) ─────────────────────────────────
def _forecast_family() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Three forecasters differing only in dispersion: (y, mu, sigma_hat)."""
    d = _many_days(n_days=300, seed=7)
    mu, sigma, y = d["mu"], d["sigma"], d["y"]
    return {
        "over-confident": (y, mu, sigma * 0.6),
        "calibrated": (y, mu, sigma),
        "conservative": (y, mu, sigma * 1.7),
    }


def reliability() -> plt.Figure:
    """Empirical against nominal coverage across the full range of levels."""
    fam = _forecast_family()
    levels = np.linspace(0.05, 0.95, 19)
    colors = {"over-confident": T.DANGER, "calibrated": T.PRIMARY, "conservative": T.WARNING}

    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.92))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=T.TRUTH, zorder=1)
    for name, (y, mu, sig) in fam.items():
        emp = [float(np.mean(np.abs(y - mu) <= stats_ppf(lv) * sig)) for lv in levels]
        ax.plot(levels, emp, "o-", ms=2.4, color=colors[name], label=name, zorder=3)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=10)
    return fig


def stats_ppf(level: float) -> float:
    """Half-width multiplier for a central interval at the given coverage."""
    from scipy import stats

    return float(stats.norm.ppf(0.5 + level / 2))


def _pit_panel(name: str) -> plt.Figure:
    """PIT histogram for one member of the forecast family."""
    from scipy import stats

    y, mu, sig = _forecast_family()[name]
    pit = stats.norm.cdf((y - mu) / sig)
    colors = {"over-confident": T.DANGER, "calibrated": T.PRIMARY, "conservative": T.WARNING}

    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.80))
    # No alpha: nothing is drawn beneath, and blending shifts the hue off-palette.
    ax.hist(pit, bins=20, range=(0, 1), density=True, color=colors[name], zorder=3)
    ax.axhline(1.0, color=T.TRUTH, ls="--", lw=0.8, zorder=4)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.2)
    return fig


def pit_overconfident() -> plt.Figure:
    """U-shaped: the outcome falls in the tails too often."""
    return _pit_panel("over-confident")


def pit_calibrated() -> plt.Figure:
    """Flat: the outcome is equally likely anywhere in the predictive distribution."""
    return _pit_panel("calibrated")


def pit_conservative() -> plt.Figure:
    """Peaked at the centre: the predictive distribution is too wide."""
    return _pit_panel("conservative")


# ── why the reference and the strata are what they are ────────────────────────
def reference_choice() -> plt.Figure:
    r"""The reference width: spread of the target, or spread of the residuals.

    Net-load's marginal spread is dominated by the diurnal swing that a forecast
    explains, so it is several times larger than the width a calibrated interval
    actually needs. Referencing it compares the interval to the size of the signal
    rather than to the irreducible uncertainty.
    """
    # Read from the experiment rather than recomputed, so that a number shown here
    # cannot drift from the same number quoted in the text.
    src = Path(__file__).resolve().parent.parent / "chapter-3-cwe-validation" / "s3523_reference_scales.csv"
    v = pd.read_csv(src).set_index("quantity")["value"]
    bars = [
        ("spread of the target $y$", float(v["kappa_target"]), T.DANGER),
        (r"spread of the residuals $y-\hat{\mu}$", float(v["kappa_residual"]), T.PRIMARY),
        ("width a calibrated\ninterval actually needs", float(v["median_interval_width"]), T.INFO),
    ]

    fig, ax = plt.subplots(figsize=S.panel(0.50, aspect=0.62))
    ypos = np.arange(len(bars))[::-1]
    for (_label, val, color), y in zip(bars, ypos, strict=False):
        ax.barh(y, val, height=0.55, color=color, zorder=3)
        ax.text(val + 0.09, y, f"{val:.2f}", va="center", ha="left", fontsize=12, color=color, zorder=4)
    ax.set_yticks(ypos, [b[0] for b in bars])
    ax.set_xlim(0, max(b[1] for b in bars) * 1.28)
    ax.set_xlabel("interval width")
    ax.tick_params(left=False, which="both")
    ax.spines["left"].set_visible(False)
    ratio = float(v["kappa_target"]) / float(v["median_interval_width"])
    ax.annotate(
        rf"${ratio:.1f}\times$",
        xy=(float(v["kappa_target"]) * 0.62, ypos[0] - 0.40),
        ha="left",
        va="center",
        fontsize=12,
        color=T.DANGER,
    )
    return fig


def factorial_matrix() -> plt.Figure:
    r"""Where each combination places its optimum, against the truth.

    Drawn against an explicit truth line rather than as a grid of bare numbers:
    the reader needs something to read $1.164$ against before it means anything.
    The two dots in each row are the two choices of reference; the arrow between
    them is the improvement that choice buys.
    """
    src = Path(__file__).resolve().parent.parent / "chapter-3-cwe-validation" / "s3523_netload_factorial.csv"
    g = pd.read_csv(src).groupby(["reference", "stratified"])["optimum"].mean()

    rows = [("marginal", "none"), ("stratified", "hour of day")]
    fig, ax = plt.subplots(figsize=S.panel(0.52, aspect=0.78))

    ax.axvline(1.0, color=T.TEXT_MUTED, ls=":", lw=1.0, zorder=1)
    ax.annotate(
        "truth",
        xy=(1.0, -0.42),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        color=T.TEXT_MUTED,
    )

    for i, (_label, strat) in enumerate(rows):
        y = 1 - i
        v_t, v_r = float(g.loc[("target", strat)]), float(g.loc[("residual", strat)])
        ax.annotate(
            "",
            xy=(v_r, y),
            xytext=(v_t, y),
            arrowprops={"arrowstyle": "->", "color": T.TEXT_MUTED, "lw": 0.9, "alpha": 0.6},
        )
        for v, color, ref in ((v_t, T.DANGER, r"from $y$"), (v_r, T.PRIMARY, r"from $y-\hat{\mu}$")):
            sound = abs(v - 1.0) < 0.005
            ax.plot([v], [y], "o", color=T.PRIMARY if sound else color, ms=9 if sound else 7, zorder=4)
            ax.annotate(
                f"{v:.3f}",
                xy=(v, y),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=10.5,
                color=T.PRIMARY if sound else color,
            )
            if i == 0:  # name the two references once, on the top row only
                ax.annotate(
                    ref,
                    xy=(v, y),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=color,
                )

    ax.set_yticks([1, 0], ["scored\nmarginally", "scored\nwithin strata"])
    ax.set_ylim(-0.75, 1.80)
    ax.set_xlim(0.985, 1.20)
    ax.set_xlabel("location of the optimum")
    ax.tick_params(left=False, which="both")
    ax.spines["left"].set_visible(False)

    return fig


def synthetic_series() -> plt.Figure:
    r"""How the synthetic net-load series is built, and what makes it net-load.

    Demand is double-peaked; a clear-sky PV bell is subtracted from it. The
    difference crosses zero, and its conditional spread is widest exactly where
    PV output is largest. Those three properties --- a subtracted generation
    term, sign changes, and variance tied to the time of day --- are what separate
    net-load from a demand series, and are why the reference width and the strata
    of the metric definition matter.
    """
    rng = np.random.default_rng(4)
    hod = np.arange(48) / 48.0
    hours = hod * 24
    demand, pv, _ = _components(hod)
    mu, sigma = _profile(hod)
    y = mu + rng.normal(0, sigma)
    half = Z * sigma

    fig, ax = plt.subplots(figsize=S.panel(0.62, aspect=0.60))
    ax.axhline(0, color=T.TEXT_MUTED, lw=0.7, ls=":", zorder=1)
    ax.plot(hours, demand, color=T.TEXT_MUTED, ls="--", lw=1.0, zorder=2)
    ax.fill_between(hours, 0, pv, color=T.tint(T.DANGER, 0.55), zorder=2)
    ax.fill_between(hours, mu - half, mu + half, color=T.PRIMARY_TINT, zorder=3)
    ax.plot(hours, mu, color=T.PRIMARY, zorder=4)
    ax.plot(hours, y, "o", ms=2.0, color=T.TEXT_MUTED, zorder=5)

    ax.annotate(
        "demand",
        xy=(hours[16], demand[16]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color=T.TEXT_MUTED,
    )
    ax.annotate("PV", xy=(12, pv.max() * 0.86), ha="center", fontsize=10, color=T.DANGER)
    ax.annotate(
        "net-load",
        xy=(hours[6], mu[6]),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color=T.PRIMARY,
    )
    imax = int(np.argmax(half))
    ax.annotate(
        "",
        xy=(hours[imax], (mu + half)[imax]),
        xytext=(hours[imax], (mu - half)[imax]),
        arrowprops={"arrowstyle": "<->", "color": T.INFO, "lw": 1.2},
    )
    ax.annotate(
        "widest\nspread",
        xy=(hours[imax], (mu - half)[imax]),
        xytext=(-10, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9.5,
        color=T.INFO,
    )

    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("hour of day")
    ax.set_ylabel("power")
    ax.margins(y=0.16)
    return fig


# ── Chapter 6: why a joint distribution is needed for a joint constraint ──────
# The claim these two panels make is the one the multi-target chapter rests on:
# two forecasts can agree on every margin and still disagree about the
# probability that a constraint is violated, because the constraint is a
# functional of the targets taken together. Drawn rather than measured, so the
# margins are identical by construction and dependence is the only difference.

_S_RATED = 130.0  # transformer nameplate, in the units of both axes
_PQ_MU = np.array([78.0, 42.0])
_PQ_SD = np.array([20.0, 14.0])


def _pq_contour(ax: plt.Axes, rho: float) -> None:
    """Two probability contours of a bivariate normal with the fixed margins."""
    cov = np.diag(_PQ_SD) @ np.array([[1.0, rho], [rho, 1.0]]) @ np.diag(_PQ_SD)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    # Mahalanobis radii for the 95 % and 50 % contours of a bivariate normal.
    for radius, alpha, width in ((2.448, 0.16, 0.9), (1.177, 0.34, 1.3)):
        ax.add_patch(
            Ellipse(
                _PQ_MU,
                2 * radius * np.sqrt(vals[-1]),
                2 * radius * np.sqrt(vals[0]),
                angle=angle,
                facecolor=T.PRIMARY_TINT,
                edgecolor=T.PRIMARY,
                lw=width,
                alpha=alpha,
                zorder=3,
            )
        )
    ax.plot(*_PQ_MU, "o", ms=3.2, color=T.PRIMARY, zorder=6)


def _pq_margin(ax: plt.Axes, i: int, horizontal: bool) -> None:
    """The marginal density on an axis, the one both panels share."""
    grid = np.linspace(0.0, _PQ_MU[i] + 4.2 * _PQ_SD[i], 240)
    dens = np.exp(-0.5 * ((grid - _PQ_MU[i]) / _PQ_SD[i]) ** 2)
    if horizontal:
        ax.fill_between(grid, 0, dens, color=T.PRIMARY_TINT, lw=0)
        ax.plot(grid, dens, color=T.PRIMARY, lw=1.0)
        ax.set_ylim(0, 1.18)
    else:
        ax.fill_betweenx(grid, 0, dens, color=T.PRIMARY_TINT, lw=0)
        ax.plot(dens, grid, color=T.PRIMARY, lw=1.0)
        ax.set_xlim(0, 1.18)
    ax.axis("off")


def _pq_exceedance(rho: float) -> float:
    r"""Pr(S > S_rated) for the drawn distribution, by simulation."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2, 2_000_000))
    p = _PQ_MU[0] + _PQ_SD[0] * z[0]
    q = _PQ_MU[1] + _PQ_SD[1] * (rho * z[0] + np.sqrt(1 - rho**2) * z[1])
    return float(np.mean(np.hypot(p, q) > _S_RATED))


def _pq_panel(rho: float, title: str, note: str) -> plt.Figure:
    fig = plt.figure(figsize=S.panel(0.46, aspect=1.02))
    # The panel is laid out by hand; an automatic engine cannot place an
    # equal-aspect axes against two marginal strips without distorting it.
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.09, top=0.93)
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 6), wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    _pq_margin(axt, 0, horizontal=True)
    _pq_margin(fig.add_subplot(gs[1, 1], sharey=ax), 1, horizontal=False)

    # Equal scales, so the apparent-power limit reads as the circular arc it is.
    ax.set_aspect("equal")
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 120)

    th = np.linspace(0, np.pi / 2, 300)
    arc_x, arc_y = _S_RATED * np.cos(th), _S_RATED * np.sin(th)
    ax.fill_between(arc_x, arc_y, 200, color=T.DANGER, alpha=0.10, lw=0, zorder=1)
    ax.plot(arc_x, arc_y, color=T.DANGER, lw=1.5, zorder=4)

    _pq_contour(ax, rho)

    ax.set_xlabel("Active power $P$ (kW)")
    ax.set_ylabel("Reactive power $Q$ (kvar)")
    # Ticks rather than a bare frame: the magnitudes are those of a small
    # secondary substation, and leaving them off made a dimensioned argument
    # look like an abstract sketch.
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0, 40, 80])

    # A transformer badge on the limit, so the arc reads as a nameplate.
    ax.add_patch(Circle((146, 74), 7.5, facecolor=T.DANGER, ec="white", lw=1.1, zorder=6))
    icon_patch(ax, "hdd-network-fill", (146, 74), 8.0, "white", zorder=7)
    ax.text(
        146,
        62,
        rf"$S = S^{{\mathrm{{rated}}}}$" "\n" rf"$= {_S_RATED:.0f}$ kVA",
        fontsize=8,
        color=T.DANGER,
        ha="center",
        va="top",
        zorder=7,
    )

    ax.text(
        164,
        114,
        rf"$\Pr(S > S^{{\mathrm{{rated}}}}) = {100 * _pq_exceedance(rho):.1f}\%$",
        fontsize=8.5,
        color=T.DANGER,
        ha="right",
        va="top",
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.text(
        164,
        100,
        note,
        fontsize=7.5,
        color=T.TEXT_MUTED,
        ha="right",
        va="top",
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    axt.set_title(title, fontsize=10, pad=4)
    return fig


def constraint_independent() -> plt.Figure:
    """Independent margins: the joint density is axis aligned."""
    return _pq_panel(0.0, "Independent margins", "P and Q peak apart")


def constraint_joint() -> plt.Figure:
    """The same margins, now coupled, so more mass sits beyond the rating."""
    return _pq_panel(0.8, "Same margins, coupled", "P and Q peak together")


# ── multi-target conformal: independent vs. pooled calibration ────────────────
# Residual correlation after conditioning on the shared backbone is typically
# weaker than P/Q's own raw correlation (fig_ch6_joint_*'s rho~0.8-0.94), so a
# moderate value is used here rather than reusing that figure's own rho --
# also gives the clearer before/after contrast: joint coverage under
# independent calibration falls as rho falls, floored at the rho=0 product
# bound (0.9 x 0.9 = 0.81).
_CONF_RHO = 0.3
_CONF_N = 150
_CONF_SEED = 3


def _conformal_sample() -> tuple[np.ndarray, np.ndarray]:
    """Schematic normalised residuals (z_P, z_Q) for a correlated target pair."""
    rng = np.random.default_rng(_CONF_SEED)
    cov = np.array([[1.0, _CONF_RHO], [_CONF_RHO, 1.0]])
    z = rng.multivariate_normal((0.0, 0.0), cov, size=_CONF_N)
    return z[:, 0], z[:, 1]


def _pooled_threshold() -> float:
    """The shared max-norm threshold solving Pr(max(|Z_P|, |Z_Q|) <= q) = NOMINAL.

    Estimated by simulation at a large sample size, the same convention
    :func:`_pq_exceedance` uses for a probability with no closed form.
    """
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, _CONF_RHO], [_CONF_RHO, 1.0]])
    z = rng.multivariate_normal((0.0, 0.0), cov, size=2_000_000)
    scores = np.abs(z).max(axis=1)
    return float(np.quantile(scores, NOMINAL))


def _conformal_panel(half_width: float, title: str) -> plt.Figure:
    """One calibration region against the same schematic (z_P, z_Q) sample.

    The coverage rate is folded into the title (above the axes) rather than
    an in-plot text box: a box placed inside the data area can land on top of
    scattered points near that corner (the earlier top-left/bottom-right
    placement did, for some half-widths), which a fixed axes-fraction
    position cannot avoid for every panel.
    """
    z_p, z_q = _conformal_sample()
    covered = (np.abs(z_p) <= half_width) & (np.abs(z_q) <= half_width)
    rate = covered.mean()

    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=1.0))
    ax.add_patch(
        plt.Rectangle(
            (-half_width, -half_width),
            2 * half_width,
            2 * half_width,
            facecolor=T.PRIMARY_TINT,
            edgecolor=T.PRIMARY,
            lw=1.4,
            zorder=2,
        )
    )
    ax.plot(z_p[covered], z_q[covered], "o", ms=3.2, color=T.PRIMARY, alpha=0.75, mew=0, zorder=3)
    ax.plot(z_p[~covered], z_q[~covered], "o", ms=3.6, color=T.DANGER, alpha=0.9, mew=0, zorder=4)

    lim = 3.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_{P} = (P - \mu_P)/\sigma_P$")
    ax.set_ylabel(r"$z_{Q} = (Q - \mu_Q)/\sigma_Q$")
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])

    S.format_axes(ax)
    ax.set_title(title + "\n" + rf"{100 * rate:.0f}\% jointly covered", fontsize=10, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def conformal_independent() -> plt.Figure:
    """Each target calibrated to 90% on its own axis; the joint event is not guaranteed."""
    return _conformal_panel(Z, "Independent per-target calibration")


def conformal_pooled() -> plt.Figure:
    """One shared max-norm threshold, calibrated so the joint event hits 90%."""
    return _conformal_panel(_pooled_threshold(), "Pooled (max-norm) calibration")


def conformal_bonferroni() -> plt.Figure:
    """Each target calibrated independently at level 1-alpha/D; the union bound recovers 90% jointly."""
    return _conformal_panel(Z_BONFERRONI, "Bonferroni-corrected calibration")


# ── multi-target CQR: the same three constructions in quantile-bound space ────
# Unlike the residual-based schematic above (a 2D scatter in standardised
# residual space, shared axis scale for both targets by construction), CQR's
# own non-conformity score (eq:ch6-cqr-score) is normalised by each target's
# *own* predicted width w_{i,h,d} -- so a target with a narrower predicted
# interval is not on the same footing as one with a wider predicted interval,
# the exact asymmetry this chapter's own quantile_joint pooling finding
# traces the failure mode to. The figure is drawn in interval-bound space
# (horizontal bars, one row per target) rather than reusing the residual
# scatter, so that each target's own raw width and its calibrated margin are
# both visible directly, instead of being collapsed onto one shared axis
# scale the way standardisation does for the residual-based score.
_CQR_W_P = 1.0  # target P's own predicted interval width (wider)
_CQR_W_Q = 0.4  # target Q's own predicted interval width (narrower)
_CQR_N = 60
_CQR_SEED = 5


def _cqr_sample() -> tuple[np.ndarray, np.ndarray]:
    """Schematic *signed* dimensionless deviations (y_d - centre_d)/w_d for a correlated target pair.

    Kept signed, not ``abs``-ed, so plotted points scatter on both sides of
    each target's interval; the CQR exceedance score itself (``abs`` of this)
    is only needed for the coverage comparison, computed where used instead.
    """
    rng = np.random.default_rng(_CQR_SEED)
    cov = np.array([[1.0, _CONF_RHO], [_CONF_RHO, 1.0]])
    z = rng.multivariate_normal((0.0, 0.0), cov, size=_CQR_N)
    return z[:, 0], z[:, 1]


def _cqr_pooled_quantile() -> float:
    """The shared pooled-CQR threshold solving Pr(max(e_P, e_Q) <= q) = NOMINAL."""
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, _CONF_RHO], [_CONF_RHO, 1.0]])
    z = rng.multivariate_normal((0.0, 0.0), cov, size=2_000_000)
    scores = np.abs(z).max(axis=1)
    return float(np.quantile(scores, NOMINAL))


def _cqr_panel(q_p: float, q_q: float, title: str) -> plt.Figure:
    """One pair of calibrated CQR intervals (half-width q_d * w_d) against the same schematic sample.

    The coverage rate is folded into the title (above the axes), not an
    in-plot text box: see :func:`_conformal_panel`'s own docstring for why.
    """
    z_p, z_q = _cqr_sample()  # signed, dimensionless (y_d - centre_d) / w_d
    half_p, half_q = q_p * _CQR_W_P, q_q * _CQR_W_Q
    covered_p, covered_q = np.abs(z_p) <= q_p, np.abs(z_q) <= q_q
    rate = (covered_p & covered_q).mean()

    fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.62))
    rows = [(0, half_p, covered_p, z_p * _CQR_W_P), (1, half_q, covered_q, z_q * _CQR_W_Q)]
    for y, half, covered, signed in rows:
        ax.add_patch(
            plt.Rectangle((-half, y - 0.3), 2 * half, 0.6, facecolor=T.PRIMARY_TINT, edgecolor=T.PRIMARY, lw=1.4)
        )
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, size=signed.size)
        ax.plot(signed[covered], y + jitter[covered], "o", ms=3.0, color=T.PRIMARY, alpha=0.75, mew=0, zorder=3)
        ax.plot(signed[~covered], y + jitter[~covered], "o", ms=3.4, color=T.DANGER, alpha=0.9, mew=0, zorder=4)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$P$", r"$Q$"])
    ax.set_xlabel(r"Signed exceedance beyond $\hat{Q}(\tau)$, in units of $w_{i,h,d}$")

    S.format_axes(ax)
    ax.set_title(title + "\n" + rf"{100 * rate:.0f}\% jointly covered", fontsize=10, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def cqr_marginal() -> plt.Figure:
    """Each target's CQR interval calibrated independently to its own 90% marginal coverage."""
    z_p, z_q = _cqr_sample()
    return _cqr_panel(
        float(np.quantile(np.abs(z_p), NOMINAL)),
        float(np.quantile(np.abs(z_q), NOMINAL)),
        "Marginal CQR calibration",
    )


def cqr_bonferroni() -> plt.Figure:
    """Each target's CQR interval calibrated independently at the Bonferroni-corrected level."""
    z_p, z_q = _cqr_sample()
    level = 1 - ALPHA / 2
    return _cqr_panel(
        float(np.quantile(np.abs(z_p), level)),
        float(np.quantile(np.abs(z_q), level)),
        "Bonferroni-corrected CQR",
    )


def cqr_pooled() -> plt.Figure:
    """Both targets' CQR intervals widened by the same pooled (max-pooled) threshold."""
    q = _cqr_pooled_quantile()
    return _cqr_panel(q, q, "Pooled CQR calibration")


FIGURES = {
    # Chapter 6: a joint constraint needs a joint distribution
    "fig_ch6_constraint_independent.pdf": constraint_independent,
    "fig_ch6_constraint_joint.pdf": constraint_joint,
    # Chapter 6: the three target-coverage constructions for multi-target
    # conformal calibration -- independent (marginal, under-covers jointly),
    # pooled (max-norm, the proposed joint score), and Bonferroni-corrected
    # (per-target union bound) -- same schematic (z_P, z_Q) sample throughout.
    "fig_ch6_conformal_independent.pdf": conformal_independent,
    "fig_ch6_conformal_bonferroni.pdf": conformal_bonferroni,
    "fig_ch6_conformal_pooled.pdf": conformal_pooled,
    # Chapter 6: the same three constructions in CQR's own quantile-bound
    # space (eq:ch6-cqr-score/eq:ch6-pooled-cqr-score/eq:ch6-bonferroni-cqr),
    # drawn as per-target interval bars rather than a residual scatter, so
    # each target's own width asymmetry (and pooling's exposure to it) is
    # visible directly.
    "fig_ch6_cqr_marginal.pdf": cqr_marginal,
    "fig_ch6_cqr_bonferroni.pdf": cqr_bonferroni,
    "fig_ch6_cqr_pooled.pdf": cqr_pooled,
    "fig_cwe_measured.pdf": cwe_measured,
    "fig_cwe_kernel.pdf": cwe_kernel,
    "fig_cwe_surface.pdf": cwe_surface,
    "fig_cond_forecast_calibrated.pdf": cond_forecast_calibrated,
    "fig_cond_forecast_constant.pdf": cond_forecast_constant,
    "fig_cond_coverage_calibrated.pdf": cond_coverage_calibrated,
    "fig_cond_coverage_constant.pdf": cond_coverage_constant,
    # validity and efficiency (Figure 2.4)
    "fig_interval_overconfident.pdf": interval_overconfident,
    "fig_interval_calibrated.pdf": interval_calibrated,
    "fig_interval_conservative.pdf": interval_conservative,
    # reliability and PIT (Figure 2.5)
    "fig_reliability.pdf": reliability,
    "fig_pit_overconfident.pdf": pit_overconfident,
    "fig_pit_calibrated.pdf": pit_calibrated,
    "fig_pit_conservative.pdf": pit_conservative,
    # the two design choices, explained then reported
    "fig_synthetic_series.pdf": synthetic_series,
    "fig_reference_choice.pdf": reference_choice,
    "fig_factorial_matrix.pdf": factorial_matrix,
}


# Which chapter's image folder each figure belongs to; Chapter-3 by default.
CHAPTER: dict[str, str] = {
    "fig_ch6_constraint_independent.pdf": "Chapter-6",
    "fig_ch6_constraint_joint.pdf": "Chapter-6",
    "fig_ch6_conformal_independent.pdf": "Chapter-6",
    "fig_ch6_conformal_pooled.pdf": "Chapter-6",
    "fig_ch6_conformal_bonferroni.pdf": "Chapter-6",
    "fig_ch6_cqr_marginal.pdf": "Chapter-6",
    "fig_ch6_cqr_bonferroni.pdf": "Chapter-6",
    "fig_ch6_cqr_pooled.pdf": "Chapter-6",
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every illustration to ``out`` and to the thesis image folder."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    images = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name)
        thesis = images / CHAPTER.get(name, "Chapter-3")
        if thesis.exists():
            fig.savefig(thesis / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    print("rendering thesis illustrations:")
    render_all()
