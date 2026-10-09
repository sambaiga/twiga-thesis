r"""Chapter 2 illustration: four ways of representing forecast uncertainty, in the order the chapter treats them.

Four panels at 0.48 of the text width, composed as a 2x2 grid of subfigures in LaTeX, in the thesis line-plot style
(:mod:`line_style`): teal forecast, teal-tint interval, small hollow orange observations, Computer Modern type.

* ``fig_uq_bnn``: sample paths from different weight draws; their spread is the epistemic uncertainty.
* ``fig_uq_parametric``: a predictive density with location and scale; an exceedance probability is its tail area.
* ``fig_uq_quantile``: nested quantile bands whose width follows the data, with no assumed density.
* ``fig_uq_conformal``: a point forecast widened by a calibrated residual quantile, with a coverage guarantee.

The BNN, quantile and conformal panels share one synthetic day, so the three representations are directly
comparable.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from twiga_thesis.plotting import ch2_uq_paradigms as U; U.render()"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from twiga_thesis.plotting import line_style as LS, tokens as T

OUT = Path(__file__).resolve().parent / "output"
FRACTION, ASPECT = 0.48, 0.6
N_DRAWS = 8
NOTE_SIZE = 8
INNER_BAND = T.tint(T.PRIMARY, 0.65)  # 25-75% band, darker than the 5-95% band


def _day(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic day: lead time, expected load, its spread (larger at midday) and observations."""
    rng = np.random.default_rng(seed)
    hours = np.linspace(0, 24, 49)
    mean = 30 + 10 * np.sin((hours - 7) / 24 * 2 * np.pi) - 6 * np.exp(-(((hours - 13) / 3) ** 2))
    spread = 1.2 + 3.0 * np.exp(-(((hours - 13) / 3.5) ** 2))
    observed = mean + spread * rng.standard_normal(hours.size)
    return hours, mean, spread, observed


def _concept_axes(ax, xlabel: str, ylabel: str) -> None:
    """Axes for a concept panel: labelled spines, no tick values."""
    LS.finish(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def bnn_panel():
    """Sample paths from different weight draws around their mean; their spread is the epistemic uncertainty."""
    fig, ax = LS.new_panel(FRACTION, ASPECT)
    hours, mean, spread, observed = _day()
    rng = np.random.default_rng(3)
    scale = 0.6 + 0.8 * spread  # weight uncertainty, present all day and largest at midday
    draws = []
    # Each weight draw shifts the forecast smoothly; amplitudes are spread evenly so the draws fan out across
    # the band, with a small random phase so they are not parallel.
    for amplitude in np.linspace(-1.0, 1.0, N_DRAWS):
        phase = rng.uniform(-0.5, 0.5)
        deviation = amplitude * (0.75 + 0.25 * np.sin(hours / 24 * 2 * np.pi + phase))
        draws.append(mean + scale * deviation)
    draws = np.array(draws)
    # Predictive interval: weight (epistemic) and noise (aleatoric) variance combined.
    sd = np.sqrt(draws.var(axis=0) + spread**2)
    centre = draws.mean(axis=0)
    ax.fill_between(hours, centre - 1.645 * sd, centre + 1.645 * sd, color=LS.BAND_COLOR, lw=0)
    for path in draws:
        ax.plot(hours, path, color=LS.FORECAST_COLOR, lw=0.5, alpha=0.6)
    ax.plot(hours, centre, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH * 1.4)
    ax.plot(hours, observed, "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=3, **LS.MARKER)
    ax.text(
        0.6,
        (centre + 1.645 * sd).max() + 3.5,
        r"$\hat{y}^{(s)} = f_{\theta^{(s)}}(\mathbf{x}),\ \theta^{(s)} \sim q(\theta)$",
        ha="left",
        va="center",
        fontsize=NOTE_SIZE,
    )
    ax.set_xlim(0, 24)
    ax.set_ylim((centre - 1.645 * sd).min() - 2, (centre + 1.645 * sd).max() + 6)
    _concept_axes(ax, "Lead time", r"Net load $y$")
    return fig


def parametric_panel():
    """Predictive density with location and scale; the shaded tail is the exceedance probability."""
    fig, ax = LS.new_panel(FRACTION, ASPECT)
    y = np.linspace(-3.6, 3.6, 400)
    density = np.exp(-(y**2) / 2) / np.sqrt(2 * np.pi)
    limit = 1.3
    tail = y >= limit
    ax.fill_between(y, density, color=LS.BAND_COLOR, lw=0)
    ax.fill_between(y[tail], density[tail], color=T.HIGHLIGHT, alpha=0.75, lw=0)
    ax.plot(y, density, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
    peak = density.max()
    ax.plot([0, 0], [0, peak], color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH)
    half = np.exp(-0.5) / np.sqrt(2 * np.pi)
    ax.annotate("", xy=(-1, half), xytext=(0, half), arrowprops={"arrowstyle": "<->", "lw": 0.6, "color": T.TEXT_DARK})
    ax.text(-0.5, half + 0.02, r"$\sigma$", ha="center", va="bottom", fontsize=NOTE_SIZE)
    ax.text(0.08, peak * 1.03, r"$\mu$", ha="left", va="bottom", fontsize=NOTE_SIZE)
    ax.annotate(
        r"$\Pr(y > y^{\star})$",
        xy=(1.75, 0.035),
        xytext=(2.0, 0.2),
        fontsize=NOTE_SIZE,
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": T.TEXT_DARK},
    )
    ax.set_xlim(y[0], y[-1])
    ax.set_ylim(0, peak * 1.18)
    _concept_axes(ax, r"Net load $y$", "Density")
    ax.set_xticks([limit])
    ax.set_xticklabels([r"$y^{\star}$"])
    return fig


def quantile_panel():
    """Nested quantile bands whose width follows the data, around the median."""
    fig, ax = LS.new_panel(FRACTION, ASPECT)
    hours, mean, spread, observed = _day()
    z = {0.05: -1.645, 0.25: -0.674, 0.75: 0.674, 0.95: 1.645}
    q = {tau: mean + score * spread for tau, score in z.items()}
    ax.fill_between(hours, q[0.05], q[0.95], color=LS.BAND_COLOR, lw=0)
    ax.fill_between(hours, q[0.25], q[0.75], color=INNER_BAND, lw=0)
    ax.plot(hours, mean, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
    ax.plot(hours, observed, "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=3, **LS.MARKER)
    # Labels spread beyond the right spine, joined to the line ends, since the band is narrow there.
    for tau, values, shift in ((0.95, q[0.95], 5.0), (0.5, mean, 0.0), (0.05, q[0.05], -5.0)):
        ax.annotate(
            rf"$q_{{{tau:g}}}$",
            xy=(24, values[-1]),
            xytext=(25.6, values[-1] + shift),
            fontsize=NOTE_SIZE,
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": T.TEXT_MUTED, "shrinkA": 0, "shrinkB": 0},
        )
    ax.set_xlim(0, 24)
    ax.set_ylim(q[0.05].min() - 2, q[0.95].max() + 2)
    _concept_axes(ax, "Lead time", r"Net load $y$")
    return fig


def conformal_panel():
    """Point forecast widened by the calibrated residual quantile: constant width, guaranteed coverage."""
    fig, ax = LS.new_panel(FRACTION, ASPECT)
    hours, mean, spread, observed = _day()
    radius = np.quantile(np.abs(observed - mean), 0.9)
    lower, upper = mean - radius, mean + radius
    ax.fill_between(hours, lower, upper, color=LS.BAND_COLOR, lw=0)
    ax.plot(hours, mean, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
    ax.plot(hours, observed, "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=3, **LS.MARKER)
    x = 24.6
    bracket = {"arrowstyle": "|-|,widthA=0.25,widthB=0.25", "lw": 0.6, "shrinkA": 0, "shrinkB": 0}
    ax.annotate("", xy=(x, upper[-1]), xytext=(x, lower[-1]), annotation_clip=False, arrowprops=bracket)
    ax.text(x + 0.6, mean[-1], r"$\pm\hat{q}_{1-\alpha}$", ha="left", va="center", fontsize=NOTE_SIZE)
    q05 = mean - 1.645 * spread
    q95 = mean + 1.645 * spread
    bottom = q05.min() - 2
    ax.text(12, bottom + 0.6, r"coverage $\geq 1 - \alpha$", ha="center", va="bottom", fontsize=NOTE_SIZE)
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom, q95.max() + 2)
    _concept_axes(ax, "Lead time", r"Net load $y$")
    return fig


def render() -> list[Path]:
    """Write the four panels to ``output/`` (PDF and PNG) and return the PDF paths."""
    LS.configure()
    OUT.mkdir(exist_ok=True)
    paths = []
    for name, build in (
        ("fig_uq_bnn", bnn_panel),
        ("fig_uq_parametric", parametric_panel),
        ("fig_uq_quantile", quantile_panel),
        ("fig_uq_conformal", conformal_panel),
    ):
        fig = build()
        LS.save(fig, OUT / name)
        paths.append(OUT / f"{name}.pdf")
    return paths
