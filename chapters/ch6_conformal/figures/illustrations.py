r"""Concept illustrations for Chapter 7.

Same production treatment as `illustrations.py`'s Chapter 6/3 figures: clean
vector lines through :func:`thesis_figures.style.style`, a title per panel, and
colour-coded inline text callouts rather than a legend box, matching
`fig_ch6_constraint_independent.pdf`. These are arguments, not measurements, so
each is captioned "Schematic" in the chapter rather than styled to look
measured.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import ch7_illustrations as F; F.render_all()"
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

from twiga_thesis.plotting import style as S, tokens as T

ALPHA = 0.10
NOMINAL = 1 - ALPHA
_CALLOUT = {"fontsize": 9, "zorder": 6}
_BBOX = {"facecolor": "white", "edgecolor": "none", "pad": 1.5}


# ── horizon-wise width: SCP against the proposed adaptive interval ────────────
def cp_width_horizon() -> plt.Figure:
    """A single global quantile against a horizon-specific one."""
    h = np.arange(1, 25)
    sigma = 0.4 + 0.10 * np.sqrt(h)  # growing predictive uncertainty with lead time
    scp_width = np.full_like(h, 1.645 * sigma.mean(), dtype=float)  # one quantile, no h dependence
    adaptive_width = 1.645 * sigma  # eqn:crf-score's per-horizon scale

    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.8))
        ax.plot(h, scp_width, color=T.WARNING, lw=1.0)
        ax.plot(h, adaptive_width, color=T.PRIMARY, lw=1.0)
        ax.text(24.3, scp_width[-1], "SCP", color=T.WARNING, ha="left", va="center", **_CALLOUT)
        ax.text(24.3, adaptive_width[-1], "Adaptive CP", color=T.PRIMARY, ha="left", va="center", **_CALLOUT)
        ax.set_xlim(1, 24)
        ax.set_xlabel("Lead time $h$")
        ax.set_ylabel("Interval half-width")
        ax.set_xticks([1, 6, 12, 18, 24])
        S.format_axes(ax)
    return fig


# ── why marginal and stratified CWE can disagree ───────────────────────────────
def cwe_marginal_vs_stratified() -> tuple[plt.Figure, float, float]:
    """Per-horizon coverage for an interval that is only marginally calibrated.

    Returns the figure and the marginal/stratified CWE values, so the caption
    can quote them rather than the plot itself.
    """
    from twiga.core.metrics.interval import get_interval_metrics

    rng = np.random.default_rng(0)
    n, h = 300, 24
    true = rng.normal(10, 2, (n, h))
    loc = true + rng.normal(0, 0.5, (n, h))
    width = np.linspace(0.5, 1.2, h)  # calibrated width is ~0.82; under- then over-covers
    lower, upper = loc - width, loc + width

    m = get_interval_metrics(true, lower, upper, loc=loc, alpha=ALPHA)
    cwe, cwe_strat = float(m["cwe"].iloc[0]), float(m["cwe_stratified"].iloc[0])
    cov = np.mean((true >= lower) & (true <= upper), axis=0)

    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.8))
        ax.axhline(NOMINAL, color=T.TEXT_MUTED, ls="--", lw=1.0)
        ax.text(1, NOMINAL + 0.015, "Nominal", color=T.TEXT_MUTED, fontsize=8.5, va="bottom")
        ax.plot(np.arange(1, h + 1), cov, "-", color=T.PRIMARY, lw=1.0)
        ax.text(
            1.5,
            0.60,
            f"Marginal CWE {cwe:.2f}\nStratified CWE {cwe_strat:.2f}",
            color=T.PRIMARY,
            ha="left",
            va="bottom",
            bbox=_BBOX,
            **_CALLOUT,
        )
        ax.set_xlim(1, h)
        ax.set_ylim(0.55, 1.02)
        ax.set_xlabel("Lead time $h$")
        ax.set_ylabel("Coverage")
        S.format_axes(ax)
    return fig, cwe, cwe_strat


# ── fixed (global) against horizon-wise non-conformity score ──────────────────
def _bump(x: np.ndarray, centre: float, spread: float) -> np.ndarray:
    """A smooth unimodal bump, for a schematic score density."""
    return np.exp(-0.5 * ((x - centre) / spread) ** 2) / spread


def score_distribution_horizon() -> plt.Figure:
    """Non-conformity score distributions at three lead times against the pooled distribution.

    Illustrates why `eqn:cqr-score`/`eqn:p-score`'s single quantile does not
    match any one horizon's true quantile once the score distribution itself
    shifts with lead time, the motivation `eqn:crf-score` addresses.
    """
    x = np.linspace(0, 3, 400)
    horizons = [(1, 0.5, 0.18, T.INFO), (12, 0.9, 0.28, T.PRIMARY), (24, 1.4, 0.40, T.DANGER)]
    pooled = sum(_bump(x, c, s) for _, c, s, _ in horizons) / len(horizons)
    global_quantile = x[np.searchsorted(np.cumsum(pooled) / pooled.sum(), NOMINAL)]

    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.8))
        ax.plot(x, pooled, color=T.TEXT_MUTED, lw=1.0)
        ax.text(2.35, float(_bump(1.4, 1.4, 0.55)) + 0.15, "Pooled (SCP)", color=T.TEXT_MUTED, ha="center", **_CALLOUT)
        for h, centre, spread, color in horizons:
            ax.plot(x, _bump(x, centre, spread), color=color, lw=1.0)
            ax.text(
                centre, float(_bump(centre, centre, spread)) + 0.05, f"$h={h}$", color=color, ha="center", **_CALLOUT
            )
        ax.axvline(global_quantile, color=T.TEXT_MUTED, ls="--", lw=1.0)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, None)
        ax.set_xlabel("Non-conformity score $\\gamma$")
        ax.set_ylabel("Density")
        ax.set_yticks([])
        S.format_axes(ax)
        S.despine(ax, left=True)
    return fig


# ── the MLPGAM feature-gating mechanism ────────────────────────────────────────
def gating_diagram() -> plt.Figure:
    """Block diagram of `eqn:mlpgam-gating`/`eqn:mlpgam-latent`: features, gate, encoder."""
    with S.style():
        fig, ax = plt.subplots(figsize=S.panel(0.46, aspect=0.55))
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 3.0)
        ax.axis("off")

        def box(x: float, w: float, label: str, color: str, y: float = 0.7, h: float = 1.6) -> None:
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0.08", facecolor=T.tint(color, 0.78), edgecolor=color, lw=1.3
                )
            )
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

        def arrow(x0: float, x1: float, y: float = 1.5) -> None:
            ax.add_patch(
                FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13, color=T.TEXT_MUTED, lw=1.1)
            )

        box(0.0, 2.2, "Features\n$\\mathbf{x}^{(m)}$", T.INFO)
        box(2.8, 2.2, "Gate\n$\\eta^{(m)}$", T.PRIMARY)
        box(5.6, 2.2, "Masked\ninput", T.INFO)
        box(8.4, 2.0, "Encoder\n$g_\\theta^{(m)}$", T.PRIMARY)
        arrow(2.2, 2.8)
        arrow(5.0, 5.6)
        arrow(7.8, 8.4)
        ax.text(9.4, 0.3, "Latent $\\mathbf{z}^{(m)}$", ha="center", color=T.PRIMARY, fontsize=9)
    return fig


FIGURES = {
    "fig_ch7_cp_width_horizon.pdf": cp_width_horizon,
    "fig_ch7_cwe_marginal_vs_stratified.pdf": cwe_marginal_vs_stratified,
    "fig_ch7_score_distribution_horizon.pdf": score_distribution_horizon,
    "fig_ch7_gating_diagram.pdf": gating_diagram,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every illustration to ``out`` and to Images/Chapter-7."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    images = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
    for name, fn in FIGURES.items():
        result = fn()
        fig = result[0] if isinstance(result, tuple) else result
        fig.savefig(out / name)
        if images.exists():
            fig.savefig(images / name)
        plt.close(fig)
        if isinstance(result, tuple):
            print(f"  {name}  (CWE {result[1]:.2f}, stratified {result[2]:.2f})")
        else:
            print(f"  {name}")


if __name__ == "__main__":
    print("rendering chapter 7 illustrations:")
    render_all()
