r"""Chapter 3 figures in the thesis line-plot style.

The style is that of :mod:`line_style`: Computer Modern through LaTeX, teal for the reference or preferred case,
orange for the case the reader should notice, dotted grey references, no titles.

Each function returns one panel; :func:`render` writes the requested panels to ``output/`` and, when asked, to the
Chapter 3 image folder.

Usage::

    MPLBACKEND=Agg uv run python -c "import sys; sys.path.insert(0, 'experiment'); \
        from twiga_thesis.plotting import ch3_figures as F; F.render(['dss_eu_disagreement'])"
"""

from __future__ import annotations

from pathlib import Path
import shutil

from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from twiga.core.data.selection import select_top_features
from twiga.utils.data_loading import _read_dataset_parquet, normalise_timestamp_column

from twiga_thesis.plotting import line_style as LS, tokens as T
from twiga_thesis.plotting.illustrations import Z, _components, _profile

OUT = Path(__file__).resolve().parent / "output"
THESIS = Path(__file__).resolve().parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-3"
DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
RESULTS = Path(__file__).resolve().parents[1] / "cwe_validation"
NOTE_SIZE = 8

# The five metrics that rank the two candidates (PPS is zero for both on this target and carries no ranking).
METRICS = {
    "abs_corr": "Pearson correlation",
    "f_score": r"\textsc{anova} F-score",
    "mi_score": "Mutual information",
    "xi_score": r"$\xi$-correlation",
    "rf_importance": "Random-forest importance",
}


def dss_eu_disagreement():
    """Per-metric scores of Ghi and Temperature for reactive power on DSS-EU, scaled to the larger of the two.

    The F-score alone prefers Temperature; the other four metrics prefer Ghi, and so does the fused Borda score.
    """
    df = _read_dataset_parquet(DATA / "DSS-EU.parquet")
    df = df[(df.feeder == "default") & (df.ev_pen == 0.4) & (df.pv_pen == 0.4)]
    df = normalise_timestamp_column(df.reset_index(drop=True), "timestamp")
    candidates = ["Ghi", "Temperature"]
    df = df[["timestamp", "q_kvar", *candidates]].dropna()
    _, scores = select_top_features(
        data=df, features=candidates, target="q_kvar", top_k=len(candidates), return_scores=True
    )
    scores = scores.set_index("feature")

    rows = [*METRICS, "borda_count"]
    labels = [*METRICS.values(), "Borda (fused)"]
    fig, ax = LS.new_panel(0.55, 0.62)
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= 0.35  # separate the fused score from its constituents
    for yi, metric in zip(y, rows, strict=True):
        ghi, temp = scores.loc["Ghi", metric], scores.loc["Temperature", metric]
        top = max(ghi, temp)
        g, t = ghi / top, temp / top
        ax.plot([g, t], [yi, yi], color=T.GRAY_400, lw=LS.LINE_WIDTH, zorder=1)
        ax.plot(g, yi, "o", color=T.PRIMARY, ms=5, zorder=3)
        ax.plot(t, yi, "s", color=T.HIGHLIGHT, ms=4.5, zorder=3)
    ax.axhline((y[-2] + y[-1]) / 2, color=T.GRID, lw=0.6, zorder=0)
    LS.finish(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Score relative to the larger of the two")
    handles = [
        Line2D([], [], marker="o", ls="", color=T.PRIMARY, ms=5, label="Ghi"),
        Line2D([], [], marker="s", ls="", color=T.HIGHLIGHT, ms=4.5, label="Temperature"),
    ]
    LS.legend_below(ax, handles, [h.get_label() for h in handles], ncol=2)
    return fig


def _legend_handle(marker: str, color: str, label: str, hollow: bool = False, ls: str = "") -> Line2D:
    face = "white" if hollow else color
    return Line2D([], [], marker=marker, ls=ls, color=color, mfc=face, ms=4.5, lw=LS.LINE_WIDTH, label=label)


def soundness_sweep():
    """Each metric across the Gaussian forecast family, as a fraction of its own best value."""
    df = pd.read_csv(RESULTS / "s3521_soundness_gaussian.csv")
    x = df["sigma"].to_numpy()
    series = [
        ("CRPS", "crps", "min", T.PRIMARY, "--"),
        ("WS", "winkler", "min", T.PRIMARY, "-"),
        ("CWE", "cwe_reduced", "max", T.WARNING, "-"),
        ("CWC", "cwc", "max", T.HIGHLIGHT, "-"),
    ]
    fig, ax = LS.new_panel(0.60, 0.56)
    ax.axvline(1.0, color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH, zorder=1)
    for label, col, better, color, ls in series:
        v = df[col].to_numpy(dtype=float)
        best = np.nanmin(v) if better == "min" else np.nanmax(v)
        v = best / v if better == "min" else v / best
        ax.plot(x, v, color=color, ls=ls, lw=LS.LINE_WIDTH * 1.3, label=label, zorder=3)
        ax.plot([x[int(np.nanargmax(v))]], [1.0], "o", color=color, ms=3.5, zorder=5)
    ax.text(1.02, 0.05, r"$\sigma = 1$", fontsize=NOTE_SIZE, color=T.TEXT_MUTED, ha="left")
    LS.finish(ax)
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks([0.5, 1, 1.5, 2])
    ax.set_xticklabels(["0.5", "1", "1.5", "2"])
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"Forecast dispersion $\sigma$")
    ax.set_ylabel("Fraction of best")
    LS.legend_below(ax, ncol=4)
    return fig


def synthetic_series():
    """One day of the synthetic net-load series: demand minus clear-sky PV, with the conditionally correct interval."""
    rng = np.random.default_rng(4)
    hod = np.arange(48) / 48.0
    hours = hod * 24
    demand, pv, _ = _components(hod)
    mu, sigma = _profile(hod)
    y = mu + rng.normal(0, sigma)
    half = Z * sigma
    fig, ax = LS.new_panel(0.62, 0.58)
    ax.axhline(0, color=LS.REFERENCE_COLOR, lw=LS.LINE_WIDTH, ls=":", zorder=1)
    ax.fill_between(hours, 0, pv, color=T.tint(T.WARNING, 0.75), lw=0, zorder=2)
    ax.plot(hours, demand, color=T.TEXT_MUTED, ls="--", lw=LS.LINE_WIDTH, zorder=2)
    ax.fill_between(hours, mu - half, mu + half, color=LS.BAND_COLOR, lw=0, zorder=3)
    ax.plot(hours, mu, color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH * 1.3, zorder=4)
    ax.plot(hours, y, "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=5, **LS.MARKER)
    ax.text(hours[16], demand[16] + 0.35, "Demand", fontsize=NOTE_SIZE, color=T.TEXT_MUTED, ha="right")
    ax.text(12, pv.max() * 0.62, "PV", fontsize=NOTE_SIZE, color=T.TEXT_DARK, ha="center")
    ax.text(hours[5], mu[5] - 0.75, "Net load", fontsize=NOTE_SIZE, color=T.TEXT_DARK, ha="center")
    imax = int(np.argmax(half))
    x_arrow = hours[imax]
    ax.annotate(
        "",
        xy=(x_arrow, (mu + half)[imax]),
        xytext=(x_arrow, (mu - half)[imax]),
        arrowprops={"arrowstyle": "|-|,widthA=0.2,widthB=0.2", "lw": 0.6, "color": T.TEXT_DARK},
        zorder=6,
    )
    ax.text(x_arrow + 1.6, (mu - half)[imax] - 0.05, "Widest interval", fontsize=NOTE_SIZE, ha="left", va="center")
    LS.finish(ax)
    LS.hour_axis(ax)
    ax.set_ylabel("Power")
    ax.margins(y=0.08)
    return fig


def reference_choice():
    """Spread of the target, spread of the residuals, and the width a calibrated interval needs."""
    v = pd.read_csv(RESULTS / "s3523_reference_scales.csv").set_index("quantity")["value"]
    bars = [
        (r"Target spread $y$", float(v["kappa_target"]), T.HIGHLIGHT),
        (r"Residual spread $y-\hat{\mu}$", float(v["kappa_residual"]), T.WARNING),
        ("Calibrated interval", float(v["median_interval_width"]), T.PRIMARY),
    ]
    fig, ax = LS.new_panel(0.48, 0.62)
    ypos = np.arange(len(bars))[::-1]
    ratio = float(v["kappa_target"]) / float(v["median_interval_width"])
    for i, ((_, val, color), yy) in enumerate(zip(bars, ypos, strict=True)):
        ax.barh(yy, val, height=0.5, color=color, zorder=3)
        note = rf"{val:.2f} (${ratio:.1f}\times$ calibrated)" if i == 0 else f"{val:.2f}"
        ax.text(val + 0.07, yy, note, va="center", ha="left", fontsize=NOTE_SIZE)
    LS.finish(ax)
    ax.set_yticks(ypos, [b[0] for b in bars])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(b[1] for b in bars) * 1.75)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Interval width")
    return fig


def factorial_matrix():
    """Where each combination of reference and stratification places its optimum, against the truth."""
    g = pd.read_csv(RESULTS / "s3523_netload_factorial.csv").groupby(["reference", "stratified"])["optimum"].mean()
    rows = [("Marginal", "none"), ("Stratified by hour", "hour of day")]
    fig, ax = LS.new_panel(0.48, 0.62)
    ax.axvline(1.0, color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH, zorder=1)
    ax.text(1.003, -0.38, "Truth", fontsize=NOTE_SIZE, color=T.TEXT_MUTED, ha="left", va="center")
    for i, (_, strat) in enumerate(rows):
        yy = 1 - i
        v_t, v_r = float(g.loc[("target", strat)]), float(g.loc[("residual", strat)])
        ax.plot([v_r, v_t], [yy, yy], color=T.GRAY_400, lw=LS.LINE_WIDTH, zorder=2)
        ax.plot(v_t, yy, "s", color=T.HIGHLIGHT, ms=4.5, zorder=3)
        ax.plot(v_r, yy, "o", color=T.PRIMARY, ms=5, zorder=3)
        for val in (v_t, v_r):
            ax.text(val, yy + 0.16, f"{val:.3f}", fontsize=NOTE_SIZE, ha="center", va="bottom")
    LS.finish(ax)
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 1.45)
    ax.set_xlim(0.99, 1.19)
    ax.set_xticks([1.0, 1.05, 1.1, 1.15])
    ax.set_xlabel(r"Location of the optimum ($\sigma$)")
    handles = [
        _legend_handle("s", T.HIGHLIGHT, r"Reference: target $y$"),
        _legend_handle("o", T.PRIMARY, r"Reference: residual $y-\hat{\mu}$"),
    ]
    LS.legend_below(ax, handles, [h.get_label() for h in handles], ncol=1)
    return fig


def contrast_coverage():
    """Coverage by hour of a conditional-width and a constant-width forecast with the same marginal coverage."""
    d = pd.read_csv(RESULTS / "s3524_contrasting_coverage.csv")
    fig, ax = LS.new_panel(0.47, 0.70)
    LS.reference_line(ax, 0.90)
    h = d["hour"] + 0.5
    ax.plot(h, d["conditional"], color=T.PRIMARY, lw=LS.LINE_WIDTH * 1.3, label="Conditional width", zorder=4)
    ax.plot(h, d["constant"], color=T.HIGHLIGHT, lw=LS.LINE_WIDTH * 1.3, label="Constant width", zorder=4)
    ax.text(0.4, 0.905, "Nominal", fontsize=NOTE_SIZE, color=T.TEXT_MUTED, va="bottom")
    LS.finish(ax)
    LS.hour_axis(ax)
    ax.set_ylim(0.75, 1.02)
    ax.set_yticks([0.8, 0.9, 1.0])
    ax.set_ylabel("Coverage")
    LS.legend_below(ax, ncol=2)
    return fig


def contrast_preference():
    """The margin by which each metric prefers the conditional forecast; positive agrees with the proper scores."""
    tab = pd.read_csv(RESULTS / "s3524_contrasting_cases.csv").set_index("case")
    rows = [
        ("WS", "winkler", "min"),
        ("CRPS", "crps", "min"),
        ("CWE (marginal)", "cwe_reduced", "max"),
        ("CWE (stratified)", "cwe_stratified", "max"),
        ("CWC", "cwc", "max"),
    ]
    labels, margins = [], []
    for name, col, better in rows:
        g, b = float(tab.loc["conditional", col]), float(tab.loc["constant width", col])
        margins.append(100 * (b - g) / abs(g) if better == "min" else 100 * (g - b) / abs(b))
        labels.append(name)
    fig, ax = LS.new_panel(0.47, 0.70)
    ypos = np.arange(len(labels))[::-1]
    for m, yy in zip(margins, ypos, strict=True):
        ax.barh(yy, m, height=0.5, color=T.PRIMARY if m > 0 else T.HIGHLIGHT, zorder=3)
        text = rf"{m:+.2f}\%" if abs(m) < 1 else rf"{m:+.1f}\%"
        ax.text(max(m, 0) + 0.8, yy, text, va="center", ha="left", fontsize=NOTE_SIZE)
    ax.axvline(0, color=LS.REFERENCE_COLOR, lw=LS.LINE_WIDTH, zorder=4)
    LS.finish(ax)
    ax.set_yticks(ypos, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-2, 38)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel(r"Margin favouring the conditional forecast (\%)")
    return fig


def margin_strip():
    """Per-fold margin favouring the conditional forecast, by metric and dataset, on a symmetric-log axis."""
    m = pd.read_csv(RESULTS / "s3524_margins.csv")
    order = [
        ("CWE marg.", "CWE (marginal)"),
        ("CWC", "CWC"),
        ("CRPS", "CRPS"),
        ("WS", "WS"),
        ("CWE strat.", "CWE (stratified)"),
    ]
    style = {"MLVS-PT": ("o", T.PRIMARY, "MLVS-PT"), "MLVS-UK": ("o", T.HIGHLIGHT, "SPS-UK")}
    fig, ax = LS.new_panel(0.72, 0.46)
    ax.axvline(0, color=LS.REFERENCE_COLOR, ls=":", lw=LS.LINE_WIDTH, zorder=2)
    for i, (metric, _) in enumerate(order):
        yy = len(order) - 1 - i
        for ds, (marker, color, _) in style.items():
            v = m[(m.metric == metric) & (m.dataset == ds)]["margin"].to_numpy()
            offset = -0.14 if ds == "MLVS-PT" else 0.14
            ax.plot(v, np.full(v.size, yy + offset), marker, ls="", color=color, mec="white", mew=0.5, ms=4.2, zorder=4)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_xlim(-12, 260)
    ax.set_xticks([-10, -1, 0, 1, 10, 100])
    ax.set_xticklabels(["$-10$", "$-1$", "0", "1", "10", "100"])
    ax.minorticks_off()
    LS.finish(ax)
    ax.set_yticks(range(len(order)), [o[1] for o in order][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"Margin favouring the conditional forecast (\%)")
    handles = [_legend_handle(mk, c, lab) for mk, c, lab in style.values()]
    LS.legend_below(ax, handles, [h.get_label() for h in handles], ncol=2)
    return fig


FIGURES = {
    "dss_eu_disagreement": ("fig_ch3_dss_eu_metric_disagreement", dss_eu_disagreement),
    "soundness_sweep": ("fig_soundness_sweep", soundness_sweep),
    "synthetic_series": ("fig_synthetic_series", synthetic_series),
    "reference_choice": ("fig_reference_choice", reference_choice),
    "factorial_matrix": ("fig_factorial_matrix", factorial_matrix),
    "contrast_coverage": ("fig_contrast_coverage", contrast_coverage),
    "contrast_preference": ("fig_contrast_preference", contrast_preference),
    "margin_strip": ("fig_margin_strip", margin_strip),
}


def render(names: list[str] | None = None, copy_to_thesis: bool = False) -> list[Path]:
    """Render the named figures (all by default) to ``output/``; copy to the Chapter 3 folder if asked."""
    LS.configure()
    OUT.mkdir(exist_ok=True)
    paths = []
    for name in names or list(FIGURES):
        stem, build = FIGURES[name]
        LS.save(build(), OUT / stem)
        if copy_to_thesis and THESIS.exists():
            shutil.copy(OUT / f"{stem}.pdf", THESIS / f"{stem}.pdf")
        paths.append(OUT / f"{stem}.pdf")
    return paths
