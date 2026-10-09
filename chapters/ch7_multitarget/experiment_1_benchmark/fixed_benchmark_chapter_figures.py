r"""Chapter 6, Experiment 1 figures for every dataset, in the thesis line-plot style.

One panel per dataset (or per dataset and target), authored at its final size
with :mod:`thesis_figures.line_style` (10 pt labels, 8 pt legends below the
axes, outward ticks, no minor ticks, the thesis palette), so the chapter can
place them as subfigures and refer to each one:

* ``significance_crps_<target>``: pairwise DM tests on fold-level CRPS
  (Benjamini-Hochberg adjusted), significant pairs in the primary colour;
* ``joint_coverage``, ``energy_relative``, ``brier_skill``: the joint metrics
  of the four heads on every dataset, one metric per panel;
* ``fold_coverage``: exact joint coverage per fold and head;
* ``correlation``: distance correlation of the observed P, Q and V beside
  that of each head's forecast errors;
* ``clouds_pv``: each head's implied joint distribution of P and V for one
  representative test window, with the observation.

Files are written as ``fig_ch6_fixed_<dataset>_<name>.pdf`` (and the
dataset-independent joint panels as ``fig_ch6_fixed_<name>.pdf``) under
``<experiment-root>/analysis/chapter_figures``; ``--copy-to-images`` also copies
them into the thesis Images/Chapter-6 folder.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_chapter_figures.py \
        --experiment-root ../experiments/ch7_multitarget/multitarget_fixed_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parent / "shared"), str(script_dir.parents[1])]

import benchmark_dependence_check as bdc  # noqa: E402
from experiment.common.comparison_plots import plot_dm_significance_matrix  # noqa: E402
from experiment.common.result_analysis import dm_and_mcs  # noqa: E402
import fixed_benchmark_tables as fbt  # noqa: E402
from thesis_figures import line_style as LS, style as S, tokens as T  # noqa: E402
from twiga.core.stats.association import distance_correlation  # noqa: E402

MAIN = fbt.MAIN
LABEL = {"fpqr_anchor": "FPQR", "qr": "QR", "bnn": "BNN", "laplace": "Laplace"}
COLOUR = {"fpqr_anchor": T.PALETTE[2], "qr": T.PALETTE[0], "bnn": T.PALETTE[4], "laplace": T.PALETTE[5]}
SLUG = {"mlvs-pt": "mlvs", "dss-eu": "dsseu", "dss-uk": "dssuk", "dss-us": "dssus"}
NAME = {"mlvs-pt": "MLVS-PT", "dss-eu": "DSS-EU", "dss-uk": "DSS-UK", "dss-us": "DSS-US"}
ORDER = ["mlvs-pt", "dss-eu", "dss-uk", "dss-us"]
NOMINAL = 0.90
# Diverging around a reference: orange below, white at the reference, teal above (the thesis palette's
# DANGER and PRIMARY, as in Chapter 7's figures).
DIVERGING = LinearSegmentedColormap.from_list("thesis_diverging", [T.DANGER, T.WHITE, T.PRIMARY])
# DM significance in the chapter's original greyscale (comparison_plots.plot_dm_significance_matrix):
# significant pairs light, the rest dark grey, a white cross on the diagonal.
SIGNIFICANT, NOT_SIGNIFICANT = "#EFEFEF", "#525252"
# Dependence strength, as the Chapter 7 gate-value heatmap: white to the thesis teal, sign in the printed value.
STRENGTH = LinearSegmentedColormap.from_list("thesis_strength", ["#FFFFFF", T.PRIMARY])
STRENGTH.set_bad("#FFFFFF")
THREE_PANELS = 0.30


def pct() -> str:
    """A percent sign that survives both renderers: escaped only when TeX typesets the figure."""
    return r"\%" if matplotlib.rcParams["text.usetex"] else "%"


def dataset_prefix(dataset_dir_name: str) -> str:
    """The short key (``mlvs-pt``, ``dss-eu``, ...) a dataset directory starts with."""
    return next(prefix for prefix in ORDER if dataset_dir_name.startswith(prefix))


def heatmap_ticks(ax: plt.Axes) -> None:
    """Heatmap axes: no ticks marks, no spines."""
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


# ── Significance ──────────────────────────────────────────────────────────────


def significance_panel(folds: pd.DataFrame, target: str, path: Path, latex: bool = True) -> None:
    """Pairwise DM significance on fold-level CRPS, drawn exactly as Chapter 7's DM figures.

    Uses the shared :func:`comparison_plots.plot_dm_significance_matrix` unchanged (binary
    greyscale, BH-adjusted at alpha = 0.05, colour bar naming both classes), in the base
    thesis style rather than the compact line-plot one, relabelled and saved at 300 dpi
    as ``ch7_architectural_analysis.plot_significance`` does.
    """
    scoped = folds[(folds["target"] == target) & folds["condition"].isin(MAIN)]
    _, p_matrix, _ = dm_and_mcs(fbt.drop_degenerate(scoped, "crps"), metric="crps")
    heads = [h for h in MAIN if h in p_matrix.index]
    S.configure(latex=latex)  # the base thesis style, as Chapter 7 draws these
    try:
        fig = plot_dm_significance_matrix(p_matrix.loc[heads, heads], "", target)
        if fig is None:
            return
        labels = [LABEL[h] for h in heads]
        ax = fig.axes[0]
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        for ext in ("pdf", "png"):
            fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        LS.configure(latex=latex)  # back to the line-plot style for the other panels


# ── Joint metrics, one metric per panel ───────────────────────────────────────


def joint_panel(joint: dict[str, pd.DataFrame], column: str, ylabel: str, path: Path) -> None:
    """Grouped bars: datasets on the x-axis, one bar per head."""
    datasets = [d for d in ORDER if d in joint]
    fig, ax = LS.new_panel(THREE_PANELS, 1.05)
    width = 0.8 / len(MAIN)
    x = np.arange(len(datasets))
    values = []
    for k, head in enumerate(MAIN):
        heights = []
        for d in datasets:
            frame = joint[d]
            value = frame.loc[head, column]
            if column == "energy":
                value = value / frame.loc["qr", "energy"]
            heights.append(value)
        values += heights
        ax.bar(x + (k - (len(MAIN) - 1) / 2) * width, heights, width, color=COLOUR[head], label=LABEL[head], lw=0)
    if column == "joint":
        LS.reference_line(ax, NOMINAL)
        LS.set_y_axis(ax, unit_interval=True)
    elif column == "energy":
        LS.reference_line(ax, 1.0)
        ax.set_ylim(min(0.8, min(values) * 0.97), max(1.1, max(values) * 1.02))
    else:
        LS.reference_line(ax, 0.0)
    ax.set_xticks(x, [NAME[d] for d in datasets], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    LS.finish(ax)
    LS.legend_below(ax, ncol=2)
    LS.save(fig, path)


def calibrated_panel(calibrated: pd.DataFrame, path: Path) -> None:
    """Joint coverage of each head natively, after max-norm, and after the three unscaled CQR constructions.

    The scaled CQR variants are reported in the calibrated table only.
    """
    methods = {
        "native": "Native",
        "max-norm": "Max-norm",
        "CQR marginal": "CQR marginal",
        "CQR Bonferroni": "CQR Bonferroni",
        "CQR pooled": "CQR pooled",
    }
    shades = dict(zip(methods, [T.PALETTE[5], T.PALETTE[0], T.PALETTE[2], T.PALETTE[1], T.PALETTE[3]], strict=True))
    fig, ax = LS.new_panel(0.96, 0.36)
    width = 0.8 / len(methods)
    x = np.arange(len(MAIN))
    for k, (method, text) in enumerate(methods.items()):
        heights = [calibrated["joint"].get((h, method), np.nan) for h in MAIN]
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, heights, width, color=shades[method], label=text, lw=0)
    LS.reference_line(ax, NOMINAL)
    LS.set_y_axis(ax, unit_interval=True)
    ax.set_xticks(x, [LABEL[h] for h in MAIN])
    ax.set_ylabel("Joint coverage")
    LS.finish(ax)
    LS.legend_below(ax, ncol=len(methods))
    LS.save(fig, path)


# ── Coverage per fold ─────────────────────────────────────────────────────────


def fold_coverage_panel(loaded: dict, targets: list[str], path: Path) -> None:
    """Exact joint coverage per fold (columns) and head (rows), diverging at the nominal 90%."""
    rows, starts = [], None
    for head in MAIN:
        preds = loaded[head][1]
        interval = preds[preds["metric_type"] == "interval"].copy()
        interval["hit"] = (interval["Actual"] >= interval["lower"]) & (interval["Actual"] <= interval["upper"])
        wide = interval.pivot_table(index=["Folds", "timestamp"], columns="target", values="hit").dropna()
        joint = wide.all(axis=1).groupby(level="Folds").mean()
        rows.append(joint.rename(LABEL[head]))
        if starts is None:
            stamps = pd.to_datetime(interval.groupby("Folds")["timestamp"].min(), utc=True)
            starts = stamps.dt.strftime("%b %y")
    grid = pd.concat(rows, axis=1).T
    fig, ax = LS.new_panel(0.48, 0.42)
    ax.imshow(
        grid.to_numpy(dtype=float),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=NOMINAL, vmax=1.0),
        aspect="auto",
    )
    for (i, j), value in np.ndenumerate(grid.to_numpy(dtype=float)):
        ink = T.TEXT_LIGHT if value < 0.35 or value > 0.97 else T.TEXT_DARK
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=ink)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xticks(range(grid.shape[1]), starts.reindex(grid.columns).to_numpy(), rotation=45, ha="right")
    heatmap_ticks(ax)
    LS.save(fig, path)


# ── Dependence ────────────────────────────────────────────────────────────────


def head_stats(loaded: dict, targets: list[str]) -> dict[str, dict]:
    """Residual correlation, scale and forecasts per head, from its saved 90% intervals."""
    stats = {}
    for head in MAIN:
        preds = loaded[head][1]
        stats[head] = bdc.head_stats(bdc.wide_frame(preds[preds["metric_type"] == "interval"], targets), targets)
    return stats


def dependence_matrix(values: np.ndarray, max_samples: int = 2000) -> np.ndarray:
    """Pairwise distance correlation of a ``(rows, targets)`` array, as Chapter 6's dependence analysis.

    Distance correlation is zero if and only if two variables are independent and
    presumes no functional form, so it also registers the non-linear coupling of
    voltage to active and reactive power; it carries no sign.
    """
    out = np.eye(values.shape[1])
    for i in range(values.shape[1]):
        for j in range(i + 1, values.shape[1]):
            out[i, j] = out[j, i] = distance_correlation(values[:, i], values[:, j], max_samples=max_samples)
    return out


def correlation_panel(stats: dict[str, dict], targets: list[str], path: Path) -> None:
    """Distance correlation of the observed P, Q and V beside that of each head's forecast errors.

    Drawn as the Chapter 7 gate-value heatmap: white to the thesis teal, with the
    value printed in each off-diagonal cell.
    """
    roles = [f"${fbt.ROLE[t]}$" for t in targets]
    first = next(iter(stats.values()))
    matrices = {"Observed": dependence_matrix(first["actual"])}
    matrices |= {LABEL[h]: dependence_matrix(s["actual"] - s["forecast"]) for h, s in stats.items()}
    fig, grid = plt.subplots(
        1, len(matrices) + 1, figsize=S.panel(1.0, 0.24), gridspec_kw={"width_ratios": [1] * len(matrices) + [0.06]}
    )
    axes, colour_axis = grid[:-1], grid[-1]
    for ax, (title, matrix) in zip(axes, matrices.items(), strict=True):
        shown = np.where(np.eye(len(roles), dtype=bool), np.nan, matrix)
        image = ax.imshow(np.ma.masked_invalid(shown), cmap=STRENGTH, vmin=0.0, vmax=1.0)
        for (i, j), value in np.ndenumerate(matrix):
            if i != j:
                ink = "white" if value > 0.55 else T.TEXT_DARK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=ink)
        ax.set_xticks(range(len(roles)), roles)
        ax.set_yticks(range(len(roles)), roles if ax is axes[0] else [])
        ax.set_title(title, fontsize=LS.LEGEND_FONTSIZE + 1)
        ax.minorticks_off()
        heatmap_ticks(ax)
    colorbar = fig.colorbar(image, cax=colour_axis, ticks=[0, 0.5, 1])
    colorbar.set_label("Distance correlation", fontsize=LS.LEGEND_FONTSIZE)
    colorbar.outline.set_visible(False)
    LS.save(fig, path)


def clouds_panel(stats: dict[str, dict], targets: list[str], path: Path, seed: int = 20260906) -> None:
    """Each head's implied P-V cloud for one representative test window, with its 90% ellipse and the observation."""
    rng = np.random.default_rng(seed)
    actual = next(iter(stats.values()))["actual"]
    row = int(np.abs(actual - actual.mean(0)).sum(1).argmin())
    p_idx, v_idx = 0, len(targets) - 1
    fig, axes = plt.subplots(1, len(MAIN), figsize=S.panel(1.0, 0.27), sharex=False, sharey=True)
    for ax, head in zip(axes, MAIN, strict=True):
        pts = bdc.sample_cloud(stats[head], 2000, rng, row)[:, [p_idx, v_idx]]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=COLOUR[head], alpha=0.25, lw=0, rasterized=True)
        bdc.ellipse(ax, pts, COLOUR[head], f"90{pct()} region")
        ax.plot(
            actual[row, p_idx], actual[row, v_idx], "o", mfc=T.WHITE, mec=T.TEXT_DARK, ms=4, mew=0.9, label="Observed"
        )
        rho = np.corrcoef(pts.T)[0, 1]
        ax.set_title(f"{LABEL[head]} ($\\rho = {rho:+.2f}$)", fontsize=LS.LEGEND_FONTSIZE + 1)
        ax.set_xlabel("$P$")
        LS.finish(ax)
    axes[0].set_ylabel("$V$")
    observed = plt.Line2D([], [], marker="o", ls="", mfc=T.WHITE, mec=T.TEXT_DARK, ms=4, mew=0.9)
    region = plt.Line2D([], [], color=T.TEXT_DARK, lw=1.2)
    labels = ["Observed", f"90{pct()} region of each head"]
    LS.legend_below(axes[len(axes) // 2 - 1], [observed, region], labels, ncol=2)
    LS.save(fig, path)


# ── Driver ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Draw every Experiment 1 chapter figure for every dataset."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--no-latex", action="store_true")
    ap.add_argument("--copy-to-images", action="store_true")
    args = ap.parse_args()
    LS.configure(latex=not args.no_latex)
    out = args.experiment_root / "analysis" / "chapter_figures"
    out.mkdir(parents=True, exist_ok=True)
    joint: dict[str, pd.DataFrame] = {}
    for name, dataset_dir in fbt.newest_dataset_dirs(args.experiment_root).items():
        key = dataset_prefix(name)
        print(f"== {NAME[key]}")
        built = fbt.build(dataset_dir, args.experiment_root / "analysis" / name, args.experiment_root)
        loaded, targets = built["loaded"], built["targets"]
        joint[key] = built["joint"]
        folds = pd.concat(
            [fbt.per_fold(m, p).assign(condition=h) for h, (m, p) in loaded.items() if h in MAIN], ignore_index=True
        )
        prefix = f"fig_ch6_fixed_{SLUG[key]}"
        for target in targets:
            significance_panel(
                folds, target, out / f"{prefix}_significance_crps_{fbt.ROLE[target].lower()}", not args.no_latex
            )
        fold_coverage_panel(loaded, targets, out / f"{prefix}_fold_coverage")
        stats = head_stats(loaded, targets)
        correlation_panel(stats, targets, out / f"{prefix}_correlation")
        clouds_panel(stats, targets, out / f"{prefix}_clouds_pv")
        if not built["calibrated"].empty:
            calibrated_panel(built["calibrated"], out / f"{prefix}_joint_coverage_calibrated")
    joint_panel(joint, "joint", "Joint coverage", out / "fig_ch6_fixed_joint_coverage")
    joint_panel(joint, "energy", "Relative energy score", out / "fig_ch6_fixed_energy_relative")
    joint_panel(joint, "brier_skill", "Brier skill", out / "fig_ch6_fixed_brier_skill")
    if args.copy_to_images:
        images = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
        for pdf in out.glob("*.pdf"):
            shutil.copy(pdf, images / pdf.name)
        print(f"copied into {images}")
    print(f"written under {out}")


if __name__ == "__main__":
    main()
