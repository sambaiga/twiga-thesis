r"""Chapter 6, Experiment 1 report: statistics, conclusions and every figure, per dataset.

Builds on ``fixed_benchmark_tables.py`` (tables, fold-level frames and
significance) and follows the practices of ``ch6_analyse_results.py`` and
``ch7_analyse_results.py``:

* fold-then-mean aggregation; DM and MCS on fold-level losses through
  ``twiga_thesis.analysis.dm_and_mcs``, after dropping heads that
  are degenerate on the tested metric;
* ``metric_table`` per target (mean, sample std, per-fold ranks, arithmetic,
  Borda and diversity-weighted Borda ranks), PICP ranked by its distance from
  the nominal 0.90;
* an exact Kupiec unconditional-coverage test per target and for joint
  coverage, from the saved hit sequences (Chapter 7 had to approximate it);
* a markdown conclusion whose decision rests on each section's deciding metric
  (lower mean wins, DM and MCS qualify it), never on the blended rank;
* the shared ``comparison_plots`` figures (forest grid, metric bars, DM
  matrices) and Chapter 7's per-fold coverage heatmap, coloured about the
  nominal level, plus the Chapter 6 dependence figures (residual correlation,
  sample clouds), calibrated joint coverage and a cross-dataset panel of the
  joint metrics.

Everything is written under ``<experiment-root>/analysis/``. Nothing is
copied into the thesis unless ``--copy-to-images`` is given.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_report.py \
        --experiment-root <experiments>/ch7_multitarget/multitarget_fixed_benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe, as in ch6_analyse_results.py / ch7_analyse_results.py
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parents[2]))
sys.path.insert(0, str(script_dir.parent / "shared"))
sys.path.insert(0, str(script_dir.parents[1]))
from twiga.core.metrics.calibration import kupiec_pof_test  # noqa: E402

import chapters.ch7_multitarget.experiment_1_benchmark.fixed_benchmark_analysis as fba  # noqa: E402
import chapters.ch7_multitarget.experiment_1_benchmark.fixed_benchmark_tables as fbt  # noqa: E402
import chapters.ch7_multitarget.shared.benchmark_dependence_check as bdc  # noqa: E402
from twiga_thesis.analysis import dm_and_mcs, metric_table  # noqa: E402
from twiga_thesis.plotting import style as S, tokens as T  # noqa: E402
from twiga_thesis.plotting.comparison_plots import (  # noqa: E402
    plot_dm_significance_matrix,
    plot_forest_grid,
    plot_metric_bar_grid,
)

NOMINAL = 0.90
REFERENCE = fbt.REFERENCE
MAIN = fbt.MAIN
LABEL = {
    "fpqr_anchor": "FPQR",
    "qr": "QR",
    "bnn": "BNN",
    "laplace": "Laplace",
    "fpqr": "FPQR (learned)",
    "fpqr_ch5": "FPQR (Ch.~5)",
    "fpqr_anchor_tuned": "FPQR (tuned)",
    "qr_tuned": "QR (tuned)",
    "bnn_tuned": "BNN (tuned)",
}
COLOUR = {"fpqr_anchor": T.PALETTE[2], "qr": T.PALETTE[0], "bnn": T.PALETTE[4], "laplace": T.PALETTE[5]}
# benchmark_dependence_check.py's figures key their labels and colours by head name.
bdc.HEAD_LABEL.update({h: LABEL[h] for h in MAIN})
bdc.HEAD_COLOUR.update(COLOUR)
SHORT = {"mlvs-pt": "MLVS-PT", "dss-eu": "DSS-EU", "dss-uk": "DSS-UK", "dss-us": "DSS-US"}
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    "coverage", [(0.0, T.DANGER), (0.5, T.DANGER), (NOMINAL, "#FFFFFF"), (1.0, T.PRIMARY)]
)
# The shared comparison_plots figures label conditions through display_name, which
# splits on "_" and title-cases ("fpqr_anchor" -> "FPQR Anchor"). Plotting on the
# display labels themselves, with each label registered as its own token, keeps
# them exactly as written here.
METRIC_DISPLAY = {"nrmse": "NRMSE", "crps": "CRPS", "winkler_score": "WS", "cwe": "CWE", "picp": "PICP"}


def pct() -> str:
    """A percent sign that survives both renderers: escaped only when TeX typesets the figure."""
    return r"\%" if matplotlib.rcParams["text.usetex"] else "%"


def label(head: str) -> str:
    """Display label for a head, with TeX-only spacing removed when TeX is off."""
    text = LABEL.get(head, head)
    return text if matplotlib.rcParams["text.usetex"] else text.replace("~", " ")


def displayed(frame: pd.DataFrame) -> pd.DataFrame:
    """The fold frame with conditions renamed to their display labels."""
    return frame.assign(condition=frame["condition"].map(lambda c: label(c)))


def display_tokens() -> dict[str, str]:
    """display_name lookup that maps every display label to itself."""
    return {label(h).lower(): label(h) for h in LABEL}


METRIC_DIRECTIONS = {
    "lower": {"nrmse", "crps", "winkler_score", "nmpi"},
    "higher": {"cwe"},
}


def short(name: str) -> str:
    return next((label for prefix, label in SHORT.items() if name.startswith(prefix)), name)


# ── statistics ────────────────────────────────────────────────────────────────


def fold_frames(loaded: dict, targets: list[str]) -> pd.DataFrame:
    """One row per head, fold and target: the four reported metrics plus PICP and NMPI."""
    rows = []
    for head, (metrics, preds) in loaded.items():
        base = fbt.per_fold(metrics, preds).assign(condition=head)
        interval = metrics[metrics.metric_type == "interval"].groupby(["Folds", "target"])[["picp", "nmpi"]].mean()
        rows.append(base.merge(interval.reset_index(), on=["Folds", "target"], how="left"))
    frame = pd.concat(rows, ignore_index=True)
    return frame[frame["target"].isin(targets)]


def hits(preds: pd.DataFrame) -> pd.DataFrame:
    """Coverage indicators per step, one column per target, indexed by (fold, timestamp)."""
    interval = preds[preds.metric_type == "interval"].copy()
    interval["hit"] = (interval.Actual >= interval.lower) & (interval.Actual <= interval.upper)
    return interval.pivot_table(index=["Folds", "timestamp"], columns="target", values="hit").dropna().astype(bool)


def hpo_importance(hpo_dir: Path) -> pd.DataFrame:
    """PED-ANOVA importances the head-tuned searches saved, one row per head and parameter.

    Reads ``<model>_head_tuned_importance.json`` from the dataset's HPO cache
    folder; empty when the head_tuned group has not run there.
    """
    rows = [
        {"head": path.stem.removesuffix("_head_tuned_importance"), "parameter": param, "importance": value}
        for path in sorted(hpo_dir.glob("*_head_tuned_importance.json"))
        for param, value in json.loads(path.read_text()).items()
    ]
    return pd.DataFrame(rows, columns=["head", "parameter", "importance"])


def kupiec(loaded: dict, targets: list[str]) -> pd.DataFrame:
    """Exact Kupiec POF test per head, per target and for joint coverage, at the nominal 90% level.

    Joint coverage of three targets at 90% each is not itself a 90% event, so
    the joint row tests against the nominal 90% the chapter reports it against,
    as a statement about operational reliability, not a calibration claim.
    """
    rows = []
    for head, (_, preds) in loaded.items():
        h = hits(preds)
        series = {fbt.ROLE[t]: h[t] for t in targets} | {"joint": h[targets].all(axis=1)}
        for name, covered in series.items():
            result = kupiec_pof_test((~covered).to_numpy().astype(float), alpha=1 - NOMINAL)
            rows.append(
                {
                    "condition": head,
                    "target": name,
                    "n": int(covered.size),
                    "coverage": float(covered.mean()),
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "calibrated": result.calibrated,
                }
            )
    return pd.DataFrame(rows)


def joint_fold_frame(loaded: dict, targets: list[str]) -> pd.DataFrame:
    """Per-fold joint coverage, energy score and Brier score for every head."""
    rows = []
    for head, (metrics, preds) in loaded.items():
        joint = hits(preds)[targets].all(axis=1).groupby(level="Folds").mean().rename("joint_coverage")
        window = metrics[(metrics.metric_type == "window") & (metrics.target == "all")]
        scores = window.groupby("Folds")[["energy_score", "brier"]].mean()
        rows.append(pd.concat([joint, scores], axis=1).reset_index().assign(condition=head))
    return pd.concat(rows, ignore_index=True)


def decision(table: pd.DataFrame, dm: pd.DataFrame, mcs: set, metric: str, lower: bool = True) -> str:
    """One sentence on the deciding metric: lowest mean wins, DM and MCS qualify it (never the blended rank)."""
    best = str(table.sort_values(metric, ascending=lower).iloc[0]["condition"])
    included = ", ".join(LABEL.get(n, n) for n in sorted(mcs))
    if best == REFERENCE:
        text = f"**{LABEL[REFERENCE]}** has the best mean `{metric}`."
        return text + (f" The MCS also retains {included}." if len(mcs) > 1 else "")
    row = dm[
        ((dm.condition_a == REFERENCE) & (dm.condition_b == best))
        | ((dm.condition_a == best) & (dm.condition_b == REFERENCE))
    ]
    p_adj = float(row["p_adjusted"].iloc[0]) if not row.empty else float("nan")
    verdict = "significantly better than" if p_adj < 0.05 else "not significantly different from"
    return f"**{LABEL.get(best, best)}** has the best mean `{metric}` and is {verdict} **{LABEL[REFERENCE]}** (DM p_adj={p_adj:.3f}); MCS: {included}."  # noqa: E501


# ── figures ───────────────────────────────────────────────────────────────────


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_coverage(folds_joint: pd.DataFrame, per_target: pd.DataFrame, starts: pd.Series, out: Path) -> None:
    """Chapter 7's per-fold coverage heatmap: rows heads, columns folds; panels for joint coverage and each target."""
    panels = [("Joint", folds_joint.pivot(index="condition", columns="Folds", values="joint_coverage"))]
    for role in ("P", "Q", "V"):
        scoped = per_target[per_target["role"] == role]
        panels.append((role, scoped.pivot(index="condition", columns="Folds", values="picp")))
    fig, axes = plt.subplots(
        len(panels), 1, figsize=S.panel(1.0, aspect=0.22 * len(panels)), sharex=True, layout="constrained"
    )
    image = None
    for ax, (title, wide) in zip(axes, panels, strict=True):
        wide = wide.reindex(MAIN)
        image = ax.imshow(
            wide.to_numpy(), cmap=COVERAGE_CMAP, norm=Normalize(0.0, 1.0), aspect="auto", interpolation="none"
        )
        for i in range(wide.shape[0]):
            for j in range(wide.shape[1]):
                value = wide.iat[i, j]
                if np.isfinite(value):
                    colour = "white" if value < 0.65 or value > 0.985 else T.TEXT_DARK
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=colour)
        ax.set_yticks(range(len(MAIN)), [LABEL[h] for h in MAIN])
        ax.set_ylabel(title if title == "Joint" else f"${title}$", rotation=0, ha="right", va="center")
        ax.tick_params(length=0)
        ax.minorticks_off()
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xticks(range(len(starts)), [s.strftime("%b %Y") for s in starts], rotation=30, ha="right")
    axes[-1].set_xlabel("Start of the test window")
    bar = fig.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02, location="right")
    bar.set_label(r"$\mathrm{PICP}$")
    bar.ax.axhline(NOMINAL, color=T.TEXT_DARK, lw=0.6)
    bar.set_ticks([0.0, 0.5, 1.0])
    bar.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    bar.outline.set_visible(False)
    save(fig, out, "fold_coverage")


# Per-target comparison with the target role on the x-axis. Metrics already on a
# 0-1 scale (PICP, stratified CWE) are shown as they are; WS and CRPS are in the
# target's own units, so each is divided by the largest observed value of that
# target over the test period, which puts P, Q and V on one scale without
# min-max normalisation (that stretches differences of a fraction of a percent
# to the full axis). The scaled panels use a log axis so V, whose scores are
# an order of magnitude smaller after scaling, stays readable beside P and Q.
SCALED = {"winkler_score", "crps"}
NATIVE_METRICS = [("winkler_score", "lower"), ("crps", "lower"), ("cwe", "higher")]
CONFORMAL_METRICS = [("winkler_score", "lower"), ("picp", "nominal"), ("cwe", "higher")]
NORMALISED_TITLE = {
    "winkler_score": r"WS / $\max y$",
    "crps": r"CRPS / $\max y$",
    "cwe": "CWE (stratified)",
    "picp": "PICP",
}
DIRECTION_NOTE = {"lower": "lower is better", "higher": "higher is better", "nominal": "nominal 0.90"}
CONFORMAL_VARIANTS = {"max_norm": "max-norm", "quantile_unscaled_marginal": "CQR"}
ROLES = ("P", "Q", "V")


def target_scale(loaded: dict, targets: list[str]) -> dict[str, float]:
    """Largest absolute observed value per target role over the test period."""
    preds = loaded[REFERENCE][1]
    actual = preds[preds["metric_type"] == "interval"].groupby("target")["Actual"].agg(lambda a: a.abs().max())
    return {fbt.ROLE[t]: float(actual[t]) for t in targets}


def plot_normalised(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str]],
    series: dict[str, tuple[str, str, str | None]],
    scale: dict[str, float],
    name: str,
    out: Path,
) -> None:
    """Panels per metric, the target role on the x-axis, one bar per series.

    ``series`` maps a condition key to (legend label, colour, hatch).
    """
    means = frame.groupby(["condition", "role"])[[m for m, _ in metrics]].mean()
    keys = [k for k in series if k in means.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(metrics), figsize=S.panel(1.0, aspect=0.32), squeeze=False)
    width = 0.8 / len(keys)
    x = np.arange(len(ROLES))
    for ax, (metric, direction) in zip(axes[0], metrics, strict=True):
        for k, key in enumerate(keys):
            values = [means[metric].get((key, role), np.nan) for role in ROLES]
            if metric in SCALED:
                values = [v / scale[role] for v, role in zip(values, ROLES, strict=True)]
            text, colour, hatch = series[key]
            ax.bar(
                x + (k - (len(keys) - 1) / 2) * width,
                values,
                width,
                color=colour,
                hatch=hatch,
                edgecolor="white" if hatch else None,
                lw=0,
                label=text,
            )
        if direction == "nominal":
            ax.axhline(NOMINAL, color="k", lw=0.7, ls="--")
        if metric in SCALED:
            # Log scale: busbar voltage sits far from zero, so its scores divided by
            # max y are an order of magnitude below P's and Q's and vanish on a linear axis.
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(0, 1)
        ax.set_xticks(x, [f"${r}$" for r in ROLES])
        ax.set_title(f"{NORMALISED_TITLE[metric]} ({DIRECTION_NOTE[direction]})", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12 if len(labels) > 4 else 0.08, 1, 1))
    save(fig, out, name)


def normalised_figures(loaded: dict, dataset_dir: Path, targets: list[str], out: Path) -> None:
    """Unconformalised heads on WS, CRPS and CWE; conformal variants on WS, PICP and CWE."""
    scale = target_scale(loaded, targets)
    heads = compared_heads(loaded)
    native = fold_frames({h: loaded[h] for h in heads}, targets)
    native = native.assign(role=native["target"].map(fbt.ROLE))
    series = {h: (label(h), COLOUR.get(h, T.PALETTE[1]), None) for h in heads}
    plot_normalised(native, NATIVE_METRICS, series, scale, "normalised_unconformalised", out)

    rows, series = [], {}
    for head in MAIN:
        for suffix, method in CONFORMAL_VARIANTS.items():
            path = dataset_dir / "conformal" / f"{head}_{suffix}"
            if not (path / "metrics.parquet").exists():
                continue
            metrics, preds = fba.load_condition(path)
            key = f"{head}_{suffix}"
            frame = fbt.per_fold(metrics, preds)
            picp = metrics[metrics.metric_type == "interval"].groupby(["Folds", "target"])["picp"].mean()
            frame = frame.drop(columns=["picp"], errors="ignore")
            frame = frame.merge(picp.reset_index(), on=["Folds", "target"], how="left").assign(condition=key)
            rows.append(frame)
            # Head colour; CQR hatched so the two calibrations of one head sit side by side.
            series[key] = (f"{label(head)} + {method}", COLOUR[head], "///" if suffix != "max_norm" else None)
    if rows:
        conformal = pd.concat(rows, ignore_index=True)
        conformal = conformal.assign(role=conformal["target"].map(fbt.ROLE))
        plot_normalised(conformal, CONFORMAL_METRICS, series, scale, "normalised_conformal", out)


def plot_calibrated(calibrated: pd.DataFrame, out: Path) -> None:
    """Joint coverage by head before and after each conformal variant."""
    methods = list(fbt.CALIBRATION)
    fig, ax = plt.subplots(figsize=S.panel(0.7, aspect=0.6))
    width = 0.8 / len(methods)
    x = np.arange(len(MAIN))
    shades = [T.PALETTE[k % len(T.PALETTE)] for k in range(len(methods))]
    for k, method in enumerate(methods):
        values = [calibrated["joint"].get((h, method), np.nan) for h in MAIN]
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, values, width, color=shades[k], label=method)
    ax.axhline(NOMINAL, color="k", lw=0.8, ls="--", label=f"nominal 90{pct()}")
    ax.set_xticks(x, [label(h) for h in MAIN])
    ax.set_ylabel("Joint coverage")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, out, "joint_coverage_calibrated")


def plot_cross_dataset(joint_by_dataset: dict[str, pd.DataFrame], out: Path) -> None:
    """Joint metrics across datasets: joint coverage, energy score relative to QR, and Brier skill."""
    names = list(joint_by_dataset)
    rows = [
        ("joint", "Joint coverage"),
        ("energy_rel", "Energy score / QR"),
        ("brier_skill", "Brier skill"),
    ]
    fig, axes = plt.subplots(len(rows), len(names), figsize=S.panel(1.0, aspect=0.75), squeeze=False)
    x = np.arange(len(MAIN))
    for j, name in enumerate(names):
        frame = joint_by_dataset[name].loc[MAIN].copy()
        frame["energy_rel"] = frame["energy"] / frame.loc["qr", "energy"]
        for i, (column, row_label) in enumerate(rows):
            ax = axes[i, j]
            ax.bar(x, frame[column], color=[COLOUR[h] for h in MAIN])
            if column == "joint":
                ax.axhline(NOMINAL, color="k", lw=0.7, ls="--")
                ax.set_ylim(0, 1)
            if column == "energy_rel":
                ax.axhline(1.0, color="k", lw=0.7, ls="--")
                ax.set_ylim(min(0.85, float(frame[column].min()) * 0.97), max(1.15, float(frame[column].max()) * 1.02))
            if column == "brier_skill":
                ax.axhline(0.0, color="k", lw=0.7)
            ax.set_xticks(
                x, [label(h) for h in MAIN] if i == len(rows) - 1 else [], rotation=45, ha="right", fontsize=7
            )
            if i == 0:
                ax.set_title(short(name), fontsize=9)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    save(fig, out, "joint_metrics_by_dataset")


def dependence_figures(loaded: dict, targets: list[str], out: Path, seed: int = 20260906) -> pd.DataFrame:
    """Residual-correlation matrices and sample clouds, Chapter 6's construction (see benchmark_dependence_check.py)."""
    wide = {}
    for head in MAIN:
        preds = loaded[head][1]
        wide[head] = bdc.wide_frame(preds[preds.metric_type == "interval"], targets)
    stats = {h: bdc.head_stats(w, targets) for h, w in wide.items()}
    observed = np.corrcoef(next(iter(stats.values()))["actual"].T)
    bdc.plot_correlation_matrices(observed, {h: s["corr"] for h, s in stats.items()}, targets, out)
    rng = np.random.default_rng(seed)
    actual = next(iter(stats.values()))["actual"]
    row = int(np.abs(actual - actual.mean(0)).sum(1).argmin())
    bdc.plot_sample_clouds(
        {h: bdc.sample_cloud(s, 4000, rng, row) for h, s in stats.items()}, actual[row], targets, out
    )
    for old, new in (
        ("fig_ch6_benchmark_correlation_matrices", "correlation_matrices"),
        ("fig_ch6_benchmark_sample_clouds", "sample_clouds"),
    ):
        (out / f"{old}.pdf").rename(out / f"{new}.pdf")
    roles = [fbt.ROLE[t] for t in targets]
    rows = [
        {
            "matrix": "observed",
            **{f"{a}-{b}": observed[i, k] for i, a in enumerate(roles) for k, b in enumerate(roles) if i < k},
        }
    ]
    rows += [
        {
            "matrix": LABEL[h],
            **{f"{a}-{b}": s["corr"][i, k] for i, a in enumerate(roles) for k, b in enumerate(roles) if i < k},
        }
        for h, s in stats.items()
    ]
    return pd.DataFrame(rows)


# ── per-dataset report ────────────────────────────────────────────────────────


def compared_heads(loaded: dict) -> list[str]:
    """The fixed heads, then any head-tuned heads the runs hold."""
    return [h for h in [*MAIN, *fbt.TUNED] if h in loaded]


def report(
    name: str, dataset_dir: Path, out: Path, experiment_root: Path | None = None, include_tuned: bool = False
) -> pd.DataFrame:
    """Tables, statistics, conclusion and figures for one dataset; returns its joint metrics."""
    result = fbt.build(dataset_dir, out, experiment_root, include_tuned)
    loaded, targets = result["loaded"], result["targets"]
    compared = compared_heads(loaded)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    folds = fold_frames(loaded, targets)
    folds["role"] = folds["target"].map(fbt.ROLE)
    lines = [f"# Experiment 1, {short(name)} ({dataset_dir.parent.parent.name})", ""]
    lines += ["Fold-level losses throughout (Chapters 5 and 7's convention); MCS at alpha=0.05.", ""]
    main_folds = folds[folds["condition"].isin(compared)]
    for role in ("P", "Q", "V"):
        scoped = main_folds[main_folds["role"] == role]
        table = metric_table(
            scoped,
            metric_cols=["nrmse", "crps", "winkler_score", "cwe", "picp", "nmpi"],
            lower_better=METRIC_DIRECTIONS["lower"],
            higher_better=METRIC_DIRECTIONS["higher"],
            target_values={"picp": NOMINAL},
        )
        table.to_csv(out / f"metric_table_{role}.csv", index=False)
        lines += [f"## Target {role}", "", table.round(4).to_markdown(index=False), ""]
        for metric in ("crps", "winkler_score", "nrmse"):
            dm, p_matrix, model_set = dm_and_mcs(fbt.drop_degenerate(scoped, metric), metric=metric)
            dm.to_csv(out / f"dm_{metric}_{role}.csv", index=False)
            lines.append(f"- {decision(table, dm, set(map(str, model_set.included)), metric)}")
            if metric != "nrmse":
                shown = p_matrix.loc[compared, compared].rename(index=label, columns=label)
                fig = plot_dm_significance_matrix(shown, name, f"{metric}_{role}")
                if fig is not None:
                    save(fig, figs, f"significance_{metric}_{role}")
        lines.append("")
        for plotter, stem in ((plot_forest_grid, "forest"), (plot_metric_bar_grid, "metric_bars")):
            kwargs = {
                "metric_cols": ["nrmse", "crps", "winkler_score", "cwe", "picp"],
                "metric_display": METRIC_DISPLAY,
                "extra_acronyms": display_tokens(),
            }
            if plotter is plot_metric_bar_grid:
                # Own y-axis per metric: NRMSE and PICP sit near 1, WS in the tens.
                kwargs |= {
                    "lower_better": METRIC_DIRECTIONS["lower"],
                    "higher_better": METRIC_DIRECTIONS["higher"],
                    "target_values": {"picp": NOMINAL},
                    "sharey": False,
                }
            try:
                fig = plotter(displayed(scoped), name, f"benchmark_{role}", label(REFERENCE), **kwargs)
                if fig is not None:
                    save(fig, figs, f"{stem}_{role}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [figure skipped] {stem}_{role}: {exc}")

    joint_folds = joint_fold_frame({h: loaded[h] for h in compared}, targets)
    joint_table = metric_table(
        joint_folds,
        metric_cols=["joint_coverage", "energy_score", "brier"],
        lower_better={"energy_score", "brier"},
        higher_better=set(),
        target_values={"joint_coverage": NOMINAL},
    )
    joint_table.to_csv(out / "metric_table_joint.csv", index=False)
    lines += ["## Joint", "", joint_table.round(4).to_markdown(index=False), ""]
    for metric in ("energy_score", "brier"):
        dm, _, model_set = dm_and_mcs(fbt.drop_degenerate(joint_folds, metric), metric=metric)
        dm.to_csv(out / f"dm_{metric}_joint.csv", index=False)
        lines.append(f"- {decision(joint_table, dm, set(map(str, model_set.included)), metric)}")
    lines.append("")

    importance = hpo_importance(experiment_root / "backbone_hpo" / name) if experiment_root else pd.DataFrame()
    if not importance.empty:
        importance.to_csv(out / "hpo_importance.csv", index=False)
        table = importance.round(3).to_markdown(index=False)
        lines += ["## Head tuning: PED-ANOVA parameter importance", "", table, ""]

    kup = kupiec(loaded, targets)
    kup.to_csv(out / "kupiec.csv", index=False)
    lines += ["## Kupiec unconditional coverage (nominal 90%)", "", kup.round(4).to_markdown(index=False), ""]
    correlations = dependence_figures(loaded, targets, figs)
    correlations.to_csv(out / "residual_correlation.csv", index=False)
    lines += [
        "## Residual correlation (observed vs each head's errors)",
        "",
        correlations.round(3).to_markdown(index=False),
        "",
    ]

    starts = (
        loaded[REFERENCE][1]
        .groupby("Folds")["timestamp"]
        .min()
        .sort_index()
        .pipe(lambda s: pd.to_datetime(s).dt.tz_localize(None))
    )
    plot_fold_coverage(joint_folds, folds.dropna(subset=["picp"]), starts, figs)
    if not result["calibrated"].empty:
        plot_calibrated(result["calibrated"], figs)
    normalised_figures(loaded, dataset_dir, targets, figs)
    fans = fba._fan_frames({("benchmark", h): v for h, v in loaded.items()})
    if fans:
        best, worst = fba.pick_days(fans)
        names = {h: label(h) for h in fans}
        fba.plot_day(fans, best, "best day", figs / "forecast_best", names=names)
        fba.plot_day(fans, worst, "worst day", figs / "forecast_worst", names=names)
    (out / "conclusion.md").write_text("\n".join(lines) + "\n")
    return result["joint"]


def main() -> None:
    """Report every dataset, then the cross-dataset joint-metrics figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, required=True, help="folder holding the fixed benchmark's runs/")
    ap.add_argument("--no-latex", action="store_true")
    ap.add_argument(
        "--include-tuned", action="store_true", help="add the head-tuned heads to every table and test family"
    )
    ap.add_argument(
        "--tuned-root",
        type=Path,
        default=None,
        help="folder holding the head-tuned runs/ when apart from the fixed ones (default: --experiment-root)",
    )
    ap.add_argument("--output", type=Path, default=None, help="analysis folder (default: <experiment-root>/analysis)")
    ap.add_argument(
        "--copy-to-images",
        action="store_true",
        help="also copy the PDFs into the thesis Images/Chapter-6 as fig_ch6_fixed_<dataset>_<name>.pdf",
    )
    args = ap.parse_args()
    S.configure(latex=not args.no_latex)
    analysis = args.output or args.experiment_root / "analysis"
    tuned_root = args.tuned_root or args.experiment_root
    joint_by_dataset = {}
    for name, dataset_dir in fbt.newest_dataset_dirs(args.experiment_root).items():
        print(f"== {name}")
        joint_by_dataset[name] = report(name, dataset_dir, analysis / name, tuned_root, args.include_tuned)
    order = sorted(joint_by_dataset, key=lambda n: list(SHORT).index(next(p for p in SHORT if n.startswith(p))))
    plot_cross_dataset({n: joint_by_dataset[n] for n in order}, analysis)
    if args.copy_to_images:
        images = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
        for pdf in analysis.glob("*/figures/*.pdf"):
            slug = short(pdf.parent.parent.name).lower().replace("-", "_")
            shutil.copy(pdf, images / f"fig_ch6_fixed_{slug}_{pdf.stem}.pdf")
        shutil.copy(analysis / "joint_metrics_by_dataset.pdf", images / "fig_ch6_fixed_joint_metrics_by_dataset.pdf")
        print(f"copied into {images}")
    print(f"written under {analysis}")


if __name__ == "__main__":
    main()
