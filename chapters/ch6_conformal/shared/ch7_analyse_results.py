r"""Analyze Chapter 7 benchmark results and produce thesis-ready comparisons.

The report covers architectural and probabilistic benchmarks for both
datasets. It normalizes model result frames, reports point and interval metric
tables, compares point-only conditions on MAE and interval-producing
conditions on Winkler score, computes coverage diagnostics, and writes CSV
reports and thesis-styled figures. Shared analysis and comparison functions
live in ``experiment.common``; Chapter 7 retains its experiment definitions,
metric sets, and CWE comparison.

Usage:

    uv run python experiment/ch6_conformal/shared/ch7_analyse_results.py \
        --experiment architectural_evaluation probabilistic_benchmark --dataset all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe, as in Chapter 6's ch6_analyse_results.py
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
# The chapter directory (for ``shared.*``) and ``experiment/`` (for ``twiga_thesis.plotting``).
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir.parents[1]))

from chapters.ch6_conformal.shared.protocol import CHAPTER_DATASETS, DEFAULT_OUTPUT_ROOT  # noqa: E402

RESULTS_ROOT = DEFAULT_OUTPUT_ROOT
OUT_DIR = RESULTS_ROOT / "analysis"
FIG_DIR = OUT_DIR / "figures"
IMAGES_DIR = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"

from chapters.ch6_conformal.shared.metrics import cwe_comparison  # noqa: E402
from twiga_thesis.analysis import dm_and_mcs, fold_level, kupiec_table, metric_table
from twiga_thesis.plotting import (  # noqa: E402
    style as S,
    tokens as T,
)
from twiga_thesis.plotting.comparison_plots import (
    display_name,
    head_label,
    plot_dm_significance_matrix,
    plot_forest_grid,
    plot_good_bad_day,
    plot_metric_bar_grid,
    plot_metric_bars,
    plot_normalized_comparison,
    plot_primary_metric_comparison,
    save_metric_figures,
)

DATASETS = list(CHAPTER_DATASETS)

# ── Chapter 7's own metric set (see module docstring) ───────────────────────
CH7_METRIC_COLS = ["nrmse", "winkler_score", "cwe", "cwe_stratified", "picp", "nmpi"]
CH7_PRIMARY_METRIC_COLS = ["nrmse", "winkler_score", "cwe_stratified"]
CH7_LOWER_BETTER = {"nrmse", "winkler_score", "nmpi"}
CH7_HIGHER_BETTER = {"cwe", "cwe_stratified"}
NOMINAL_PICP = 0.90  # Chapter 7 reuses Chapter 5's stated nominal coverage level.

# ── Per-experiment configuration ─────────────────────────────────────────────
EXPERIMENTS: dict[str, dict] = {
    "architectural_evaluation": {
        "csv": "benchmark_point_full.csv",  # superset: has NRMSE for all 4 conditions,
        # interval metrics NaN for the two point-only ones -- see module docstring.
        "interval_conditions": ["mlpf_scp", "mlpgam_scp"],
        "reference": "mlpgam_scp",  # this chapter's own architecture choice.
    },
    "probabilistic_benchmark": {
        "csv": "benchmark_point_full.csv",
        "interval_conditions": ["mlpgam_bnn", "mlpgam_crc", "mlpgam_laplace", "mlpgam_qr"],
        "reference": "mlpgam_crc",  # the chapter's headline proposed framework.
        # mlpgam_fpqr is run but not reported: its head search resumed an Optuna study holding trials from the
        # unpinned (Kumaraswamy proposal, spline QVN) configuration, so the trained model combined the pinned
        # architecture with hyperparameters tuned for a different one and does not represent FPQR fairly.
        "excluded": ["mlpgam_fpqr"],
    },
}


def _to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """``df.to_markdown()``, falling back to a fenced ``to_string()`` block if ``tabulate`` is absent."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def winsorize_nrmse(full: pd.DataFrame, quantile: float = 0.97) -> pd.DataFrame:
    """Clip ``nrmse`` at its own upper quantile per (dataset, group, condition).

    NRMSE divides by the scoring window's own range, so a near-constant window
    can explode one instance's value and dominate the mean over instances.
    Ported from Chapter 6's ``ch6_analyse_results._winsorize_nrmse``.
    """
    if "nrmse" not in full.columns:
        return full
    group_cols = [c for c in ("dataset", "group", "condition") if c in full.columns]
    full = full.copy()
    caps = full.groupby(group_cols)["nrmse"].transform(lambda s: s.quantile(quantile))
    full["nrmse"] = full["nrmse"].clip(upper=caps)
    return full


def drop_degenerate_conditions(fold_df: pd.DataFrame, metric: str, label: str) -> pd.DataFrame:
    """Drop conditions whose ``metric`` is NaN in every fold, so one collapsed condition cannot empty the DM/MCS pivot.

    Ported from Chapter 6's ``ch6_analyse_results._drop_degenerate_conditions``.
    """
    all_nan = fold_df.groupby("condition")[metric].apply(lambda s: s.isna().all())
    degenerate = all_nan[all_nan].index.tolist()
    if not degenerate:
        return fold_df
    print(f"  [excluded from significance] {label}: {metric} is NaN in every fold for {degenerate}")
    return fold_df[~fold_df["condition"].isin(degenerate)]


def write_conclusion(
    out_path: Path,
    *,
    dataset: str,
    experiment: str,
    reference: str,
    sections: list[dict],
    kupiec: pd.DataFrame,
) -> None:
    """Write one dataset's metric tables, DM/MCS significance, Kupiec test and decision sentences as markdown.

    Adapted from Chapter 6's ``write_conclusion``. The decision is taken on each
    section's decisive metric alone (lower mean wins; DM ``p_adjusted`` and the
    MCS qualify it), never on the blended Borda rank. Each section is a dict with
    ``name``, ``metric``, ``table``, ``dm``, ``mcs``.
    """
    lines = [f"## {dataset}", ""]
    for sec in sections:
        metric, table, dm, mcs = sec["metric"], sec["table"], sec["dm"], sec["mcs"]
        best = str(table.sort_values(metric).iloc[0]["condition"])
        included = sorted(str(name) for name in mcs.included)
        row = dm[
            ((dm["condition_a"] == reference) & (dm["condition_b"] == best))
            | ((dm["condition_a"] == best) & (dm["condition_b"] == reference))
        ]
        p_adj = float(row["p_adjusted"].iloc[0]) if not row.empty else float("nan")
        ref_name, best_name = display_name(reference), display_name(best)
        if best == reference:
            decision = f"On `{metric}`, no alternative has a lower mean than the reference (**{ref_name}**)."
            if len(included) > 1:
                decision += f" The MCS nevertheless retains {', '.join(display_name(n) for n in included)}."
        elif pd.isna(p_adj) or p_adj >= 0.05:
            decision = (
                f"On `{metric}`, **{best_name}** has the lowest mean, but its difference from the reference "
                f"**{ref_name}** is not significant (DM p_adjusted={p_adj:.3f})."
            )
        else:
            decision = (
                f"On `{metric}`, **{best_name}** is significantly better than the reference **{ref_name}** "
                f"(DM p_adjusted={p_adj:.3f}); the MCS retains {', '.join(display_name(n) for n in included)}."
            )
        lines += [
            f"### {sec['name']}",
            "",
            _to_markdown(table.round(4)),
            "",
            f"DM significance (BH-adjusted, `{metric}`):",
            "",
            _to_markdown(dm.round(4)),
            "",
            f"Model Confidence Set (alpha=0.05): {', '.join(display_name(n) for n in included) or '(none)'}",
            "",
            f"**Decision.** {decision}",
            "",
        ]
    lines += [
        "### Kupiec coverage test",
        "",
        _to_markdown(kupiec.round(4)) if not kupiec.empty else "Not computed.",
        "",
    ]
    mode = "a" if out_path.exists() else "w"
    with out_path.open(mode) as handle:
        if mode == "w":
            handle.write(f"# {experiment}\n\n")
        handle.write("\n".join(lines) + "\n")


def latest_engine_full(experiment: str, dataset: str) -> pd.DataFrame:
    """Each condition's rows from the most recent ExperimentEngine run of ``experiment`` that covers it on ``dataset``.

    Engine runs live at ``<root>/<experiment>/<spec slug>/runs/<run id>/results/<prefix>_full.csv``; run ids start with
    a timestamp, so later directories are newer. A partial rerun (``--conditions``) therefore replaces only the
    conditions it ran, and the remaining ones come from the latest earlier run.
    """
    frames, seen = [], set()
    for csv in sorted((RESULTS_ROOT / experiment).glob("*/runs/*/results/*_full.csv"), reverse=True):
        full = pd.read_csv(csv)
        if "dataset" not in full.columns:
            continue
        rows = full[(full["dataset"] == dataset) & ~full["condition"].isin(seen)]
        if rows.empty:
            continue
        print(f"[run] {experiment} / {dataset} <- {csv.parent.parent.name}: {sorted(rows['condition'].unique())}")
        seen |= set(rows["condition"])
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_full(experiment: str, dataset: str) -> pd.DataFrame:
    """Read one experiment/dataset's per-window results into the shared result schema.

    Prefers the latest ExperimentEngine run (:func:`latest_engine_full`); falls back to the earlier
    ``<dataset>/results/benchmark_point_full.csv`` layout, renaming ``model`` to ``condition``. Either way ``group``
    is set to ``experiment`` so the common result-analysis functions scope the rows consistently.
    """
    df = latest_engine_full(experiment, dataset)
    if df.empty:
        base = RESULTS_ROOT / experiment
        # Windows-produced runs write lowercase dataset directories (``mlvs-pt``).
        dataset_dir = next((d for d in (dataset, dataset.lower()) if (base / d).exists()), dataset)
        path = base / dataset_dir / "results" / EXPERIMENTS[experiment]["csv"]
        if not path.exists():
            print(f"[missing] {experiment} / {dataset}: no engine run and no {path}")
            return pd.DataFrame()
        df = pd.read_csv(path).rename(columns={"model": "condition"})
        df["dataset"] = dataset
    excluded = EXPERIMENTS[experiment].get("excluded", [])
    if excluded:
        print(f"[excluded] {experiment} / {dataset}: {excluded}")
        df = df[~df["condition"].isin(excluded)]
    df["group"] = experiment
    return df


def representative_from_full(full: pd.DataFrame, dataset: str, group: str) -> pd.DataFrame:
    """Best/worst calendar-day NRMSE per condition, derived from ``full`` directly.

    Unlike Chapter 5's design ablation, Experiments 2/3 have no pre-built
    ``..._representative.csv``. Not a gap needing a rerun: ``full``'s rows
    are already at forecast-origin grain (confirmed against
    ``evaluate_interval_forecast``'s own docstring -- one row per origin,
    horizon-averaged, at whatever cadence the CV origins fall on here, finer
    than Chapter 5's once-daily convention), so grouping by calendar date
    first and taking the mean reduces it to the same one-row-per-day grain
    ``plot_good_bad_day`` expects, matching Chapter 5's
    ``ch5_fpqr_representative.csv`` schema (``condition``, ``dataset``,
    ``group``, ``kind``, ``nrmse``).
    """
    scoped = full[(full["dataset"] == dataset) & (full["group"] == group)].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped["date"] = pd.to_datetime(scoped["timestamp"]).dt.date
    daily = scoped.groupby(["condition", "date"], as_index=False)["nrmse"].mean()

    rows = []
    for condition, condition_rows in daily.groupby("condition"):
        best_nrmse = condition_rows["nrmse"].min()
        worst_nrmse = condition_rows["nrmse"].max()
        base = {"condition": condition, "dataset": dataset, "group": group}
        rows.append({**base, "kind": "best", "nrmse": best_nrmse})
        rows.append({**base, "kind": "worst", "nrmse": worst_nrmse})
    return pd.DataFrame(rows)


def plot_inference_time(full: pd.DataFrame, dataset: str, group: str) -> plt.Figure | None:
    """Inference time only, from the ``inference-time`` column.

    Chapter 5's ``plot_computation_time`` additionally derives approximate
    training time from ``runs/<timestamp>/conditions/.../lightning_logs``
    directory-name timestamps -- Experiments 2/3 here do not use that
    ``runs/<timestamp>`` wrapper (confirmed against the results tree), so
    that half is not reusable as-is. Rather than pass a fabricated
    ``coverage``/``run_dir`` to force a mismatched path through
    ``plot_computation_time``, this is a small, honest standalone
    inference-time-only figure instead.
    """
    group_rows = full[(full["dataset"] == dataset) & (full["group"] == group)]
    if group_rows.empty:
        return None
    inference = group_rows.groupby(["condition", "Folds"])["inference-time"].mean().groupby("condition").mean()
    if inference.empty:
        return None
    order = inference.sort_values().index
    fig, ax = plt.subplots(figsize=(5, 0.4 * len(order) + 1.5))
    ax.barh(range(len(order)), [inference[condition] for condition in order], color=T.INFO)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([S.tex_safe(head_label(condition)) for condition in order])
    ax.set_xlabel("Inference Time (s)")
    S.format_axes(ax)
    fig.tight_layout()
    return fig


def analyse_experiment(full: pd.DataFrame, dataset: str, experiment: str) -> None:
    """Print and save the metric table, DM/MCS significance (MAE + Winkler score), Kupiec test, and figures."""
    cfg = EXPERIMENTS[experiment]
    interval_conditions = cfg["interval_conditions"]

    fold_df = fold_level(full, dataset, experiment, metric_cols=[*CH7_METRIC_COLS, "mae"])
    if fold_df.empty:
        print(f"[skip] {dataset} / {experiment}: no rows")
        return

    print(f"\n=== {dataset} / {experiment} ===")

    # Point-accuracy table + DM/MCS on MAE, across every condition (incl. point-only ones).
    point_table = metric_table(
        fold_df,
        metric_cols=["nrmse", "mae"],
        lower_better={"nrmse", "mae"},
        higher_better=set(),
    )
    print("-- point accuracy --")
    print(point_table.round(4).to_string(index=False))
    point_table.to_csv(OUT_DIR / f"{dataset}_{experiment}_point_metrics.csv", index=False)

    dm_mae, p_matrix_mae, mcs_mae = dm_and_mcs(
        drop_degenerate_conditions(fold_df, "mae", f"{dataset}/{experiment}"), metric="mae"
    )
    print(dm_mae.to_string(index=False))
    print(f"MCS on MAE (alpha=0.05): {mcs_mae.included}")
    sections = [{"name": "Point accuracy", "metric": "mae", "table": point_table, "dm": dm_mae, "mcs": mcs_mae}]
    dm_mae.to_csv(OUT_DIR / f"{dataset}_{experiment}_point_significance.csv", index=False)

    # Probabilistic-quality table + DM/MCS on Winkler score, interval-producing conditions only.
    interval_fold_df = fold_df[fold_df["condition"].isin(interval_conditions)].dropna(subset=["winkler_score"])
    if interval_fold_df["condition"].nunique() >= 2:
        interval_table = metric_table(
            interval_fold_df,
            metric_cols=CH7_METRIC_COLS,
            lower_better=CH7_LOWER_BETTER,
            higher_better=CH7_HIGHER_BETTER,
            target_values={"picp": NOMINAL_PICP},
        )
        print("-- probabilistic quality --")
        print(interval_table.round(4).to_string(index=False))
        interval_table.to_csv(OUT_DIR / f"{dataset}_{experiment}_interval_metrics.csv", index=False)

        dm_ws, p_matrix_ws, mcs_ws = dm_and_mcs(
            drop_degenerate_conditions(interval_fold_df, "winkler_score", f"{dataset}/{experiment}"),
            metric="winkler_score",
        )
        print(dm_ws.to_string(index=False))
        print(f"MCS on Winkler score (alpha=0.05): {mcs_ws.included}")
        dm_ws.to_csv(OUT_DIR / f"{dataset}_{experiment}_interval_significance.csv", index=False)
        sections.append(
            {
                "name": "Probabilistic quality",
                "metric": "winkler_score",
                "table": interval_table,
                "dm": dm_ws,
                "mcs": mcs_ws,
            }
        )

        cwe_gap = cwe_comparison(interval_fold_df, group_cols=["condition"])
        print("-- CWE marginal vs. stratified --")
        print(cwe_gap.round(4).to_string())
        cwe_gap.to_csv(OUT_DIR / f"{dataset}_{experiment}_cwe_comparison.csv")
    else:
        p_matrix_ws = None

    # Point-only conditions (mlpf_point/mlpgam_point in Experiment 2) have no
    # PICP at all -- restrict to the interval-producing conditions so
    # kupiec_table's fold_size/exception-count arithmetic never divides a
    # NaN PICP.
    full_interval_only = full[full["condition"].isin(interval_conditions)]
    # Engine runs write one row per metric type per window; count each window once (its interval row).
    if "metric_type" in full_interval_only.columns and (full_interval_only["metric_type"] == "interval").any():
        full_interval_only = full_interval_only[full_interval_only["metric_type"] == "interval"]
    kupiec = kupiec_table(full_interval_only, dataset, experiment)
    if not kupiec.empty:
        print(f"\nKupiec POF (unconditional coverage, alpha={1 - NOMINAL_PICP:.2f}):")
        print(kupiec.round(4).to_string(index=False))
        kupiec.to_csv(OUT_DIR / f"{dataset}_{experiment}_kupiec.csv", index=False)
    conclusion = OUT_DIR / f"{experiment}_conclusion.md"
    write_conclusion(
        conclusion, dataset=dataset, experiment=experiment, reference=cfg["reference"], sections=sections, kupiec=kupiec
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reference = cfg["reference"]

    plot_metrics = [*CH7_METRIC_COLS, "mae"]
    fig = plot_forest_grid(
        fold_df,
        dataset,
        experiment,
        reference,
        plot_metrics,
        backbone_prefixes=("mlpgam_", "mlpf_"),
    )
    fig.savefig(FIG_DIR / f"{dataset}_{experiment}_forest.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metric_figures = plot_metric_bars(
        fold_df,
        dataset,
        experiment,
        reference,
        plot_metrics,
        lower_better=CH7_LOWER_BETTER | {"mae"},
        higher_better=CH7_HIGHER_BETTER,
        target_values={"picp": NOMINAL_PICP},
        backbone_prefixes=("mlpgam_", "mlpf_"),
    )
    save_metric_figures(metric_figures, FIG_DIR, f"{dataset}_{experiment}", kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        dataset,
        experiment,
        reference,
        plot_metrics,
        lower_better=CH7_LOWER_BETTER | {"mae"},
        higher_better=CH7_HIGHER_BETTER,
        target_values={"picp": NOMINAL_PICP},
        backbone_prefixes=("mlpgam_", "mlpf_"),
    )
    if metric_grid is not None:
        metric_grid.savefig(FIG_DIR / f"{dataset}_{experiment}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    if p_matrix_ws is not None:
        fig = plot_dm_significance_matrix(p_matrix_ws, dataset, experiment)
        if fig is not None:
            fig.savefig(FIG_DIR / f"{dataset}_{experiment}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

    fig = plot_inference_time(full, dataset, experiment)
    if fig is not None:
        fig.savefig(FIG_DIR / f"{dataset}_{experiment}_inference_time.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    representative = representative_from_full(full, dataset, experiment)
    fig = plot_good_bad_day(representative, dataset, experiment)
    if fig is not None:
        fig.savefig(FIG_DIR / f"{dataset}_{experiment}_good_bad_day.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  figures written under {FIG_DIR}")


def plot_fold_coverage(
    fold_df: pd.DataFrame, full: pd.DataFrame, conditions: list[str], dataset: str, out: Path
) -> None:
    """Per-fold coverage (PICP) of each head as a heatmap on a linear [0, 1] scale, annotated with the PICP.

    Rows are heads, columns folds labelled by the start of their test window, and the colour is the PICP itself on a
    linear [0, 1] scale whose colour turns at the nominal 0.90 (marked on the colour bar): under-coverage (the
    failure the conformal guarantee guards against) shades towards the danger colour, over-coverage towards the
    primary colour, so the seasonal shortfall and each head's systematic bias read at a glance. Full text width.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: PLC0415
    from matplotlib.ticker import MultipleLocator  # noqa: PLC0415

    from chapters.ch6_conformal.shared.ch7_architectural_analysis import _tex_label  # noqa: PLC0415

    starts = pd.to_datetime(full[full["dataset"] == dataset].groupby("Folds")["timestamp"].min())
    conditions = sorted(conditions, key=lambda c: c != "mlpgam_crc")  # the proposed head first
    wide = fold_df.pivot(index="condition", columns="Folds", values="picp").reindex(conditions)
    # Linear [0, 1] PICP scale (even ticks 0, 0.2, ..., 1) whose colour turns at the nominal level: white at 0.90,
    # blue above it, orange below, saturating at 0.5 and below so the under-coverage that matters stays legible.
    cmap = LinearSegmentedColormap.from_list(
        "coverage", [(0.0, T.DANGER), (0.5, T.DANGER), (NOMINAL_PICP, "#FFFFFF"), (1.0, T.PRIMARY)]
    )
    fig, ax = plt.subplots(figsize=S.panel(1.0, aspect=0.30))
    image = ax.imshow(wide.to_numpy(), cmap=cmap, norm=Normalize(0.0, 1.0), aspect="auto", interpolation="none")
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            value = wide.iat[i, j]
            if np.isfinite(value):
                color = "white" if value < 0.65 or value > 0.985 else T.TEXT_DARK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([_tex_label(c) for c in conditions])
    ax.set_xticks(range(wide.shape[1]))
    ax.set_xticklabels([starts[f].strftime("%b %Y") for f in wide.columns], rotation=30, ha="right")
    ax.set_xlabel("Start of the four-month test window")
    ax.minorticks_off()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    bar.set_label(r"$\mathrm{PICP}$")
    bar.ax.axhline(NOMINAL_PICP, color=T.TEXT_DARK, lw=0.6)
    bar.set_ticks([0.0, 0.5, 1.0])  # labelled major ticks
    bar.ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # unlabelled minor ticks every 0.1
    bar.ax.tick_params(which="minor", length=2, width=0.5)
    bar.outline.set_visible(False)
    fig.tight_layout(pad=0.3)
    slug = {"MLVS-PT": "pt", "MLVS-UK": "uk"}.get(dataset, dataset.lower())
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fold_coverage_{slug}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def render_single_figures(experiment: str, datasets: list[str], copy_to_images: bool = False) -> None:
    """Single-plot thesis figures for ``experiment``, in the style of the architectural evaluation's figures.

    Chapter 5-style cross-dataset bars per metric (error metrics as the paired relative score), Chapter 6-style
    significance matrices on the decisive metrics, and per-fold coverage. Written to
    ``<analysis>/twiga_thesis.plotting/<experiment>``; ``copy_to_images`` copies the PDFs to the thesis as
    ``fig_ch7_<short>_<name>.pdf`` (``short`` = ``bench`` for the probabilistic benchmark).
    """
    import shutil  # noqa: PLC0415

    from chapters.ch6_conformal.shared import ch7_architectural_analysis as arch  # noqa: PLC0415

    conditions = EXPERIMENTS[experiment]["interval_conditions"]
    out = OUT_DIR / "twiga_thesis.plotting" / experiment
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for dataset in datasets:
        full = load_full(experiment, dataset)
        if full.empty:
            continue
        fold_df = fold_level(full, dataset, experiment, metric_cols=arch.METRICS)
        fold_df = fold_df[fold_df["condition"].isin(conditions)]
        frames[dataset] = fold_df
        for metric, name in (("winkler_score", "ws"), ("mae", "mae")):
            _, p_matrix, _ = dm_and_mcs(fold_df.dropna(subset=[metric]), metric=metric)
            arch.plot_significance(p_matrix.loc[conditions, conditions], dataset, f"{name}", out)
        plot_fold_coverage(fold_df, full, conditions, dataset, out)
    if frames:
        arch.plot_dataset_comparison(frames, conditions, out, "metric")
    if copy_to_images:
        short = {"probabilistic_benchmark": "bench"}.get(experiment, experiment)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in out.glob("*.pdf"):
            shutil.copy(pdf, IMAGES_DIR / f"fig_ch7_{short}_{pdf.name}")
    print(f"thesis figures written under {out}")


def run_analysis(
    experiments: list[str],
    datasets: list[str],
    results_root: Path = DEFAULT_OUTPUT_ROOT,
    winsorize: bool = False,
    latex: bool = True,
    thesis: bool = False,
    copy_to_images: bool = False,
) -> None:
    """Analyse ``experiments`` on ``datasets`` read from ``results_root``; write to ``<results_root>/analysis``."""
    global RESULTS_ROOT, OUT_DIR, FIG_DIR
    RESULTS_ROOT = Path(results_root).resolve()
    OUT_DIR = RESULTS_ROOT / "analysis"
    FIG_DIR = OUT_DIR / "figures"

    S.configure(latex=latex)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:
        full = pd.concat([load_full(experiment, dataset) for dataset in datasets], ignore_index=True)
        if winsorize and not full.empty:
            full = winsorize_nrmse(full)
        if full.empty:
            print(f"[skip] {experiment}: nothing to analyse")
            continue

        (OUT_DIR / f"{experiment}_conclusion.md").unlink(missing_ok=True)
        for dataset in datasets:
            analyse_experiment(full, dataset, experiment)
        if thesis:
            render_single_figures(experiment, datasets, copy_to_images=copy_to_images)

        # Cross-dataset comparison figures, interval-producing conditions only.
        interval_conditions = EXPERIMENTS[experiment]["interval_conditions"]
        interval_full = full[full["condition"].isin(interval_conditions)]
        if set(datasets) == set(DATASETS) and not interval_full.empty:
            fig = plot_primary_metric_comparison(interval_full, experiment, primary_metric_cols=CH7_PRIMARY_METRIC_COLS)
            if fig is not None:
                out = FIG_DIR / f"{experiment}_primary_metrics.pdf"
                fig.savefig(out, dpi=480, bbox_inches="tight")
                plt.close(fig)
                print(f"wrote {out}")

            fig = plot_normalized_comparison(
                interval_full, experiment, metrics=[("winkler_score", True), ("cwe_stratified", False)]
            )
            if fig is not None:
                out = FIG_DIR / f"{experiment}_normalised_comparison.pdf"
                fig.savefig(out, dpi=480, bbox_inches="tight")
                plt.close(fig)
                print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", default=["all"], choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--dataset", nargs="+", default=["all"], choices=[*DATASETS, "all"])
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory holding <experiment>/<dataset>/results; analysis is written to <root>/analysis",
    )
    parser.add_argument(
        "--winsorize-nrmse",
        action="store_true",
        help="clip NRMSE at its 97th percentile per condition before aggregating (ported from Chapter 6)",
    )
    parser.add_argument(
        "--no-latex", action="store_true", help="fall back to matplotlib's own text renderer (no LaTeX toolchain)"
    )
    parser.add_argument("--thesis-figures", action="store_true", help="also write single-plot thesis figures")
    parser.add_argument("--copy-to-images", action="store_true", help="copy the thesis figures into the thesis")
    args = parser.parse_args()
    run_analysis(
        list(EXPERIMENTS) if "all" in args.experiment else args.experiment,
        DATASETS if "all" in args.dataset else args.dataset,
        results_root=args.results_root,
        winsorize=args.winsorize_nrmse,
        latex=not args.no_latex,
        thesis=args.twiga_thesis.plotting,
        copy_to_images=args.copy_to_images,
    )


if __name__ == "__main__":
    main()
