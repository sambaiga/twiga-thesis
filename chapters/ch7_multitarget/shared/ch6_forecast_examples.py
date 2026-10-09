r"""Good/bad-day forecast panels for Chapter 6's two foundational ablations.

Two halves, mirroring ``ch6_conformal/mlpgam_forecast_examples.py`` and
``chapter-5-fpqr/qvn_forecast_examples.py``'s good/bad-day pattern
(``GROUP_CONDITIONS``, ``predictions_path``, ``load``, ``pick_days``,
``day_frame``/``day_metric``, one panel via ``experiment/common/forecast_panels.py``'s
``draw``) rather than reimplementing it a second time:

- **fpqr_gating_ablation** (parquet-based): ``fpqr_gating_ablation.py``'s
  engine writes exactly ``predictions.parquet``/``metrics.parquet`` per
  condition, split by ``metric_type`` the same way
  ``chapter-5-fpqr/qvn_forecast_examples.py``'s does (no condition here sets
  a ``ConformalConfig``, so this run's engine output follows that split, not
  ``mlpgam_forecast_examples.py``'s flat one). Unlike either precedent, every
  row here also carries a ``target`` column (``LP``/``LQ``/``LV``, Chapter 6
  is multi-target) -- panels are drawn for one target at a time
  (``--target``, default ``LP``, the active-power series). Verified directly
  against the on-disk parquet: ``metric_type == "quantile"`` rows carry
  ``lower``/``upper`` as ``NaN`` (no per-tau fan saved), so panels fall back
  to the same degenerate 3-level fan (``[0.05, 0.5, 0.95]`` stacked from
  ``lower``/``forecast``/``upper``) ``mlpgam_forecast_examples.py`` already
  uses for its own single-interval conditions.

- **copula_variants** (CSV-based): ``copula_variants.py`` never wrote a
  ``predictions.parquet`` (see its own module docstring: checkpoints go to a
  throwaway ``tempfile.mkdtemp()`` directory, and only the scored CSVs
  survive). ``pick_copula_days`` reads the real per-forecast-day losses from
  ``copula_losses.csv`` to report which (fold, day) the *formal* backtest
  found best/worst, for reference. The panels themselves reuse
  ``forecast_examples.py``'s own ``predictions.npz`` prediction cache instead
  -- confirmed to hold every ``copula_variants.VARIANTS`` name under the same
  strings (``"fpqr shared grid"``, ``"fpqr shared + copula"``, etc.) via its
  ``model_names`` array. **This is a different fit from ``copula_variants.py``'s
  own formal backtest** (``forecast_examples.py``'s own default scenario/seed/
  CV protocol, not ``copula_variants.py``'s rolling-weeks backtest folds), so
  these panels illustrate the method -- what a shared vs. copula-coupled
  quantile fan actually looks like on a day of data -- rather than reproduce
  ``copula_per_fold.csv``'s exact numbers. If the cache is missing, or does
  not cover the requested conditions, panel drawing is skipped with a clear
  printed message rather than crashing or retraining.

Usage::

    uv run python experiment/ch7_multitarget/ch6_forecast_examples.py \
        --experiment fpqr_gating_ablation --group grid
    uv run python experiment/ch7_multitarget/ch6_forecast_examples.py \
        --experiment multitarget_experiment --group benchmark
    uv run python experiment/ch7_multitarget/ch6_forecast_examples.py \
        --experiment copula_variants --dataset dss-eu_default_0_4_0_4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twiga_thesis.plotting.comparison_plots import display_name  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT = script_dir.parent / "results" / "analysis" / "forecast_examples"

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent / "experiment_2_backbone_conformal"))
sys.path.insert(0, str(script_dir.parent / "experiment_1_benchmark"))
import chapters.ch7_multitarget.experiment_1_benchmark.multitarget_experiment as mte  # noqa: E402
import chapters.ch7_multitarget.experiment_2_backbone_conformal.backbone_conformal_study as bcs  # noqa: E402
import chapters.ch7_multitarget.experiment_2_backbone_conformal.fpqr_gating_ablation as fga  # noqa: E402
from chapters.ch7_multitarget.shared.forecast_examples import build_windows  # noqa: E402

sys.path.insert(0, str(script_dir.parent.parent))
from twiga_thesis.plotting.style import configure, pad_limits, panel, tex_safe  # noqa: E402

sys.path.insert(0, str(script_dir.parent.parent / "common"))
from twiga_thesis.plotting.forecast_panels import draw  # noqa: E402

_FAN_LEVELS = np.array([0.05, 0.5, 0.95])

# ── FPQR gating ablation configuration ────────────────────────────────────────
# Derived from fpqr_gating_ablation.CONDITIONS rather than retyped, so a
# future condition added/renamed there is picked up automatically.
FPQR_GROUP_CONDITIONS: dict[str, list[str]] = {}
for _condition in fga.CONDITIONS:
    FPQR_GROUP_CONDITIONS.setdefault(_condition.group, []).append(_condition.name)

# ── Backbone conformal study configuration ────────────────────────────────────
# Derived from backbone_conformal_study.CONDITIONS the same way, one group
# ("backbone", every condition name "{backbone}_{conformal}").
BACKBONE_CONFORMAL_CONDITIONS: list[str] = [c.name for c in bcs.CONDITIONS]

# ── Multitarget experiment configuration ──────────────────────────────────────
# Derived from multitarget_experiment.CONDITIONS the same way as
# FPQR_GROUP_CONDITIONS -- "benchmark"/"grid"/"conformal" groups.
MULTITARGET_GROUP_CONDITIONS: dict[str, list[str]] = {}
for _condition in mte.CONDITIONS:
    MULTITARGET_GROUP_CONDITIONS.setdefault(_condition.group, []).append(_condition.name)


def predictions_path(run_dir: Path, dataset: str, group: str, condition: str) -> Path:
    """Path to one condition's saved ``predictions.parquet``, matching the engine's own layout."""
    return run_dir / "conditions" / dataset.lower() / group / condition / "predictions.parquet"


def load(
    run_dir: Path, dataset: str, group: str, conditions: list[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Saved predictions/metrics per condition, and the conditions with no completed run.

    Args:
        run_dir: The run directory to read from (see :func:`fpqr_run_dir`).
        dataset: Dataset name (e.g. ``"MLVS-PT-3PHASE"``).
        group: Condition group name.
        conditions: Condition names to load.

    Returns:
        ``(predictions, metrics, missing)`` -- per-condition parquet frames,
        and the subset of ``conditions`` with no completed backtest.
    """
    preds, metrics, missing = {}, {}, []
    for condition in conditions:
        path = predictions_path(run_dir, dataset, group, condition)
        if not path.exists():
            missing.append(condition)
            continue
        preds[condition] = pd.read_parquet(path)
        metrics[condition] = pd.read_parquet(path.parent / "metrics.parquet")
    return preds, metrics, missing


def fpqr_run_dir(group: str, run_dir_override: Path | None) -> Path | None:
    """The run directory to read ``group``'s predictions from: ``--run-dir`` if given, else the latest."""
    if run_dir_override is not None:
        return run_dir_override
    run = fga.latest_run_for_group(fga.discover_runs(), group)
    return run["run_dir"] if run is not None else None


# Day-selection metrics available on the fpqr_gating_ablation/multitarget_experiment
# schema, keyed by the CLI's --day-metric name: (metric_type row to read it
# from, column name). "nrmse" (point accuracy) is the original/default
# choice; "winkler"/"crps" pick the day by probabilistic sharpness+coverage
# instead, which can and does disagree with the NRMSE-picked day for a
# poorly-calibrated condition (see module docstring).
DAY_METRICS: dict[str, tuple[str, str]] = {
    "nrmse": ("point", "nrmse"),
    "winkler": ("interval", "winkler_score"),
    "crps": ("quantile", "crps"),
}


def _metric_type_mask(df: pd.DataFrame, metric_type: str) -> pd.Series:
    """``df["metric_type"] == metric_type``, or all-``True`` when the column is absent entirely.

    A conformal-wrapped condition's ``predictions.parquet``/``metrics.parquet``
    (e.g. ``laplace_conformal``/``bnn_conformal``, ``mlpf_quantile``) has no
    ``metric_type`` column at all -- every metric/point/interval value
    already lives on one flat row per timestamp (see
    ``ch6_analyse_results.py``'s ``_merge_point_interval`` docstring for the
    same shape distinction) -- so filtering by an exact ``metric_type``
    match would raise a ``KeyError`` rather than simply matching every row,
    which is what a flat-row frame's own single row per timestamp already
    represents.
    """
    if "metric_type" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["metric_type"] == metric_type


def pick_days(
    metrics: dict[str, pd.DataFrame], target: str, day_metric_name: str = "nrmse"
) -> tuple[tuple[pd.Timestamp, int], tuple[pd.Timestamp, int]]:
    """Shared good/bad (timestamp, fold) key for one target: mean per-day metric pooled across conditions.

    Args:
        metrics: Per-condition ``metrics.parquet`` frames.
        target: Target to score (``"LP"``, ``"LQ"``, or ``"LV"``).
        day_metric_name: Which column decides "good"/"bad", one of
            ``DAY_METRICS``'s keys -- ``"nrmse"`` (point accuracy, the
            original default) or a probabilistic metric (``"winkler"``,
            ``"crps"``). A day picked as "good" by NRMSE is not
            necessarily "good" by Winkler/CRPS for a condition whose
            intervals are miscalibrated even when its median is accurate
            (see the ``head_joint_parametric`` case in the module
            docstring), so the two selections can and do disagree.

    Only rows of the metric_type that ``day_metric_name`` maps to are used,
    and only ``(timestamp, Folds)`` pairs where every requested condition has
    a *real, non-NaN* value are eligible. NaN rows are dropped before the
    eligibility check (not just skipped inside the mean), otherwise a
    condition that failed to predict on a given day -- e.g. a diverged
    training run -- would silently drop out of the pooled average instead
    of disqualifying that day, and a "shared" day could end up with no
    real value for that condition at all.
    """
    metric_type, column = DAY_METRICS[day_metric_name]
    frames = []
    for name, m in metrics.items():
        rows = m[_metric_type_mask(m, metric_type) & (m["target"] == target)][["timestamp", "Folds", column]].copy()
        rows["condition"] = name
        frames.append(rows.dropna(subset=[column]))
    pooled = pd.concat(frames, ignore_index=True)
    grouped = pooled.groupby(["timestamp", "Folds"])[column]
    day_err = grouped.mean()
    complete = day_err[grouped.size() == len(metrics)]
    if complete.empty:
        raise RuntimeError(
            f"no (timestamp, fold) pair for target={target!r} has a real {day_metric_name} value "
            "for every requested condition"
        )
    return complete.idxmin(), complete.idxmax()


def day_frame(preds: pd.DataFrame, timestamp: pd.Timestamp, fold: int, target: str) -> pd.DataFrame:
    """One condition's interval-typed rows for one backtest day/target, sorted for plotting.

    ``timestamp`` is normalised here too, not just ``preds["timestamp"]``:
    ``pick_days`` returns a metrics-frame timestamp (that frame's own
    per-day row key, not necessarily midnight -- found 2026-09-26 to
    sometimes carry a mid-day time, e.g. ``13:00``), so comparing it
    unnormalised against ``preds``'s normalised timestamp silently matched
    zero rows every time.
    """
    mask = (
        (preds["Folds"] == fold)
        & _metric_type_mask(preds, "interval")
        & (preds["target"] == target)
        & (preds["timestamp"].dt.normalize() == pd.Timestamp(timestamp).normalize())
    )
    return preds[mask].sort_values("timestamp")


def day_metric(
    metrics: pd.DataFrame, timestamp: pd.Timestamp, fold: int, target: str, metric_type: str, column: str
) -> float:
    """One scalar metric value for one backtest day/target/metric_type, or NaN if absent."""
    mask = (
        (metrics["Folds"] == fold)
        & _metric_type_mask(metrics, metric_type)
        & (metrics["target"] == target)
        & (metrics["timestamp"].dt.normalize() == pd.Timestamp(timestamp).normalize())
    )
    day = metrics[mask]
    return float(day[column].iloc[0]) if not day.empty else float("nan")


def draw_fpqr_panel(ax, day: pd.DataFrame, label: str, small: bool = False) -> None:
    """One condition's forecast panel: the degenerate 3-level fan from lower/forecast/upper.

    No in-figure title (a caption, or ``forecast_examples_summary.csv``,
    carries the condition name and its NRMSE/Winkler/PICP) -- only a compact
    in-axes corner label identifying which condition/day this panel is,
    matching the thesis's own no-banner-title convention (see
    :func:`~significance_plots.plot_dm_significance_matrix`).

    ``small=True`` is for a sub-panel inside the 3-column overview grid
    (~2.1in wide), which needs tighter fonts than the thesis's own
    ``configure()``/``THESIS_RC`` defaults to avoid collisions, so it
    overrides them explicitly. The default (``False``) is for a standalone
    full-textwidth (6.3in) export and takes no explicit fontsize at all,
    inheriting ``style.py``'s own ``configure()`` rcParams (``font.size``/
    ``axes.labelsize``/``xtick.labelsize``/``ytick.labelsize``, all 10pt)
    the same way every other panel in this module already does -- the
    previous hardcoded 7/8pt override (reused unchanged from the overview
    grid) is what made ``good_<condition>.pdf`` read as undersized at
    publication size.
    """
    t_axis = day["timestamp"].to_numpy()
    q = np.stack([day["lower"].to_numpy(), day["forecast"].to_numpy(), day["upper"].to_numpy()])
    y = day["Actual"].to_numpy()
    ylim = pad_limits(y, q)
    draw(ax, t_axis, y, q, _FAN_LEVELS, t_i=0, ylim=ylim)
    if small:
        ax.text(0.02, 0.96, tex_safe(label), transform=ax.transAxes, va="top", ha="left", fontsize=7)
        ax.set_ylabel("kW / kVAr / p.u.", fontsize=8)
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.02, 0.96, tex_safe(label), transform=ax.transAxes, va="top", ha="left")
        ax.set_ylabel("kW / kVAr / p.u.")


def run_fpqr_forecast_examples(
    group: str,
    dataset: str,
    conditions: list[str],
    target: str,
    run_dir_override: Path | None,
    latex: bool,
    day_metric_name: str = "nrmse",
) -> None:
    """Good/bad-day overview + per-condition panels for one FPQR group/dataset/target."""
    run_dir = fpqr_run_dir(group, run_dir_override)
    if run_dir is None:
        print(f"no synced run covers group={group}; skipping")
        return

    out_dir = OUT / "fpqr_gating_ablation" / group / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, metrics, missing = load(run_dir, dataset, group, conditions)
    if missing:
        print(f"no completed backtest for: {', '.join(missing)} (skipped, not silently dropped)")
    if not preds:
        print(f"no completed {group} backtest found for {dataset} under {run_dir}; skipping")
        return

    (good_ts, good_fold), (bad_ts, bad_fold) = pick_days(metrics, target, day_metric_name)
    print(f"day-selection metric: {day_metric_name}")
    print(f"{len(preds)} conditions with a completed backtest: {', '.join(preds)}")
    print(f"good day  {good_ts:%Y-%m-%d}  fold {good_fold}  (target {target})")
    print(f"bad day   {bad_ts:%Y-%m-%d}  fold {bad_fold}  (target {target})")

    configure(latex=latex)
    rows = []
    for label, ts, fold in (("good", good_ts, good_fold), ("bad", bad_ts, bad_fold)):
        present = [c for c in conditions if c in preds]
        ncol = min(len(present), 3)
        nrow = -(-len(present) // ncol) if ncol else 0
        pw, ph = panel(1.0 / max(ncol, 1), aspect=0.75)
        fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, ph * nrow), squeeze=False)
        axes_flat = axes.flatten()

        for ax, condition in zip(axes_flat, present, strict=False):
            day = day_frame(preds[condition], ts, fold, target)
            m = metrics[condition]
            nrmse = day_metric(m, ts, fold, target, "point", "nrmse")
            winkler = day_metric(m, ts, fold, target, "interval", "winkler_score")
            picp = day_metric(m, ts, fold, target, "interval", "picp")
            rows.append(
                {
                    "day": label,
                    "condition": condition,
                    "target": target,
                    "NRMSE": nrmse,
                    "Winkler": winkler,
                    "PICP": picp,
                }
            )

            panel_label = f"{display_name(condition)} ({label.capitalize()} Day, {ts:%Y-%m-%d})"
            draw_fpqr_panel(ax, day, panel_label, small=True)

            fig_single, ax_single = plt.subplots(figsize=panel(1, aspect=0.7))
            draw_fpqr_panel(ax_single, day, display_name(condition))
            fig_single.savefig(out_dir / f"{label}_{condition}.pdf", dpi=600, bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes_flat[len(present) :]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"overview_{label}_day.pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "forecast_examples_summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}")


# ── backbone_conformal_study half ─────────────────────────────────────────────
# Engine/parquet-based, exactly like the fpqr_gating_ablation half above
# (backbone_conformal_study.py runs through the same ExperimentEngine, with
# save_predictions=True the same way) -- reuses predictions_path/load/
# pick_days/day_frame/day_metric/draw_fpqr_panel unchanged. The one real
# difference: fga's discover_runs/latest_run_for_group select by *group*
# (fpqr_gating_ablation.py runs several groups per dataset in one engine
# call), but backbone_conformal_study.py has exactly one group ("backbone")
# and a separate ExperimentEngine call per dataset instead, so its run must
# be selected by *dataset* -- latest_run_for_group's own group filter would
# otherwise match every dataset's run interchangeably.


def backbone_conformal_run_dir(dataset: str, run_dir_override: Path | None) -> Path | None:
    """The run directory to read ``dataset``'s backbone-conformal-study predictions from."""
    if run_dir_override is not None:
        return run_dir_override
    candidates = [run for run in bcs.discover_runs() if dataset in run["datasets"]]
    if not candidates:
        return None
    return max(candidates, key=lambda run: run["timestamp"])["run_dir"]


def run_backbone_conformal_forecast_examples(
    dataset: str,
    conditions: list[str],
    target: str,
    run_dir_override: Path | None,
    latex: bool,
    day_metric_name: str = "nrmse",
) -> None:
    """Good/bad-day overview + per-condition panels for one backbone-conformal-study dataset/target."""
    group = "backbone"  # backbone_conformal_study.py's own one and only Condition group
    run_dir = backbone_conformal_run_dir(dataset, run_dir_override)
    if run_dir is None:
        print(f"no synced run covers dataset={dataset}; skipping")
        return

    out_dir = OUT / "backbone_conformal_study" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, metrics, missing = load(run_dir, dataset, group, conditions)
    if missing:
        print(f"no completed backtest for: {', '.join(missing)} (skipped, not silently dropped)")
    if not preds:
        print(f"no completed backbone-conformal-study backtest found for {dataset} under {run_dir}; skipping")
        return

    (good_ts, good_fold), (bad_ts, bad_fold) = pick_days(metrics, target, day_metric_name)
    print(f"day-selection metric: {day_metric_name}")
    print(f"{len(preds)} conditions with a completed backtest: {', '.join(preds)}")
    print(f"good day  {good_ts:%Y-%m-%d}  fold {good_fold}  (target {target})")
    print(f"bad day   {bad_ts:%Y-%m-%d}  fold {bad_fold}  (target {target})")

    configure(latex=latex)
    rows = []
    for label, ts, fold in (("good", good_ts, good_fold), ("bad", bad_ts, bad_fold)):
        present = [c for c in conditions if c in preds]
        ncol = min(len(present), 3)
        nrow = -(-len(present) // ncol) if ncol else 0
        pw, ph = panel(1.0 / max(ncol, 1), aspect=0.75)
        fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, ph * nrow), squeeze=False)
        axes_flat = axes.flatten()

        for ax, condition in zip(axes_flat, present, strict=False):
            day = day_frame(preds[condition], ts, fold, target)
            m = metrics[condition]
            nrmse = day_metric(m, ts, fold, target, "point", "nrmse")
            winkler = day_metric(m, ts, fold, target, "interval", "winkler_score")
            picp = day_metric(m, ts, fold, target, "interval", "picp")
            rows.append(
                {
                    "day": label,
                    "condition": condition,
                    "target": target,
                    "NRMSE": nrmse,
                    "Winkler": winkler,
                    "PICP": picp,
                }
            )

            panel_label = f"{display_name(condition)} ({label.capitalize()} Day, {ts:%Y-%m-%d})"
            draw_fpqr_panel(ax, day, panel_label, small=True)

            fig_single, ax_single = plt.subplots(figsize=panel(1, aspect=0.7))
            draw_fpqr_panel(ax_single, day, display_name(condition))
            fig_single.savefig(out_dir / f"{label}_{condition}.pdf", dpi=600, bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes_flat[len(present) :]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"overview_{label}_day.pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "forecast_examples_summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}")


# ── multitarget_experiment half ───────────────────────────────────────────────
# Engine/parquet-based, exactly like the backbone_conformal_study half above
# (multitarget_experiment.py runs through the same ExperimentEngine, with
# save_predictions=True the same way) -- reuses predictions_path/load/
# pick_days/day_frame/day_metric/draw_fpqr_panel unchanged. Selected by
# *group* (multitarget_experiment.py runs "benchmark"/"grid"/"conformal" as
# separate Condition groups within one engine call per dataset, mirroring
# fpqr_gating_ablation.py's own convention, not backbone_conformal_study.py's
# one-group-per-dataset one), read from its own results tree rather than
# ch6_analyse_results.py's MULTITARGET_ROOT constant to avoid importing that
# module here.
MULTITARGET_RESULTS_ROOT = (
    script_dir.parent / "results" / "ch7_multitarget" / "multitarget_experiment" / "multitarget_forecasting_benchmark"
)


def multitarget_run_dir(group: str, run_dir_override: Path | None) -> Path | None:
    """The run directory to read ``group``'s multitarget-experiment predictions from."""
    if run_dir_override is not None:
        return run_dir_override
    candidates = [
        (json.loads(meta_path.read_text()), meta_path.parent)
        for meta_path in MULTITARGET_RESULTS_ROOT.glob("runs/*/meta.json")
    ]
    candidates = [(meta, run_dir) for meta, run_dir in candidates if group in meta["groups"]]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0]["timestamp"])[1]


def run_multitarget_forecast_examples(
    group: str,
    dataset: str,
    conditions: list[str],
    target: str,
    run_dir_override: Path | None,
    latex: bool,
    day_metric_name: str = "nrmse",
) -> None:
    """Good/bad-day overview + per-condition panels for one multitarget-experiment group/dataset/target."""
    run_dir = multitarget_run_dir(group, run_dir_override)
    if run_dir is None:
        print(f"no synced run covers group={group}; skipping")
        return

    out_dir = OUT / "multitarget_experiment" / group / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, metrics, missing = load(run_dir, dataset, group, conditions)
    if missing:
        print(f"no completed backtest for: {', '.join(missing)} (skipped, not silently dropped)")
    if not preds:
        print(f"no completed {group} backtest found for {dataset} under {run_dir}; skipping")
        return

    (good_ts, good_fold), (bad_ts, bad_fold) = pick_days(metrics, target, day_metric_name)
    print(f"day-selection metric: {day_metric_name}")
    print(f"{len(preds)} conditions with a completed backtest: {', '.join(preds)}")
    print(f"good day  {good_ts:%Y-%m-%d}  fold {good_fold}  (target {target})")
    print(f"bad day   {bad_ts:%Y-%m-%d}  fold {bad_fold}  (target {target})")

    configure(latex=latex)
    rows = []
    for label, ts, fold in (("good", good_ts, good_fold), ("bad", bad_ts, bad_fold)):
        present = [c for c in conditions if c in preds]
        ncol = min(len(present), 3)
        nrow = -(-len(present) // ncol) if ncol else 0
        pw, ph = panel(1.0 / max(ncol, 1), aspect=0.75)
        fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, ph * nrow), squeeze=False)
        axes_flat = axes.flatten()

        for ax, condition in zip(axes_flat, present, strict=False):
            day = day_frame(preds[condition], ts, fold, target)
            m = metrics[condition]
            nrmse = day_metric(m, ts, fold, target, "point", "nrmse")
            winkler = day_metric(m, ts, fold, target, "interval", "winkler_score")
            picp = day_metric(m, ts, fold, target, "interval", "picp")
            rows.append(
                {
                    "day": label,
                    "condition": condition,
                    "target": target,
                    "NRMSE": nrmse,
                    "Winkler": winkler,
                    "PICP": picp,
                }
            )

            panel_label = f"{display_name(condition)} ({label.capitalize()} Day, {ts:%Y-%m-%d})"
            draw_fpqr_panel(ax, day, panel_label, small=True)

            fig_single, ax_single = plt.subplots(figsize=panel(1, aspect=0.7))
            draw_fpqr_panel(ax_single, day, display_name(condition))
            fig_single.savefig(out_dir / f"{label}_{condition}.pdf", dpi=600, bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes_flat[len(present) :]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"overview_{label}_day.pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "forecast_examples_summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}")


# ── copula_variants half ──────────────────────────────────────────────────────

COPULA_RESULTS_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "copula_variants"

# forecast_examples.py's own predictions.npz cache -- a DIFFERENT fit from
# copula_variants.py's formal backtest (see module docstring). Assumed to
# have been built with forecast_examples.py's own defaults (--scenario
# "DSS-EU:default:0.4:0.4" --lookback 48 --horizon 48 --backbone mlpf), since
# the cache carries no scenario/lookback/horizon metadata of its own to
# confirm this from -- true of the cache actually on disk at the time this
# script was written (OUT_ROOT, not a --backbone subdirectory).
FORECAST_EXAMPLES_CACHE = script_dir.parent / "forecast_examples" / "predictions.npz"
CACHE_SCENARIO = "DSS-EU:default:0.4:0.4"
CACHE_LOOKBACK = 48
CACHE_HORIZON = 48


def _latest_copula_run_dir(dataset_slug: str) -> Path | None:
    """The most recently timestamped run directory for one copula dataset slug."""
    import json

    metas = []
    for meta_path in (COPULA_RESULTS_ROOT / dataset_slug).glob("runs/*/meta.json"):
        meta = json.loads(meta_path.read_text())
        metas.append((meta, meta_path.parent))
    if not metas:
        return None
    return max(metas, key=lambda pair: pair[0]["timestamp"])[1]


def pick_copula_days(
    losses_df: pd.DataFrame, conditions: list[str], metric: str = "CRPS"
) -> tuple[tuple[str, int], tuple[str, int]]:
    """Best/worst (fold, day) from the real backtest's per-forecast losses, for reference only.

    Args:
        losses_df: ``copula_losses.csv``, already day-indexed (``day``,
            ``model``, ``metric``, ``value``, ``fold`` columns).
        conditions: Condition names to restrict the comparison to.
        metric: Loss column to rank by (``"CRPS"``, ``"Winkler"``, ``"Energy
            score"``, or ``"Brier score"``).

    Returns:
        ``(fold, day)`` for the best (lowest mean loss) and worst day.
    """
    sub = losses_df[(losses_df["metric"] == metric) & (losses_df["model"].isin(conditions))]
    wide = sub.pivot_table(index=["fold", "day"], columns="model", values="value")
    wide = wide[[c for c in conditions if c in wide.columns]].dropna()
    if wide.empty:
        raise RuntimeError(f"no (fold, day) pair has {metric} for every requested condition")
    mean_loss = wide.mean(axis=1)
    return mean_loss.idxmin(), mean_loss.idxmax()


def pick_cache_days(blob, truth: np.ndarray, conditions: list[str], target_idx: int) -> tuple[int, int]:
    """Best/worst midnight-aligned test window, from the cache's own pass-1 (no-sampled-path) quantiles.

    Mirrors ``forecast_examples.py``'s own pass-1 day selection: mean
    absolute median error, scaled by the target's test-set std, averaged
    across ``conditions``, restricted to midnight-aligned windows so a full
    day is shown.
    """
    medians = {}
    for name in conditions:
        q = blob[f"q::{name}"]
        lv = blob[f"lv::{name}"]
        target_levels = lv[:, target_idx] if lv.ndim == 2 else lv
        med_idx = int(np.abs(target_levels - 0.5).argmin())
        medians[name] = q[:, med_idx, :, target_idx]
    target_truth = truth[..., target_idx]
    scale = target_truth.std()
    errs = np.mean([np.abs(m - target_truth).mean(axis=1) / scale for m in medians.values()], axis=0)
    return errs, target_truth


def draw_copula_panel(ax, t_axis, y, q, lv, label: str) -> None:
    """One copula-variant's forecast panel, from the ``predictions.npz`` cache's own quantile fan.

    No in-figure title, matching :func:`draw_fpqr_panel` -- only a compact
    in-axes corner label.
    """
    ylim = pad_limits(y, q)
    draw(ax, t_axis, y, q, lv, t_i=0, ylim=ylim)
    ax.text(0.02, 0.96, tex_safe(label), transform=ax.transAxes, va="top", ha="left", fontsize=7)
    ax.set_ylabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)


def run_copula_forecast_examples(
    dataset_slug: str, conditions: list[str], latex: bool, day_metric: str = "CRPS"
) -> None:
    """Reference good/bad day from the real backtest, plus illustrative panels from the prediction cache."""
    out_dir = OUT / "copula_variants" / dataset_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _latest_copula_run_dir(dataset_slug)
    if run_dir is not None and (run_dir / "results" / "copula_losses.csv").exists():
        losses = pd.read_csv(run_dir / "results" / "copula_losses.csv")
        available = [c for c in conditions if c in losses["model"].unique()]
        if len(available) >= 2:
            (good_fold, good_day), (bad_fold, bad_day) = pick_copula_days(losses, available, metric=day_metric)
            print(f"day-selection metric: {day_metric}")
            print(f"formal backtest good day: fold={good_fold} day={good_day}")
            print(f"formal backtest bad day:  fold={bad_fold} day={bad_day}")
        else:
            print(f"[note] {dataset_slug}: fewer than 2 requested conditions in copula_losses.csv")
    else:
        print(f"[note] {dataset_slug}: no copula_losses.csv found; skipping the formal-backtest reference day")

    if not FORECAST_EXAMPLES_CACHE.exists():
        print(f"no prediction cache at {FORECAST_EXAMPLES_CACHE}; panels skipped (see module docstring)")
        return
    blob = np.load(FORECAST_EXAMPLES_CACHE, allow_pickle=False)
    names = [str(n) for n in blob["model_names"]]
    present = [c for c in conditions if c in names]
    missing = [c for c in conditions if c not in names]
    if missing:
        print(f"cache lacks: {', '.join(missing)} (skipped, not silently dropped)")
    if len(present) < 2:
        print(f"cache covers fewer than 2 requested conditions for {dataset_slug}; panels skipped")
        return

    print(f"loading truth/timestamps for the cache's own fit ({CACHE_SCENARIO}) ...")
    d = build_windows(CACHE_SCENARIO, CACHE_LOOKBACK, CACHE_HORIZON)
    truth = d["y_te"] * d["sd"] + d["mu"]
    targets = d["targets"]
    stamps = pd.to_datetime(d["stamps_te"][:, 0])
    midnight = np.flatnonzero((stamps.hour == 0) & (stamps.minute == 0))
    if midnight.size == 0:
        print("no midnight-aligned test window in the cache's fit; panels skipped")
        return

    configure(latex=latex)
    target_idx = 0  # first target (active power) -- see module docstring's --target note for fpqr's analogue
    target = targets[target_idx]
    errs, target_truth = pick_cache_days(blob, truth, present, target_idx)
    good = int(midnight[errs[midnight].argmin()])
    bad = int(midnight[errs[midnight].argmax()])
    print(f"cache-fit good window {good} ({pd.to_datetime(d['stamps_te'][good, 0]):%Y-%m-%d}), target={target}")
    print(f"cache-fit bad window  {bad} ({pd.to_datetime(d['stamps_te'][bad, 0]):%Y-%m-%d}), target={target}")

    for label, window in (("good", good), ("bad", bad)):
        ncol = min(len(present), 3)
        nrow = -(-len(present) // ncol)
        pw, ph = panel(1.0 / max(ncol, 1), aspect=0.75)
        fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, ph * nrow), squeeze=False)
        axes_flat = axes.flatten()
        t_axis = pd.to_datetime(d["stamps_te"][window])
        y = target_truth[window]

        for ax, name in zip(axes_flat, present, strict=False):
            q = blob[f"q::{name}"][window][:, :, target_idx]
            lv = blob[f"lv::{name}"][:, target_idx]
            ts0 = t_axis[0]
            panel_label = f"{display_name(name)} ({ts0:%Y-%m-%d}, Target={target})"
            draw_copula_panel(ax, t_axis, y, q, lv, panel_label)

            fig_single, ax_single = plt.subplots(figsize=panel(1, aspect=0.7))
            draw_copula_panel(ax_single, t_axis, y, q, lv, panel_label)
            fig_single.savefig(out_dir / f"{label}_{name.replace(' ', '_')}.pdf", dpi=600, bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes_flat[len(present) :]:
            ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"overview_{label}_day.pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(f"\nwrote {out_dir}")


def main() -> None:
    """Parse CLI arguments and draw the requested experiment's forecast-example panels."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--experiment",
        required=True,
        choices=["fpqr_gating_ablation", "backbone_conformal_study", "multitarget_experiment", "copula_variants"],
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="fpqr_gating_ablation/backbone_conformal_study/multitarget_experiment: dataset name; "
        "copula_variants: dataset slug",
    )
    ap.add_argument(
        "--group",
        default=None,
        choices=sorted({*FPQR_GROUP_CONDITIONS, *MULTITARGET_GROUP_CONDITIONS}),
        help="fpqr_gating_ablation/multitarget_experiment only",
    )
    ap.add_argument("--conditions", nargs="+", default=None, help="override the group/experiment's own condition list")
    ap.add_argument("--target", default="LP", help="fpqr_gating_ablation/backbone_conformal_study only: LP, LQ, or LV")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="fpqr_gating_ablation/backbone_conformal_study only: override run discovery",
    )
    ap.add_argument(
        "--day-metric",
        default=None,
        help=(
            "which metric decides the good/bad day: fpqr_gating_ablation/backbone_conformal_study accept "
            f"{sorted(DAY_METRICS)} (default 'nrmse'); copula_variants accepts a "
            "copula_losses.csv metric name, e.g. 'CRPS' (default), 'Winkler', 'Brier score'"
        ),
    )
    ap.add_argument("--latex", action="store_true", default=False)
    args = ap.parse_args()

    if args.experiment == "fpqr_gating_ablation":
        group = args.group or "grid"
        dataset = args.dataset or "MLVS-PT-3PHASE"
        conditions = args.conditions or FPQR_GROUP_CONDITIONS[group]
        day_metric_name = args.day_metric or "nrmse"
        run_fpqr_forecast_examples(group, dataset, conditions, args.target, args.run_dir, args.latex, day_metric_name)
    elif args.experiment == "backbone_conformal_study":
        dataset = args.dataset or "MLVS-PT-3PHASE"
        conditions = args.conditions or BACKBONE_CONFORMAL_CONDITIONS
        day_metric_name = args.day_metric or "nrmse"
        run_backbone_conformal_forecast_examples(
            dataset, conditions, args.target, args.run_dir, args.latex, day_metric_name
        )
    elif args.experiment == "multitarget_experiment":
        group = args.group or "benchmark"
        dataset = args.dataset or "MLVS-PT-3PHASE"
        conditions = args.conditions or MULTITARGET_GROUP_CONDITIONS[group]
        day_metric_name = args.day_metric or "nrmse"
        run_multitarget_forecast_examples(
            group, dataset, conditions, args.target, args.run_dir, args.latex, day_metric_name
        )
    else:
        slugs = (
            sorted(p.name for p in COPULA_RESULTS_ROOT.iterdir() if p.is_dir()) if COPULA_RESULTS_ROOT.exists() else []
        )
        dataset_slug = args.dataset or (slugs[0] if slugs else None)
        if dataset_slug is None:
            raise SystemExit(f"no copula_variants dataset found under {COPULA_RESULTS_ROOT}")
        # copula_variants.VARIANTS' default four (shared/per-target grid, shared/per-target
        # Gaussian copula) -- the clearest single contrast, not every one of the eight variants.
        conditions = args.conditions or [
            "fpqr shared grid",
            "fpqr per-target grid",
            "fpqr shared + copula",
            "fpqr per-target + copula",
        ]
        day_metric = args.day_metric or "CRPS"
        run_copula_forecast_examples(dataset_slug, conditions, args.latex, day_metric)


if __name__ == "__main__":
    main()
