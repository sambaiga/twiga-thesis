r"""Analyze Chapter 6 multi-target experiments and write tables and conclusions.

The report combines FPQR gating, multi-target benchmarks, copula variants,
backbone/conformal studies, and operational decision experiments. It adapts
their distinct CSV and run layouts into shared fold-level analysis functions,
while keeping metric sets, references, joint-coverage calculations, and
decision conclusions specific to Chapter 6. Figures use the thesis theme and
shared comparison plot functions.

Usage:

    uv run python experiment/ch7_multitarget/ch6_analyse_results.py --experiment all
    uv run python experiment/ch7_multitarget/ch6_analyse_results.py \
        --experiment fpqr_gating_ablation --group proposal qvn
    uv run python experiment/ch7_multitarget/ch6_analyse_results.py \
        --experiment copula_variants --dataset dss-eu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

# Headless-safe regardless of what a later import (e.g. twiga.experiment,
# pulled in below for joint_interval_coverage) does to matplotlib's own
# backend auto-detection -- found 2026-09-26: adding that import made
# pyplot's lazy backend resolution fall through to TkAgg on first
# plt.subplots() call, which then crashed on this machine's broken Tcl
# install. Set before pyplot is ever imported, matching
# forecast_examples.py/ch6_forecast_examples.py's own defensive convention.
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT_DIR = script_dir.parent / "results" / "analysis"

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent / "experiment_2_backbone_conformal"))
import chapters.ch7_multitarget.experiment_2_backbone_conformal.backbone_conformal_study as bcs  # noqa: E402
import chapters.ch7_multitarget.experiment_2_backbone_conformal.fpqr_gating_ablation as fga  # noqa: E402
from twiga_thesis.analysis import (  # noqa: E402
    dm_and_mcs,
    fold_level,
    kupiec_table,
    metric_table,
)
from twiga_thesis.plotting.comparison_plots import (  # noqa: E402
    display_name as _base_display_name,
    plot_dm_significance_matrix,
    plot_good_bad_day,
    plot_metric_bar_grid,
    plot_metric_bars,
    plot_normalized_comparison,
    save_metric_figures,
)

sys.path.insert(0, str(script_dir.parent.parent))
from twiga.core.utils import slugify  # noqa: E402
from twiga.experiment.analysis import aggregate_fold_metrics, joint_interval_coverage  # noqa: E402
from twiga.experiment.run_catalog import (  # noqa: E402
    discover_runs as discover_experiment_runs,
    latest_run_for_group,
)

from twiga_thesis.plotting import style as S  # noqa: E402

# ── Chapter 6 acronym tokens, layered onto shared comparison labels ─────────
_EXTRA_ACRONYMS = {
    "fpqr": "FPQR",
    "qvn": "QVN",
    "crps": "CRPS",
    "picp": "PICP",
    "nmpi": "NMPI",
    "cwe": "CWE",
    "nrmse": "NRMSE",
    "mae": "MAE",
}


def display_name(name: str) -> str:
    """Format a Chapter 6 label with chapter-specific acronym casing."""
    return _base_display_name(name, extra_acronyms=_EXTRA_ACRONYMS)


# ── FPQR gating ablation configuration ────────────────────────────────────────
FPQR_GROUPS = ["proposal", "qvn", "grid", "multigate", "head"]  # fixed hypothesis order
# Copied from fpqr_gating_ablation.SPEC_MLVS_PT_3PHASE.reference_conditions
# (identical to SPEC_DSS_EU's -- both specs share _SPEC_COMMON's
# reference_conditions dict) rather than re-typed, so a future edit to that
# script's reference conditions cannot silently drift out of sync here.
FPQR_REFERENCE: dict[str, str] = dict(fga.SPEC_MLVS_PT_3PHASE.reference_conditions)
FPQR_METRIC_COLS = ["nrmse", "winkler_score", "picp", "nmpi", "cwe_stratified", "mae"]
FPQR_LOWER_BETTER = {"nrmse", "winkler_score", "nmpi", "mae"}
FPQR_HIGHER_BETTER = {"cwe_stratified"}
FPQR_METRIC_DISPLAY = {
    "nrmse": "NRMSE",
    "winkler_score": "WS",
    "picp": "PICP",
    "nmpi": "NMPI",
    "cwe_stratified": "CWE (Stratified)",
    "mae": "MAE",
}
FPQR_DECISIVE_METRIC = "winkler_score"

# ── Copula variants configuration ─────────────────────────────────────────────
# Condition names in copula_per_fold.csv carry spaces ("fpqr shared grid"),
# not underscores -- display_name/head_label/plot_forest all tokenise on "_",
# so every copula condition name is normalised to underscores once, here,
# rather than leaking raw spaced strings into labels that would otherwise
# title-case the whole string as one token (e.g. "Fpqr shared grid").
COPULA_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "copula_variants"
COPULA_REFERENCE = "fpqr_shared_grid"  # underscore form of "fpqr shared grid"
COPULA_METRIC_COLS = ["Winkler", "CRPS", "Brier score"]
COPULA_LOWER_BETTER = {"Winkler", "CRPS", "Brier score"}
COPULA_HIGHER_BETTER: set[str] = set()
COPULA_METRIC_DISPLAY = {"Winkler": "Winkler", "CRPS": "CRPS", "Brier score": "Brier Score"}


def _slugify_condition(name: str) -> str:
    """``"fpqr shared grid"`` -> ``"fpqr_shared_grid"``, so display helpers tokenise it correctly."""
    return name.replace(" ", "_")


# ── Multitarget benchmark configuration (Experiment 1: FPQR vs QR/BNN/Laplace) ──
# multitarget_experiment.py has no discover_runs/latest_run_for_group of its
# own (unlike fpqr_gating_ablation.py) -- it is a pure ExperimentEngine
# runner -- so this module reimplements the same meta.json-glob discovery
# against its own results root, one level shallower than
# fpqr_gating_ablation's (no per-experiment nesting between "runs" and the
# results root itself).
MULTITARGET_ROOT = (
    script_dir.parent / "results" / "ch7_multitarget" / "multitarget_experiment" / "multitarget_forecasting_benchmark"
)
MULTITARGET_GROUPS = ["benchmark", "grid", "conformal"]
# Copied from multitarget_experiment._SPEC_COMMON["reference_conditions"].
# "benchmark" is "fpqr_anchor_outer", not the plain "fpqr" this once was --
# multitarget_experiment.py's own 52cfe3e promoted fpqr_anchor_outer to be
# the benchmark group's FPQR condition; this copy had drifted out of sync
# with that rename, silently producing a NaN DM p-value (no condition named
# "fpqr" exists in the data to compare against) rather than a real
# significance test against the chapter's actual FPQR configuration.
MULTITARGET_REFERENCE = {
    "benchmark": "fpqr_anchor_outer",
    "grid": "fpqr_shared_grid",
    "conformal": "laplace_max_norm",
}
MULTITARGET_METRIC_COLS = ["nrmse", "winkler_score", "picp", "nmpi", "cwe_stratified", "cwe", "mae"]
MULTITARGET_LOWER_BETTER = {"nrmse", "winkler_score", "nmpi", "mae"}
MULTITARGET_HIGHER_BETTER = {"cwe_stratified", "cwe"}
MULTITARGET_METRIC_DISPLAY = FPQR_METRIC_DISPLAY
# "joint_coverage" -- the simultaneous P/Q/V coverage rate, same metric and
# same reasoning as BACKBONE_CONFORMAL_METRIC_COLS's own addition below --
# is only meaningful for the "conformal" group (benchmark/grid conditions
# calibrate nothing jointly), so load_multitarget_full merges it in only
# for that group rather than adding it to MULTITARGET_METRIC_COLS itself,
# which every group's metric_table/plot call shares.
MULTITARGET_CONFORMAL_METRIC_COLS = [*MULTITARGET_METRIC_COLS, "joint_coverage"]
MULTITARGET_CONFORMAL_HIGHER_BETTER = MULTITARGET_HIGHER_BETTER | {"joint_coverage"}
MULTITARGET_CONFORMAL_METRIC_DISPLAY = {**MULTITARGET_METRIC_DISPLAY, "joint_coverage": "Joint Coverage", "cwe": "CWE"}
MULTITARGET_DECISIVE_METRIC = "winkler_score"

# ── Backbone conformal study configuration (follow-on: backbone x conformal, ──
# marginal metrics/coverage on all four datasets) ─────────────────────────────
# backbone_conformal_study.py mirrors fpqr_gating_ablation.py's own
# discover_runs()/latest_run_for_group() (not multitarget_experiment.py's
# hand-rolled _discover_meta_runs), so this module's own copies are reused
# directly rather than reimplemented a third time.
BACKBONE_CONFORMAL_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "backbone_conformal_study"
BACKBONE_CONFORMAL_GROUP = "backbone"  # the study's one and only Condition group
# Copied from backbone_conformal_study._SPEC_COMMON["reference_conditions"].
BACKBONE_CONFORMAL_REFERENCE = dict(bcs.SPEC_MLVS_PT_3PHASE.reference_conditions)["backbone"]
BACKBONE_CONFORMAL_LOWER_BETTER = MULTITARGET_LOWER_BETTER
BACKBONE_CONFORMAL_DECISIVE_METRIC = "winkler_score"
# "joint_coverage" -- the simultaneous P/Q/V coverage rate Chapter 6's own
# methodology names as its calibration-performance metric (subsec:ch6-metrics)
# -- is specific to this script (only load_backbone_conformal_full computes
# it, from each condition's raw predictions.parquet via
# twiga.experiment.analysis.joint_interval_coverage), so it is added here
# rather than onto the shared MULTITARGET_* config
# multitarget_experiment.py's own analysis also uses.
BACKBONE_CONFORMAL_METRIC_COLS = [*MULTITARGET_METRIC_COLS, "joint_coverage"]
BACKBONE_CONFORMAL_HIGHER_BETTER = MULTITARGET_HIGHER_BETTER | {"joint_coverage"}
BACKBONE_CONFORMAL_METRIC_DISPLAY = {**MULTITARGET_METRIC_DISPLAY, "joint_coverage": "Joint Coverage"}

# ── Operational decision configuration (Experiment 3: hosting-capacity / ────
# thermal-exceedance Brier score) ─────────────────────────────────────────────
# operational_decision.py writes directly under RESULTS_ROOT/runs/<id>/,
# no per-dataset subdirectory -- each run's meta.json instead names which
# decision(s) it covers (e.g. ["DSS-UK"]), so a run is looked up by decision
# membership, not by a slug directory the way copula_variants is.
OPERATIONAL_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "operational_decision"
# "joint" (the simplest parametric baseline) is the reference: the chapter's
# operational claim is specifically that FPQR/copula beats the plain
# parametric baselines on the decisive Brier score, so that is the condition
# an alternative must significantly beat, not an arbitrary FPQR variant.
OPERATIONAL_REFERENCE = "joint"
OPERATIONAL_METRIC_COLS = [
    "Brier score",
    "Brier skill score",
    "Reliability",
    "Resolution",
    "Peak relative economic value",
    # Interval-based metrics -- the same marginal calibration surface
    # Experiment 2 (backbone_conformal_study.py) reports, added here so a
    # backbone/conformal condition that wins the *decision* comparison can
    # also be checked against a sane *marginal* interval (e.g. the
    # NHiTS/DSS-UK interval collapse would show up directly here as
    # picp/nmpi/winkler_score, not just inferred from the operational Brier
    # score's own degenerate 0/1 pattern).
    "rmse",
    "mae",
    "picp",
    "nmpi",
    "winkler_score",
    "cwe",
    "cwe_stratified",
]
OPERATIONAL_LOWER_BETTER = {"Brier score", "Reliability", "rmse", "mae", "winkler_score", "nmpi"}
OPERATIONAL_HIGHER_BETTER = {
    "Brier skill score",
    "Resolution",
    "Peak relative economic value",
    "cwe",
    "cwe_stratified",
}
OPERATIONAL_METRIC_DISPLAY = {
    "Brier score": "Brier Score",
    "Brier skill score": "Brier Skill Score",
    "Reliability": "Reliability",
    "Resolution": "Resolution",
    "Peak relative economic value": "Peak REV",
    "rmse": "RMSE",
    "mae": "MAE",
    "picp": "PICP",
    "nmpi": "NMPI",
    "winkler_score": "WS",
    "cwe": "CWE",
    "cwe_stratified": "CWE (Stratified)",
}
OPERATIONAL_DECISIVE_METRIC = "Brier score"
# "picp"/"cwe" are deliberately not in either better/worse set, same
# convention as MULTITARGET_HIGHER_BETTER: PICP has a nominal target rather
# than a monotone direction, and marginal CWE is diagnostic-only project-wide
# (see CAVEATS_TEXT).
# "Pr(event) predicted"/"Pr(event) realised"/"Event rate error (%)" are
# deliberately not ranked columns -- they have no "better" direction on their
# own (a rate is diagnostic, not a score) -- but are readable straight from
# the raw `ch6_backbone_conformal_decision_{decision}_full.csv`/
# `ch6_operational_full.csv`: a `Pr(event) realised` of 0.0 or 1.0 in every
# fold is exactly the degenerate-label signature that makes
# "Brier skill score"/"Peak relative economic value" NaN (found 2026-09-26,
# DSS-UK).

# ── Backbone conformal decision configuration (follow-on: does calibration ────
# help the DSS-UK/DSS-US decision) ─────────────────────────────────────────────
# backbone_conformal_decision.py's own output shape matches
# operational_decision.py's exactly (per-decision `{decision}_per_fold.csv`
# with tier/metric/target/model/value/fold columns, `meta.json`'s "decisions"
# key), so its loader mirrors load_operational_full's below rather than
# introducing a third schema.
BACKBONE_CONFORMAL_DECISION_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "backbone_conformal_decision"
# "mlpgam_none" -- MLPGAM is this chapter's own reference backbone throughout
# fpqr_gating_ablation.py/multitarget_experiment.py, and "none" conformal
# matches operational_decision.py's own uncalibrated baseline.
BACKBONE_CONFORMAL_DECISION_REFERENCE = "mlpgam_none"
BACKBONE_CONFORMAL_DECISION_METRIC_COLS = OPERATIONAL_METRIC_COLS
BACKBONE_CONFORMAL_DECISION_LOWER_BETTER = OPERATIONAL_LOWER_BETTER
BACKBONE_CONFORMAL_DECISION_HIGHER_BETTER = OPERATIONAL_HIGHER_BETTER
BACKBONE_CONFORMAL_DECISION_METRIC_DISPLAY = OPERATIONAL_METRIC_DISPLAY
BACKBONE_CONFORMAL_DECISION_DECISIVE_METRIC = "Brier score"

# ── Multitarget decision: MLPF-only, four-head companion to the above ────────
# multitarget_decision.py (2026-10-02) mirrors backbone_conformal_decision.py's
# output shape exactly (same per-decision `ch6_multitarget_decision_
# {decision}_full.csv`, tier/metric/target/model/value/Folds/condition/group/
# dataset schema), varying head (fpqr_anchor_outer/qr/bnn/laplace) instead of
# backbone (MLPF/MLPGAM/NHiTS) -- same metric configuration reused as-is.
MULTITARGET_DECISION_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "multitarget_decision"
# "laplace_max_norm" -- matches multitarget_decision.py's own
# ExperimentSpec.reference_conditions["decision"].
MULTITARGET_DECISION_REFERENCE = "laplace_max_norm"
MULTITARGET_DECISION_METRIC_COLS = OPERATIONAL_METRIC_COLS
MULTITARGET_DECISION_LOWER_BETTER = OPERATIONAL_LOWER_BETTER
MULTITARGET_DECISION_HIGHER_BETTER = OPERATIONAL_HIGHER_BETTER
MULTITARGET_DECISION_METRIC_DISPLAY = OPERATIONAL_METRIC_DISPLAY
MULTITARGET_DECISION_DECISIVE_METRIC = "Brier score"


# ── Caveats, verbatim from ANALYSIS_PLAN.md SS6 ──────────────────────────────
CAVEATS_TEXT = """## Caveats

- Fold test windows overlap (fold 0 tests June-August, fold 1 July-September), so the reported fold
  standard deviation understates uncertainty.
- Evaluation covers June-December only; winter, which matters for an EV/PV-penetration study, is never
  in a test fold.
- One fixed seed across conditions -- the reported spread is fold variation, not seed variation.
- Both completed FPQR-gating-ablation runs cover MLVS-PT-3PHASE only; any table built now should say so
  rather than imply cross-dataset coverage.
- CWE is never the deciding metric here -- it has no per-observation loss differential to feed a
  Diebold-Mariano test, and is reported only as a diagnostic alongside PICP and nMPI.
"""


def _to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """``df.to_markdown()``, falling back to a fenced ``to_string()`` block if ``tabulate`` is absent."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def _winsorize_nrmse(full: pd.DataFrame, quantile: float = 0.97) -> pd.DataFrame:
    """Clip the ``nrmse`` column at its own upper quantile, per (dataset, group, condition).

    ``twiga.core.metrics.point.nrmse`` divides RMSE by the target's own
    range (max - min) within the window being scored -- correct, and the
    normalisation this project's methodology specifies (see that function's
    own docstring) -- but a rare, near-constant window (range close to
    zero) makes that one instance's NRMSE explode by orders of magnitude.
    Confirmed 2026-09-27 on the MLVS-PT-3PHASE benchmark rerun: active
    power's NRMSE reached 104,208 on <1.1% of instances, against a median
    of ~0.29 -- a naive mean over instances (what ``fold_level``/
    ``metric_table`` compute) is then dominated by that handful of
    near-zero-scale windows rather than reflecting typical performance.

    This is not a bug in ``nrmse()`` itself, and not fixed there -- it is
    used correctly, and consistently, across the whole project. It is an
    aggregation-robustness gap in how this script's own metric tables
    summarise it, matching the same class of issue
    ``twiga.experiment.analysis.aggregate()`` already documents and reports
    a MEDIAN column alongside MEAN for. Clipping at the 97th percentile
    (chosen empirically against the confirmed <1.1% outlier fraction, with
    margin) per (dataset, group, condition) -- not globally -- keeps every
    condition judged against its own tail, not another's, and was verified
    to preserve the qualitative ranking a median-based comparison already
    gives (only the magnitude changes).
    """
    if "nrmse" not in full.columns:
        return full
    group_cols = [c for c in ("dataset", "group", "condition") if c in full.columns]
    if not group_cols:
        return full
    full = full.copy()
    caps = full.groupby(group_cols)["nrmse"].transform(lambda s: s.quantile(quantile))
    full["nrmse"] = full["nrmse"].clip(upper=caps)
    return full


def _ensure_metric_columns(pivot: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Add any ``metric_cols`` missing from ``pivot`` as an all-NaN column.

    A metric pivoted from ``df[df["metric"].isin(metric_cols)]`` is simply
    absent as a column when a run predates that metric being computed at
    all (e.g. an older ``backbone_conformal_decision`` run, before the
    interval-metric/Peak-REV rows this script now also requests) -- not
    NaN-valued, genuinely not a column, which crashes
    ``fold_level``/``metric_table``'s ``groupby(...)[metric_cols]`` selection
    with a ``KeyError`` rather than reporting the metric as unavailable.
    Matches the project's own established convention (see
    ``twiga.core.metrics.interval.get_interval_metrics``'s own docstring):
    a metric that cannot be computed/found is reported as a NaN column, not
    silently dropped from the schema.
    """
    for metric in metric_cols:
        if metric not in pivot.columns:
            pivot[metric] = float("nan")
    return pivot


def _drop_degenerate_conditions(fold_df: pd.DataFrame, metric: str, label: str) -> pd.DataFrame:
    """Drop conditions whose ``metric`` is ``NaN`` in every fold before a DM/MCS comparison.

    ``dm_and_mcs``'s own pivot (folds x conditions, one column per
    condition) calls ``.dropna()`` on the whole wide frame -- correct when
    every condition has at least an occasional NaN fold, but a condition
    that is NaN in *every* fold (e.g. a collapsed predictive interval
    leaving Winkler score undefined every time) makes every row NaN
    somewhere, so ``.dropna()`` empties the frame entirely and
    ``pairwise_dm_table`` then raises "loss arrays must not be empty" --
    for every OTHER condition too, not just the degenerate one. Found
    2026-09-26 the moment the conformal-condition-filtering fix above
    stopped silently hiding NHiTS's DSS-UK PICP=0 collapse from this table.
    Excluding only the genuinely-degenerate condition(s) here, with a
    printed reason, keeps the comparison running for everything else
    instead of losing the whole dataset's significance test to one
    condition's failure.
    """
    by_condition = fold_df.groupby("condition")[metric]
    all_nan = by_condition.apply(lambda s: s.isna().all())
    degenerate = all_nan[all_nan].index.tolist()
    if not degenerate:
        return fold_df
    print(f"  [excluded from significance] {label}: {metric} is NaN in every fold for {degenerate}")
    return fold_df[~fold_df["condition"].isin(degenerate)]


_ENGINE_KEY = ["dataset", "group", "condition", "Folds", "timestamp", "target"]
_ENGINE_POINT_COLS = ["nrmse", "mae"]
_ENGINE_INTERVAL_COLS = ["winkler_score", "picp", "nmpi", "cwe_stratified", "cwe", "joint_coverage"]


def _merge_point_interval(df: pd.DataFrame) -> pd.DataFrame:
    """One wide row per :data:`_ENGINE_KEY`, from the engine's metric_type-split long rows.

    Most engine-produced conditions carry a ``metric_type`` column
    (``"point"``/``"interval"``/``"quantile"``), one row per type per
    timestamp, so point and interval metrics must be merged back together.
    Conditions that wrap a base model in a ``ConformalConfig`` (e.g.
    ``multitarget_experiment.py``'s ``conformal`` group, or
    ``backbone_conformal_study.py``'s own ``*_max_norm``/``*_mahalanobis``/
    ``*_quantile``/``*_quantile_joint`` conditions) take a different path in
    :class:`~twiga.experiment.ExperimentEngine` that writes every metric onto
    a single row per timestamp instead, leaving ``metric_type`` entirely
    null for those rows. That flat shape is already the wide row this
    function otherwise builds, so it is returned as-is.

    Regression test for a real bug found 2026-09-26: this used to decide
    which shape to use via ``df["metric_type"].isna().all()`` on the *whole*
    input frame, which only works when every row shares one shape --
    ``multitarget_experiment.py``'s own ``conformal`` group is always called
    on its own, uniformly flat-row group. ``backbone_conformal_study.py``'s
    single ``"backbone"`` group mixes both shapes in one call
    (``*_none`` conditions keep the point/interval/quantile split;
    every calibrated condition is flat-row) -- with the old whole-frame
    check, the merge-by-``metric_type`` branch ran, and every calibrated
    condition's rows (``metric_type`` NaN, matching neither ``"point"`` nor
    ``"interval"``) were silently dropped from both sides of the merge and
    lost entirely: 12 of 15 conditions across every dataset, present and
    complete in the raw CSV, never appeared in the printed comparison table.
    Splitting by row instead of by whole-frame check handles both shapes
    correctly whether they arrive separately or mixed together.
    """
    # Filtered by presence, not just assumed: _ENGINE_INTERVAL_COLS now
    # includes "joint_coverage" (added 2026-09-26 for
    # backbone_conformal_study.py's own merge), which no other caller's CSV
    # has at all -- an unfiltered selection here raised a hard KeyError for
    # every other dataset/experiment this function serves, not just a NaN
    # column.
    point_cols = [c for c in _ENGINE_POINT_COLS if c in df.columns]
    interval_cols = [c for c in _ENGINE_INTERVAL_COLS if c in df.columns]
    flat_mask = df["metric_type"].isna()
    flat = df.loc[flat_mask, [*_ENGINE_KEY, *point_cols, *interval_cols]]
    typed = df.loc[~flat_mask]
    if typed.empty:
        return flat
    point = typed[typed["metric_type"] == "point"][[*_ENGINE_KEY, *point_cols]]
    interval = typed[typed["metric_type"] == "interval"][[*_ENGINE_KEY, *interval_cols]]
    merged = point.merge(interval, on=_ENGINE_KEY, how="outer")
    return pd.concat([merged, flat], ignore_index=True) if not flat.empty else merged


# ── FPQR gating ablation: run discovery + loading ────────────────────────────


def load_fpqr_full(groups: list[str]) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Latest-run ``ch6_fpqr_gating_full.csv`` rows for ``groups``, point+interval merged wide.

    Reuses ``fpqr_gating_ablation.discover_runs``/``latest_run_for_group`` for
    run discovery, exactly as that module's own (thin) ``significance_report``
    does. Point-metric_type rows (``nrmse``/``mae``) and interval-metric_type
    rows (``winkler_score``/``picp``/``nmpi``/``cwe_stratified``) are merged
    back into one wide row per ``(dataset, group, condition, Folds, timestamp,
    target)`` key, so :func:`~analyse_results.fold_level`/``metric_table`` see
    every metric column at once instead of the engine's own long,
    metric_type-split rows.

    Args:
        groups: Condition groups to load (a subset of ``FPQR_GROUPS``).

    Returns:
        The merged wide frame (empty if nothing was found), and a coverage
        log of ``(group, run_dir_name, run_timestamp)`` for groups that were
        found.
    """
    runs = fga.discover_runs()
    frames, coverage = [], []
    for group in groups:
        run = fga.latest_run_for_group(runs, group)
        if run is None:
            print(f"[missing] group={group}: no run covers it")
            continue
        csv_path = run["run_dir"] / "results" / "ch6_fpqr_gating_full.csv"
        if not csv_path.exists():
            print(f"[missing] group={group}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["group"] == group]
        if df.empty:
            print(f"[missing] group={group}: 0 rows in {csv_path}")
            continue
        frames.append(_merge_point_interval(df))
        coverage.append((group, run["run_dir"].name, run["timestamp"]))
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _winsorize_nrmse(full), coverage


# ── Multitarget benchmark: run discovery + loading ───────────────────────────


def _discover_meta_runs(root: Path) -> list[tuple[dict, Path]]:
    """``(meta, run_dir)`` for every ``root/runs/*/meta.json``, oldest to newest on disk order."""
    runs = discover_experiment_runs(root / "runs")
    return [(run["metadata"], run["run_dir"]) for run in runs]


def _latest_multitarget_run(group: str) -> tuple[dict, Path] | None:
    """The most recently timestamped multitarget-benchmark run covering ``group``."""
    run = latest_run_for_group(discover_experiment_runs(MULTITARGET_ROOT / "runs"), group)
    if run is None:
        return None
    return run["metadata"], run["run_dir"]


def load_multitarget_full(
    groups: list[str],
) -> tuple[pd.DataFrame, list[tuple[str, str, str]], pd.DataFrame]:
    """Latest-run ``ch6_benchmark_full.csv`` rows for ``groups``, point+interval merged wide.

    Same engine-produced schema and point/interval merge as
    :func:`load_fpqr_full` (this experiment shares
    :class:`~twiga.experiment.ExperimentEngine` with
    ``fpqr_gating_ablation.py``, just without that module's own
    ``discover_runs`` helper), so this is that function's logic against
    ``MULTITARGET_ROOT`` instead.

    Args:
        groups: Condition groups to load (a subset of ``MULTITARGET_GROUPS``).

    Returns:
        The merged wide frame (empty if nothing was found), a coverage log
        of ``(group, run_dir_name, run_timestamp)`` for groups that were
        found, and the same runs' ``ch6_benchmark_representative.csv`` rows
        (empty frame if none exist), for :func:`plot_good_bad_day`.
    """
    frames, coverage, representative_frames = [], [], []
    for group in groups:
        found = _latest_multitarget_run(group)
        if found is None:
            print(f"[missing] group={group}: no run covers it")
            continue
        meta, run_dir = found
        csv_path = run_dir / "results" / "ch6_benchmark_full.csv"
        if not csv_path.exists():
            print(f"[missing] group={group}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["group"] == group]
        if df.empty:
            print(f"[missing] group={group}: 0 rows in {csv_path}")
            continue
        if group == "conformal" and "dataset" in df.columns:
            joint_frames = [
                _load_joint_coverage(run_dir, dataset, group="conformal") for dataset in df["dataset"].unique()
            ]
            joint_frames = [jf for jf in joint_frames if not jf.empty]
            if joint_frames:
                joint_coverage = pd.concat(joint_frames, ignore_index=True)
                df = df.merge(joint_coverage, on=["condition", "Folds"], how="left")
            elif "joint_coverage" not in df.columns:
                # No predictions.parquet available for this run (e.g. a
                # synced copy of only the summary CSVs) -- report as an
                # all-NaN column rather than letting metric_table's
                # downstream groupby(...)[metric_cols] crash with a
                # KeyError, matching get_interval_metrics' own "a metric
                # that cannot be computed is NaN, not silently dropped"
                # convention (see _ensure_metric_columns' own docstring).
                df["joint_coverage"] = float("nan")
        frames.append(_merge_point_interval(df))
        coverage.append((group, run_dir.name, meta["timestamp"]))
        representative_path = run_dir / "results" / "ch6_benchmark_representative.csv"
        if representative_path.exists():
            representative_df = pd.read_csv(representative_path)
            representative_frames.append(representative_df[representative_df["group"] == group])
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    representative = pd.concat(representative_frames, ignore_index=True) if representative_frames else pd.DataFrame()
    return _winsorize_nrmse(full), coverage, representative


# ── Backbone conformal study: run discovery + loading ────────────────────────


def _load_joint_coverage(run_dir: Path, dataset: str, group: str = BACKBONE_CONFORMAL_GROUP) -> pd.DataFrame:
    """Simultaneous P/Q/V coverage per condition/fold, from this dataset's own saved ``predictions.parquet``.

    Chapter 6's own methodology (subsec:ch6-metrics) states calibration
    performance is assessed via "the simultaneous coverage rate of the
    multi-target conformal prediction region... the proportion of forecast
    instances for which [every target] is jointly contained" -- but
    neither ``backbone_conformal_study.py``'s nor ``multitarget_experiment.py``'s
    own per-target metric CSV computes this (only marginal per-target PICP).
    Recovered here directly from each condition's raw predictions (already
    saved, ``metric_type == "interval"`` rows carry ``lower``/``upper``/``Actual``
    per target/timestamp) via :func:`twiga.experiment.analysis.joint_interval_coverage`
    -- no rerun needed. ``group`` defaults to ``backbone_conformal_study``'s
    own single group; ``load_multitarget_full`` passes ``"conformal"`` instead,
    since that script's conditions live under a differently-named group.
    Returns an empty frame if no condition under this dataset/group has a
    saved ``predictions.parquet`` (e.g. a run predating ``save_predictions=True``).
    """
    dataset_dirname = slugify(dataset)  # matches the engine's own on-disk directory name for this dataset
    conditions_root = run_dir / "conditions" / dataset_dirname / group
    frames = [pd.read_parquet(path) for path in sorted(conditions_root.glob("*/predictions.parquet"))]
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    return joint_interval_coverage(predictions, group_cols=["condition", "Folds", "timestamp"])


def load_backbone_conformal_full(
    datasets: list[str],
) -> tuple[pd.DataFrame, list[tuple[str, str, str]], pd.DataFrame]:
    """Latest-run ``ch6_backbone_conformal_full.csv`` rows for ``datasets``, point+interval merged wide.

    Reuses ``backbone_conformal_study.discover_runs`` for run discovery
    (that module mirrors ``fpqr_gating_ablation.py``'s own
    ``discover_runs``/``latest_run_for_group``, not
    ``multitarget_experiment.py``'s hand-rolled ``_discover_meta_runs``), but
    selects by *dataset* rather than by group: the study has one Condition
    group (``"backbone"``, every condition name ``"{backbone}_{conformal}"``)
    run once per dataset via a separate ``ExperimentEngine`` call, so a run
    is looked up by ``dataset in run["datasets"]`` instead.

    Args:
        datasets: Dataset keys to load (a subset of
            ``backbone_conformal_study.DATASETS``).

    Returns:
        The merged wide frame (empty if nothing was found), a coverage log
        of ``(dataset, run_dir_name, run_timestamp)`` for datasets that were
        found, and the same run's ``ch6_backbone_conformal_representative.csv``
        rows (empty frame if none exist), for :func:`plot_good_bad_day`.
    """
    runs = bcs.discover_runs()
    frames, coverage, representative_frames = [], [], []
    for dataset in datasets:
        candidates = [run for run in runs if dataset in run["datasets"]]
        if not candidates:
            print(f"[missing] dataset={dataset}: no run covers it")
            continue
        run = max(candidates, key=lambda r: r["timestamp"])
        csv_path = run["run_dir"] / "results" / "ch6_backbone_conformal_full.csv"
        if not csv_path.exists():
            print(f"[missing] dataset={dataset}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["dataset"] == dataset] if "dataset" in df.columns else df
        if df.empty:
            print(f"[missing] dataset={dataset}: 0 rows in {csv_path}")
            continue
        joint_coverage = _load_joint_coverage(run["run_dir"], dataset)
        if not joint_coverage.empty:
            df = df.merge(joint_coverage, on=["condition", "Folds"], how="left")
        frames.append(_merge_point_interval(df))
        coverage.append((dataset, run["run_dir"].name, run["timestamp"]))
        representative_path = run["run_dir"] / "results" / "ch6_backbone_conformal_representative.csv"
        if representative_path.exists():
            representative_df = pd.read_csv(representative_path)
            representative_frames.append(representative_df[representative_df["dataset"] == dataset])
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    representative = pd.concat(representative_frames, ignore_index=True) if representative_frames else pd.DataFrame()
    return _winsorize_nrmse(full), coverage, representative


# ── Backbone conformal decision: run discovery + loading ─────────────────────


def discover_backbone_conformal_decisions() -> list[str]:
    """Every decision name (lower-cased, e.g. ``"dss-uk"``) with a run directory under this root.

    Unlike :func:`discover_operational_decisions`/the old
    ``operational_decision.py`` layout this mirrors in spirit,
    ``backbone_conformal_decision.py`` (migrated to ``ExperimentEngine``
    2026-09-25) writes one *separate* nested run tree per decision --
    ``{root}/{decision}/backbone_conformal_decision___{decision}/runs/<run_id>/``
    -- rather than one shared top-level ``runs/*/meta.json`` whose own
    ``meta.json`` names every decision it covers (that top-level
    ``runs/<run_id>/`` directory exists too, written once by this script's
    own ``main()``, but only ever holds
    ``results/backbone_conformal_decision_config.json`` -- a run-config
    record, not any decision's actual results). Discovery here is therefore
    "which ``{decision}/...`` subfolder exists", not "which meta.json lists
    this decision".
    """
    if not BACKBONE_CONFORMAL_DECISION_ROOT.exists():
        return []
    names = []
    for p in BACKBONE_CONFORMAL_DECISION_ROOT.iterdir():
        if not p.is_dir() or p.name == "runs":
            continue
        pattern = f"*/runs/*/results/ch6_backbone_conformal_decision_{p.name}_full.csv"
        if next(p.glob(pattern), None) is not None:
            names.append(p.name)
    return sorted(names)


def _latest_backbone_conformal_decision_run(decision: str) -> tuple[dict, Path] | None:
    """The most recently timestamped run under ``{root}/{decision}/*/runs/*`` for one decision.

    One glob level deeper than :func:`_discover_meta_runs` handles: the real
    layout nests an experiment-slug directory (e.g.
    ``backbone_conformal_decision___dss-uk``) between the decision folder
    and ``runs/``, so ``_discover_meta_runs(root / decision)`` -- which only
    globs ``runs/*/meta.json`` directly under its argument -- never finds
    anything.
    """
    candidates = [
        (json.loads(meta_path.read_text()), meta_path.parent)
        for meta_path in (BACKBONE_CONFORMAL_DECISION_ROOT / decision.lower()).glob("*/runs/*/meta.json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0]["timestamp"])


def load_backbone_conformal_decision_full(decisions: list[str]) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Latest-run ``ch6_backbone_conformal_decision_{decision}_full.csv`` rows for ``decisions``, pivoted wide.

    Real schema, confirmed directly against a completed run (not assumed):
    ``tier, metric, target, model, value, Folds, metric_type, condition,
    group, dataset`` -- ``condition`` (e.g. ``"mlpf_none"``,
    ``"nhits_mahalanobis"``) is already its own column, so unlike an
    earlier version of this function there is no ``"model"`` column to
    rename into it (renaming an already-present column onto itself would
    silently create a duplicate-named column instead); ``Folds`` is already
    present under that exact capitalised name, not a lowercase ``"fold"``
    needing ``pd.factorize``.

    Args:
        decisions: Decision names (from
            :func:`discover_backbone_conformal_decisions`).

    Returns:
        The pivoted wide frame (empty if nothing was found), and a coverage
        log of ``(decision, run_dir_name, run_timestamp)``.
    """
    frames, coverage = [], []
    for decision in decisions:
        found = _latest_backbone_conformal_decision_run(decision)
        if found is None:
            print(f"[missing] decision={decision}: no run found")
            continue
        meta, run_dir = found
        csv_path = run_dir / "results" / f"ch6_backbone_conformal_decision_{decision.lower()}_full.csv"
        if not csv_path.exists():
            print(f"[missing] decision={decision}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df["dataset"] = decision
        df["group"] = "backbone_conformal_decision"

        keep = df[df["metric"].isin(BACKBONE_CONFORMAL_DECISION_METRIC_COLS)].copy()
        pivot = keep.pivot_table(
            index=["dataset", "group", "condition", "Folds"],
            columns="metric",
            values="value",
            aggfunc="mean",
        ).reset_index()
        pivot = _ensure_metric_columns(pivot, BACKBONE_CONFORMAL_DECISION_METRIC_COLS)
        pivot["timestamp"] = pivot["Folds"]
        frames.append(pivot)
        coverage.append((decision, run_dir.name, meta["timestamp"]))
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return full, coverage


def discover_multitarget_decisions() -> list[str]:
    """Every decision name with a run directory under ``MULTITARGET_DECISION_ROOT``.

    Same layout as :func:`discover_backbone_conformal_decisions`
    (``{root}/{decision}/multitarget_decision___{decision}/runs/<run_id>/``) --
    ``multitarget_decision.py`` reuses ``ExperimentEngine``'s own
    ``start_run``/output-directory convention identically.
    """
    if not MULTITARGET_DECISION_ROOT.exists():
        return []
    names = []
    for p in MULTITARGET_DECISION_ROOT.iterdir():
        if not p.is_dir() or p.name == "runs":
            continue
        pattern = f"*/runs/*/results/ch6_multitarget_decision_{p.name}_full.csv"
        if next(p.glob(pattern), None) is not None:
            names.append(p.name)
    return sorted(names)


def _latest_multitarget_decision_run(decision: str) -> tuple[dict, Path] | None:
    """The most recently timestamped run under ``{root}/{decision}/*/runs/*`` for one decision."""
    candidates = [
        (json.loads(meta_path.read_text()), meta_path.parent)
        for meta_path in (MULTITARGET_DECISION_ROOT / decision.lower()).glob("*/runs/*/meta.json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0]["timestamp"])


def load_multitarget_decision_full(decisions: list[str]) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Latest-run ``ch6_multitarget_decision_{decision}_full.csv`` rows for ``decisions``, pivoted wide.

    Identical schema and pivot logic to
    :func:`load_backbone_conformal_decision_full` -- ``multitarget_decision.py``
    writes the same shape, just under its own filename/group/root.
    """
    frames, coverage = [], []
    for decision in decisions:
        found = _latest_multitarget_decision_run(decision)
        if found is None:
            print(f"[missing] decision={decision}: no run found")
            continue
        meta, run_dir = found
        csv_path = run_dir / "results" / f"ch6_multitarget_decision_{decision.lower()}_full.csv"
        if not csv_path.exists():
            print(f"[missing] decision={decision}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df["dataset"] = decision
        df["group"] = "multitarget_decision"

        keep = df[df["metric"].isin(MULTITARGET_DECISION_METRIC_COLS)].copy()
        pivot = keep.pivot_table(
            index=["dataset", "group", "condition", "Folds"],
            columns="metric",
            values="value",
            aggfunc="mean",
        ).reset_index()
        pivot = _ensure_metric_columns(pivot, MULTITARGET_DECISION_METRIC_COLS)
        pivot["timestamp"] = pivot["Folds"]
        frames.append(pivot)
        coverage.append((decision, run_dir.name, meta["timestamp"]))
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return full, coverage


# ── Operational decision: run discovery + loading ────────────────────────────


def discover_operational_decisions() -> list[str]:
    """Every decision name (lower-cased, e.g. ``"dss-uk"``) any run under ``OPERATIONAL_ROOT`` covers."""
    names: set[str] = set()
    for meta, _run_dir in _discover_meta_runs(OPERATIONAL_ROOT):
        names.update(decision.lower() for decision in meta.get("decisions", []))
    return sorted(names)


def _latest_operational_run(decision: str) -> tuple[dict, Path] | None:
    """The most recently timestamped operational-decision run covering ``decision`` (case-insensitive)."""
    candidates = [
        (meta, run_dir)
        for meta, run_dir in _discover_meta_runs(OPERATIONAL_ROOT)
        if decision.lower() in {d.lower() for d in meta.get("decisions", [])}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0]["timestamp"])


def load_operational_full(decisions: list[str]) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Latest-run ``{decision}_per_fold.csv`` rows for ``decisions``, pivoted wide.

    Same normalisation as :func:`load_copula_full` (``model`` -> ``condition``,
    ``fold`` string label -> integer ``Folds`` via ``pd.factorize``,
    ``metric`` pivoted into wide columns) -- this experiment's
    ``dss-uk_per_fold.csv``/``dss-us_per_fold.csv`` share ``copula_per_fold.csv``'s
    ``tier``/``metric``/``target``/``model``/``value``/``fold`` shape, just
    with ``tier`` always ``"operational"`` and a different metric set
    (Brier score, Brier skill score, reliability, resolution, event-rate
    error, Pr(event) predicted/realised).

    Args:
        decisions: Decision names (from :func:`discover_operational_decisions`).

    Returns:
        The pivoted wide frame (empty if nothing was found), and a coverage
        log of ``(decision, run_dir_name, run_timestamp)``.
    """
    frames, coverage = [], []
    for decision in decisions:
        found = _latest_operational_run(decision)
        if found is None:
            print(f"[missing] decision={decision}: no run found")
            continue
        meta, run_dir = found
        csv_path = run_dir / "results" / f"{decision}_per_fold.csv"
        if not csv_path.exists():
            print(f"[missing] decision={decision}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"model": "condition"})
        df["condition"] = df["condition"].map(_slugify_condition)
        df["dataset"] = decision
        df["group"] = "operational_decision"
        df["Folds"] = pd.factorize(df["fold"])[0] + 1

        keep = df[df["metric"].isin(OPERATIONAL_METRIC_COLS)].copy()
        pivot = keep.pivot_table(
            index=["dataset", "group", "condition", "Folds", "fold"],
            columns="metric",
            values="value",
            aggfunc="mean",
        ).reset_index()
        pivot = _ensure_metric_columns(pivot, OPERATIONAL_METRIC_COLS)
        pivot["timestamp"] = pivot["fold"]
        frames.append(pivot)
        coverage.append((decision, run_dir.name, meta["timestamp"]))
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return full, coverage


# ── Copula variants: run discovery + loading ─────────────────────────────────


def discover_copula_dataset_slugs() -> list[str]:
    """Every dataset-slug directory under ``COPULA_ROOT`` that has at least one run.

    Datasets are discovered from disk rather than hardcoded, since coverage
    (currently DSS-EU/DSS-UK/DSS-US/MLVS-PT-3PHASE) is still growing.
    """
    slugs = []
    if not COPULA_ROOT.exists():
        return slugs
    for candidate in sorted(COPULA_ROOT.iterdir()):
        if candidate.is_dir() and list(candidate.glob("runs/*/meta.json")):
            slugs.append(candidate.name)
    return slugs


def _latest_copula_run(slug: str) -> tuple[dict, Path] | None:
    """The most recently timestamped run directory for one copula dataset slug."""
    metas = []
    for meta_path in (COPULA_ROOT / slug).glob("runs/*/meta.json"):
        meta = json.loads(meta_path.read_text())
        metas.append((meta, meta_path.parent))
    if not metas:
        return None
    return max(metas, key=lambda pair: pair[0]["timestamp"])


def load_copula_full(dataset_slugs: list[str]) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Latest-run ``copula_per_fold.csv`` rows for ``dataset_slugs``, pivoted wide.

    ``model`` is renamed to ``condition`` (then underscore-slugified via
    :func:`_slugify_condition`) and ``dataset``/``group="copula_variants"``
    columns are added, matching the schema :func:`~analyse_results.fold_level`/
    ``metric_table`` expect. ``Folds`` is derived from the string ``fold``
    label (e.g. ``"2021-04..2021-05"``) via ``pd.factorize`` -- there is no
    integer fold index in ``copula_per_fold.csv``, unlike
    ``ch6_fpqr_gating_full.csv``'s own ``Folds`` column.

    ``tier``+``metric`` is pivoted into wide columns: ``Winkler``/``CRPS``
    from ``tier == "marginal"`` (averaged across the P/Q/V targets --
    ``copula_per_fold.csv`` has no single combined marginal score, so this is
    a per-fold mean over targets, matching this chapter's convention of
    reporting one number per condition rather than a three-target
    breakdown here), and ``Brier score`` from ``tier == "operational"``
    (already single-target, ``target == "S"``). ``picp`` is also pivoted
    (from ``"PICP (diagnostic)"``) for the metric table's diagnostic column,
    though see :func:`analyse_copula_dataset`'s note on why the formal
    Kupiec test is not run against it here.

    Args:
        dataset_slugs: Dataset-slug directory names (from
            :func:`discover_copula_dataset_slugs`).

    Returns:
        The pivoted wide frame (empty if nothing was found), and a coverage
        log of ``(slug, run_dir_name, run_timestamp)``.
    """
    frames, coverage = [], []
    for slug in dataset_slugs:
        found = _latest_copula_run(slug)
        if found is None:
            print(f"[missing] dataset={slug}: no run found")
            continue
        meta, run_dir = found
        csv_path = run_dir / "results" / "copula_per_fold.csv"
        if not csv_path.exists():
            print(f"[missing] dataset={slug}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"model": "condition"})
        df["condition"] = df["condition"].map(_slugify_condition)
        df["dataset"] = slug
        df["group"] = "copula_variants"
        df["Folds"] = pd.factorize(df["fold"])[0] + 1

        keep = df[
            ((df["tier"] == "marginal") & (df["metric"].isin(["Winkler", "CRPS", "PICP (diagnostic)"])))
            | ((df["tier"] == "operational") & (df["metric"] == "Brier score"))
        ].copy()
        keep["metric"] = keep["metric"].replace({"PICP (diagnostic)": "picp"})

        pivot = keep.pivot_table(
            index=["dataset", "group", "condition", "Folds", "fold"],
            columns="metric",
            values="value",
            aggfunc="mean",
        ).reset_index()
        pivot["timestamp"] = pivot["fold"]
        frames.append(pivot)
        coverage.append((slug, run_dir.name, meta["timestamp"]))
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return full, coverage


# ── Conclusion writing ────────────────────────────────────────────────────────


def write_conclusion(
    out_path: Path,
    *,
    dataset: str,
    group: str,
    reference: str,
    table: pd.DataFrame,
    dm_table: pd.DataFrame,
    mcs,
    decisive_metric: str,
    kupiec: pd.DataFrame,
    append: bool = False,
) -> dict:
    """Write (or append) one dataset's metric table, significance, and decision sentence.

    Decision rule (module docstring, restated): adopt the reference unless an
    alternative wins on ``decisive_metric`` at DM/MCS significance --

    - best condition (by ``weighted_borda``) == reference -> "no alternative unseats the reference."
    - best ranks above reference but DM ``p_adjusted >= 0.05`` -> "adopt reference; not significant."
    - DM significant but MCS retains more than one condition -> "adopt best provisionally; flag for a
      larger run."
    - DM significant and MCS retains only the best -> "adopt best at DM/MCS significance."

    Args:
        out_path: Markdown file to write. Appended to (with a blank-line
            separator) rather than overwritten when ``append`` is True, so
            multiple datasets within one group/experiment accumulate into a
            single ``conclusion.md`` in the order they were analysed.
        dataset: Dataset name/slug this section covers.
        group: Condition group / experiment name this section covers.
        reference: The reference (simplest) condition for this axis.
        table: :func:`~analyse_results.metric_table`'s output, sorted by
            ``weighted_borda`` descending.
        dm_table: :func:`~analyse_results.dm_and_mcs`'s pairwise DM table.
        mcs: :func:`~analyse_results.dm_and_mcs`'s Model Confidence Set.
        decisive_metric: The metric name the DM/MCS table was run on.
        kupiec: :func:`~analyse_results.kupiec_table`'s output, or an empty
            frame when the Kupiec test was not computed (noted as such).
        append: Append to an existing file instead of overwriting it.

    Returns:
        A small dict of decision facts (``best``, ``reference``, ``adopt``,
        ``p_adjusted``, ``mcs_included``) for the caller's own bookkeeping
        (e.g. ``index.md``'s summary table).
    """
    best = str(table.iloc[0]["condition"])
    included = sorted(str(name) for name in getattr(mcs, "included", []))

    row = dm_table[
        ((dm_table["condition_a"] == reference) & (dm_table["condition_b"] == best))
        | ((dm_table["condition_a"] == best) & (dm_table["condition_b"] == reference))
    ]
    p_adj = float(row["p_adjusted"].iloc[0]) if not row.empty else float("nan")

    if best == reference:
        decision = f"No alternative unseats the reference (**{display_name(reference)}**)."
        adopt = reference
    elif pd.isna(p_adj) or p_adj >= 0.05:
        decision = (
            f"**{display_name(best)}** ranks above the reference on the blended Borda score, but the "
            f"difference against **{display_name(reference)}** on `{decisive_metric}` is not significant "
            f"(DM p_adjusted={p_adj:.3f} >= 0.05); adopt the reference, **{display_name(reference)}**."
        )
        adopt = reference
    elif len(included) > 1:
        decision = (
            f"DM finds **{display_name(best)}** significantly different from the reference "
            f"**{display_name(reference)}** on `{decisive_metric}` (p_adjusted={p_adj:.3f}), but the Model "
            f"Confidence Set retains {len(included)} conditions "
            f"({', '.join(display_name(name) for name in included)}) -- no single clear winner; adopt "
            f"**{display_name(best)}** provisionally and flag for a larger run."
        )
        adopt = best
    else:
        decision = (
            f"**{display_name(best)}** is adopted at DM/MCS significance over the reference "
            f"**{display_name(reference)}** on `{decisive_metric}` (p_adjusted={p_adj:.3f}; the Model "
            f"Confidence Set retains only {display_name(best)})."
        )
        adopt = best

    lines = [
        f"## {dataset}",
        "",
        "### Metric table",
        "",
        _to_markdown(table.round(4)),
        "",
        f"### DM significance (BH-adjusted p-values, `{decisive_metric}`)",
        "",
        _to_markdown(dm_table.round(4)),
        "",
        f"Model Confidence Set (alpha=0.05): {', '.join(display_name(name) for name in included) or '(none)'}",
        "",
        "### Kupiec coverage test",
        "",
        _to_markdown(kupiec.round(4)) if not kupiec.empty else "Not computed -- see note below.",
        "",
        "### Decision",
        "",
        decision,
        "",
    ]
    mode = "a" if append and out_path.exists() else "w"
    with out_path.open(mode) as handle:
        if mode == "w":
            handle.write(f"# {display_name(group)}\n\n")
        handle.write("\n".join(lines))
        handle.write("\n")

    return {
        "dataset": dataset,
        "group": group,
        "reference": reference,
        "best": best,
        "adopt": adopt,
        "p_adjusted": p_adj,
        "mcs_included": included,
    }


def _finalise_conclusion(out_path: Path) -> None:
    """Append the shared caveats footer once, after every dataset section has been written."""
    with out_path.open("a") as handle:
        handle.write("\n" + CAVEATS_TEXT + "\n")


# ── Per-hypothesis analysis ───────────────────────────────────────────────────


def analyse_fpqr_group(full: pd.DataFrame, dataset: str, group: str, out_dir: Path, fig_dir: Path) -> dict:
    """Metric table, DM/MCS significance, Kupiec test, figures, and conclusion for one FPQR group/dataset.

    Args:
        full: The merged frame from :func:`load_fpqr_full`.
        dataset: Dataset name to scope to.
        group: FPQR condition group to scope to (one of ``FPQR_GROUPS``).
        out_dir: Directory for this group's CSVs/conclusion (already exists).
        fig_dir: Directory for this group's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    fold_df = fold_level(full, dataset, group, metric_cols=FPQR_METRIC_COLS)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] {dataset} / {group}: insufficient data to compare")
        return {}

    print(f"\n=== fpqr_gating_ablation / {group} / {dataset} ===")
    table = metric_table(
        fold_df, metric_cols=FPQR_METRIC_COLS, lower_better=FPQR_LOWER_BETTER, higher_better=FPQR_HIGHER_BETTER
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{dataset}_metrics.csv", index=False)

    fold_df = _drop_degenerate_conditions(fold_df, FPQR_DECISIVE_METRIC, f"fpqr_gating_ablation/{dataset}")
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=FPQR_DECISIVE_METRIC)
    dm_table.to_csv(out_dir / f"{dataset}_significance.csv", index=False)

    kupiec = kupiec_table(full, dataset, group)
    kupiec.to_csv(out_dir / f"{dataset}_kupiec.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = FPQR_REFERENCE[group]
    figures = plot_metric_bars(
        fold_df,
        dataset,
        group,
        reference,
        metric_cols=FPQR_METRIC_COLS,
        metric_display=FPQR_METRIC_DISPLAY,
        lower_better=FPQR_LOWER_BETTER,
        higher_better=FPQR_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, dataset, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        dataset,
        group,
        reference,
        FPQR_METRIC_COLS,
        metric_display=FPQR_METRIC_DISPLAY,
        lower_better=FPQR_LOWER_BETTER,
        higher_better=FPQR_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{dataset}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    fig = plot_dm_significance_matrix(p_matrix, dataset, group)
    if fig is not None:
        fig.savefig(fig_dir / f"{dataset}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=dataset,
        group=group,
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=FPQR_DECISIVE_METRIC,
        kupiec=kupiec,
        append=True,
    )
    print(f"  wrote {out_dir}")
    return facts


def analyse_multitarget_group(
    full: pd.DataFrame, representative: pd.DataFrame, dataset: str, group: str, out_dir: Path, fig_dir: Path
) -> dict:
    """Metric table, DM/MCS significance, Kupiec test, figures, and conclusion for one benchmark group/dataset.

    Same shape as :func:`analyse_fpqr_group`, against ``MULTITARGET_*``
    configuration instead -- this is Experiment 1 (``group="benchmark"``:
    FPQR vs QR/BNN/Laplace), plus the same script's own ``"grid"`` and
    ``"conformal"`` groups.

    Args:
        full: The merged frame from :func:`load_multitarget_full`.
        representative: The representative-instance frame from
            :func:`load_multitarget_full`, for :func:`plot_good_bad_day`.
        dataset: Dataset name to scope to.
        group: Condition group to scope to (one of ``MULTITARGET_GROUPS``).
        out_dir: Directory for this group's CSVs/conclusion (already exists).
        fig_dir: Directory for this group's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    # "conformal" additionally carries joint_coverage (merged in by
    # load_multitarget_full) -- the other two groups calibrate nothing
    # jointly, so MULTITARGET_METRIC_COLS stays their own metric set.
    metric_cols = MULTITARGET_CONFORMAL_METRIC_COLS if group == "conformal" else MULTITARGET_METRIC_COLS
    higher_better = MULTITARGET_CONFORMAL_HIGHER_BETTER if group == "conformal" else MULTITARGET_HIGHER_BETTER
    metric_display = MULTITARGET_CONFORMAL_METRIC_DISPLAY if group == "conformal" else MULTITARGET_METRIC_DISPLAY

    fold_df = fold_level(full, dataset, group, metric_cols=metric_cols)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] {dataset} / {group}: insufficient data to compare")
        return {}

    print(f"\n=== multitarget_experiment / {group} / {dataset} ===")
    table = metric_table(
        fold_df,
        metric_cols=metric_cols,
        lower_better=MULTITARGET_LOWER_BETTER,
        higher_better=higher_better,
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{dataset}_metrics.csv", index=False)

    # Per-target breakdown (2026-09-29), supplementary to the table above --
    # that one is averaged across P/Q/V (aggregate_fold_metrics's own
    # group_columns=("condition","Folds") default folds "target" away along
    # with everything else). Significance/Kupiec/the conclusion stay pooled
    # (this project's own convention: DM needs a per-observation loss
    # differential, not a per-target-split one), so this is diagnostic only,
    # surfaced to catch exactly the kind of target-specific pattern the
    # pooled table hides -- e.g. one target's forecast having compressed
    # amplitude relative to actual while another doesn't, invisible in an
    # average across three physically different quantities.
    if "target" in full.columns:
        scoped = full.loc[(full["dataset"] == dataset) & (full["group"] == group)]
        by_target_fold = aggregate_fold_metrics(scoped, metric_cols, group_columns=("condition", "target", "Folds"))
        by_target = by_target_fold.groupby(["condition", "target"])[metric_cols].mean().round(4)
        print(f"\n--- {dataset} / {group}: per-target breakdown ---")
        print(by_target.to_string())
        by_target.to_csv(out_dir / f"{dataset}_metrics_by_target.csv")

    fold_df = _drop_degenerate_conditions(fold_df, MULTITARGET_DECISIVE_METRIC, f"multitarget_experiment/{dataset}")
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=MULTITARGET_DECISIVE_METRIC)
    dm_table.to_csv(out_dir / f"{dataset}_significance.csv", index=False)

    kupiec = kupiec_table(full, dataset, group)
    kupiec.to_csv(out_dir / f"{dataset}_kupiec.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = MULTITARGET_REFERENCE[group]
    figures = plot_metric_bars(
        fold_df,
        dataset,
        group,
        reference,
        metric_cols=metric_cols,
        metric_display=metric_display,
        lower_better=MULTITARGET_LOWER_BETTER,
        higher_better=higher_better,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, dataset, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        dataset,
        group,
        reference,
        metric_cols,
        metric_display=metric_display,
        lower_better=MULTITARGET_LOWER_BETTER,
        higher_better=higher_better,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{dataset}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    if not representative.empty:
        good_bad_fig = plot_good_bad_day(representative, dataset, group, metric="nrmse")
        if good_bad_fig is not None:
            good_bad_fig.savefig(fig_dir / f"{dataset}_good_bad_day.pdf", dpi=600, bbox_inches="tight")
            plt.close(good_bad_fig)

    fig = plot_dm_significance_matrix(p_matrix, dataset, group)
    if fig is not None:
        fig.savefig(fig_dir / f"{dataset}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=dataset,
        group=group,
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=MULTITARGET_DECISIVE_METRIC,
        kupiec=kupiec,
        append=True,
    )
    print(f"  wrote {out_dir}")
    return facts


def analyse_backbone_conformal(
    full: pd.DataFrame, representative: pd.DataFrame, dataset: str, out_dir: Path, fig_dir: Path
) -> dict:
    """Metric table, DM/MCS significance, Kupiec test, figures, and conclusion for one dataset.

    Same shape as :func:`analyse_multitarget_group`, against
    ``BACKBONE_CONFORMAL_*`` configuration -- the study has one Condition
    group (``BACKBONE_CONFORMAL_GROUP``, every condition name
    ``"{backbone}_{conformal}"``), so ``group`` is fixed rather than a
    parameter.

    Args:
        full: The merged frame from :func:`load_backbone_conformal_full`.
        representative: The representative-instance frame from
            :func:`load_backbone_conformal_full`, for :func:`plot_good_bad_day`.
        dataset: Dataset name to scope to.
        out_dir: Directory for this dataset's CSVs/conclusion (already exists).
        fig_dir: Directory for this dataset's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    group = BACKBONE_CONFORMAL_GROUP
    fold_df = fold_level(full, dataset, group, metric_cols=BACKBONE_CONFORMAL_METRIC_COLS)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] backbone_conformal_study / {dataset}: insufficient data to compare")
        return {}

    print(f"\n=== backbone_conformal_study / {dataset} ===")
    table = metric_table(
        fold_df,
        metric_cols=BACKBONE_CONFORMAL_METRIC_COLS,
        lower_better=BACKBONE_CONFORMAL_LOWER_BETTER,
        higher_better=BACKBONE_CONFORMAL_HIGHER_BETTER,
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{dataset}_metrics.csv", index=False)

    fold_df = _drop_degenerate_conditions(
        fold_df, BACKBONE_CONFORMAL_DECISIVE_METRIC, f"backbone_conformal_study/{dataset}"
    )
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=BACKBONE_CONFORMAL_DECISIVE_METRIC)
    dm_table.to_csv(out_dir / f"{dataset}_significance.csv", index=False)

    kupiec = kupiec_table(full, dataset, group)
    kupiec.to_csv(out_dir / f"{dataset}_kupiec.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = BACKBONE_CONFORMAL_REFERENCE
    figures = plot_metric_bars(
        fold_df,
        dataset,
        group,
        reference,
        metric_cols=BACKBONE_CONFORMAL_METRIC_COLS,
        metric_display=BACKBONE_CONFORMAL_METRIC_DISPLAY,
        lower_better=BACKBONE_CONFORMAL_LOWER_BETTER,
        higher_better=BACKBONE_CONFORMAL_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, dataset, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        dataset,
        group,
        reference,
        BACKBONE_CONFORMAL_METRIC_COLS,
        metric_display=BACKBONE_CONFORMAL_METRIC_DISPLAY,
        lower_better=BACKBONE_CONFORMAL_LOWER_BETTER,
        higher_better=BACKBONE_CONFORMAL_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{dataset}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    if not representative.empty:
        good_bad_fig = plot_good_bad_day(representative, dataset, group, metric="nrmse")
        if good_bad_fig is not None:
            good_bad_fig.savefig(fig_dir / f"{dataset}_good_bad_day.pdf", dpi=600, bbox_inches="tight")
            plt.close(good_bad_fig)

    fig = plot_dm_significance_matrix(p_matrix, dataset, group)
    if fig is not None:
        fig.savefig(fig_dir / f"{dataset}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=dataset,
        group=group,
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=BACKBONE_CONFORMAL_DECISIVE_METRIC,
        kupiec=kupiec,
        append=True,
    )
    print(f"  wrote {out_dir}")
    return facts


def analyse_backbone_conformal_decision(full: pd.DataFrame, decision: str, out_dir: Path, fig_dir: Path) -> dict:
    """Metric table, DM/MCS significance, figures, and conclusion for one backbone-conformal decision dataset.

    Same shape as :func:`analyse_operational_dataset` (no Kupiec test, same
    reasoning: ``{decision}_per_fold.csv`` is already fold-aggregated), with
    a different reference (``"mlpgam_none"``, this study's own uncalibrated
    baseline condition) and a wider condition set (12 backbone x conformal
    cells rather than 10 head/copula variants).

    Args:
        full: The pivoted frame from :func:`load_backbone_conformal_decision_full`.
        decision: Decision name (dataset) to scope to.
        out_dir: Directory for this dataset's CSVs/conclusion (already exists).
        fig_dir: Directory for this dataset's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    group = "backbone_conformal_decision"
    fold_df = fold_level(full, decision, group, metric_cols=BACKBONE_CONFORMAL_DECISION_METRIC_COLS)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] backbone_conformal_decision / {decision}: insufficient data to compare")
        return {}

    print(f"\n=== backbone_conformal_decision / {decision} ===")
    table = metric_table(
        fold_df,
        metric_cols=BACKBONE_CONFORMAL_DECISION_METRIC_COLS,
        lower_better=BACKBONE_CONFORMAL_DECISION_LOWER_BETTER,
        higher_better=BACKBONE_CONFORMAL_DECISION_HIGHER_BETTER,
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{decision}_metrics.csv", index=False)

    fold_df = _drop_degenerate_conditions(
        fold_df, BACKBONE_CONFORMAL_DECISION_DECISIVE_METRIC, f"backbone_conformal_decision/{decision}"
    )
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=BACKBONE_CONFORMAL_DECISION_DECISIVE_METRIC)
    dm_table.to_csv(out_dir / f"{decision}_significance.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = BACKBONE_CONFORMAL_DECISION_REFERENCE
    figures = plot_metric_bars(
        fold_df,
        decision,
        group,
        reference,
        metric_cols=BACKBONE_CONFORMAL_DECISION_METRIC_COLS,
        metric_display=BACKBONE_CONFORMAL_DECISION_METRIC_DISPLAY,
        lower_better=BACKBONE_CONFORMAL_DECISION_LOWER_BETTER,
        higher_better=BACKBONE_CONFORMAL_DECISION_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, decision, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        decision,
        group,
        reference,
        BACKBONE_CONFORMAL_DECISION_METRIC_COLS,
        metric_display=BACKBONE_CONFORMAL_DECISION_METRIC_DISPLAY,
        lower_better=BACKBONE_CONFORMAL_DECISION_LOWER_BETTER,
        higher_better=BACKBONE_CONFORMAL_DECISION_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{decision}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    fig = plot_dm_significance_matrix(p_matrix, decision, group)
    if fig is not None:
        fig.savefig(fig_dir / f"{decision}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=decision,
        group=group,
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=BACKBONE_CONFORMAL_DECISION_DECISIVE_METRIC,
        kupiec=pd.DataFrame(),
        append=False,
    )
    print(f"  wrote {out_dir}")
    return facts


def analyse_multitarget_decision(full: pd.DataFrame, decision: str, out_dir: Path, fig_dir: Path) -> dict:
    """Metric table, DM/MCS significance, figures, and conclusion for one multitarget-decision dataset.

    Identical shape to :func:`analyse_backbone_conformal_decision` -- same
    schema, same metric config (``OPERATIONAL_METRIC_COLS`` via
    ``MULTITARGET_DECISION_*``) -- varying only the group name/reference
    condition (``"laplace_max_norm"``, this script's own uncalibrated-head
    comparison point) and the condition set (20 head x conformal cells:
    8 each for fpqr_anchor_outer/qr, 2 each for bnn/laplace).

    Args:
        full: The pivoted frame from :func:`load_multitarget_decision_full`.
        decision: Decision name (dataset) to scope to.
        out_dir: Directory for this dataset's CSVs/conclusion (already exists).
        fig_dir: Directory for this dataset's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    group = "multitarget_decision"
    fold_df = fold_level(full, decision, group, metric_cols=MULTITARGET_DECISION_METRIC_COLS)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] multitarget_decision / {decision}: insufficient data to compare")
        return {}

    print(f"\n=== multitarget_decision / {decision} ===")
    table = metric_table(
        fold_df,
        metric_cols=MULTITARGET_DECISION_METRIC_COLS,
        lower_better=MULTITARGET_DECISION_LOWER_BETTER,
        higher_better=MULTITARGET_DECISION_HIGHER_BETTER,
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{decision}_metrics.csv", index=False)

    fold_df = _drop_degenerate_conditions(
        fold_df, MULTITARGET_DECISION_DECISIVE_METRIC, f"multitarget_decision/{decision}"
    )
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=MULTITARGET_DECISION_DECISIVE_METRIC)
    dm_table.to_csv(out_dir / f"{decision}_significance.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = MULTITARGET_DECISION_REFERENCE
    figures = plot_metric_bars(
        fold_df,
        decision,
        group,
        reference,
        metric_cols=MULTITARGET_DECISION_METRIC_COLS,
        metric_display=MULTITARGET_DECISION_METRIC_DISPLAY,
        lower_better=MULTITARGET_DECISION_LOWER_BETTER,
        higher_better=MULTITARGET_DECISION_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, decision, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        decision,
        group,
        reference,
        MULTITARGET_DECISION_METRIC_COLS,
        metric_display=MULTITARGET_DECISION_METRIC_DISPLAY,
        lower_better=MULTITARGET_DECISION_LOWER_BETTER,
        higher_better=MULTITARGET_DECISION_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{decision}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    fig = plot_dm_significance_matrix(p_matrix, decision, group)
    if fig is not None:
        fig.savefig(fig_dir / f"{decision}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=decision,
        group=group,
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=MULTITARGET_DECISION_DECISIVE_METRIC,
        kupiec=pd.DataFrame(),
        append=False,
    )
    print(f"  wrote {out_dir}")
    return facts


def analyse_operational_dataset(full: pd.DataFrame, decision: str, out_dir: Path, fig_dir: Path) -> dict:
    """Metric table, DM/MCS significance, figures, and conclusion for one operational-decision dataset.

    Same shape as :func:`analyse_copula_dataset` (no Kupiec test, same
    reasoning: ``{decision}_per_fold.csv`` is already fold-aggregated). The
    decisive metric is always ``"Brier score"`` here -- this experiment's
    entire purpose is the hosting-capacity/thermal-exceedance Brier score,
    unlike ``copula_variants`` where Brier score is only sometimes present.

    Args:
        full: The pivoted frame from :func:`load_operational_full`.
        decision: Decision name (dataset) to scope to.
        out_dir: Directory for this dataset's CSVs/conclusion (already exists).
        fig_dir: Directory for this dataset's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    fold_df = fold_level(full, decision, "operational_decision", metric_cols=OPERATIONAL_METRIC_COLS)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] operational_decision / {decision}: insufficient data to compare")
        return {}

    print(f"\n=== operational_decision / {decision} ===")
    table = metric_table(
        fold_df,
        metric_cols=OPERATIONAL_METRIC_COLS,
        lower_better=OPERATIONAL_LOWER_BETTER,
        higher_better=OPERATIONAL_HIGHER_BETTER,
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{decision}_metrics.csv", index=False)

    fold_df = _drop_degenerate_conditions(fold_df, OPERATIONAL_DECISIVE_METRIC, f"operational_decision/{decision}")
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=OPERATIONAL_DECISIVE_METRIC)
    dm_table.to_csv(out_dir / f"{decision}_significance.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = OPERATIONAL_REFERENCE
    figures = plot_metric_bars(
        fold_df,
        decision,
        "operational_decision",
        reference,
        metric_cols=OPERATIONAL_METRIC_COLS,
        metric_display=OPERATIONAL_METRIC_DISPLAY,
        lower_better=OPERATIONAL_LOWER_BETTER,
        higher_better=OPERATIONAL_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, decision, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        decision,
        "operational_decision",
        reference,
        OPERATIONAL_METRIC_COLS,
        metric_display=OPERATIONAL_METRIC_DISPLAY,
        lower_better=OPERATIONAL_LOWER_BETTER,
        higher_better=OPERATIONAL_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{decision}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    fig = plot_dm_significance_matrix(p_matrix, decision, "operational_decision")
    if fig is not None:
        fig.savefig(fig_dir / f"{decision}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=decision,
        group="operational_decision",
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=OPERATIONAL_DECISIVE_METRIC,
        kupiec=pd.DataFrame(),
        append=False,
    )
    print(f"  wrote {out_dir}")
    return facts


def analyse_copula_dataset(full: pd.DataFrame, dataset_slug: str, out_dir: Path, fig_dir: Path) -> dict:
    """Metric table, DM/MCS significance, figures, and conclusion for one copula-variants dataset.

    The decisive metric is ``"Brier score"`` (the operational-tier score
    ``COPULA_METRIC_COLS`` carries) when present, falling back to
    ``"Winkler"`` otherwise -- per the module docstring's decision rule, the
    Winkler table stays a secondary diagnostic when an operational metric is
    available to decide on.

    No Kupiec test is run here: ``copula_per_fold.csv`` is already
    fold-aggregated (one row per condition/fold/metric/target, not one row
    per timestamp), so there is no per-timestamp exception count to
    reconstruct the way :func:`~analyse_results.kupiec_table` does for
    ``ch6_fpqr_gating_full.csv`` -- pretending fold-count as timestamp-count
    would silently fabricate a test with the wrong sample size rather than
    report an honest "not computed".

    Args:
        full: The pivoted frame from :func:`load_copula_full`.
        dataset_slug: Dataset-slug directory name to scope to.
        out_dir: Directory for this dataset's CSVs/conclusion (already exists).
        fig_dir: Directory for this dataset's figures.

    Returns:
        :func:`write_conclusion`'s decision-facts dict, or ``{}`` if there
        was not enough data to analyse.
    """
    fold_df = fold_level(full, dataset_slug, "copula_variants", metric_cols=COPULA_METRIC_COLS)
    if fold_df.empty or fold_df["condition"].nunique() < 2:
        print(f"[skip] copula_variants / {dataset_slug}: insufficient data to compare")
        return {}

    print(f"\n=== copula_variants / {dataset_slug} ===")
    table = metric_table(
        fold_df, metric_cols=COPULA_METRIC_COLS, lower_better=COPULA_LOWER_BETTER, higher_better=COPULA_HIGHER_BETTER
    )
    print(table.round(4).to_string(index=False))
    table.to_csv(out_dir / f"{dataset_slug}_metrics.csv", index=False)

    decisive_metric = "Brier score" if fold_df["Brier score"].notna().any() else "Winkler"
    fold_df = _drop_degenerate_conditions(fold_df, decisive_metric, f"copula_variants/{dataset_slug}")
    dm_table, p_matrix, mcs = dm_and_mcs(fold_df, metric=decisive_metric)
    dm_table.to_csv(out_dir / f"{dataset_slug}_significance.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    reference = COPULA_REFERENCE
    figures = plot_metric_bars(
        fold_df,
        dataset_slug,
        "copula_variants",
        reference,
        metric_cols=COPULA_METRIC_COLS,
        metric_display=COPULA_METRIC_DISPLAY,
        lower_better=COPULA_LOWER_BETTER,
        higher_better=COPULA_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    save_metric_figures(figures, fig_dir, dataset_slug, kind="bar")
    metric_grid = plot_metric_bar_grid(
        fold_df,
        dataset_slug,
        "copula_variants",
        reference,
        COPULA_METRIC_COLS,
        metric_display=COPULA_METRIC_DISPLAY,
        lower_better=COPULA_LOWER_BETTER,
        higher_better=COPULA_HIGHER_BETTER,
        extra_acronyms=_EXTRA_ACRONYMS,
    )
    if metric_grid is not None:
        metric_grid.savefig(fig_dir / f"{dataset_slug}_metric_bars.pdf", dpi=480, bbox_inches="tight")
        plt.close(metric_grid)

    fig = plot_dm_significance_matrix(p_matrix, dataset_slug, "copula_variants")
    if fig is not None:
        fig.savefig(fig_dir / f"{dataset_slug}_significance_matrix.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    conclusion_path = out_dir / "conclusion.md"
    facts = write_conclusion(
        conclusion_path,
        dataset=dataset_slug,
        group="copula_variants",
        reference=reference,
        table=table,
        dm_table=dm_table,
        mcs=mcs,
        decisive_metric=decisive_metric,
        kupiec=pd.DataFrame(),
        append=False,
    )
    print(f"  wrote {out_dir}")
    return facts


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_fpqr_analysis(groups: list[str], dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested FPQR group, in ``FPQR_GROUPS``'s fixed order.

    Args:
        groups: Requested groups (already resolved from ``--group``, "all" expanded).
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``(group, dataset, conclusion_path)`` triples, in analysis order.
    """
    ordered_groups = [group for group in FPQR_GROUPS if group in groups]
    full, coverage = load_fpqr_full(ordered_groups)
    print("\n=== fpqr_gating_ablation coverage ===")
    for group, run_name, timestamp in coverage:
        print(f"  {group:12s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] fpqr_gating_ablation: nothing to analyse")
        return []

    results = []
    for group in ordered_groups:
        group_datasets = sorted(full.loc[full["group"] == group, "dataset"].unique())
        if "all" not in dataset_filter:
            group_datasets = [d for d in group_datasets if any(token.lower() in d.lower() for token in dataset_filter)]
        out_dir = OUT_DIR / "fpqr_gating_ablation" / group
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        # A stale conclusion.md from an earlier run (possibly covering a
        # different set of datasets) must not survive into this run's
        # append=True loop below, or its sections would linger alongside --
        # or interleave with -- this run's own.
        (out_dir / "conclusion.md").unlink(missing_ok=True)
        for dataset in group_datasets:
            facts = analyse_fpqr_group(full, dataset, group, out_dir, fig_dir)
            if facts:
                results.append((group, dataset, out_dir / "conclusion.md"))
        if group_datasets:
            _finalise_conclusion(out_dir / "conclusion.md")
    return results


def run_multitarget_analysis(groups: list[str], dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested multitarget-benchmark group, in ``MULTITARGET_GROUPS``'s fixed order.

    Args:
        groups: Requested groups (already resolved from ``--group``, "all" expanded).
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``(group, dataset, conclusion_path)`` triples, in analysis order.
    """
    ordered_groups = [group for group in MULTITARGET_GROUPS if group in groups]
    full, coverage, representative = load_multitarget_full(ordered_groups)
    print("\n=== multitarget_experiment coverage ===")
    for group, run_name, timestamp in coverage:
        print(f"  {group:12s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] multitarget_experiment: nothing to analyse")
        return []

    results = []
    for group in ordered_groups:
        group_datasets = sorted(full.loc[full["group"] == group, "dataset"].unique())
        if "all" not in dataset_filter:
            group_datasets = [d for d in group_datasets if any(token.lower() in d.lower() for token in dataset_filter)]
        out_dir = OUT_DIR / "multitarget_experiment" / group
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "conclusion.md").unlink(missing_ok=True)
        for dataset in group_datasets:
            facts = analyse_multitarget_group(full, representative, dataset, group, out_dir, fig_dir)
            if facts:
                results.append((group, dataset, out_dir / "conclusion.md"))
        if group_datasets:
            _finalise_conclusion(out_dir / "conclusion.md")
    return results


def run_backbone_conformal_analysis(dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested backbone-conformal-study dataset, in sorted dataset-name order.

    Args:
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``(BACKBONE_CONFORMAL_GROUP, dataset, conclusion_path)`` triples, in
        analysis order.
    """
    datasets = (
        list(bcs.DATASETS)
        if "all" in dataset_filter
        else [d for d in bcs.DATASETS if any(token.lower() in d.lower() for token in dataset_filter)]
    )
    full, coverage, representative = load_backbone_conformal_full(datasets)
    print("\n=== backbone_conformal_study coverage ===")
    for dataset, run_name, timestamp in coverage:
        print(f"  {dataset:16s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] backbone_conformal_study: nothing to analyse")
        return []

    results = []
    out_dir = OUT_DIR / "backbone_conformal_study" / BACKBONE_CONFORMAL_GROUP
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conclusion.md").unlink(missing_ok=True)
    found_datasets = sorted(full["dataset"].unique())
    for dataset in found_datasets:
        facts = analyse_backbone_conformal(full, representative, dataset, out_dir, fig_dir)
        if facts:
            results.append((BACKBONE_CONFORMAL_GROUP, dataset, out_dir / "conclusion.md"))
    if found_datasets:
        _finalise_conclusion(out_dir / "conclusion.md")
        normalized_fig = plot_normalized_comparison(
            full,
            BACKBONE_CONFORMAL_GROUP,
            metrics=[
                (BACKBONE_CONFORMAL_DECISIVE_METRIC, True),
                ("nrmse", True),
                ("picp", False),
            ],
            legend_title="Condition:",
            metric_display=BACKBONE_CONFORMAL_METRIC_DISPLAY,
        )
        if normalized_fig is not None:
            normalized_fig.savefig(fig_dir / "normalized_comparison.pdf", dpi=600, bbox_inches="tight")
            plt.close(normalized_fig)
    return results


def run_operational_analysis(dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested operational-decision dataset, in sorted decision-name order.

    Args:
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``("operational_decision", decision, conclusion_path)`` triples, in
        analysis order.
    """
    decisions = discover_operational_decisions()
    if "all" not in dataset_filter:
        decisions = [d for d in decisions if any(token.lower() in d.lower() for token in dataset_filter)]
    full, coverage = load_operational_full(decisions)
    print("\n=== operational_decision coverage ===")
    for decision, run_name, timestamp in coverage:
        print(f"  {decision:12s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] operational_decision: nothing to analyse")
        return []

    results = []
    for decision in decisions:
        out_dir = OUT_DIR / "operational_decision" / decision
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        facts = analyse_operational_dataset(full, decision, out_dir, fig_dir)
        if facts:
            _finalise_conclusion(out_dir / "conclusion.md")
            results.append(("operational_decision", decision, out_dir / "conclusion.md"))
    return results


def run_backbone_conformal_decision_analysis(dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested backbone-conformal-decision dataset, in sorted decision-name order.

    Args:
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``("backbone_conformal_decision", decision, conclusion_path)``
        triples, in analysis order.
    """
    decisions = discover_backbone_conformal_decisions()
    if "all" not in dataset_filter:
        decisions = [d for d in decisions if any(token.lower() in d.lower() for token in dataset_filter)]
    full, coverage = load_backbone_conformal_decision_full(decisions)
    print("\n=== backbone_conformal_decision coverage ===")
    for decision, run_name, timestamp in coverage:
        print(f"  {decision:12s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] backbone_conformal_decision: nothing to analyse")
        return []

    results = []
    for decision in decisions:
        out_dir = OUT_DIR / "backbone_conformal_decision" / decision
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        facts = analyse_backbone_conformal_decision(full, decision, out_dir, fig_dir)
        if facts:
            _finalise_conclusion(out_dir / "conclusion.md")
            results.append(("backbone_conformal_decision", decision, out_dir / "conclusion.md"))
    return results


def run_multitarget_decision_analysis(dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested multitarget-decision dataset, in sorted decision-name order.

    Mirrors :func:`run_backbone_conformal_decision_analysis` exactly, against
    ``multitarget_decision.py``'s own output tree.

    Args:
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``("multitarget_decision", decision, conclusion_path)`` triples, in
        analysis order.
    """
    decisions = discover_multitarget_decisions()
    if "all" not in dataset_filter:
        decisions = [d for d in decisions if any(token.lower() in d.lower() for token in dataset_filter)]
    full, coverage = load_multitarget_decision_full(decisions)
    print("\n=== multitarget_decision coverage ===")
    for decision, run_name, timestamp in coverage:
        print(f"  {decision:12s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] multitarget_decision: nothing to analyse")
        return []

    results = []
    for decision in decisions:
        out_dir = OUT_DIR / "multitarget_decision" / decision
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        facts = analyse_multitarget_decision(full, decision, out_dir, fig_dir)
        if facts:
            _finalise_conclusion(out_dir / "conclusion.md")
            results.append(("multitarget_decision", decision, out_dir / "conclusion.md"))
    return results


def run_copula_analysis(dataset_filter: list[str]) -> list[tuple[str, str, Path]]:
    """Fully analyse every requested copula-variants dataset, in sorted slug order.

    Args:
        dataset_filter: Requested dataset substrings (already resolved from
            ``--dataset``; ``["all"]`` matches everything discovered).

    Returns:
        ``("copula_variants", dataset_slug, conclusion_path)`` triples, in
        analysis order.
    """
    slugs = discover_copula_dataset_slugs()
    if "all" not in dataset_filter:
        slugs = [slug for slug in slugs if any(token.lower() in slug.lower() for token in dataset_filter)]
    full, coverage = load_copula_full(slugs)
    print("\n=== copula_variants coverage ===")
    for slug, run_name, timestamp in coverage:
        print(f"  {slug:24s} <- {run_name}  ({timestamp})")
    if full.empty:
        print("[skip] copula_variants: nothing to analyse")
        return []

    results = []
    for slug in slugs:
        out_dir = OUT_DIR / "copula_variants" / slug
        fig_dir = out_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        facts = analyse_copula_dataset(full, slug, out_dir, fig_dir)
        if facts:
            _finalise_conclusion(out_dir / "conclusion.md")
            results.append(("copula_variants", slug, out_dir / "conclusion.md"))
    return results


def write_index(entries: list[tuple[str, str, Path]]) -> None:
    """Write ``index.md`` last, linking every ``conclusion.md`` in the fixed analysis order."""
    lines = ["# Chapter 6 Analysis Index", ""]
    for group, dataset, path in entries:
        rel = path.relative_to(OUT_DIR)
        lines.append(f"- [{display_name(group)} / {dataset}]({rel.as_posix()})")
    index_path = OUT_DIR / "index.md"
    index_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {index_path}")


def main() -> None:
    """Parse CLI arguments and run the fixed-order FPQR-then-copula analysis."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "fpqr_gating_ablation",
            "multitarget_experiment",
            "copula_variants",
            "backbone_conformal_study",
            "backbone_conformal_decision",
            "multitarget_decision",
            "operational_decision",
        ],
    )
    parser.add_argument("--group", nargs="+", default=["all"], choices=["all", *FPQR_GROUPS, *MULTITARGET_GROUPS])
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument(
        "--no-latex", action="store_true", help="fall back to matplotlib's own text renderer (no LaTeX toolchain)"
    )
    args = parser.parse_args()

    S.configure(latex=not args.no_latex)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_experiments = {
        "fpqr_gating_ablation",
        "multitarget_experiment",
        "copula_variants",
        "backbone_conformal_study",
        "backbone_conformal_decision",
        "multitarget_decision",
        "operational_decision",
    }
    experiments = all_experiments if "all" in args.experiment else set(args.experiment)
    fpqr_groups = FPQR_GROUPS if "all" in args.group else [g for g in args.group if g in FPQR_GROUPS]
    multitarget_groups = (
        MULTITARGET_GROUPS if "all" in args.group else [g for g in args.group if g in MULTITARGET_GROUPS]
    )

    # Order is hardcoded, not CLI-controlled -- each experiment always
    # completes fully before the next starts, whichever order --experiment
    # listed them in. Sequence follows ANALYSIS_PLAN.md's own hypothesis
    # order: the FPQR mechanism ablation (H3) resolves what the marginal
    # architecture is before the benchmark (H1a) cites it, the benchmark
    # resolves point/probabilistic accuracy before the copula family
    # question (H2) builds on it. backbone_conformal_study/_decision are a
    # later follow-on (added 2026-09-21, not one of Chapter-6.tex's three
    # named experiments): they build on both the mechanism (fpqr_gating_
    # ablation) and dependence (copula_variants) conclusions, so they run
    # after both, and the calibrated decision result sits immediately before
    # operational_decision's own uncalibrated one so the two are read
    # together. operational_decision comes last since it is downstream of
    # every earlier condition choice.
    entries: list[tuple[str, str, Path]] = []
    if "fpqr_gating_ablation" in experiments:
        entries.extend(run_fpqr_analysis(fpqr_groups, args.dataset))
    if "multitarget_experiment" in experiments:
        entries.extend(run_multitarget_analysis(multitarget_groups, args.dataset))
    if "copula_variants" in experiments:
        entries.extend(run_copula_analysis(args.dataset))
    if "backbone_conformal_study" in experiments:
        entries.extend(run_backbone_conformal_analysis(args.dataset))
    if "backbone_conformal_decision" in experiments:
        entries.extend(run_backbone_conformal_decision_analysis(args.dataset))
    if "multitarget_decision" in experiments:
        entries.extend(run_multitarget_decision_analysis(args.dataset))
    if "operational_decision" in experiments:
        entries.extend(run_operational_analysis(args.dataset))

    if entries:
        write_index(entries)
    else:
        print("Nothing to analyse.")


if __name__ == "__main__":
    main()
