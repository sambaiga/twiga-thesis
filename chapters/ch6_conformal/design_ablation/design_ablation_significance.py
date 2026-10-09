r"""Significance, effect-size, and forest-plot views for the Chapter 7 design ablation.

Generates, per dataset and for a chosen set of groups (``best_config`` and
``gating`` by default — two of the review comments this script answers: is the
"proposed configuration wins on MLVS-PT only" finding actually distinguishable
from noise, and is the gating trade-off claim backed by anything beyond a raw
percentage). Passing ``--group training_x_conformal`` additionally runs a
two-way ANOVA additivity check (see ``_factorial_interaction`` below), which
answers a third: ``Chapter-7.tex``'s methods section promises this factorial
"tests whether [training-schedule and conformal-score] effects are additive",
but the results section never reports the test.

1. A **pairwise DM significance matrix**, on per-fold Winkler score,
   Benjamini-Hochberg corrected across the group's pairs — never CWE,
   matching ``significance.py``'s rule and ``Chapter-7.tex``'s own stated
   protocol (the WS decides comparisons). Rendered via
    ``experiment/common/comparison_plots.py``'s ``plot_dm_significance_matrix``
   (matplotlib + ``thesis_figures``, shared with Chapter 5 and any future
   chapter that needs the same figure), not ``twiga.core.plot``'s lets-plot
   version this script used previously — every other Chapter 7 figure
   (calibration timing, design-ablation summary, forest plots, forecast
   examples) is matplotlib/``thesis_figures``-styled, and the lets-plot
   output looked visibly out of place (different palette, different
   "Twiga Forecast" watermark branding) next to them.
2. The **Model Confidence Set** at the same group, so a null DM result on one
   pair is not mistaken for "no significant difference anywhere in the group".
3. A **forest plot** (``experiment/common/comparison_plots.py``'s
   ``plot_forest``, same matplotlib/``thesis_figures`` convention) with
   bootstrap 95% CI per condition vs. the group's reference, one panel per
   metric.
4. A **Cohen's d effect-size table** for Winkler score, condition vs.
   reference, printed to the console — the module's own `enrich_summary`
   hardcodes CWE as the "primary" interval metric it enriches, which is the
   wrong metric for a chapter-7 decision, so this computes Winkler's d
   directly with the same underlying `bootstrap_ci`/`cohen_d` primitives.

Run:
    uv run python experiment/ch6_conformal/design_ablation/design_ablation_significance.py \\
        --run-dir experiment/ch6_conformal/results/20260913_083254_08cf63
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from experiment.common.comparison_plots import bootstrap_ci, plot_dm_significance_matrix, plot_forest  # noqa: E402
from experiment.common.significance import mcs_ranking, pairwise_dm_table
import numpy as np
import pandas as pd
from twiga.core.plot import cohen_d  # noqa: E402
from twiga.experiment.analysis import aggregate_fold_metrics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conditions import ABLATION_SPEC  # noqa: E402

DATASETS = ["MLVS-PT", "MLVS-UK"]
GROUPS = ["best_config", "gating"]

# Sourced from conditions.ABLATION_SPEC.reference_conditions (the single
# place these are defined) plus the "gating" override mlpgam_design_ablation.py
# applies on top of it, rather than hand-duplicating a subset here — a second,
# incomplete copy is what silently made every group but best_config/gating/
# training_x_conformal unrunnable via --group (KeyError on lookup below).
REFERENCE_CONDITIONS = {**ABLATION_SPEC.reference_conditions, "gating": "gate_default"}

# Same registry `twiga.experiment.analysis` uses, so column selection matches
# what actually produced the CSV.
POINT_COLS = ["mae", "rmse", "nrmse", "smape", "corr", "nbias"]
INTERVAL_COLS = ["cwe", "cwe_stratified", "picp", "nmpi", "winkler_score", "ace"]


def _fold_level(full: pd.DataFrame, dataset: str, group: str) -> pd.DataFrame:
    """One row per (condition, Folds): the correct resampling unit.

    ``ch7_design_ablation_full.csv`` has one row per (fold, timestamp), not
    per fold — bootstrapping or DM-testing on the raw rows would treat
    thousands of within-fold timestamps as independent samples. Averaging to
    one row per fold first is what `twiga.experiment.analysis.aggregate`
    itself does before computing the summary CSV's std columns; this mirrors
    that step so the numbers here agree with the summary.
    """
    sub = full[(full["dataset"] == dataset) & (full["group"] == group)]
    metric_cols = [c for c in POINT_COLS + INTERVAL_COLS if c in sub.columns]
    return aggregate_fold_metrics(sub, metric_cols)


def _winkler_effect_table(fold_df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Cohen's d + bootstrap CI for Winkler score, every condition vs. the group's reference."""
    ref_cond = REFERENCE_CONDITIONS[group]
    ref_vals = fold_df.loc[fold_df["condition"] == ref_cond, "winkler_score"].to_numpy()
    rng = np.random.default_rng(42)

    rows = []
    for cond, sub in fold_df.groupby("condition"):
        vals = sub["winkler_score"].to_numpy()
        lo, hi = bootstrap_ci(vals, rng=rng)
        d = 0.0 if cond == ref_cond else cohen_d(vals, ref_vals)
        rows.append(
            {
                "condition": cond,
                "winkler_mean": round(float(vals.mean()), 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "cohen_d_vs_ref": round(d, 3) if not np.isnan(d) else np.nan,
                "is_reference": cond == ref_cond,
            }
        )
    return pd.DataFrame(rows).sort_values("winkler_mean")


def _dm_and_mcs(fold_df: pd.DataFrame, group: str, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """DM pairwise table + square p-value matrix + MCS, on per-fold Winkler score."""
    conditions = sorted(fold_df["condition"].unique())
    pivot = fold_df.pivot(index="Folds", columns="condition", values="winkler_score").dropna()
    losses = {c: pivot[c].to_numpy() for c in conditions}

    dm_table = pairwise_dm_table(losses, alpha=alpha)
    mcs = mcs_ranking(losses, alpha=alpha)

    p_matrix = pd.DataFrame(np.nan, index=conditions, columns=conditions)
    for _, row in dm_table.iterrows():
        p_matrix.loc[row["condition_a"], row["condition_b"]] = row["p_adjusted"]
        p_matrix.loc[row["condition_b"], row["condition_a"]] = row["p_adjusted"]

    return dm_table, p_matrix, mcs


def _factorial_interaction(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Two-way ANOVA on Winkler score: training_mode x conformal_score, per fold.

    ``training_x_conformal`` condition names are ``f"{training_mode}_x_{conformal_score}"``
    (see ``FACTORIAL_TRAINING_MODES``/``FACTORIAL_CONFORMAL_SCORES`` in
    ``design_ablation/conditions.py``); "_x_" only ever appears as this separator, so a
    plain split is unambiguous. The interaction term's p-value is the direct
    answer to ``Chapter-7.tex``'s own stated question -- "tests whether their
    effects are additive" -- which the results section never actually reports.

    Returns:
        The type-II ANOVA table (``sum_sq``, ``df``, ``F``, ``PR(>F)``) for
        ``training_mode``, ``conformal_score``, and their interaction.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    df = fold_df.copy()
    parts = df["condition"].str.split("_x_", n=1, expand=True)
    df["training_mode"], df["conformal_score"] = parts[0], parts[1]

    model = ols("winkler_score ~ C(training_mode) * C(conformal_score)", data=df).fit()
    return sm.stats.anova_lm(model, typ=2)


def run(run_dir: Path, groups: list[str], datasets: list[str], out_dir: Path) -> None:
    """Generate significance/effect-size/forest-plot views for every (dataset, group) pair.

    Args:
        run_dir: Design-ablation run directory (must contain ``results/ch7_design_ablation_full.csv``).
        groups: Ablation groups to analyse.
        datasets: Datasets to analyse.
        out_dir: Directory to write figures and per-condition effect CSVs to.
    """
    full = pd.read_csv(run_dir / "results" / "ch7_design_ablation_full.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for group in groups:
            fold_df = _fold_level(full, dataset, group)
            if fold_df.empty:
                print(f"[skip] {dataset} / {group}: no rows")
                continue
            n_conditions = fold_df["condition"].nunique()
            if n_conditions < 2:
                print(f"[skip] {dataset} / {group}: only {n_conditions} condition(s), nothing to compare")
                continue

            print(f"\n=== {dataset} / {group} ===")

            # 1-2. Significance + MCS, on Winkler score.
            dm_table, p_matrix, mcs = _dm_and_mcs(fold_df, group)
            print(dm_table.to_string(index=False))
            print(f"MCS (alpha=0.10): {mcs.included}")

            # matplotlib's imshow has no analogue of lets-plot's
            # ContinuousTransformWithLimits crash on a degenerate (all-equal)
            # fill scale -- a uniform-p-value matrix just renders as one flat
            # colour, which is accurate (the DM rows above already say why)
            # rather than something to special-case around.
            sig_fig = plot_dm_significance_matrix(p_matrix, dataset, group)
            if sig_fig is None:
                print(f"  ({n_conditions} condition(s): nothing to compare, skipping heatmap)")
            else:
                sig_path = out_dir / f"{dataset}_{group}_significance.pdf"
                sig_fig.savefig(sig_path, bbox_inches="tight")
                sig_fig.savefig(sig_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
                print(f"  saved {sig_path}")

            # 3. Forest plot, Winkler score only (this chapter's deciding metric).
            forest_figures = plot_forest(
                fold_df, dataset, group, reference=REFERENCE_CONDITIONS[group], metric_cols=["winkler_score"]
            )
            forest_fig = forest_figures["winkler_score"]
            forest_path = out_dir / f"{dataset}_{group}_forest.pdf"
            forest_fig.savefig(forest_path, bbox_inches="tight")
            forest_fig.savefig(forest_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
            print(f"  saved {forest_path}")

            # 4. Cohen's d effect-size table for Winkler score.
            effect_table = _winkler_effect_table(fold_df, group)
            print(effect_table.to_string(index=False))
            effect_table.to_csv(out_dir / f"{dataset}_{group}_winkler_effect.csv", index=False)

            # 5. Factorial additivity check (training_x_conformal only).
            if group == "training_x_conformal":
                anova = _factorial_interaction(fold_df)
                print("\nTwo-way ANOVA (Winkler score ~ training_mode * conformal_score):")
                print(anova.to_string())
                anova.to_csv(out_dir / f"{dataset}_{group}_anova.csv")


def main() -> None:
    """Parse CLI args and run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Design-ablation run directory (contains results/)")
    parser.add_argument("--group", nargs="+", default=GROUPS)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=DATASETS,
        choices=[*DATASETS, "all"],
        help="Dataset(s) to analyse, or 'all' for every dataset in DATASETS.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <run-dir>/figures/significance")
    args = parser.parse_args()

    dataset_keys = DATASETS if "all" in args.dataset else args.dataset
    out_dir = args.out_dir or args.run_dir / "figures" / "significance"
    run(args.run_dir, args.group, dataset_keys, out_dir)


if __name__ == "__main__":
    main()
