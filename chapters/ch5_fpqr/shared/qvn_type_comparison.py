r"""Consolidated comparison table for a Chapter 5 group: metrics, DM significance, and calibration.

`analyse_results.py` already writes three separate per-dataset CSVs for a
given group -- `{dataset}_{group}_metrics.csv` (NRMSE/CRPS/Winkler/CWE
stratified plus a Borda-count ranking), `{dataset}_{group}_significance.csv`
(pairwise Diebold-Mariano with Benjamini-Hochberg correction), and
`{dataset}_{group}_kupiec.csv` (unconditional coverage test). This script
does not recompute any of that -- reused as-is, per this session's "reuse
what the engine already wrote" convention -- it only joins the three into
one table per dataset and adds a DM win-count column, so a single significant
loss is not lost inside a blended average rank (the significance table is
the deciding evidence, not the Borda count -- see this repo's own
"significance table first, not avg-rank" convention).

Only conditions with a completed formal backtest appear. For `qvn_type`
(the default) that is 5 of 8 conditions on both datasets (`qvn_relu`,
`qvn_softplus`, `qvn_monotone`, `qvn_monotone_cal`, `qvn_lattice` --
`qvn_spline`/`qvn_umnn`/`qvn_attention` have not finished one; see
`README.md` item 4). `--group proposal` (the FPN/proposal-network ablation:
`prop_softmax`/`prop_horizon`/`prop_kumaraswamy`) has all three complete on
both datasets, from its original 2026-09-13 run.

`analyse_results.py`'s own `METRIC_COLS` (NRMSE/CRPS/Winkler/CWE-stratified/
PICP/NMPI/calibration-error) does not include the pinball loss, weighted
quantile loss, sharpness, or tail pinball loss (`metric_type == "quantile"`
rows) -- pulled in here two ways depending on what a group's run actually
saved: from each condition's own `metrics.parquet` when
`qvn_forecast_examples.py`'s `latest_predictions` finds one (fold-then-mean,
same aggregation `analyse_results.py::fold_level`/`metric_table` use for
every other metric here), falling back to `analyse_results.py`'s own
`load_csv`/`fold_level` against the run-level `ch5_fpqr_full.csv` otherwise
-- needed for `proposal`, whose 2026-09-13 run predates `save_predictions`
existing on the engine and so never wrote a per-condition `metrics.parquet`
at all, only the run-level aggregate (same gap `README.md` already documents
for `benchmark`'s original run). The per-condition path is preferred when
available since it does not depend on which run happens to be
most-recently-timestamped, and it is what caught `qvn_monotone` on MLVS-PT
only having completed in an older run than `qvn_type`'s current latest.

Run:  uv run python experiment/chapter-5-fpqr/qvn_type_comparison.py
      uv run python experiment/chapter-5-fpqr/qvn_type_comparison.py --group proposal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from twiga.experiment.analysis import aggregate_fold_metrics
from twiga.experiment.run_catalog import load_group_csv

from chapters.ch5_fpqr.qvn_forecast_examples import GROUP_CONDITIONS, latest_predictions

script_dir = Path(__file__).resolve().parent
RUNS_DIR = script_dir / "results" / "chapter_5_fpqr" / "runs"
ANALYSIS = script_dir / "results" / "analysis"
DATASETS = ["MLVS-PT", "MLVS-UK"]

METRIC_COLS = ["nrmse", "crps", "winkler_score", "cwe_stratified", "picp", "nmpi", "calibration_error"]
# quantile-typed rows only (metric_type == "quantile"); NaN on point/interval
# rows, so the mean below skips them.
QUANTILE_METRIC_COLS = ["pinball", "wql", "sharpness", "tail_pinball"]


def dm_win_counts(sig: pd.DataFrame) -> pd.DataFrame:
    """Per condition: how many pairwise DM comparisons it significantly won/lost."""
    sig = sig[sig["significant"]]
    wins = sig["winner"].value_counts()
    losers = sig.apply(lambda r: r["condition_b"] if r["winner"] == r["condition_a"] else r["condition_a"], axis=1)
    losses = losers.value_counts()
    conditions = set(sig["condition_a"]) | set(sig["condition_b"])
    return pd.DataFrame(
        {
            "condition": sorted(conditions),
            "dm_significant_wins": [int(wins.get(c, 0)) for c in sorted(conditions)],
            "dm_significant_losses": [int(losses.get(c, 0)) for c in sorted(conditions)],
        }
    )


def quantile_metrics(dataset: str, group: str, conditions: list[str]) -> pd.DataFrame:
    """Pinball/WQL/sharpness/tail-pinball per condition: per-condition metrics.parquet, or the group's full CSV."""
    rows = []
    for condition in conditions:
        path = latest_predictions(dataset, condition, group)
        if path is None:
            continue
        m = pd.read_parquet(path.parent / "metrics.parquet")
        fold_means = m.groupby("Folds")[QUANTILE_METRIC_COLS].mean()
        rows.append({"condition": condition, **fold_means.mean().to_dict()})
    if rows:
        return pd.DataFrame(rows)

    # No per-condition metrics.parquet anywhere for this group (predates
    # save_predictions) -- fall back to the run-level aggregate.
    full, _coverage, missing = load_group_csv(RUNS_DIR, [group], "ch5_fpqr_full.csv")
    for issue in missing:
        print(f"[missing] {issue}")
    if full.empty:
        return pd.DataFrame(columns=["condition", *QUANTILE_METRIC_COLS])
    scoped = full.loc[(full["dataset"] == dataset) & (full["group"] == group)]
    fold_df = aggregate_fold_metrics(scoped, QUANTILE_METRIC_COLS)
    return fold_df.groupby("condition")[QUANTILE_METRIC_COLS].mean().reset_index()


def build_table(dataset: str, group: str, conditions: list[str]) -> pd.DataFrame:
    """One consolidated table for a dataset: metrics + DM win/loss counts + Kupiec calibration."""
    metrics = pd.read_csv(ANALYSIS / f"{dataset}_{group}_metrics.csv")
    significance = pd.read_csv(ANALYSIS / f"{dataset}_{group}_significance.csv")
    kupiec = pd.read_csv(ANALYSIS / f"{dataset}_{group}_kupiec.csv")[["condition", "calibrated"]]

    table = metrics[["condition", *METRIC_COLS, "borda_count", "weighted_borda"]].copy()
    table = table.merge(quantile_metrics(dataset, group, conditions), on="condition", how="left")
    table = table.merge(dm_win_counts(significance), on="condition", how="left").fillna(
        {"dm_significant_wins": 0, "dm_significant_losses": 0}
    )
    table = table.merge(kupiec, on="condition", how="left")
    table["dataset"] = dataset
    return table.sort_values("weighted_borda", ascending=False).reset_index(drop=True)


def main() -> None:
    """Build and print the consolidated table for both datasets, and save one combined CSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="qvn_type", choices=list(GROUP_CONDITIONS))
    args = ap.parse_args()
    conditions = GROUP_CONDITIONS[args.group]

    tables = [build_table(ds, args.group, conditions) for ds in DATASETS]
    combined = pd.concat(tables, ignore_index=True)
    out_path = ANALYSIS / f"{args.group}_comparison_both_datasets.csv"
    combined.to_csv(out_path, index=False)

    for ds, table in zip(DATASETS, tables, strict=True):
        print(f"\n=== {ds}: {args.group} comparison ===")
        print(table.to_string(index=False))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
