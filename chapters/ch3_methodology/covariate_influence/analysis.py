"""Chapter 3, RQ2: covariate-ablation analysis -- NRMSE/CWE per condition per dataset, fold-then-mean.

Sized like ``cwe_validation/run_all.py``: one experiment's own orchestrator, not a multi-experiment aggregator.

Run:  uv run python chapters/ch3_methodology/covariate_influence/analysis.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chapters.ch3_methodology.covariate_influence.experiment import DATASETS, DEFAULT_OUTPUT_ROOT
from twiga_thesis.analysis import dm_and_mcs, fold_level, gw_test, metric_table

METRIC_COLS = ["nrmse", "cwe", "cwe_stratified", "picp", "nmpi"]
LOWER_BETTER = {"nrmse"}
HIGHER_BETTER = {"cwe", "cwe_stratified"}
TARGET_VALUES = {"picp": 0.90, "nmpi": 0.0}


def load_results(output_root: Path | None = None) -> pd.DataFrame:
    """Load the combined covariate-ablation results CSV written by ``experiment.py``."""
    root = output_root or DEFAULT_OUTPUT_ROOT
    path = root / "covariate_influence" / "results_covariate_influence.csv"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}; run experiment.py first")
    return pd.read_csv(path)


def condition_metric_table(results: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per-condition NRMSE/CWE/PICP/nMPI table (fold-then-mean) for one dataset."""
    folds = fold_level(results, dataset, "covariate_influence", METRIC_COLS)
    return metric_table(folds, METRIC_COLS, LOWER_BETTER, HIGHER_BETTER, TARGET_VALUES)


def significance(results: pd.DataFrame, dataset: str, metric: str = "nrmse") -> dict[str, object]:
    """DM and GW pairwise significance across the four covariate conditions, on ``metric``."""
    folds = fold_level(results, dataset, "covariate_influence", METRIC_COLS)
    dm_table, dm_p_matrix, mcs = dm_and_mcs(folds, metric=metric)
    gw_table, gw_p_matrix = gw_test(folds, metric=metric)
    return {
        "dm_table": dm_table,
        "dm_p_matrix": dm_p_matrix,
        "mcs": mcs,
        "gw_table": gw_table,
        "gw_p_matrix": gw_p_matrix,
    }


def main() -> None:
    """Print the per-dataset metric and significance tables."""
    results = load_results()
    for dataset in DATASETS:
        if dataset not in results["dataset"].unique():
            continue
        print(f"\n{'=' * 78}\nCOVARIATE INFLUENCE -- {dataset}\n{'=' * 78}")
        print(condition_metric_table(results, dataset).to_string(index=False))
        sig = significance(results, dataset)
        print("\nDM table:")
        print(sig["dm_table"].to_string(index=False))
        print("\nGW table:")
        print(sig["gw_table"].to_string(index=False))
        print(sig["mcs"])


if __name__ == "__main__":
    main()
