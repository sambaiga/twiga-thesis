"""Chapter 3: protocol-comparison analysis (single split vs expanding-window backtest).

Sized like ``chapters/ch3_methodology/cwe_validation/run_all.py`` -- one chapter-3 experiment's own orchestrator,
not a multi-experiment aggregator (``ch7_multitarget/shared/ch6_analyse_results.py``'s ~2000-line pattern does not
fit here).

Three outputs, matching the plan:

  protocol_agreement_table    Each model's rank under each protocol, side by side, plus the rank difference.
  metric_table_by_scenario    The five-metric table (``conditions.POINT_METRICS``) full-sample and, when raw
                               predictions are available (``true`` column named explicitly, see
                               :func:`extreme_scenario_mask`), restricted to the extreme-scenario rows only
                               (``true < 0`` or ``true`` at/above its own ~90th percentile).
  dm_gw_table                  DM (``twiga_thesis.analysis.dm_and_mcs``) and GW (``twiga_thesis.analysis.gw_test``)
                               pairwise significance on the Winkler-score-free point-loss metric this experiment
                               actually has -- NRMSE, the lower-is-better entry in ``POINT_METRICS`` -- both L1
                               (NRMSE itself is already an L2-style loss; MAE stands in for the L1 comparison).

Run:  uv run python chapters/ch3_methodology/best_practices/analysis.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chapters.ch3_methodology.best_practices.conditions import POINT_METRICS
from chapters.ch3_methodology.best_practices.protocol import DEFAULT_OUTPUT_ROOT, experiment_output_dir
from twiga_thesis.analysis import dm_and_mcs, fold_level, gw_test, metric_table

LOWER_BETTER = {"nrmse", "mae", "mape", "smape"}
HIGHER_BETTER: set[str] = set()
TARGET_VALUES = {"nbias": 0.0}


def load_protocol_summary(protocol: str, output_root: Path | None = None) -> pd.DataFrame:
    """Load one protocol's combined results CSV, written by ``experiment.py``."""
    path = experiment_output_dir(protocol, output_root) / f"results_{protocol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No results for protocol={protocol!r} at {path}; run experiment.py first")
    return pd.read_csv(path)


def protocol_agreement_table(
    single_split_folds: pd.DataFrame, expanding_folds: pd.DataFrame, metric: str = "nrmse"
) -> pd.DataFrame:
    """Each model's mean-rank under each protocol, side by side, plus how far the two protocols disagree.

    Args:
        single_split_folds: Fold-level frame for the single-split protocol (one row per condition; ``cv_folds=1``
            gives exactly one row per condition, so "mean rank" here is just that one fold's rank).
        expanding_folds: Fold-level frame for the expanding-window protocol.
        metric: The metric ranked on (lower is better, per ``LOWER_BETTER``).

    Returns:
        One row per condition: ``rank_single_split``, ``rank_expanding``, ``rank_delta``
        (``rank_single_split - rank_expanding``; positive means the single split ranks a model worse than the
        expanding backtest does).
    """
    single_rank = single_split_folds.groupby("condition")[metric].mean().rank(ascending=True)
    expanding_rank = expanding_folds.groupby("condition")[metric].mean().rank(ascending=True)
    table = pd.DataFrame({"rank_single_split": single_rank, "rank_expanding": expanding_rank}).reset_index()
    table["rank_delta"] = table["rank_single_split"] - table["rank_expanding"]
    return table.sort_values("rank_expanding")


def extreme_scenario_mask(true_values: pd.Series, lower_quantile_pct: float = 0.90) -> pd.Series:
    """Rows where net load is negative (export) or in its own top decile (peak import), per the plan's definition.

    Args:
        true_values: The ``actual`` column of a raw predictions frame (``forecaster.result.ForecastResult
            .to_dataframe()``'s own column name -- see that method's docstring for the full raw-prediction
            schema; ``timestamp``/``target``/``model``/``forecast``/``actual``).
        lower_quantile_pct: The percentile defining "peak"; 0.90 per the plan.

    Returns:
        Boolean mask, same index as ``true_values``.
    """
    threshold = true_values.quantile(lower_quantile_pct)
    return (true_values < 0) | (true_values >= threshold)


def metric_table_by_scenario(
    full_sample_folds: pd.DataFrame,
    raw_predictions: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Full-sample and (when raw predictions are supplied) extreme-scenario-only metric tables.

    Args:
        full_sample_folds: Fold-level frame (``twiga_thesis.analysis.fold_level``'s own output) with
            ``POINT_METRICS`` already computed over the whole test window by the engine.
        raw_predictions: Optional raw per-timestamp predictions frame with ``actual``/``forecast`` columns
            (``ForecastResult.to_dataframe()``'s schema), one condition's predictions concatenated with a
            ``condition``/``Folds`` column added -- only available when ``Condition.save_raw_predictions=True``
            was set for the run (not set by this experiment's own ``conditions.py``, so this branch stays unused
            until a caller reruns with it enabled and recomputes metrics on the masked rows directly, e.g. via
            ``twiga.core.metrics.point.get_pointwise_metrics``).

    Returns:
        ``{"full_sample": ..., "extreme_scenario": ...}`` -- the second key is only present when
        ``raw_predictions`` is given.
    """
    tables = {"full_sample": metric_table(full_sample_folds, POINT_METRICS, LOWER_BETTER, HIGHER_BETTER, TARGET_VALUES)}
    if raw_predictions is not None:
        from twiga.core.metrics.point import get_pointwise_metrics  # noqa: PLC0415

        masked = raw_predictions[extreme_scenario_mask(raw_predictions["actual"])]
        rows = []
        for (condition, fold), group in masked.groupby(["condition", "Folds"]):
            metrics = get_pointwise_metrics(
                y=group["actual"].to_numpy(), y_hat=group["forecast"].to_numpy(), metric_names=POINT_METRICS
            )
            metrics["condition"] = condition
            metrics["Folds"] = fold
            rows.append(metrics)
        extreme_folds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=full_sample_folds.columns)
        tables["extreme_scenario"] = metric_table(
            extreme_folds, POINT_METRICS, LOWER_BETTER, HIGHER_BETTER, TARGET_VALUES
        )
    return tables


def dm_gw_table(fold_frame: pd.DataFrame, metric: str = "nrmse") -> dict[str, object]:
    """DM (unconditional) and GW (conditional) pairwise significance on ``metric``, both reported per the plan."""
    dm_table, dm_p_matrix, mcs = dm_and_mcs(fold_frame, metric=metric)
    gw_table, gw_p_matrix = gw_test(fold_frame, metric=metric)
    return {
        "dm_table": dm_table,
        "dm_p_matrix": dm_p_matrix,
        "mcs": mcs,
        "gw_table": gw_table,
        "gw_p_matrix": gw_p_matrix,
    }


def main() -> None:
    """Load both protocols' results, print the three comparison tables."""
    output_root = DEFAULT_OUTPUT_ROOT
    single_split_summary = load_protocol_summary("single_split", output_root)
    expanding_summary = load_protocol_summary("expanding", output_root)

    for dataset in single_split_summary["dataset"].unique():
        single_split_folds = fold_level(single_split_summary, dataset, "baselines", POINT_METRICS)
        expanding_folds = fold_level(expanding_summary, dataset, "baselines", POINT_METRICS)

        print(f"\n{'=' * 78}\nPROTOCOL AGREEMENT -- {dataset}\n{'=' * 78}")
        print(protocol_agreement_table(single_split_folds, expanding_folds).to_string(index=False))

        for protocol, folds in (("single_split", single_split_folds), ("expanding", expanding_folds)):
            print(f"\n{'=' * 78}\nFIVE-METRIC TABLE -- {dataset} -- {protocol}\n{'=' * 78}")
            tables = metric_table_by_scenario(folds)
            print(tables["full_sample"].to_string(index=False))

            print(f"\nDM/GW significance (NRMSE) -- {dataset} -- {protocol}")
            significance = dm_gw_table(folds)
            print(significance["dm_table"].to_string(index=False))
            print(significance["gw_table"].to_string(index=False))
            print(significance["mcs"])


if __name__ == "__main__":
    main()
