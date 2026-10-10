"""Shared tabulation and significance orchestration for experiment results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from twiga.core.metrics import mcs_ranking, pairwise_dm_table
from twiga.core.metrics.calibration import kupiec_pof_test
from twiga.core.metrics.stats import benjamini_hochberg, giacomini_white_from_losses
from twiga.experiment.analysis import aggregate_fold_metrics, rank_models


def fold_level(
    full: pd.DataFrame,
    dataset: str,
    group: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Aggregate timestamp-level results to one metric row per condition and fold."""
    required_columns = {"dataset", "group"}
    missing_columns = required_columns.difference(full.columns)
    if missing_columns:
        raise ValueError(f"Result frame is missing scope columns: {sorted(missing_columns)}")
    scoped = full.loc[(full["dataset"] == dataset) & (full["group"] == group)]
    return aggregate_fold_metrics(scoped, metric_cols)


def metric_table(
    fold_frame: pd.DataFrame,
    metric_cols: list[str],
    lower_better: set[str],
    higher_better: set[str],
    target_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Summarize fold metrics and rank conditions with diversity-weighted Borda scores.

    Metrics in ``target_values`` are ranked by distance from their target;
    other metrics must be listed in exactly one direction set. The returned
    table includes means, sample standard deviations, mean per-fold ranks,
    and the arithmetic, Borda, and diversity-weighted aggregate ranks.
    """
    target_values = {"picp": 0.90} if target_values is None else target_values
    configured_metrics = lower_better | higher_better | set(target_values)
    unconfigured_metrics = set(metric_cols).difference(configured_metrics)
    conflicting_metrics = lower_better & higher_better
    if unconfigured_metrics or conflicting_metrics:
        raise ValueError(
            "Metric directions must be configured exactly once; "
            f"unconfigured={sorted(unconfigured_metrics)}, conflicting={sorted(conflicting_metrics)}"
        )

    missing_columns = set(metric_cols).difference(fold_frame.columns)
    if missing_columns:
        raise ValueError(f"Fold frame is missing metric columns: {sorted(missing_columns)}")
    if "condition" not in fold_frame.columns or "Folds" not in fold_frame.columns:
        raise ValueError("Fold frame requires 'condition' and 'Folds' columns")

    mean_frame = fold_frame.groupby("condition")[metric_cols].mean()
    std_frame = fold_frame.groupby("condition")[metric_cols].std(ddof=1).rename(columns=lambda column: f"{column}_std")
    ranks = pd.DataFrame(index=mean_frame.index)
    for metric in metric_cols:
        values = fold_frame.pivot(index="Folds", columns="condition", values=metric)
        if metric in lower_better:
            metric_ranks = values.rank(axis=1, ascending=True)
        elif metric in higher_better:
            metric_ranks = values.rank(axis=1, ascending=False)
        else:
            metric_ranks = (values - target_values[metric]).abs().rank(axis=1, ascending=True)
        ranks[f"{metric}_rank"] = metric_ranks.mean()

    scores = mean_frame.copy()
    for metric, target in target_values.items():
        if metric in scores.columns:
            scores[metric] = (scores[metric] - target).abs()
    effective_lower_is_better = lower_better | (set(target_values) & set(metric_cols))
    aggregate_ranks = rank_models(scores, lower_is_better=effective_lower_is_better)
    aggregate_ranks = aggregate_ranks[["arith_rank", "borda_count", "weighted_borda"]]
    table = pd.concat([mean_frame, std_frame, ranks, aggregate_ranks], axis=1).reset_index()
    return table.sort_values("weighted_borda", ascending=False)


def dm_and_mcs(
    fold_frame: pd.DataFrame,
    alpha: float = 0.05,
    metric: str = "winkler_score",
    fold_column: str = "Folds",
    condition_column: str = "condition",
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Compute pairwise DM results, a symmetric adjusted-p matrix, and the MCS.

    The input must contain one loss value per condition and fold. Rows missing
    the selected loss are excluded together so every compared condition uses
    the same folds.
    """
    required_columns = {fold_column, condition_column, metric}
    missing_columns = required_columns.difference(fold_frame.columns)
    if missing_columns:
        raise ValueError(f"Fold frame is missing comparison columns: {sorted(missing_columns)}")

    complete = fold_frame.dropna(subset=[metric])
    loss_frame = complete.pivot(index=fold_column, columns=condition_column, values=metric).dropna()
    if loss_frame.shape[1] < 2:
        raise ValueError(f"At least two conditions with complete {metric} losses are required")

    losses = {str(condition): loss_frame[condition].to_numpy() for condition in loss_frame.columns}
    dm_table = pairwise_dm_table(losses, alpha=alpha)
    model_set = mcs_ranking(losses, alpha=alpha)
    conditions = list(losses)
    p_matrix = pd.DataFrame(np.nan, index=conditions, columns=conditions)
    for row in dm_table.itertuples(index=False):
        p_matrix.loc[row.condition_a, row.condition_b] = row.p_adjusted
        p_matrix.loc[row.condition_b, row.condition_a] = row.p_adjusted
    return dm_table, p_matrix, model_set


def gw_test(
    fold_frame: pd.DataFrame,
    alpha: float = 0.05,
    metric: str = "winkler_score",
    fold_column: str = "Folds",
    condition_column: str = "condition",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the all-pairs Giacomini-White conditional-accuracy table and a symmetric adjusted-p matrix.

    Same input contract as :func:`dm_and_mcs` (one loss value per condition and fold; rows missing the selected
    loss are excluded together so every compared condition uses the same folds), but tests *conditional*
    predictive accuracy rather than DM's unconditional one: :func:`twiga.core.metrics.stats.giacomini_white_from_losses`
    conditions on the lagged loss differential, so a model that is only occasionally much better is not washed
    out by DM-style averaging the way an unconditionally-equal-on-average model would be. ``twiga.core.metrics``
    has no ready-made pairwise GW table (only ``pairwise_dm_table``'s DM-specific one), so this function builds
    one directly from the per-pair primitive, mirroring ``pairwise_dm_table``'s own all-pairs/BH-correction
    structure rather than introducing a different convention.

    ``giacomini_white_from_losses`` is already a one-sided test (H1: the second series is conditionally more
    accurate); the raw one-sided p-value for each ordered pair is folded into a two-sided equivalent via
    ``2 * min(p, 1 - p)``, the same transform ``pairwise_dm_table`` applies to DM's own one-sided primitive,
    before Benjamini-Hochberg correction across all pairs.

    Args:
        fold_frame: One row per condition and fold, as returned by :func:`fold_level`.
        alpha: Target false-discovery rate for the Benjamini-Hochberg correction.
        metric: The loss column to compare; lower must be better.
        fold_column: Column identifying the fold.
        condition_column: Column identifying the condition.

    Returns:
        A tuple ``(gw_table, p_matrix)``: ``gw_table`` has one row per unordered pair with
        ``condition_a``, ``condition_b``, ``p_value``, ``p_adjusted``, ``significant``, and ``winner``
        (the condition with the lower mean loss), sorted by adjusted p-value; ``p_matrix`` is the
        symmetric adjusted-p-value matrix indexed and columned by condition name, matching
        :func:`dm_and_mcs`'s own ``p_matrix`` shape.

    Raises:
        ValueError: If required columns are missing or fewer than two conditions have complete losses.
    """
    required_columns = {fold_column, condition_column, metric}
    missing_columns = required_columns.difference(fold_frame.columns)
    if missing_columns:
        raise ValueError(f"Fold frame is missing comparison columns: {sorted(missing_columns)}")

    complete = fold_frame.dropna(subset=[metric])
    loss_frame = complete.pivot(index=fold_column, columns=condition_column, values=metric).dropna()
    if loss_frame.shape[1] < 2:
        raise ValueError(f"At least two conditions with complete {metric} losses are required")

    conditions = list(loss_frame.columns)
    pairs = [(conditions[i], conditions[j]) for i in range(len(conditions)) for j in range(i + 1, len(conditions))]
    raw_p_values = np.array(
        [
            float(
                giacomini_white_from_losses(
                    loss_frame[condition_a].to_numpy().reshape(-1, 1),
                    loss_frame[condition_b].to_numpy().reshape(-1, 1),
                    version="multivariate",
                )
            )
            for condition_a, condition_b in pairs
        ]
    )
    two_sided_p_values = np.clip(2 * np.minimum(raw_p_values, 1 - raw_p_values), 0.0, 1.0)
    _, adjusted_p_values = benjamini_hochberg(two_sided_p_values, alpha=alpha)

    rows = []
    pairwise = zip(pairs, two_sided_p_values, adjusted_p_values, strict=True)
    for (condition_a, condition_b), p_value, p_adjusted in pairwise:
        mean_a = float(loss_frame[condition_a].mean())
        mean_b = float(loss_frame[condition_b].mean())
        rows.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "p_value": float(p_value),
                "p_adjusted": float(p_adjusted),
                "significant": bool(p_adjusted < alpha),
                "winner": condition_b if mean_b < mean_a else condition_a,
            }
        )
    gw_table = pd.DataFrame(rows).sort_values("p_adjusted", ignore_index=True)

    p_matrix = pd.DataFrame(np.nan, index=conditions, columns=conditions)
    for row in gw_table.itertuples(index=False):
        p_matrix.loc[row.condition_a, row.condition_b] = row.p_adjusted
        p_matrix.loc[row.condition_b, row.condition_a] = row.p_adjusted
    return gw_table, p_matrix


def kupiec_table(
    full: pd.DataFrame,
    dataset: str,
    group: str,
    nominal_alpha: float = 0.10,
) -> pd.DataFrame:
    """Approximate the unconditional-coverage test from per-origin PICP rows.

    Exception counts are reconstructed per fold from mean PICP and the row
    count, then pooled by condition. This is an approximation when raw hit
    sequences are unavailable; callers should prefer an exact test whenever
    they retain those sequences.
    """
    scoped = full.loc[(full["dataset"] == dataset) & (full["group"] == group)]
    rows = []
    for condition, condition_frame in scoped.groupby("condition"):
        hit_parts = []
        for _, fold_frame in condition_frame.groupby("Folds"):
            fold_size = len(fold_frame)
            picp = fold_frame["picp"].mean()
            if pd.isna(picp):
                continue
            exception_count = int(round((1 - picp) * fold_size))
            hit_parts.append(np.concatenate((np.ones(exception_count), np.zeros(fold_size - exception_count))))
        hits = np.concatenate(hit_parts) if hit_parts else np.array([])
        if hits.size == 0:
            continue
        result = kupiec_pof_test(hits, alpha=nominal_alpha)
        rows.append(
            {
                "condition": condition,
                "n": int(hits.size),
                "observed_rate": result.detail["observed_rate"],
                "expected_rate": result.detail["expected_rate"],
                "statistic": result.statistic,
                "p_value": result.p_value,
                "calibrated": result.calibrated,
            }
        )
    columns = ["condition", "n", "observed_rate", "expected_rate", "statistic", "p_value", "calibrated"]
    return pd.DataFrame(rows, columns=columns)
