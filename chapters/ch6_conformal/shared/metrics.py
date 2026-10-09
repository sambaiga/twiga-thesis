"""Canonical metric lists for the Chapter 7 experiments.

``crc_experiment.py`` and ``crc_calib_improvement.py`` both request
``["cwe", "picp", "nmpi", "winkler_score", "ace"]`` — the marginal CWE only.
Chapter 3 establishes that the marginal form can fail to distinguish a
conditionally-adaptive interval from a constant-width one, and recommends the
stratified form wherever CWE is computed from that point onward; Chapter 5
predates that guidance and reports the marginal form alone. Chapter 7 reports
both, side by side, so a reader can see when they disagree rather than trusting
a single number that may be blind to exactly the horizon-adaptive behaviour
this chapter's method is built to produce.

``cwe_stratified`` is not new code — it already exists in
``twiga.core.metrics.interval.get_interval_metrics`` and is a valid
``metric_names``/``metrics=[...]`` entry for
:meth:`twiga.forecaster.core.TwigaForecaster.backtesting`. It was simply never
requested by the two existing scripts.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "INTERVAL_METRICS",
    "POINT_METRICS",
    "cwe_comparison",
]

# Same four point metrics the existing scripts request — kept identical so
# point-accuracy tables remain directly comparable to crc_experiment.py's.
POINT_METRICS = ["mae", "rmse", "smape", "corr"]

# The existing five, plus cwe_stratified. Winkler score, not either CWE, is
# what significance.py's design-choice tests run on — see that module's
# docstring. Stratification defaults to lead time (get_interval_metrics'
# behaviour when `strata` is not passed and the forecast array is
# multi-horizon), matching what Chapter 3 recommends CWE condition on.
INTERVAL_METRICS = ["cwe", "cwe_stratified", "picp", "nmpi", "winkler_score", "ace"]


def cwe_comparison(metrics: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Marginal against stratified CWE, side by side, with their gap.

    Args:
        metrics: A metrics table containing ``cwe`` and ``cwe_stratified``
            columns — e.g. the frame ``TwigaForecaster.backtesting()`` returns
            when ``INTERVAL_METRICS`` was requested, or a summary already
            aggregated across folds.
        group_cols: Columns identifying a row (e.g. ``["condition"]`` or
            ``["dataset", "condition"]``). When given, the two CWE columns are
            averaged within each group before comparing; when ``None`` every
            row of ``metrics`` is compared as-is.

    Returns:
        A frame with ``cwe`` (marginal), ``cwe_stratified``, and ``gap`` —
        ``cwe_stratified - cwe``. A large positive gap means the marginal score
        was blind to adaptive behaviour the stratified score credits; a large
        negative gap means the marginal score overstated a conditionally weak
        interval by averaging it with strata where it happens to fit well.

    Raises:
        KeyError: If either CWE column is missing from ``metrics``.
    """
    missing = [c for c in ("cwe", "cwe_stratified") if c not in metrics.columns]
    if missing:
        raise KeyError(f"metrics is missing {missing}; request INTERVAL_METRICS, not a partial list")

    frame = (
        metrics.groupby(group_cols)[["cwe", "cwe_stratified"]].mean().reset_index()
        if group_cols
        else metrics[["cwe", "cwe_stratified"]].copy()
    )
    frame["gap"] = frame["cwe_stratified"] - frame["cwe"]
    return frame
