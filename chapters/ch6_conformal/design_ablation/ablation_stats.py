"""Statistical protocol for MLPGAM-CRC rolling-origin ablation results.

Reproduces the analysis behind ``experiment/CRC_ABLATION_ANALYSIS.md``: for each
ablation group, a paired Wilcoxon signed-rank test against the group's reference
condition (FDR-corrected via Benjamini-Hochberg), a Kruskal-Wallis global test
across all conditions in the group, bootstrap 95% confidence intervals on the
mean, matched-pairs rank-biserial effect sizes, and fold-level coefficient of
variation (CoV) per condition.

Consumes the per-fold ``*_full.csv`` written by
:func:`twiga.experiment.analysis.aggregate` (one row per fold per condition),
e.g. ``results/mlpgam-crc_design_ablation/runs/<run_id>/results/crc_ablation_full.csv``.

Usage::

    python experiment/ch6_conformal/design_ablation/ablation_stats.py \
        --csv path/to/crc_ablation_full.csv --metric cwe
    python experiment/ch6_conformal/design_ablation/ablation_stats.py --csv path/to/crc_ablation_full.csv \
        --metric winkler_score --group conformal training_mode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon

# Mirrors ABLATION_SPEC.reference_conditions in design_ablation/conditions.py — kept in
# sync manually since this script is a standalone post-hoc analysis tool.
REFERENCE_CONDITIONS: dict[str, str] = {
    "training_mode": "two_stage_batch",
    "sigma_loss": "sigma_hybrid_sqrt",
    "monitor": "monitor_sigma",
    "lasso": "lasso_1e6",
    "conformal": "conf_90_res",
    "calib_isolation": "ci_batch",
    "best_config": "best_config",
    "alpha_level": "alpha_90_sign",
    "training_x_conformal": "two_stage_batch_x_sign",
}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (step-up procedure)."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size for the Wilcoxon signed-rank test."""
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank().to_numpy()
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom > 0 else 0.0


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean."""
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def coefficient_of_variation(x: np.ndarray) -> float:
    """Fold-level CoV (std / mean); nan if mean is zero or fewer than 2 samples."""
    if len(x) < 2:
        return float("nan")
    mean = x.mean()
    return float(x.std(ddof=1) / mean) if mean != 0 else float("nan")


def analyse_group(df: pd.DataFrame, group: str, metric: str, ref: str) -> pd.DataFrame:
    """Compute the full statistical protocol for one ablation group.

    Args:
        df: Long-form per-fold metrics (must have ``group``, ``condition``,
            ``dataset``, ``Folds``, and *metric* columns).
        group: Ablation group name (e.g. ``"training_mode"``).
        metric: Metric column to analyse (e.g. ``"cwe"``, ``"winkler_score"``).
        ref: Reference condition name within *group*.

    Returns:
        One row per condition: mean, bootstrap CI, CoV, Kruskal-Wallis
        group-level stat/p, and (for non-reference conditions) paired
        Wilcoxon p-value, FDR-corrected p-value, and rank-biserial effect size.
    """
    sub = df[df["group"] == group]
    conditions = sub["condition"].unique().tolist()

    samples = [sub[sub["condition"] == c][metric].dropna().to_numpy() for c in conditions]
    samples = [s for s in samples if len(s) > 0]
    kw_stat, kw_p = kruskal(*samples) if len(samples) > 1 else (float("nan"), float("nan"))

    ref_series = sub[sub["condition"] == ref].set_index(["dataset", "Folds"])[metric]

    rows = []
    for cond in conditions:
        cond_df = sub[sub["condition"] == cond]
        values = cond_df[metric].dropna().to_numpy()
        lo, hi = bootstrap_ci(values)

        row: dict[str, object] = {
            "group": group,
            "condition": cond,
            "n": len(values),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
            "cov": coefficient_of_variation(values),
            "kw_stat": kw_stat,
            "kw_p": kw_p,
            "wilcoxon_p": float("nan"),
            "effect_r": float("nan"),
            "n_paired": float("nan"),
        }

        if cond != ref:
            cond_series = cond_df.set_index(["dataset", "Folds"])[metric]
            paired = pd.concat([cond_series, ref_series], axis=1, join="inner").dropna()
            paired.columns = ["cond", "ref"]
            if len(paired) >= 5:
                stat, p = wilcoxon(paired["cond"], paired["ref"])
                row["wilcoxon_p"] = p
                row["effect_r"] = rank_biserial(paired["cond"].to_numpy(), paired["ref"].to_numpy())
            row["n_paired"] = len(paired)

        rows.append(row)

    out = pd.DataFrame(rows)
    mask = out["condition"] != ref
    if mask.any():
        out.loc[mask, "wilcoxon_p_fdr"] = bh_fdr(out.loc[mask, "wilcoxon_p"].to_numpy())
    else:
        out["wilcoxon_p_fdr"] = float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Path to the per-fold ablation *_full.csv")
    parser.add_argument("--metric", default="cwe", help="Metric column to analyse (default: cwe)")
    parser.add_argument("--group", nargs="+", default=None, help="Restrict to these groups (default: all present)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: alongside --csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f"Metric '{args.metric}' not found in {args.csv}. Available: {list(df.columns)}")

    groups = args.group or sorted(df["group"].unique())

    results = []
    for group in groups:
        ref = REFERENCE_CONDITIONS.get(group)
        available = df[df["group"] == group]["condition"].unique() if group in df["group"].values else []
        if ref is None or ref not in available:
            print(f"Skipping group='{group}' — no reference condition found in data")
            continue
        results.append(analyse_group(df, group, args.metric, ref))

    if not results:
        print("No groups analysed — check --group names against REFERENCE_CONDITIONS and the CSV's 'group' column.")
        return

    combined = pd.concat(results, ignore_index=True)
    out_path = Path(args.output) if args.output else Path(args.csv).with_name(f"ablation_stats_{args.metric}.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(combined.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
