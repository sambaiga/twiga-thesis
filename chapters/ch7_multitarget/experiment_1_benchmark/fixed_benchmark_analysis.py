"""Tables and significance tests for a ``fixed_benchmark.py`` run.

Reads each condition's saved ``metrics.parquet``/``predictions.parquet`` and
writes, per dataset:

  summary.csv       NRMSE, CRPS, Winkler, PICP, NMPI, CWE-stratified (the
                    engine's one-sided gate and the two-sided gate), energy
                    score and the thermal-event Brier score, per target
  significance.csv  Diebold-Mariano (BH-corrected) and Model Confidence Set
                    membership on per-day losses: Winkler and CRPS per
                    target, energy score and thermal Brier across targets

Significance comes first: a ranking on means alone is not a result.

Two-sided CWE-stratified is scored on the pooled interval predictions, not
per fold: a one-month fold holds about 30 windows, too few for a coverage
tolerance tighter than about 0.05. The tolerance is one standard error of
stratum coverage, ``sqrt(alpha (1 - alpha) deff / n)``, with the design
effect ``deff`` estimated from how strongly misses cluster within a window.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_analysis.py
    uv run python .../fixed_benchmark_analysis.py --run-root <experiments>/.../runs/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from twiga.core.metrics.interval import conditional_cwe

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[2]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from experiment.common.forecast_panels import draw  # noqa: E402
from experiment.common.significance import mcs_ranking, pairwise_dm_table  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(script_dir.parent.parent))
from thesis_figures.style import configure, pad_limits  # noqa: E402

sys.path.insert(0, str(script_dir))
from fixed_benchmark import ALPHA, DEFAULT_OUTPUT_ROOT, FAN_LEVELS  # noqa: E402

N_STRATA = 10


def dataset_dirs(output_root: Path, run_root: Path | None) -> dict[str, Path]:
    """Each dataset's condition directory, from the newest run that holds it.

    fixed_benchmark.py runs one engine call per dataset, so every dataset lands
    in its own run directory; reading only the latest run would drop all but
    the last dataset. With ``run_root`` given, only that run is read.
    """
    if run_root is not None:
        runs = [run_root]
    else:
        experiment_root = output_root / "fixed_benchmark" / "multitarget_fixed_benchmark"
        # Run ids start with a timestamp, so reverse name order is newest first.
        runs = sorted((experiment_root / "runs").glob("*"), reverse=True)
    found: dict[str, Path] = {}
    for run in runs:
        for dataset_dir in sorted((run / "conditions").glob("*")):
            if dataset_dir.is_dir() and any(dataset_dir.glob("*/*/metrics.parquet")):
                found.setdefault(dataset_dir.name, dataset_dir)
    if not found:
        raise SystemExit(f"no run with results found (output root {output_root})")
    return found


def load_condition(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A condition's metrics and predictions frames."""
    return pd.read_parquet(path / "metrics.parquet"), pd.read_parquet(path / "predictions.parquet")


def _window_hits(interval: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Interval rows of one target with a lead-time column, and their (windows, horizon) hit matrix.

    Windows are keyed by (fold, forecast origin) and leads by timestamp order
    within a window, so the matrix does not depend on row order in the file.
    """
    rows = interval.sort_values(["Folds", "forecast_origin", "timestamp"]).copy()
    rows["lead"] = rows.groupby(["Folds", "forecast_origin"]).cumcount()
    rows["hit"] = ((rows["Actual"] >= rows["lower"]) & (rows["Actual"] <= rows["upper"])).astype(float)
    hits = rows.pivot_table(index=["Folds", "forecast_origin"], columns="lead", values="hit").dropna().to_numpy()
    return rows, hits


def coverage_tolerance(hits: np.ndarray) -> float:
    """One standard error of a stratum's coverage, inflated by within-window clustering.

    Args:
        hits: (windows, steps) array of 0/1 coverage indicators for one stratum.
    """
    n_windows, m = hits.shape
    p = hits.mean()
    # Variance of the per-window hit rate against its value under independence.
    clustered = p not in (0.0, 1.0) and m >= 2 and n_windows >= 2
    deff = max(1.0, hits.mean(axis=1).var(ddof=1) / (p * (1 - p) / m)) if clustered else 1.0
    return float(np.sqrt(ALPHA * (1 - ALPHA) * deff / (n_windows * m)))


def two_sided_cwe(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-target CWE-stratified with the one-sided and the two-sided gate, plus the tolerance used."""
    interval = preds[preds["metric_type"] == "interval"]
    out = []
    for target, g in interval.groupby("target"):
        rows, hits = _window_hits(g)
        tol = float(
            np.median([coverage_tolerance(hits[:, s]) for s in np.array_split(np.arange(hits.shape[1]), N_STRATA)])
        )
        y, lo, hi, lead = (rows[c].to_numpy(float) for c in ("Actual", "lower", "upper", "lead"))
        kw = {"strata": lead, "alpha": ALPHA, "n_bins": N_STRATA, "min_count": 1}
        out.append(
            {
                "target": target,
                "cwe_s_one_sided": conditional_cwe(y, lo, hi, **kw),
                "cwe_s_two_sided": conditional_cwe(y, lo, hi, **kw, tol=tol, tol_over=tol),
                "cwe_tolerance": tol,
            }
        )
    return pd.DataFrame(out)


def summarise(metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One row per target: the marginal metrics, then the joint scores repeated on each.

    ``crps`` is the engine's own, so it is defined identically for every row
    (a conformal condition's equals its head's); ``crps_window`` is the
    scorer's exact per-window value, present for the benchmark heads only.
    Brier skill uses the pooled base rate across folds.
    """
    by_type = dict(list(metrics.groupby("metric_type")))
    out = by_type["point"].groupby("target")[["nrmse"]].mean()
    out = out.join(by_type["interval"].groupby("target")[["winkler_score", "picp", "nmpi"]].mean())
    prob = by_type.get("quantile", by_type.get("parametric"))
    if prob is not None and "crps" in prob:
        out = out.join(prob.groupby("target")[["crps"]].mean())
    out = out.join(two_sided_cwe(preds).set_index("target"))
    window = by_type.get("window")
    if window is not None:
        out = out.join(window[window["target"] != "all"].groupby("target")[["crps"]].mean().add_suffix("_window"))
        joint = window[window["target"] == "all"]
        base = joint["event_rate_realised"].mean()
        brier = joint["brier"].mean()
        out["energy_score"] = joint["energy_score"].mean()
        out["brier"] = brier
        out["brier_skill"] = 1.0 - brier / (base * (1.0 - base)) if 0.0 < base < 1.0 else np.nan
        out["event_rate_predicted"] = joint["event_rate_predicted"].mean()
        out["event_rate_realised"] = base
    return out.reset_index()


def _series(metrics: pd.DataFrame, metric_type: str, column: str, target: str, key: str) -> pd.Series:
    """One loss per forecast day, keyed by (fold, day) so conditions align on the same days."""
    rows = metrics[(metrics["metric_type"] == metric_type) & (metrics["target"] == target)]
    return rows.set_index(["Folds", key])[column].sort_index()


def significance(per_condition: dict[str, pd.DataFrame], targets: list[str]) -> pd.DataFrame:
    """DM and MCS on per-day losses: Winkler and CRPS per target, energy score and Brier across targets.

    CRPS, energy and Brier come from the scorer's per-window rows, which only
    the benchmark heads carry: a conformal condition's would repeat its head's.
    """
    tables = []

    def run(label: str, target: str, series: dict[str, pd.Series]) -> None:
        series = {k: v.dropna() for k, v in series.items() if not v.dropna().empty}
        if len(series) < 2:
            return
        common = sorted(set.intersection(*(set(v.index) for v in series.values())))
        if len(common) < 2:
            return
        losses = {k: v.loc[common].to_numpy(float)[:, None] for k, v in series.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm = pairwise_dm_table(losses, h=1)
            mcs = mcs_ranking({k: v.ravel() for k, v in losses.items()})
        dm.insert(0, "target", target)
        dm.insert(0, "metric", label)
        dm["n"] = len(common)
        dm["mcs_set"] = ", ".join(sorted(mcs.included))
        tables.append(dm)

    windowed = {k: m for k, m in per_condition.items() if "window" in set(m["metric_type"])}
    for target in targets:
        run(
            "winkler",
            target,
            {k: _series(m, "interval", "winkler_score", target, "timestamp") for k, m in per_condition.items()},
        )
        run("crps", target, {k: _series(m, "window", "crps", target, "forecast_origin") for k, m in windowed.items()})
    for column in ("energy_score", "brier"):
        run(column, "all", {k: _series(m, "window", column, "all", "forecast_origin") for k, m in windowed.items()})
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _fan_frames(loaded: dict) -> dict[str, pd.DataFrame]:
    """Each benchmark head's saved fan rows, keyed by condition."""
    return {
        cond: preds[preds["metric_type"] == "fan"]
        for (group, cond), (_, preds) in loaded.items()
        if group == "benchmark" and "fan" in set(preds["metric_type"])
    }


def pick_days(fans: dict[str, pd.DataFrame]) -> tuple[tuple, tuple]:
    """The (fold, origin) windows with the lowest and highest median error, over every head and target.

    Error is the mean absolute error of the predictive median, normalised by
    each target's range so P, Q and V count equally, averaged over heads.
    Windows starting at midnight are preferred so the panels read 00 to 23.
    Windows holding a metering outage (any observation below 1% of that
    target's median magnitude, e.g. MLVS-PT on 27-28 March 2021, where P and Q
    read about 0.002 for hours) are skipped: their error measures the outage,
    not the forecast.
    """
    errors = []
    reference = next(iter(fans.values()))
    floor = reference.groupby("target")["Actual"].transform(lambda a: 0.01 * a.abs().median())
    outage = (reference["Actual"].abs() < floor).groupby([reference["Folds"], reference["forecast_origin"]]).any()
    for frame in fans.values():
        span = frame.groupby("target")["Actual"].transform(lambda a: max(np.ptp(a.to_numpy()), 1e-9))
        err = (
            ((frame["q0.50"] - frame["Actual"]).abs() / span).groupby([frame["Folds"], frame["forecast_origin"]]).mean()
        )
        errors.append(err)
    score = pd.concat(errors, axis=1).dropna().mean(axis=1)
    score = score[~outage.reindex(score.index, fill_value=False)]
    origins = pd.to_datetime(score.index.get_level_values("forecast_origin"))
    midnight = score[(origins.hour == 0) & (origins.minute == 0)]
    score = midnight if not midnight.empty else score
    return score.idxmin(), score.idxmax()


def plot_day(
    fans: dict[str, pd.DataFrame], window: tuple, title_stem: str, path: Path, names: dict[str, str] | None = None
) -> None:
    """Rows are targets, columns heads: the observed series over each head's predictive fan.

    ``names`` maps a condition to its column title; rows are labelled by target role (P, Q, V).
    """
    from fixed_benchmark_tables import ROLE  # noqa: PLC0415

    heads = list(fans)
    targets = sorted(next(iter(fans.values()))["target"].unique())
    fig, axes = plt.subplots(
        len(targets), len(heads), figsize=(2.6 * len(heads), 2.2 * len(targets)), squeeze=False, sharex=True
    )
    quantile_cols = [f"q{level:.2f}" for level in FAN_LEVELS]
    for row, target in enumerate(targets):
        panels = []
        for head in heads:
            frame = fans[head]
            day = frame[
                (frame["Folds"] == window[0]) & (frame["forecast_origin"] == window[1]) & (frame["target"] == target)
            ].sort_values("timestamp")
            panels.append(day)
        ylim = pad_limits(*[d[["Actual", *quantile_cols]].to_numpy() for d in panels])
        for col, (head, day) in enumerate(zip(heads, panels, strict=True)):
            ax = axes[row, col]
            t_axis = pd.to_datetime(day["timestamp"]).dt.tz_localize(None).to_numpy()
            draw(ax, t_axis, day["Actual"].to_numpy(), day[quantile_cols].to_numpy().T, FAN_LEVELS, 0, ylim=ylim)
            if row == 0:
                ax.set_title((names or {}).get(head, head), fontsize=9)
            if col == 0:
                ax.set_ylabel(f"${ROLE.get(target, target)}$")
            if row < len(targets) - 1:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_name(f"{path.name}{suffix}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {title_stem}: fold {window[0]}, {pd.Timestamp(window[1]).date()} -> {path.name}.png/.pdf")


def main() -> None:
    """Write summary.csv and significance.csv per dataset and print both."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()
    out_dir = (args.run_root or args.output_root / "fixed_benchmark" / "multitarget_fixed_benchmark") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        configure(latex=False)
    except Exception as exc:  # noqa: BLE001
        print(f"thesis style unavailable ({exc}); using matplotlib defaults")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    for dataset_dir in dataset_dirs(args.output_root, args.run_root).values():
        print(f"\n{dataset_dir.name}: {dataset_dir.parent.parent.name}")
        loaded = {
            (path.parent.name, path.name): load_condition(path)
            for path in sorted(dataset_dir.glob("*/*"))
            if (path / "metrics.parquet").exists()
        }
        if not loaded:
            continue
        summary = pd.concat(
            [summarise(m, p).assign(group=g, condition=c) for (g, c), (m, p) in loaded.items()], ignore_index=True
        )
        summary = summary[["group", "condition", *[c for c in summary.columns if c not in ("group", "condition")]]]
        summary.to_csv(out_dir / f"{dataset_dir.name}_summary.csv", index=False)
        targets = sorted(summary["target"].unique())

        sig = []
        for group in sorted({g for g, _ in loaded}):
            sig_group = significance({c: m for (g, c), (m, _) in loaded.items() if g == group}, targets)
            if not sig_group.empty:
                sig.append(sig_group.assign(group=group))
        significance_table = pd.concat(sig, ignore_index=True) if sig else pd.DataFrame()
        significance_table.to_csv(out_dir / f"{dataset_dir.name}_significance.csv", index=False)

        fans = _fan_frames(loaded)
        if fans:
            best, worst = pick_days(fans)
            print(f"\n== Forecast examples ({dataset_dir.name}) ==")
            plot_day(fans, best, "best day", out_dir / f"{dataset_dir.name}_forecast_best")
            plot_day(fans, worst, "worst day", out_dir / f"{dataset_dir.name}_forecast_worst")

        print(f"\n######## {dataset_dir.name} ########")
        if not significance_table.empty:
            print("\n== Model Confidence Set (90%) per group, metric, target ==")
            print(significance_table.groupby(["group", "metric", "target"])[["n", "mcs_set"]].first().to_string())
        print("\n== Summary ==")
        print(summary.round(4).to_string(index=False))
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
