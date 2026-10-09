r"""Good/bad-day forecast panels for a Chapter 5 group's conditions, from saved backtest output.

`qvn_pit.py` answers whether each `qvn_type` mechanism is calibrated on
average (PIT/reliability), from a single freshly-trained train/test split
per condition -- it has to retrain, since PIT needs the full learned
quantile array and `predictions.parquet` only ever carries the reduced
`forecast`/`lower`/`upper` form (see that script's own docstring). This
script does not have that problem: `fpqr_experiment.py`'s formal 10-fold
backtest already wrote exactly `forecast`/`lower`/`upper`/`Actual` for
every fold of every condition, plus a `metrics.parquet` with per-day
NRMSE/CRPS/Winkler already computed -- reused here as-is, with no
retraining and no new metric computation, following the same "reuse
what the engine already wrote" rule `qvn_pit.py` itself violates only
because PIT genuinely has no other source.

A day is picked once, pooled over every condition that has a completed
backtest for the requested dataset (mean per-day NRMSE from each
condition's own `metrics.parquet`, `metric_type == "point"` rows), so
every panel plots the same day and the comparison across `qvn_type`
mechanisms is fair -- the same principle `ch7_multitarget/forecast_examples.py`
uses for its own best/worst day. `twiga.core.plot.plot_forecast` draws
each panel directly from `predictions.parquet`'s own column names
(`Actual`, `forecast`, `lower`, `upper`), needing no renaming.

Three of the eight `qvn_type` conditions (`qvn_spline`, `qvn_umnn`, and
`qvn_attention` on MLVS-PT specifically) have never completed the formal
backtest -- confirmed directly, no `predictions.parquet` exists for them
under any run directory -- and are skipped here with a printed note
rather than silently dropped; `qvn_pit.py`'s PIT/reliability figures are
still the only diagnostic available for those.

Generalised beyond `qvn_type` via `--group` (`GROUP_CONDITIONS` below has
the condition list for each group this repo has run) -- e.g. `proposal`
(`prop_softmax`/`prop_horizon`/`prop_kumaraswamy`, the FPN/proposal-network
ablation). Not every group can produce panels this way: `proposal` was run
2026-09-13, before `save_predictions` existed on the engine (same gap
`README.md` already documents for `benchmark`'s original run), so no
`predictions.parquet` was ever written for it and this script will report
every condition missing rather than silently draw an empty panel -- getting
panels for that group means rerunning it
(`fpqr_experiment.py --group proposal --dataset all --skip-hpo`), same as
`benchmark`'s own fix.

Run:  uv run python experiment/chapter-5-fpqr/qvn_forecast_examples.py
      uv run python experiment/chapter-5-fpqr/qvn_forecast_examples.py --dataset MLVS-UK
      uv run python experiment/chapter-5-fpqr/qvn_forecast_examples.py --group proposal
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lets_plot import gggrid, ggsave
import pandas as pd
from twiga.core.plot import plot_forecast

script_dir = Path(__file__).resolve().parent
RUNS = script_dir / "results" / "chapter_5_fpqr" / "runs"
OUT = script_dir / "qvn_forecast_examples"

GROUP_CONDITIONS = {
    "qvn_type": [
        "qvn_relu",
        "qvn_softplus",
        "qvn_monotone",
        "qvn_monotone_cal",
        "qvn_lattice",
        "qvn_spline",
        "qvn_umnn",
        "qvn_attention",
    ],
    "proposal": ["prop_softmax", "prop_horizon", "prop_kumaraswamy"],
}
QVN_ORDER = GROUP_CONDITIONS["qvn_type"]  # kept for existing callers/imports


def latest_predictions(dataset: str, condition: str, group: str = "qvn_type") -> Path | None:
    """The newest run directory's `predictions.parquet` for this condition/dataset, if any completed."""
    hits = sorted(RUNS.glob(f"*/conditions/{dataset}/{group}/{condition}/predictions.parquet"))
    return hits[-1] if hits else None


def load(
    dataset: str, conditions: list[str], group: str = "qvn_type"
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Saved predictions/metrics per condition, and the conditions with no completed run."""
    preds, metrics, missing = {}, {}, []
    for condition in conditions:
        path = latest_predictions(dataset, condition, group)
        if path is None:
            missing.append(condition)
            continue
        preds[condition] = pd.read_parquet(path)
        metrics[condition] = pd.read_parquet(path.parent / "metrics.parquet")
    return preds, metrics, missing


def pick_days(metrics: dict[str, pd.DataFrame]) -> tuple[tuple[pd.Timestamp, int], tuple[pd.Timestamp, int]]:
    """Shared good/bad (timestamp, fold) key: mean per-day NRMSE pooled across every available condition.

    Only (timestamp, fold) pairs every condition has a row for are eligible, so
    the picked day is comparable across the whole panel rather than favouring
    whichever condition happened to have the most folds.
    """
    frames = []
    for name, m in metrics.items():
        point = m.loc[m["metric_type"] == "point", ["timestamp", "Folds", "nrmse"]].copy()
        point["condition"] = name
        frames.append(point)
    pooled = pd.concat(frames, ignore_index=True)
    grouped = pooled.groupby(["timestamp", "Folds"])["nrmse"]
    day_err = grouped.mean()
    complete = day_err[grouped.size() == len(metrics)]
    if complete.empty:
        raise RuntimeError("no (timestamp, fold) pair is shared by every requested condition")
    return complete.idxmin(), complete.idxmax()


def _day_rows(df: pd.DataFrame, timestamp: pd.Timestamp, fold: int, metric_type: str) -> pd.DataFrame:
    """Rows for one backtest day/fold/metric_type, shared by `day_frame` and the metric lookups below."""
    mask = (df["Folds"] == fold) & (df["metric_type"] == metric_type) & (df["timestamp"].dt.normalize() == timestamp)
    return df[mask]


def day_frame(preds: pd.DataFrame, timestamp: pd.Timestamp, fold: int) -> pd.DataFrame:
    """One condition's interval-typed rows (forecast/Actual/lower/upper) for one backtest day."""
    return _day_rows(preds, timestamp, fold, "interval").sort_values("timestamp")


def day_metric(metrics: pd.DataFrame, timestamp: pd.Timestamp, fold: int, metric_type: str, column: str) -> float:
    """One scalar metric value for one backtest day, or NaN if that day has no such row."""
    day = _day_rows(metrics, timestamp, fold, metric_type)
    return float(day[column].iloc[0]) if not day.empty else float("nan")


def main() -> None:
    """Load a group's saved backtest output, pick a shared good/bad day, and draw panels."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MLVS-PT")
    ap.add_argument("--group", default="qvn_type", choices=list(GROUP_CONDITIONS))
    ap.add_argument("--conditions", nargs="+", default=None, help="defaults to every condition in --group")
    args = ap.parse_args()
    conditions = args.conditions or GROUP_CONDITIONS[args.group]

    # qvn_type keeps its original flat path (already referenced by README.md
    # and prior output); every other group nests under its own name.
    out_dir = OUT / args.dataset if args.group == "qvn_type" else OUT / args.group / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, metrics, missing = load(args.dataset, conditions, args.group)
    if missing:
        print(f"no completed backtest for: {', '.join(missing)} (skipped, not silently dropped)")
    if not preds:
        raise SystemExit(
            f"no completed {args.group} backtest found for {args.dataset} -- "
            f"rerun: uv run python experiment/chapter-5-fpqr/fpqr_experiment.py "
            f"--group {args.group} --dataset all --skip-hpo"
        )

    (good_ts, good_fold), (bad_ts, bad_fold) = pick_days(metrics)
    print(f"{len(preds)} conditions with a completed backtest: {', '.join(preds)}")
    print(f"good day  {good_ts:%Y-%m-%d}  fold {good_fold}")
    print(f"bad day   {bad_ts:%Y-%m-%d}  fold {bad_fold}")

    rows = []
    for label, ts, fold in (("good", good_ts, good_fold), ("bad", bad_ts, bad_fold)):
        panels = []
        for condition in conditions:
            if condition not in preds:
                continue
            day = day_frame(preds[condition], ts, fold)
            m = metrics[condition]
            nrmse = day_metric(m, ts, fold, "point", "nrmse")
            crps = day_metric(m, ts, fold, "quantile", "crps")
            winkler = day_metric(m, ts, fold, "interval", "winkler_score")
            rows.append({"day": label, "condition": condition, "NRMSE": nrmse, "CRPS": crps, "Winkler": winkler})

            fig = plot_forecast(
                day,
                date_col="timestamp",
                lower_col="lower",
                upper_col="upper",
                title=f"{condition}  ({label} day, {ts:%Y-%m-%d})",
                subtitle=f"NRMSE {nrmse:.3f}, CRPS {crps:.2f}, Winkler {winkler:.1f}",
                y_label="NetLoad (kW)",
            )
            ggsave(fig, f"{label}_{condition}.png", path=str(out_dir))
            panels.append(fig)

        if panels:
            grid = gggrid(panels, ncol=min(len(panels), 3))
            ggsave(grid, f"overview_{label}_day.png", path=str(out_dir))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "forecast_examples_summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
