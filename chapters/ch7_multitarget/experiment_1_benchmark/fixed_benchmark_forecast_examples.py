r"""Experiment 1 forecast examples: one good day per record, P, Q and V, for every head and its conformal variants.

Reads a fixed benchmark run (``fixed_benchmark.py``: the four heads and, on MLVS-PT, their conformal variants), or
the combined tuned set (``combine_tuned_benchmark.py``) with ``--fpqr tuned``. One day per record, good for every
head: among the whole calendar days every condition has, the day whose worst head has the highest joint hit rate
(the share of steps with P, Q and V all inside their 90% intervals), ties broken by the lowest Winkler score averaged
over the four heads, each target's divided by its mean absolute observation so P, Q and V weigh alike.
``--suggest`` lists the best candidates and ``--day`` pins a choice. Every condition is drawn on that same day, so
the panels compare like with like.

One single-plot figure per (record, condition, target), in the panel style of Chapter 7's
``calibration_forecast_example.py`` (``thesis_figures.line_style.forecast_panel``), with one y-range per (record,
target) across the four heads, and one per head across its native forecast (saved again as ``<head>_none``) and
its conformal variants, so interval widths read across the panels of one subfigure grid; no title and no model
label (the thesis's subfigure captions carry them, the file name carries the model).

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_forecast_examples.py \
        --experiment-root <experiments>/ch7_multitarget/fixed_benchmark/multitarget_fixed_benchmark
    # pin a day per record (from the printed candidates):
    uv run python .../fixed_benchmark_forecast_examples.py --experiment-root ... --day dssuk=2019-09-14:3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parents[1])]

import fixed_benchmark_tables as fbt  # noqa: E402
from thesis_figures import line_style as LS  # noqa: E402

ALPHA = 0.1
IMAGES_DIR = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
SLUG = {"mlvs-pt": "mlvs", "dss-eu": "dsseu", "dss-uk": "dssuk", "dss-us": "dssus"}
UNITS = {
    "LP": "P (kW)",
    "LQ": "Q (kvar)",
    "LV": "V (V)",
    "p_kw": "P (kW)",
    "q_kvar": "Q (kvar)",
    "v_lv_pu": "V (pu)",
}
REFERENCE = "fpqr_anchor"
HEADS = ("fpqr_anchor", "qr", "bnn", "laplace")


def dataset_slug(name: str) -> str:
    """Short file slug of a dataset directory name, e.g. ``dss-uk_weak_0_4_0_4`` -> ``dssuk``."""
    return next(slug for prefix, slug in SLUG.items() if name.startswith(prefix))


def condition_paths(dataset_dir: Path, fpqr: str = "tuned") -> dict[str, Path]:
    """Every head and conformal variant the record holds: benchmark heads first, then their calibrations.

    ``fpqr="fixed"`` takes FPQR and its conformal variants from the fixed-parameter run the combined set keeps
    under ``benchmark_fixed/`` and ``conformal_fixed/``, in place of the head-tuned FPQR.
    """
    # A fixed-only result set has no benchmark_fixed/: its benchmark/ already holds the fixed FPQR.
    suffix = "_fixed" if fpqr == "fixed" and (dataset_dir / "benchmark_fixed").exists() else ""
    paths = {h: dataset_dir / ("benchmark" + suffix if h == REFERENCE else "benchmark") / h for h in HEADS}
    for head in HEADS:
        conformal = dataset_dir / ("conformal" + suffix if head == REFERENCE else "conformal")
        if conformal.exists():
            paths |= {p.name: p for p in sorted(conformal.glob(f"{head}_*")) if not p.name.endswith("_refit")}
    return {c: p for c, p in paths.items() if (p / "predictions.parquet").exists()}


def intervals(path: Path) -> pd.DataFrame:
    """A condition's interval rows: one per (fold, timestamp, target), with the calendar date."""
    frame = pd.read_parquet(
        path / "predictions.parquet",
        columns=["timestamp", "Folds", "target", "Actual", "forecast", "lower", "upper", "metric_type"],
    )
    frame = frame[frame["metric_type"] == "interval"].drop(columns="metric_type")
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    return frame


def daily_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (date, fold): joint hit rate, scale-free mean Winkler score and number of steps."""
    frame = frame.copy()
    frame["hit"] = (frame["Actual"] >= frame["lower"]) & (frame["Actual"] <= frame["upper"])
    penalty = (2.0 / ALPHA) * (
        np.clip(frame["lower"] - frame["Actual"], 0, None) + np.clip(frame["Actual"] - frame["upper"], 0, None)
    )
    scale = frame.groupby("target")["Actual"].transform(lambda a: a.abs().mean())
    frame["ws"] = (frame["upper"] - frame["lower"] + penalty) / scale
    joint = frame.pivot_table(index=["date", "Folds", "timestamp"], columns="target", values="hit").all(axis=1)
    return pd.DataFrame(
        {
            "joint": joint.groupby(level=["date", "Folds"]).mean(),
            "ws": frame.groupby(["date", "Folds"])["ws"].mean(),
            "steps": joint.groupby(level=["date", "Folds"]).size(),
        }
    )


def rank_days(frames: dict[str, pd.DataFrame], candidates: pd.Index) -> pd.DataFrame:
    """Candidate days ranked for every head at once: worst head's joint hit rate, then mean scale-free Winkler score.

    Only whole days count: a fold's first or last date can be cut short by the window start.
    """
    scores = {head: daily_scores(frame) for head, frame in frames.items()}
    full = max(int(sc["steps"].max()) for sc in scores.values())
    joint = pd.DataFrame({head: sc.loc[sc["steps"] == full, "joint"] for head, sc in scores.items()})
    ws = pd.DataFrame({head: sc.loc[sc["steps"] == full, "ws"] for head, sc in scores.items()})
    table = joint.add_prefix("joint_").assign(min_joint=joint.min(axis=1), mean_ws=ws.mean(axis=1))
    table = table.reindex(candidates).dropna()
    return table.sort_values(["min_joint", "mean_ws"], ascending=[False, True])


def run(
    dataset_dir: Path,
    conditions: list[str] | None,
    out: Path,
    fpqr: str = "fixed",
    day: tuple[pd.Timestamp, int] | None = None,
    suggest: int = 5,
) -> pd.DataFrame:
    """Draw every chosen condition's P, Q and V on one day good for every head; returns the per-panel summary."""
    slug = dataset_slug(dataset_dir.name)
    paths = condition_paths(dataset_dir, fpqr)
    frames = {c: intervals(p) for c, p in paths.items()}
    shared = pd.concat({c: f.groupby(["date", "Folds"]).size() for c, f in frames.items()}, axis=1).dropna().index
    ranked = rank_days({h: frames[h] for h in HEADS if h in frames}, shared)
    ranked.to_csv(out / f"day_candidates_{slug}.csv")
    print(f"\n{dataset_dir.name}: best days for every head")
    print(ranked.head(suggest).round(3).to_string())
    date, fold = day if day is not None else ranked.index[0]
    chosen = [c for c in paths if conditions is None or c in conditions]
    targets = fbt.ordered_targets(frames[REFERENCE])
    days = {c: frames[c][(frames[c]["date"] == date) & (frames[c]["Folds"] == fold)] for c in chosen}
    rows = []
    for target in targets:
        # One y-range per (record, target) across the four heads, and one per head across its native forecast
        # and its conformal variants: each set is one subfigure grid in the thesis. A head with conformal
        # variants also gets its native panel on that second range, saved as <head>_none.
        frame = {c: days[c][days[c]["target"] == target].sort_values("timestamp") for c in chosen}
        heads = [c for c in chosen if c in HEADS]
        groups = {"heads": [(c, c) for c in heads]}
        for head in heads:
            variants = [c for c in chosen if c.startswith(f"{head}_")]
            if variants:
                groups[head] = [(head, f"{head}_none"), *[(c, c) for c in variants]]
        for members in groups.values():
            span = pd.concat([frame[c][["Actual", "lower", "upper"]] for c, _ in members])
            for condition, name in members:
                day = frame[condition]
                fig = LS.forecast_panel(day, UNITS.get(target, target))
                LS.set_y_axis(fig.axes[0], span["Actual"], span["lower"], span["upper"])
                LS.save(fig, out / f"fig_ch6_example_{slug}_{name}_{fbt.ROLE[target].lower()}")
        for condition in chosen:
            day = frame[condition]
            hit = (day["Actual"] >= day["lower"]) & (day["Actual"] <= day["upper"])
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "date": date.date(),
                    "fold": fold,
                    "condition": condition,
                    "target": fbt.ROLE[target],
                    "coverage": float(hit.mean()),
                    "mean_width": float((day["upper"] - day["lower"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Draw the forecast examples for the chosen records and conditions."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment-root", type=Path, required=True, help="the combined result set")
    ap.add_argument("--datasets", nargs="*", choices=list(SLUG.values()), default=None)
    ap.add_argument("--conditions", nargs="*", default=None, help="e.g. fpqr_anchor fpqr_anchor_max_norm")
    ap.add_argument("--fpqr", choices=["fixed", "tuned"], default="fixed", help="which FPQR to draw")
    ap.add_argument(
        "--day",
        nargs="*",
        default=[],
        metavar="SLUG=DATE:FOLD",
        help="pin a record's day, e.g. dssuk=2019-09-14:3; otherwise the top-ranked day",
    )
    ap.add_argument("--suggest", type=int, default=5, help="candidate days to print per record")
    ap.add_argument("--copy-to-images", action="store_true", help=f"copy the PDFs to {IMAGES_DIR}")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    LS.configure(latex=not args.no_latex)
    out = args.experiment_root / "analysis" / ("forecast_examples" + ("_tuned_fpqr" if args.fpqr == "tuned" else ""))
    out.mkdir(parents=True, exist_ok=True)
    pinned = {}
    for item in args.day:
        slug, when = item.split("=")
        date, fold = when.split(":")
        pinned[slug] = (pd.Timestamp(date, tz="UTC"), int(fold))
    summaries = [
        run(dataset_dir, args.conditions, out, args.fpqr, pinned.get(dataset_slug(name)), args.suggest)
        for name, dataset_dir in fbt.newest_dataset_dirs(args.experiment_root).items()
        if args.datasets is None or dataset_slug(name) in args.datasets
    ]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "forecast_examples_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    if args.copy_to_images:
        for pdf in out.glob("*.pdf"):
            shutil.copy(pdf, IMAGES_DIR / pdf.name)
    print(f"written under {out}")


if __name__ == "__main__":
    main()
