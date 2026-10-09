r"""Forecast example for H3: fixed (pooled), horizon-wise, and CRC conformal intervals on one day.

Uses the architectural study's ``calibration`` group, where the conditions share one trained model per fold and
differ only in the calibration step: ``conf_scp_pooled`` estimates one quantile from the unscaled residuals of every
horizon, so its interval has the same width at every lead time; ``conf_scp`` estimates one quantile per horizon, so
its width follows the uncertainty of each period; ``conf_crc`` (CRC, the proposed calibration) also scales the
residual by the learned sigma, so its width varies from day to day within a lead time. Nothing is retrained: the
saved ``predictions.parquet`` and ``metrics.parquet`` of each condition are read from the latest run that has it.

The day is CRC's best interval day: the (date, fold) with the highest daily CWE of ``conf_crc``, ties broken by the
lowest daily Winkler score, among the days every condition has. One single-plot figure per (dataset, condition),
authored at the half-text-width size it occupies in the thesis, in the panel style of
``experiment/common/forecast_panels.draw``, with a shared y-axis per dataset so the interval widths are directly
comparable; no title (the thesis caption carries it).

Usage::

    uv run python experiment/ch6_conformal/experiment_1_architectural/calibration_forecast_example.py \
        --copy-to-images
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[1]))  # experiment/ (twiga_thesis.plotting, common)
sys.path.insert(0, str(script_dir.parent))  # chapter directory (shared.*)
sys.path.insert(0, str(script_dir.parents[1] / "common"))

from chapters.ch6_conformal.shared.forecast_examples import (  # noqa: E402
    best_interval_day,
    condition_dir,
    daily_error,
    day_frame,
)
from chapters.ch6_conformal.shared.protocol import (  # noqa: E402
    ALPHA,
    CHAPTER_DATASETS,
    DEFAULT_OUTPUT_ROOT,
    experiment_output_dir,
)
from twiga_thesis.plotting import line_style as LS  # noqa: E402
from twiga_thesis.plotting.style import tex_safe  # noqa: E402

IMAGES_DIR = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
CONDITIONS = {"conf_scp_pooled": "fixed", "conf_scp": "adaptive", "conf_crc": "crc"}
REFERENCE = "conf_crc"
SLUG = {"MLVS-PT": "pt", "MLVS-UK": "uk"}
UNITS = {"NetLoad(kW)": "Net load (kW)", "NetLoad(MW)": "Net load (MW)"}


def day_stats(day: pd.DataFrame) -> dict:
    """Empirical coverage, mean interval width and Winkler score of one day's interval."""
    actual, lower, upper = day["Actual"].to_numpy(), day["lower"].to_numpy(), day["upper"].to_numpy()
    width = upper - lower
    penalty = (2.0 / ALPHA) * (np.clip(lower - actual, 0, None) + np.clip(actual - upper, 0, None))
    return {
        "coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "mean_width": float(width.mean()),
        "min_width": float(width.min()),
        "max_width": float(width.max()),
        "winkler_score": float((width + penalty).mean()),
    }


def run(dataset: str, experiment_dir: Path, out: Path) -> pd.DataFrame:
    paths = {c: condition_dir(experiment_dir, dataset, c) for c in CONDITIONS}
    missing = [c for c, p in paths.items() if p is None]
    if missing:
        raise SystemExit(f"{dataset}: no study run with {missing} under {experiment_dir}")
    metrics = {c: pd.read_parquet(p / "metrics.parquet") for c, p in paths.items()}
    shared = pd.concat({c: daily_error(m) for c, m in metrics.items()}, axis=1).dropna().index
    date, fold = best_interval_day(metrics[REFERENCE], shared)
    days = {c: day_frame(paths[c], date, fold) for c in CONDITIONS}

    rows = []
    for condition, kind in CONDITIONS.items():
        day = days[condition]
        target = str(day["target"].iloc[0])
        fig = LS.forecast_panel(day, tex_safe(UNITS.get(target, target)))
        LS.save(fig, out / f"calibration_example_{kind}_{SLUG[dataset]}")
        rows.append({"dataset": dataset, "date": date.date(), "fold": fold, "calibration": kind, **day_stats(day)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--experiment-name", default="architectural_study")
    ap.add_argument("--datasets", nargs="+", default=list(CHAPTER_DATASETS))
    ap.add_argument("--copy-to-images", action="store_true", help=f"copy the PDFs to {IMAGES_DIR}")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    LS.configure(latex=not args.no_latex)
    out = args.output_root / "analysis" / "architectural_study" / "calibration_example"
    out.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_output_dir(args.experiment_name, args.output_root)
    summary = pd.concat([run(d, experiment_dir, out) for d in args.datasets], ignore_index=True)
    summary.to_csv(out / "calibration_example_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    if args.copy_to_images:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in out.glob("*.pdf"):
            shutil.copy(pdf, IMAGES_DIR / f"fig_ch7_arch_{pdf.name}")


if __name__ == "__main__":
    main()
