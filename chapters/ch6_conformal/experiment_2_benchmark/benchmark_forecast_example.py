r"""Forecast example for the probabilistic benchmark: one CRC good day, shown for every reported head.

The day is CRC's best interval day: the (date, fold) with the highest daily CWE of ``mlpgam_crc``, ties broken by the
lowest daily Winkler score, restricted to days for which every reported head has a forecast.
The same day is then drawn for CRC, QR, BNN and Laplace, one single-plot figure per head in the compact
thesis line-plot style of ``twiga_thesis.plotting.line_style`` (observed values, forecast, 90% interval),
each panel's y-axis fitted to its own data; no title (the thesis caption carries it). Nothing is
retrained: each head's saved ``predictions.parquet``/``metrics.parquet`` are read from the latest
benchmark run that contains it.

Usage::

    uv run python experiment/ch6_conformal/experiment_2_benchmark/benchmark_forecast_example.py \
        --experiment-name probabilistic_benchmark_v4 --dataset MLVS-PT --copy-to-images
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
HEADS = {"mlpgam_crc": "crc", "mlpgam_qr": "qr", "mlpgam_bnn": "bnn", "mlpgam_laplace": "laplace"}
SLUG = {"MLVS-PT": "pt", "MLVS-UK": "uk"}
UNITS = {"NetLoad(kW)": "Net load (kW)", "NetLoad(MW)": "Net load (MW)"}


def crc_good_day(metrics: dict[str, pd.DataFrame], reference: str = "mlpgam_crc") -> tuple[pd.Timestamp, int]:
    """The reference head's best interval day, among the days every head has a forecast for."""
    shared = pd.concat({c: daily_error(m) for c, m in metrics.items()}, axis=1).dropna().index
    return best_interval_day(metrics[reference], shared)


def day_stats(day: pd.DataFrame) -> dict:
    """Daily NRMSE-free summary of one head's interval: coverage, mean width, Winkler score."""
    actual, lower, upper = day["Actual"].to_numpy(), day["lower"].to_numpy(), day["upper"].to_numpy()
    width = upper - lower
    penalty = (2.0 / ALPHA) * (np.clip(lower - actual, 0, None) + np.clip(actual - upper, 0, None))
    return {
        "coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "mean_width": float(width.mean()),
        "winkler_score": float((width + penalty).mean()),
        "mae": float(np.abs(day["forecast"].to_numpy() - actual).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--experiment-name", default="probabilistic_benchmark")
    ap.add_argument("--dataset", default="MLVS-PT", choices=list(CHAPTER_DATASETS))
    ap.add_argument("--copy-to-images", action="store_true", help=f"copy the PDFs to {IMAGES_DIR}")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    LS.configure(latex=not args.no_latex)
    experiment_dir = experiment_output_dir(args.experiment_name, args.output_root)
    paths = {c: condition_dir(experiment_dir, args.dataset, c) for c in HEADS}
    missing = [c for c, p in paths.items() if p is None]
    if missing:
        raise SystemExit(f"{args.dataset}: no benchmark run with {missing} under {experiment_dir}")
    metrics = {c: pd.read_parquet(p / "metrics.parquet") for c, p in paths.items()}
    date, fold = crc_good_day(metrics)
    days = {c: day_frame(paths[c], date, fold) for c in HEADS}

    out = args.output_root / "analysis" / "twiga_thesis.plotting" / "probabilistic_benchmark" / "forecast_example"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, short in HEADS.items():
        day = days[condition]
        target = str(day["target"].iloc[0])
        fig = LS.forecast_panel(day, tex_safe(UNITS.get(target, target)))
        LS.save(fig, out / f"forecast_example_{short}_{SLUG[args.dataset]}")
        rows.append({"dataset": args.dataset, "date": date.date(), "fold": fold, "head": condition, **day_stats(day)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"forecast_example_{SLUG[args.dataset]}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    if args.copy_to_images:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in out.glob(f"*_{SLUG[args.dataset]}.pdf"):
            shutil.copy(pdf, IMAGES_DIR / f"fig_ch7_bench_{pdf.name}")


if __name__ == "__main__":
    main()
