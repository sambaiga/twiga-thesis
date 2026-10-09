r"""Chapter 7 interpretability analysis of the proposed MLPGAM-CRC model (architectural study, ``calibration`` group).

Reads the ``conf_crc`` condition, whose ``custom_scorer`` (:mod:`shared.mlpgam_interpretability`) saved, per fold, the
gate value, gradient sensitivity and permutation importance of every input feature, and, per timestamp, the exact
additive branch decomposition of the forecast (``contrib_past``, ``contrib_future``, ``contrib_offset``). Produces:

* ``interpretability_features_<ds>.csv``: per feature, the fold mean and standard deviation of the gate value, the
  permutation importance (relative increase in test MAE) and the gradient sensitivity, with the permutation rank;
* ``interpretability_summary.csv``: per dataset, the fold mean and standard deviation of the gate-permutation and
  gate-sensitivity Spearman correlations, the covariate-branch share of the forecast and the largest decomposition
  error;
* ``interp_importance_<ds>``: gate value against permutation importance per feature, each normalised by its maximum;
* ``interp_branch_profile_<ds>``: mean history-branch and covariate-branch terms by lead time, with the mean observed
  value;
* ``interp_decomposition_<ds>``: the decomposition of one day's forecast (the day of lowest NRMSE).

One single-axes figure per file, no title (the thesis caption carries it).

Usage::

    uv run python experiment/ch6_conformal/shared/ch7_interpretability_analysis.py --copy-to-images
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir.parents[1]))

from shared.forecast_examples import condition_dir, daily_error  # noqa: E402
from shared.protocol import CHAPTER_DATASETS, DEFAULT_OUTPUT_ROOT, experiment_output_dir  # noqa: E402
from thesis_figures import line_style as LS, style as S, tokens as T  # noqa: E402

IMAGES_DIR = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
SLUG = {"MLVS-PT": "pt", "MLVS-UK": "uk"}
UNITS = {"NetLoad(kW)": "kW", "NetLoad(MW)": "MW"}
CONDITION = "conf_crc"
BRANCH_LABELS = {"past": "history", "future": "covariates"}
SUMMARY_COLUMNS = ["gate_rho_perm", "gate_rho_spearman", "branch_share_future", "decomp_max_rel_error"]


def feature_label(branch: str, name: str) -> str:
    """Readable feature name with its branch, e.g. ``Net load, lag 96 (history)``."""
    label = re.sub(r"^NetLoad\([^)]*\)", "Net load", name)
    label = label.replace("_lag_", ", lag ").replace("_rolling_mean_win_", ", rolling mean ")
    label = {
        "Ghi": "GHI",
        "yday_cos": "Day of year (cos)",
        "yweek_cos": "Week of year (cos)",
        "hour_cosin": "Hour of day",
    }.get(label, label)
    label = label.replace("yday_", "Day of year (").replace("yweek_", "Week of year (")
    if label.count("(") > label.count(")"):
        label += ")"
    return f"{label} ({BRANCH_LABELS[branch]})"


def fold_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per fold: the fold-level scalars, which the scorer repeats on every row of the fold."""
    cols = [c for c in metrics.columns if c.startswith(("gate", "sens__", "perm__", "branch_", "decomp_"))]
    return metrics.groupby("Folds")[cols].first()


def feature_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in folds.columns if c.startswith("perm__")]:
        _, branch, name = col.split("__", 2)
        gate, sens = folds.get(f"gate__{branch}__{name}"), folds.get(f"sens__{branch}__{name}")
        rows.append(
            {
                "branch": branch,
                "feature": name,
                "label": feature_label(branch, name),
                "perm_mean": folds[col].mean(),
                "perm_std": folds[col].std(ddof=1),
                "gate_mean": gate.mean() if gate is not None else np.nan,
                "gate_std": gate.std(ddof=1) if gate is not None else np.nan,
                "sens_mean": sens.mean() if sens is not None else np.nan,
            }
        )
    table = pd.DataFrame(rows).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    table["perm_rank"] = np.arange(1, len(table) + 1)
    table["gate_rank"] = table["gate_mean"].rank(ascending=False).astype(int)
    return table


def plot_importance(table: pd.DataFrame, path: Path) -> None:
    """Normalised permutation importance and gate value per feature, sorted by permutation importance.

    Grouped horizontal bars in the two contrasting thesis colours, permutation importance (the model-agnostic
    reference) in the primary colour and gate value in the highlight colour, each normalised by its maximum, with
    thin one-standard-deviation error bars; thesis-theme axis type, 8 pt feature labels and legend below the axes.
    """
    from matplotlib.ticker import MultipleLocator  # noqa: PLC0415

    order = table.iloc[::-1]
    y = np.arange(len(order))
    series = [
        ("Permutation importance", order["perm_mean"], order["perm_std"], T.PRIMARY, 0.19),
        ("Gate value", order["gate_mean"], order["gate_std"], T.HIGHLIGHT, -0.19),
    ]
    with plt.rc_context(LS.COMPACT_RC):
        fig, ax = plt.subplots(figsize=S.panel(0.48, aspect=0.22 + 0.075 * len(order)))
        right = 1.0
        for label, mean, std, color, offset in series:
            value, err = mean / mean.max(), std / mean.max()
            ax.barh(y + offset, value, 0.36, color=color, label=label, zorder=2, lw=0)
            ax.errorbar(
                value,
                y + offset,
                xerr=err,
                fmt="none",
                ecolor=T.TEXT_DARK,
                elinewidth=0.6,
                capsize=1.5,
                capthick=0.6,
                zorder=3,
            )
            right = max(right, float((value + err).max()))
        ax.set_yticks(y)
        short = order["label"].str.replace(" (history)", " (H)").str.replace(" (covariates)", " (C)")
        ax.set_yticklabels([S.tex_safe(label) for label in short], fontsize=LS.LEGEND_FONTSIZE)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(0, np.ceil(right * 4) / 4)
        ax.set_xlabel("Normalised importance")
        LS.finish(ax)
        ax.tick_params(axis="y", length=0, labelsize=LS.LEGEND_FONTSIZE)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", which="minor", length=1.5, width=0.5)
        LS.legend_below(ax, ncol=2)
        LS.save(fig, path)


def branch_profile(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean branch terms and observed value by lead time over every test window."""
    frame = preds.dropna(subset=["contrib_past"]).sort_values(["Folds", "timestamp"]).copy()
    frame["day"] = frame["timestamp"].dt.normalize()
    frame["h"] = frame.groupby(["Folds", "day"]).cumcount() + 1
    return frame.groupby("h")[["contrib_past", "contrib_future", "contrib_offset", "forecast", "Actual"]].mean()


def plot_branch_profile(profile: pd.DataFrame, unit: str, path: Path) -> None:
    """Mean branch terms and observed value by time of day, in the compact thesis line-plot style."""
    hours = (profile.index.to_numpy() - 1) / 2.0  # every window starts at 00:00
    with plt.rc_context(LS.COMPACT_RC):
        fig, ax = LS.new_panel()
        ax.plot(hours, profile["contrib_past"], color=T.INFO, ls="--", label="History")
        ax.plot(hours, profile["contrib_future"], color=T.SECONDARY, ls="-.", label="Covariates")
        ax.plot(hours, profile["contrib_offset"], color=T.TEXT_DARK, ls=(0, (1, 1.5)), label="Offset")
        ax.plot(hours, profile["Actual"], color=LS.OBSERVED_COLOR, label="Observed (mean)")
        LS.reference_line(ax, 0.0)
        LS.hour_axis(ax)
        LS.set_y_axis(ax, *(profile[c] for c in ("contrib_past", "contrib_future", "contrib_offset", "Actual")))
        ax.set_ylabel(f"Contribution ({unit})")
        LS.finish(ax)
        LS.legend_below(ax, ncol=2)
        LS.save(fig, path)


def plot_decomposition(day: pd.DataFrame, unit: str, path: Path) -> None:
    """One day's observed values and forecast, with the forecast's history (with offset) and covariate terms."""
    t = LS.naive_times(day)
    history = day["contrib_past"] + day["contrib_offset"]
    with plt.rc_context(LS.COMPACT_RC):
        fig, ax = LS.new_panel()
        ax.plot(t, day["forecast"], color=LS.FORECAST_COLOR, label="Forecast")
        ax.plot(t, history, color=T.INFO, ls="--", label="History + offset")
        ax.plot(t, day["contrib_future"], color=T.SECONDARY, ls="-.", label="Covariates")
        ax.plot(t, day["Actual"], "o", mec=LS.OBSERVED_COLOR, mfc="white", label="Observed", zorder=3, **LS.MARKER)
        LS.reference_line(ax, 0.0)
        LS.time_axis(ax, t)
        LS.set_y_axis(ax, day["Actual"], day["forecast"], history, day["contrib_future"])
        ax.set_ylabel(f"Net load ({unit})")
        LS.finish(ax)
        LS.legend_below(ax, ncol=2)
        LS.save(fig, path)


def _save(fig, path: Path) -> None:
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(dataset: str, experiment_dir: Path, out: Path) -> dict:
    path = condition_dir(experiment_dir, dataset, CONDITION)
    if path is None:
        raise SystemExit(f"{dataset}: no run with {CONDITION} under {experiment_dir}")
    metrics = pd.read_parquet(path / "metrics.parquet")
    if not any(c.startswith("perm__") for c in metrics.columns):
        raise SystemExit(f"{dataset}: {path} has no interpretability columns; rerun the calibration group")
    preds = pd.read_parquet(path / "predictions.parquet")
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    unit = UNITS.get(str(preds["target"].iloc[0]), "")
    slug = SLUG[dataset]

    folds = fold_frame(metrics)
    table = feature_table(folds)
    table.to_csv(out / f"interpretability_features_{slug}.csv", index=False)
    plot_importance(table, out / f"interp_importance_{slug}")

    profile = branch_profile(preds)
    profile.to_csv(out / f"interpretability_branch_profile_{slug}.csv")
    plot_branch_profile(profile, unit, out / f"interp_branch_profile_{slug}")

    date, fold = daily_error(metrics).idxmin()
    date = pd.Timestamp(date)
    date = date.tz_localize("UTC") if date.tz is None else date.tz_convert("UTC")
    day = preds[(preds["Folds"] == fold) & (preds["timestamp"].dt.normalize() == date)]
    day = day.dropna(subset=["contrib_past"]).sort_values("timestamp")
    plot_decomposition(day, unit, out / f"interp_decomposition_{slug}")

    summary = {"dataset": dataset, "run": path.parents[3].name, "decomposition_day": str(date.date()), "fold": fold}
    for col in SUMMARY_COLUMNS:
        if col in folds:
            summary[f"{col}_mean"] = folds[col].mean()
            summary[f"{col}_std"] = folds[col].std(ddof=1)
    summary["decomp_max_rel_error_max"] = folds.get("decomp_max_rel_error", pd.Series([np.nan])).max()
    summary["gate_perm_rank_agreement_top3"] = len(
        set(table.nsmallest(3, "perm_rank")["label"]) & set(table.nsmallest(3, "gate_rank")["label"])
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--experiment-name", default="architectural_study")
    ap.add_argument("--datasets", nargs="+", default=list(CHAPTER_DATASETS))
    ap.add_argument("--copy-to-images", action="store_true", help=f"copy the PDFs to {IMAGES_DIR}")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    S.configure(latex=not args.no_latex)
    out = args.output_root / "analysis" / "architectural_study" / "interpretability"
    out.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_output_dir(args.experiment_name, args.output_root)
    summary = pd.DataFrame([run(d, experiment_dir, out) for d in args.datasets])
    summary.to_csv(out / "interpretability_summary.csv", index=False)
    print(summary.round(4).T.to_string())
    if args.copy_to_images:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in out.glob("interp_*.pdf"):
            shutil.copy(pdf, IMAGES_DIR / f"fig_ch7_arch_{pdf.name}")


if __name__ == "__main__":
    main()
