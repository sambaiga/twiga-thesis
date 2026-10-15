"""Two supporting panels for Chapter 6's voltage-difficulty diagnosis (subsec:ch6-results).

Generates the evidence cited in the "Voltage also represents the most
challenging target..." paragraph: (a) LV's monthly-mean drift over the
MLVS-PT-3PHASE record, showing the raw level is not stationary in mean, and
(b) LV's linear correlation with every other candidate predictor in the
feature set, showing the drift is not explained by anything currently
available to the model. Both numbers are recomputed here directly from
load_dataset()'s own output, not hand-copied from an earlier exploratory
session, so a reader can reproduce them exactly.

Usage:

    uv run python experiment/ch7_multitarget/shared/ch6_voltage_nonstationarity_figure.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent))  # repo "experiment" package root

from twiga.utils.data_loading import load_dataset  # noqa: E402

from twiga_thesis.paths import thesis_data_dir
from twiga_thesis.plotting import style as S  # noqa: E402
from twiga_thesis.plotting.tokens import HIGHLIGHT, PRIMARY, TEXT_MUTED  # noqa: E402

OUT_DIR = script_dir.parent / "results" / "analysis" / "multitarget_experiment" / "benchmark" / "figures"


def main() -> None:
    S.configure()

    data, target, _cal, _exog, _hist, _lat, _lon, _lookback = load_dataset(
        dataset="MLVS-PT-3PHASE", data_path=str(thesis_data_dir()), top_k=2, weather_top_k=1, n_samples=48
    )
    if target != ["LP", "LQ", "LV"]:
        raise ValueError(f"expected load_dataset('MLVS-PT-3PHASE') target_series=['LP','LQ','LV'], got {target}")
    data = data.set_index("timestamp")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── (a) monthly-mean drift ──────────────────────────────────────────────
    monthly = data["LV"].resample("MS").mean()
    fig, ax = plt.subplots(figsize=S.panel(0.47))
    ax.plot(monthly.index, monthly.to_numpy(), color=PRIMARY, lw=1.2)
    ax.axhline(data["LV"].mean(), color=TEXT_MUTED, lw=0.8, ls="--", label=S.tex_safe("Record mean"))
    ax.set_ylabel(S.tex_safe("LV monthly mean (V)"))
    ax.set_xlabel(S.tex_safe("Year"))
    ax.legend(loc="lower right", frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    S.format_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "MLVS-PT-3PHASE_lv_monthly_drift.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── (b) correlation with every other candidate predictor ───────────────
    data["LV_lag_96"] = data["LV"].shift(96)
    predictors = ["LP", "LQ", "Ghi", "Temperature", "LV_lag_96"]
    corr = data[predictors + ["LV"]].corr()["LV"].drop("LV").abs().sort_values()
    labels = [S.tex_safe(p) for p in corr.index]

    fig, ax = plt.subplots(figsize=S.panel(0.47))
    colors = [HIGHLIGHT if v < 0.2 else PRIMARY for v in corr.to_numpy()]
    ax.barh(labels, corr.to_numpy(), color=colors)
    ax.axvline(0.2, color=TEXT_MUTED, lw=0.8, ls="--")
    ax.set_xlabel(r"Absolute correlation with LV")
    ax.set_xlim(0, 1)
    S.format_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "MLVS-PT-3PHASE_lv_correlation.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUT_DIR / 'MLVS-PT-3PHASE_lv_monthly_drift.pdf'}")
    print(f"wrote {OUT_DIR / 'MLVS-PT-3PHASE_lv_correlation.pdf'}")
    print("\nMonthly means:")
    print(monthly.to_string())
    print("\nCorrelations with LV (abs, sorted):")
    print(corr.to_string())
    print(f"\nLV coefficient of variation: {data['LV'].std() / data['LV'].mean():.4f}")
    print(f"LP coefficient of variation: {data['LP'].std() / data['LP'].mean():.4f}")


if __name__ == "__main__":
    main()
