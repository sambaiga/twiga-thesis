r"""Good/bad-day forecast panels for a Chapter 7 design-ablation group, from saved backtest output.

The same idea as `experiment/chapter-5-fpqr/qvn_forecast_examples.py`'s
`qvn_type` panels, applied to Chapter 7's own design ablation
(`mlpgam_design_ablation.py`): reuse the backtest engine's own saved
`predictions.parquet` (`Actual`/`forecast`/`lower`/`upper` per day, per fold,
per condition) and `metrics.parquet` (per-day NRMSE/Winkler score/PICP,
already computed), rather than retraining or recomputing anything.

Chapter 7's `metrics.parquet` is already one row per (timestamp, fold) with
every metric as its own column -- no `metric_type` split into
point/quantile/interval rows the way Chapter 5's does -- so day selection and
lookup here are a direct filter, not a three-way join.

A day is picked once per --group, pooled over every condition in that group
that has a completed backtest for the requested dataset (mean per-day NRMSE
across conditions, restricted to (timestamp, fold) pairs every condition
shares), so every panel in one figure shows the same calendar day and the
comparison across conditions is fair -- the same principle
`qvn_forecast_examples.py` and `ch7_multitarget/forecast_examples.py`
both use for their own best/worst day.

The default group is `training_mode`, Chapter 7's dominant, most visually
compelling effect (per-batch alternation vs. joint optimisation vs. the
two-stage schedules) -- the Chapter 5 analogue of picking `qvn_type` as the
headline comparison there.

Run:  uv run python experiment/ch6_conformal/design_ablation/mlpgam_forecast_examples.py
      uv run python experiment/ch6_conformal/design_ablation/mlpgam_forecast_examples.py --dataset MLVS-UK
      uv run python experiment/ch6_conformal/design_ablation/mlpgam_forecast_examples.py --group monitor
      uv run python experiment/ch6_conformal/design_ablation/mlpgam_forecast_examples.py --group gating
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiment/ (twiga_thesis.plotting, common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chapter directory (shared.*)
from twiga_thesis.plotting.style import configure, panel, tex_safe  # noqa: E402

# draw/_format_axes are ch7_multitarget/forecast_examples.py's own
# panel style (hollow-marker actual, dashed prediction, translucent band),
# extracted unchanged into experiment/common/forecast_panels.py so this
# module reuses exactly that implementation rather than a second copy tuned
# to the thesis palette. This engine's saved output is one nominal interval
# per step, not chapter 6's full quantile fan, so a single (lower, forecast,
# upper) triple is stacked as a degenerate 3-level "fan": draw() resolves the
# median to the middle level and draws one band from the outer two, exactly
# reproducing the same visual convention for one interval.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from twiga_thesis.plotting.forecast_panels import draw  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT = script_dir / "mlpgam_forecast_examples"

_FAN_LEVELS = np.array([0.05, 0.5, 0.95])


def draw_panel(ax, day: pd.DataFrame, title: str, subtitle: str) -> None:
    """One condition's forecast panel, via the shared chapter-6 panel style."""
    t_axis = day["timestamp"].to_numpy()
    q = np.stack([day["lower"].to_numpy(), day["forecast"].to_numpy(), day["upper"].to_numpy()])
    draw(ax, t_axis, day["Actual"].to_numpy(), q, _FAN_LEVELS, t_i=0)
    ax.set_title(tex_safe(title), fontsize=9)
    ax.set_ylabel("NetLoad (kW)", fontsize=8)
    ax.tick_params(labelsize=7)


# One run directory per dataset/group is resolved via CONDITIONS_ROOT below;
# --run-dir overrides it to point at a specific rerun rather than always the
# latest.
DEFAULT_RUN_DIR = (
    script_dir
    / "results"
    / "ch6_conformal-v2"
    / "design_ablation"
    / "mlpgam_design_evaluation__chapter_7"
    / "runs"
    / "20260918_023748_0650b2"
)

GROUP_CONDITIONS = {
    "training_mode": ["joint", "two_stage_ep_05", "two_stage_ep_10", "two_stage_ep_20", "two_stage_batch"],
    "monitor": ["monitor_loss", "monitor_mae", "monitor_sigma"],
    "gating": ["gate_default", "gate_none", "gate_unconstrained"],
    "sigma_loss": ["sigma_mse", "sigma_hybrid", "sigma_hybrid_sqrt", "sigma_laplace", "sigma_gaussian", "sigma_log"],
    "conformal": ["conf_90_res", "conf_90_sign", "conf_scp"],
}

CONDITION_LABELS = {
    "joint": "Joint",
    "two_stage_ep_05": "Two-stage (5%)",
    "two_stage_ep_10": "Two-stage (10%)",
    "two_stage_ep_20": "Two-stage (20%)",
    "two_stage_batch": "Per-batch (proposed)",
    "monitor_loss": "Monitor: combined loss",
    "monitor_mae": "Monitor: point MAE",
    "monitor_sigma": "Monitor: sigma (proposed)",
    "gate_default": "Gate: default",
    "gate_none": "Gate: none",
    "gate_unconstrained": "Gate: unconstrained",
}


def predictions_path(run_dir: Path, dataset: str, group: str, condition: str) -> Path:
    return run_dir / "conditions" / dataset.lower() / group / condition / "predictions.parquet"


def load(
    run_dir: Path, dataset: str, group: str, conditions: list[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Saved predictions/metrics per condition, and the conditions with no completed run."""
    preds, metrics, missing = {}, {}, []
    for condition in conditions:
        path = predictions_path(run_dir, dataset, group, condition)
        if not path.exists():
            missing.append(condition)
            continue
        preds[condition] = pd.read_parquet(path)
        metrics[condition] = pd.read_parquet(path.parent / "metrics.parquet")
    return preds, metrics, missing


def pick_days(metrics: dict[str, pd.DataFrame]) -> tuple[tuple[pd.Timestamp, int], tuple[pd.Timestamp, int]]:
    """Shared good/bad (timestamp, fold) key: mean per-day NRMSE pooled across every available condition."""
    frames = []
    for name, m in metrics.items():
        point = m[["timestamp", "Folds", "nrmse"]].copy()
        point["condition"] = name
        frames.append(point)
    pooled = pd.concat(frames, ignore_index=True)
    grouped = pooled.groupby(["timestamp", "Folds"])["nrmse"]
    day_err = grouped.mean()
    complete = day_err[grouped.size() == len(metrics)]
    if complete.empty:
        raise RuntimeError("no (timestamp, fold) pair is shared by every requested condition")
    return complete.idxmin(), complete.idxmax()


def day_frame(preds: pd.DataFrame, timestamp: pd.Timestamp, fold: int) -> pd.DataFrame:
    """One condition's rows for one backtest day, sorted for plotting."""
    mask = (preds["Folds"] == fold) & (preds["timestamp"].dt.normalize() == timestamp)
    return preds[mask].sort_values("timestamp")


def day_metric(metrics: pd.DataFrame, timestamp: pd.Timestamp, fold: int, column: str) -> float:
    mask = (metrics["Folds"] == fold) & (metrics["timestamp"].dt.normalize() == timestamp)
    day = metrics[mask]
    return float(day[column].iloc[0]) if not day.empty else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MLVS-PT")
    ap.add_argument("--group", default="training_mode", choices=list(GROUP_CONDITIONS))
    ap.add_argument("--conditions", nargs="+", default=None, help="defaults to every condition in --group")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--latex", action="store_true", default=True)
    args = ap.parse_args()
    conditions = args.conditions or GROUP_CONDITIONS[args.group]

    out_dir = OUT / args.group / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, metrics, missing = load(args.run_dir, args.dataset, args.group, conditions)
    if missing:
        print(f"no completed backtest for: {', '.join(missing)} (skipped, not silently dropped)")
    if not preds:
        raise SystemExit(f"no completed {args.group} backtest found for {args.dataset} under {args.run_dir}")

    (good_ts, good_fold), (bad_ts, bad_fold) = pick_days(metrics)
    print(f"{len(preds)} conditions with a completed backtest: {', '.join(preds)}")
    print(f"good day  {good_ts:%Y-%m-%d}  fold {good_fold}")
    print(f"bad day   {bad_ts:%Y-%m-%d}  fold {bad_fold}")

    configure(latex=args.latex)
    rows = []
    for label, ts, fold in (("good", good_ts, good_fold), ("bad", bad_ts, bad_fold)):
        present = [c for c in conditions if c in preds]
        ncol = min(len(present), 3)
        nrow = -(-len(present) // ncol) if ncol else 0
        fig, axes = plt.subplots(nrow, ncol, figsize=panel(ncol, aspect=0.75), squeeze=False)
        axes_flat = axes.flatten()

        for ax, condition in zip(axes_flat, present, strict=False):
            day = day_frame(preds[condition], ts, fold)
            m = metrics[condition]
            nrmse = day_metric(m, ts, fold, "nrmse")
            winkler = day_metric(m, ts, fold, "winkler_score")
            picp = day_metric(m, ts, fold, "picp")
            rows.append({"day": label, "condition": condition, "NRMSE": nrmse, "Winkler": winkler, "PICP": picp})

            display_name = CONDITION_LABELS.get(condition, condition)
            draw_panel(
                ax,
                day,
                title=f"{display_name}  ({label} day, {ts:%Y-%m-%d})",
                subtitle=f"NRMSE {nrmse:.3f}, Winkler {winkler:.1f}, PICP {picp:.2f}",
            )
            fig_single, ax_single = plt.subplots(figsize=panel(1, aspect=0.7))
            draw_panel(
                ax_single,
                day,
                title=display_name,
                subtitle=f"NRMSE {nrmse:.3f}, Winkler {winkler:.1f}, PICP {picp:.2f}",
            )
            fig_single.savefig(out_dir / f"{label}_{condition}.pdf", bbox_inches="tight")
            plt.close(fig_single)

        for ax in axes_flat[len(present) :]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"overview_{label}_day.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / f"overview_{label}_day.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "forecast_examples_summary.csv", index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
