"""Chapter 4 RQ1 — MLPF design-choice ablations on SPS-UK.

Sweeps MLPF's own existing config fields (nothing new added to
`twiga.models.nn.mlpf_model`) in four independent groups, each varying one
axis while holding the others at the architecture's own reference values
(`forecast_horizon=48` samples / 1 day, `lookback_window_size=336` samples /
7 days, `embedding_type=None`, `combination_type="addition-comb"` --
`MLPFConfig`'s own field defaults):

    horizon         forecast_horizon, 1h-7d (paper's own sweep range)
    context_window  lookback_window_size, 1d-7d
    embedding       embedding_type: None / "PosEmb" / "LearnPosEmb"
    combination     combination_type: "addition-comb" / "attn-comb" / "weighted-comb"

Deviation from the plan worth flagging: the plan describes the embedding
sweep as "None/Rotary/other positional-embedding options MLPF's own config
already exposes." `MLPFConfig.embedding_type` (`twiga/core/config/neural.py`)
is actually `Literal["PosEmb", "LearnPosEmb"] | None` -- there is no rotary
option on this field. `MLPFConfig` does carry rotary-related fields
(`patch_len`/`stride`, used by a patch-embedding path), but nothing named
"Rotary" is one of `embedding_type`'s own values, so this sweep uses the three
values the config actually exposes rather than inventing a fourth.

This is a new `twiga.experiment.ExperimentSpec`/`ExperimentEngine` script
(following `ch6_conformal/design_ablation/conditions.py`'s shape), run on one
dataset, SPS-UK (`twiga`'s internal key `MLVS-UK` -- see `data/README.md`),
HPO disabled at the spec level: this is an architecture ablation with fixed,
sensible values at every swept point, not a hyperparameter search, matching
how the paper itself reports this sweep (one architecture value changed at a
time, not independently re-tuned per value).

The dataset's date windows and CV protocol are not re-derived here: they are
read directly from `chapters.ch6_conformal.shared.protocol.DATASETS["MLVS-UK"]`,
which already documents the real SPS-UK/MLVS-UK date ranges -- re-typing them
from scratch would risk silently drifting from the one place that is already
correct.

Usage::

    uv run python chapters/ch4_mlpf/design_choices.py
    uv run python chapters/ch4_mlpf/design_choices.py --group horizon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.models.nn.mlpf_model import MLPFConfig
from twiga.utils.run_forecast import PipelineConfig, setup_environment

from chapters.ch4_mlpf.benchmark import FORECAST_HORIZON, LOOKBACK
from chapters.ch6_conformal.shared.protocol import DATASETS
from twiga_thesis.analysis import dm_and_mcs, fold_level, metric_table
from twiga_thesis.paths import thesis_data_dir
from twiga_thesis.plotting import comparison_plots, line_style, tokens

OUTPUT_DIR = Path("../experiments/ch4_mlpf/design_choices")
DATASET_KEY = "MLVS-UK"  # SPS-UK, in the thesis's own naming (see data/README.md)

POINT_METRIC_TYPES = ["point"]
# Local, proportionate metric list (this chapter's own, matching benchmark.py's METRICS):
# NRMSE drives every figure here, the rest are reported alongside for completeness.
POINT_METRICS = ["nrmse", "mae", "corr", "wmape", "smape"]
LOWER_BETTER = {"nrmse", "mae", "wmape", "smape"}
HIGHER_BETTER = {"corr"}

# ── Reference architecture values (MLPFConfig's own field defaults) ──────────
REFERENCE_HORIZON = FORECAST_HORIZON  # 48 samples = 1 day (benchmark.py's own value)
REFERENCE_LOOKBACK = LOOKBACK  # 336 samples = 7 days (benchmark.py's own value)
REFERENCE_EMBEDDING = None
REFERENCE_COMBINATION = "addition-comb"

# day-multiples at 30-min resolution (48 samples/day)
_SAMPLES_PER_DAY = 48


def _condition(name: str, group: str, description: str, **overrides: object) -> Condition:
    return Condition(
        name=name, group=group, description=description, overrides=overrides, metric_types=POINT_METRIC_TYPES
    )


HORIZON_HOURS = {"1h": 2, "6h": 12, "12h": 24, "1d": 48, "2d": 96, "3d": 144, "5d": 240, "7d": 336}
HORIZON_CONDITIONS = [
    _condition(
        f"horizon_{label}",
        "horizon",
        f"forecast_horizon={samples} samples ({label})",
        forecast_horizon=samples,
        lookback_window_size=REFERENCE_LOOKBACK,
    )
    for label, samples in HORIZON_HOURS.items()
]

CONTEXT_DAYS = {f"{d}d": d * _SAMPLES_PER_DAY for d in range(1, 8)}
CONTEXT_CONDITIONS = [
    _condition(
        f"context_{label}",
        "context_window",
        f"lookback_window_size={samples} samples ({label})",
        lookback_window_size=samples,
        forecast_horizon=REFERENCE_HORIZON,
    )
    for label, samples in CONTEXT_DAYS.items()
]

EMBEDDING_OPTIONS = {"embedding_none": None, "embedding_posemb": "PosEmb", "embedding_learnposemb": "LearnPosEmb"}
EMBEDDING_CONDITIONS = [
    _condition(name, "embedding", f"embedding_type={value!r}", embedding_type=value)
    for name, value in EMBEDDING_OPTIONS.items()
]

COMBINATION_OPTIONS = {
    "combination_addition": "addition-comb",
    "combination_attention": "attn-comb",
    "combination_weighted": "weighted-comb",
}
COMBINATION_CONDITIONS = [
    _condition(name, "combination", f"combination_type={value!r}", combination_type=value)
    for name, value in COMBINATION_OPTIONS.items()
]

DESIGN_CHOICE_CONDITIONS = [*HORIZON_CONDITIONS, *CONTEXT_CONDITIONS, *EMBEDDING_CONDITIONS, *COMBINATION_CONDITIONS]

DESIGN_CHOICE_SPEC = ExperimentSpec(
    name="MLPF Design Choices (Chapter 4, RQ1)",
    output_prefix="ch4_design_choices",
    condition_cls=MLPFConfig,
    backbone_cls=MLPFConfig,
    conditions=DESIGN_CHOICE_CONDITIONS,
    datasets={DATASET_KEY: DATASETS[DATASET_KEY]},
    hpo_enabled=False,
    reference_conditions={
        "horizon": "horizon_1d",
        "context_window": "context_7d",
        "embedding": "embedding_none",
        "combination": "combination_addition",
    },
)

DESIGN_CHOICE_ENGINE = ExperimentEngine(DESIGN_CHOICE_SPEC)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def group_metric_table(full: pd.DataFrame, group: str) -> pd.DataFrame:
    """Fold-then-mean NRMSE (and the rest of `POINT_METRICS`) per condition in one group."""
    fold_frame = fold_level(full, DATASET_KEY, group, POINT_METRICS)
    return metric_table(fold_frame, POINT_METRICS, LOWER_BETTER, HIGHER_BETTER, target_values={})


def group_significance(full: pd.DataFrame, group: str, metric: str = "nrmse", alpha: float = 0.05):
    """DM/MCS significance on `metric` across every condition in one group."""
    fold_frame = fold_level(full, DATASET_KEY, group, POINT_METRICS)
    return dm_and_mcs(fold_frame, alpha=alpha, metric=metric)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_sweep(full: pd.DataFrame, group: str, x_values: dict[str, float], x_label: str, metric: str = "nrmse"):
    """NRMSE against a swept numeric axis (horizon or context window), one line.

    Args:
        full: The engine's full per-fold result frame (`ExperimentEngine.run`'s return value).
        group: `"horizon"` or `"context_window"`.
        x_values: Condition name -> swept numeric value (e.g. `HORIZON_HOURS` with condition-name
            prefixes stripped, see callers).
        x_label: X-axis label (e.g. `"Forecast horizon (samples)"`).
        metric: Metric column plotted on the y-axis.

    Returns:
        The figure; save with `line_style.save`.
    """
    fold_frame = fold_level(full, DATASET_KEY, group, [metric])
    means = fold_frame.groupby("condition")[metric].mean()
    ordered = sorted(x_values, key=lambda name: x_values[name])
    xs = [x_values[name] for name in ordered]
    ys = [means.get(name, float("nan")) for name in ordered]

    line_style.configure(latex=False)
    fig, ax = line_style.new_panel()
    ax.plot(xs, ys, "-o", color=tokens.PALETTE[0], lw=line_style.LINE_WIDTH)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.upper())
    line_style.set_y_axis(ax, [y for y in ys if y == y])
    line_style.finish(ax)
    return fig


def save_all_figures(full: pd.DataFrame, figures_dir: Path) -> None:
    """Build and save all four RQ1 figures to `figures_dir`."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    horizon_values = {f"horizon_{label}": samples for label, samples in HORIZON_HOURS.items()}
    line_style.save(
        plot_sweep(full, "horizon", horizon_values, "Forecast horizon (30-min samples)"),
        figures_dir / "horizon_nrmse",
    )

    context_values = {f"context_{label}": samples for label, samples in CONTEXT_DAYS.items()}
    line_style.save(
        plot_sweep(full, "context_window", context_values, "Context window (30-min samples)"),
        figures_dir / "context_nrmse",
    )

    for group in ("embedding", "combination"):
        fold_frame = fold_level(full, DATASET_KEY, group, POINT_METRICS)
        figures = comparison_plots.plot_metric_bars(
            fold_frame,
            dataset=DATASET_KEY,
            group=group,
            reference=DESIGN_CHOICE_SPEC.reference_conditions[group],
            metric_cols=["nrmse"],
        )
        comparison_plots.save_metric_figures(figures, figures_dir, stem=group, kind="bar")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    base_cfg = PipelineConfig(
        output_dir=f"{output_dir}/",
        data_path=str(thesis_data_dir()),
        out_activation="Identity",
        split_freq="months",
        metrics=POINT_METRICS,
        log_level="INFO",
        log_file=f"{output_dir}/run.log",
        use_latex=False,
        num_threads=1,
    )
    setup_environment(base_cfg)

    all_groups = sorted({condition.group for condition in DESIGN_CHOICE_CONDITIONS})
    selected_groups = all_groups if "all" in args.group else args.group

    full = DESIGN_CHOICE_ENGINE.run(base_cfg=base_cfg, groups=selected_groups, dataset_keys=[DATASET_KEY])
    full.to_csv(output_dir / "design_choices_full.csv", index=False)

    for group in selected_groups:
        table = group_metric_table(full, group)
        table.to_csv(output_dir / f"{group}_metric_table.csv", index=False)
        print(f"\n--- {group} ---")
        print(table.to_string(index=False))

    save_all_figures(full, output_dir / "figures")


if __name__ == "__main__":
    main()
