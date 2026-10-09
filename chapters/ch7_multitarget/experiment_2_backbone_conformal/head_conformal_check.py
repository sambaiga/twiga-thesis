r"""Quick check: does every backbone work across every probabilistic head x conformal variant x dataset.

Not a chapter experiment -- ``multitarget_experiment.py`` (Experiment 1, MLPF
only, FPQR/QR/BNN/Laplace) and ``backbone_conformal_study.py`` (Experiment 2,
MLPF/MLPGAM/NHiTS, FPQR only) stay exactly as they are; this script exists
purely to sanity-check a proposed extension -- every backbone paired with
every head, under every conformal variant, on every dataset -- before
committing to folding any of it into either formal script. Small epoch
counts, ``--quick`` by default in spirit, run it, look at the numbers and
panels, decide.

Grid
----
Backbones: MLPF, MLPGAM, NHiTS (paired config classes for all four heads
exist for all three -- confirmed directly against ``twiga/models/nn/``
before writing this, not assumed).
Heads: FPQR, QR (quantile-kind), BNN, Laplace (parametric).
Conformal: ``none`` (always), ``max_norm``/``mahalanobis`` (any head, via
``core.py``'s quantile-derived adaptive scale for FPQR/QR, native scale for
BNN/Laplace), ``quantile``/``quantile_joint`` (CQR, quantile-kind heads
only -- BNN/Laplace have no predicted quantile interval for CQR to resize).
Datasets: DSS-EU, MLVS-PT-3PHASE, DSS-UK, DSS-US -- the same four
``backbone_conformal_study.py`` already covers, and the same per-dataset CV
numbers, cloned unchanged via ``dataclasses.replace`` rather than re-derived
(DSS-EU keeps ``cv_calib_size=0``, so its conformal-bearing conditions are
skipped, same convention as that script).

Panels
------
``forecast_examples.py``'s own drawing utilities (extracted to
``experiment/common/forecast_panels.py``: observed series as hollow markers,
predictive median dashed, prediction interval as a translucent band) are
reused unchanged for a best-day/worst-day panel per condition x target, so
this script's output reads in the same visual convention as the thesis
figures rather than a different ad hoc style. Unlike ``forecast_examples.py``
(a full quantile fan from several nested levels), this draws a single
interval band from each condition's saved ``lower``/``forecast``/``upper``
columns -- exactly the "single prediction interval" case
``forecast_panels.draw``'s own docstring describes. Best/worst is picked by
a simple per-day RMSE-normalised-by-mean-|Actual| proxy, not the engine's
own NRMSE definition -- adequate for choosing which day to look at, not
offered as a metrics table (the CSV summary is that).

Usage::

    uv run python experiment/ch7_multitarget/head_conformal_check.py --quick
    uv run python experiment/ch7_multitarget/head_conformal_check.py --dataset MLVS-PT-3PHASE \
        --backbones nhits mlpgam
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from backbone_conformal_study import (  # noqa: E402
    BACKBONES,
    CALIBRATED_CONFORMAL_VARIANTS,
    CONFORMAL_VARIANTS,
    DATASETS,
    DSS_EU,
    DSS_UK,
    DSS_US,
    SPEC_DSS_EU,
    SPEC_DSS_UK,
    SPEC_DSS_US,
    SPEC_MLVS_PT_3PHASE,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis_figures.style import (  # noqa: E402
    configure as configure_style,  # noqa: E402
    pad_limits,
    panel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from forecast_panels import draw, place_legend_below  # noqa: E402
from twiga.core.utils import configure, get_logger, slugify
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.models.nn.mlpfbnn_model import MLPFBNNConfig
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig
from twiga.models.nn.mlpfqr_model import MLPFQRConfig
from twiga.models.nn.mlpgambnn_model import MLPGAMBNNConfig
from twiga.models.nn.mlpgamlaplace_model import MLPGAMLaplaceConfig
from twiga.models.nn.mlpgamqr_model import MLPGAMQRConfig
from twiga.models.nn.nhitsbnn_model import NHITSBNNConfig
from twiga.models.nn.nhitslaplace_model import NHITSLaplaceConfig
from twiga.models.nn.nhitsqr_model import NHITSQRConfig
from twiga.utils.run_forecast import PipelineConfig

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
GROUP = "head_check"
OUTPUT_PREFIX = "head_conformal_check"
# Same external base the formal chapter scripts share
# (multitarget_experiment.py/backbone_conformal_study.py/
# backbone_conformal_decision.py all use
# .../experiments/ch7_multitarget/<script-name>/), so run output --
# checkpoints, predictions, plots, figures, all of it, not only the CSVs --
# never accumulates inside the repo. "exploratory" is this script's own
# subfolder there, keeping its output apart from a chapter result's without
# needing a different top-level location for it.
EXPLORATORY_OUT = script_dir.parent.parent.parent.parent / "experiments" / "ch7_multitarget" / "exploratory"

# ── Heads, per backbone: (config class, overrides, is-quantile-kind) ──────────
# fpqr entries reuse backbone_conformal_study.py's own BACKBONES dict
# directly (config class + overrides) rather than duplicating them here --
# that dict carries the FPQR mechanism ablation's fixed config
# (proposal_type="softmax", qvn_type="standard", the ablation-confirmed
# winners; MLPF's own crossing_penalty/rope_type restoration) and any future
# fix to it (e.g. 2026-09-26's proposal_type/qvn_type fix, made after a
# fully-collapsed NHiTS quantile grid was traced to HPO picking the
# ablation-confirmed-worst proposal_type="horizon"/qvn_type="attention" on
# DSS-UK) should not need to be applied twice.
HEADS: dict[str, dict[str, tuple[type, dict, bool]]] = {
    "mlpf": {
        "fpqr": (*BACKBONES["mlpf"], True),
        "qr": (MLPFQRConfig, {}, True),
        "bnn": (MLPFBNNConfig, {}, False),
        "laplace": (MLPFLaplaceConfig, {}, False),
    },
    "mlpgam": {
        "fpqr": (*BACKBONES["mlpgam"], True),
        "qr": (MLPGAMQRConfig, {}, True),
        "bnn": (MLPGAMBNNConfig, {}, False),
        "laplace": (MLPGAMLaplaceConfig, {}, False),
    },
    "nhits": {
        "fpqr": (*BACKBONES["nhits"], True),
        "qr": (NHITSQRConfig, {}, True),
        "bnn": (NHITSBNNConfig, {}, False),
        "laplace": (NHITSLaplaceConfig, {}, False),
    },
}
_QUANTILE_METRICS = ["point", "interval", "quantile"]
_PARAMETRIC_METRICS = ["point", "interval", "parametric"]
_CONFORMAL_METRICS = ["point", "interval"]
_QUANTILE_CQR_ONLY = ("quantile", "quantile_joint")


def build_conditions(backbones: list[str]) -> list[Condition]:
    """Every (backbone, head, conformal) cell the head's own kind supports."""
    conditions = []
    for backbone_name in backbones:
        for head_name, (config_cls, overrides, is_quantile) in HEADS[backbone_name].items():
            if is_quantile:
                applicable = CONFORMAL_VARIANTS
            else:
                applicable = {k: v for k, v in CONFORMAL_VARIANTS.items() if k not in _QUANTILE_CQR_ONLY}
            for conformal_name, conformal_config in applicable.items():
                base_metrics = _QUANTILE_METRICS if is_quantile else _PARAMETRIC_METRICS
                conditions.append(
                    Condition(
                        name=f"{backbone_name}_{head_name}_{conformal_name}",
                        group=GROUP,
                        description=f"{head_name.upper()} on {backbone_name.upper()}, conformal={conformal_name}",
                        overrides=overrides,
                        model_cls=config_cls,
                        metric_types=base_metrics if conformal_config is None else _CONFORMAL_METRICS,
                        conformal_config=conformal_config,
                    )
                )
    return conditions


def build_spec(dataset_key: str, conditions: list[Condition]) -> ExperimentSpec:
    """One ExperimentSpec per dataset, cloned from backbone_conformal_study.py's own verified specs.

    Re-deriving these CV numbers by hand is exactly how an earlier version of
    this function got DSS-EU's cv_folds wrong (8 instead of 10) and crashed
    with "window shape cannot be larger than input array shape" -- cloning
    the already-correct, already-run spec objects via dataclasses.replace and
    only swapping conditions/reference_conditions avoids that class of
    mistake entirely.
    """
    template = {
        DSS_EU: SPEC_DSS_EU,
        "MLVS-PT-3PHASE": SPEC_MLVS_PT_3PHASE,
        DSS_UK: SPEC_DSS_UK,
        DSS_US: SPEC_DSS_US,
    }[dataset_key]
    names = {c.name for c in conditions}
    reference = "mlpf_fpqr_none" if "mlpf_fpqr_none" in names else conditions[0].name
    overrides: dict = {
        "name": "Head x Conformal Check",
        "output_prefix": OUTPUT_PREFIX,
        "condition_cls": conditions[0].model_cls,
        "backbone_cls": conditions[0].model_cls,
        "conditions": conditions,
        "reference_conditions": {GROUP: reference},
    }
    if dataset_key == DSS_EU:
        # SPEC_DSS_EU inherits cv_folds=10 from _SPEC_COMMON, but DSS-EU's
        # real ~11-month span (confirmed against the parquet) only fits 8
        # non-overlapping-enough folds at this cv_test_size/cv_stride --
        # TimeBasedCV silently reduces to 8 and logs a WARNING rather than
        # erroring (backbone_conformal_study.py's own DSS-EU runs hit the
        # identical warning; not something this script introduced). Stated
        # explicitly here instead of relying on that silent reduction, same
        # as every other dataset's fold count already matches what its spec
        # states.
        overrides["cv_folds"] = 8
    return dc_replace(template, **overrides)


def base_config(epochs: int) -> PipelineConfig:
    return PipelineConfig(
        dataset_name=DSS_EU,
        data_path="examples/data",
        output_dir=str(EXPLORATORY_OUT),
        epochs=epochs,
        num_trials=1,
        hpo_epochs=5,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )


def _latest_run_root(output_dir: str, spec_name: str) -> Path | None:
    """The most recently created ``runs/<run_id>`` directory, without relying on the `latest` symlink.

    ExperimentEngine also writes a ``<experiment_root>/latest`` symlink
    pointing here, but that update is best-effort and swallows failure at
    DEBUG level (``twiga.experiment.engine._update_latest_symlink``).
    Creating a symlink on Windows requires Administrator privileges or
    Developer Mode enabled, neither of which is the default -- confirmed
    directly: on a Windows run, ``latest`` was silently never created, so
    every panel lookup against it found nothing and was silently skipped.
    ``run_id`` is timestamp-prefixed (``YYYYMMDD_HHMMSS_<gitsha>``), so a
    lexicographic sort of ``runs/*`` is a chronological one -- this needs no
    symlink support at all, on any platform.
    """
    runs_dir = Path(output_dir) / slugify(spec_name) / "runs"
    run_dirs = sorted(p for p in runs_dir.glob("*") if p.is_dir())
    if not run_dirs:
        log.warning("panels: no run directory found under %s", runs_dir)
        return None
    return run_dirs[-1]


def _day_error(group: pd.DataFrame) -> float:
    """RMSE normalised by mean |Actual|, a cheap proxy for picking a representative day."""
    resid = group["forecast"] - group["Actual"]
    denom = group["Actual"].abs().mean()
    return float(np.sqrt((resid**2).mean()) / denom) if denom > 1e-9 else float(np.sqrt((resid**2).mean()))


def plot_best_worst_panels(run_root: Path, dataset_key: str, conditions: list[Condition]) -> None:
    """One best-day and one worst-day panel per condition x target, in forecast_examples.py's own style."""
    plots_dir = run_root / "plots" / slugify(dataset_key) / "panels"
    plots_dir.mkdir(parents=True, exist_ok=True)
    conditions_root = run_root / "conditions" / slugify(dataset_key) / GROUP

    for condition in conditions:
        pred_path = conditions_root / condition.name / "predictions.parquet"
        if not pred_path.exists():
            log.warning("panels: %s has no predictions.parquet, skipped", condition.name)
            continue
        preds = pd.read_parquet(pred_path)
        if not {"forecast", "Actual", "lower", "upper", "timestamp", "target"}.issubset(preds.columns):
            log.warning("panels: %s predictions missing lower/upper columns, skipped", condition.name)
            continue
        preds = preds.dropna(subset=["lower", "upper"])
        if preds.empty:
            continue
        preds["date"] = pd.to_datetime(preds["timestamp"]).dt.date

        for target, target_df in preds.groupby("target"):
            day_errors = target_df.groupby("date").apply(_day_error, include_groups=False)
            if day_errors.empty:
                continue
            for kind, day in (("best", day_errors.idxmin()), ("worst", day_errors.idxmax())):
                day_df = target_df[target_df["date"] == day].sort_values("timestamp")
                t_axis = pd.to_datetime(day_df["timestamp"]).to_numpy()
                y = day_df["Actual"].to_numpy()
                q = np.stack([day_df["lower"].to_numpy(), day_df["forecast"].to_numpy(), day_df["upper"].to_numpy()])
                levels = np.array([0.05, 0.5, 0.95])
                ylim = pad_limits(y, q)

                fig, ax = plt.subplots(figsize=panel(1.0))
                draw(ax, t_axis, y, q, levels, 0, ylim=ylim)
                place_legend_below(fig, ax, ncol=3, fontsize=7, frameon=False)
                fig.tight_layout()
                out = plots_dir / f"{condition.name}_{target}_{kind}.pdf"
                fig.savefig(out, dpi=600, bbox_inches="tight")
                plt.close(fig)
        log.info("panels: %s saved", condition.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=[DSS_EU], choices=[*DATASETS, "all"])
    ap.add_argument("--backbones", nargs="+", default=["mlpf", "mlpgam", "nhits"], choices=list(HEADS))
    ap.add_argument("--heads", nargs="+", default=list(HEADS["mlpf"]), choices=list(HEADS["mlpf"]))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-panels", action="store_true", help="skip the best/worst-day panel plots")
    ap.add_argument("--quick", action="store_true", help="1 dataset, 1 backbone, 1 head, 5 epochs")
    args = ap.parse_args()

    configure(level="INFO")
    try:
        configure_style(latex=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("thesis style unavailable (%s); falling back to matplotlib defaults", exc)

    dataset_keys = list(DATASETS) if "all" in args.dataset else args.dataset
    backbones = args.backbones
    heads = args.heads
    epochs = args.epochs
    if args.quick:
        dataset_keys = dataset_keys[:1]
        backbones = backbones[:1]
        heads = heads[:1]
        epochs = 5

    conditions = [c for c in build_conditions(backbones) if c.name.split("_")[1] in heads]
    log.info("conditions: %s", [c.name for c in conditions])
    cfg = base_config(epochs)

    summaries = []
    for dataset_key in dataset_keys:
        ds_conditions = conditions
        if dataset_key == DSS_EU:
            ds_conditions = [c for c in conditions if not c.name.endswith(CALIBRATED_CONFORMAL_VARIANTS)]
            log.info("Skipping conformal-bearing conditions on %s (cv_calib_size=0)", dataset_key)
        spec = build_spec(dataset_key, ds_conditions)
        engine = ExperimentEngine(spec)
        summary = engine.run(base_cfg=cfg, groups=[GROUP], dataset_keys=[dataset_key], skip_hpo=True)
        if not summary.empty:
            summary["dataset"] = dataset_key
            summaries.append(summary)
            if not args.no_panels:
                run_root = _latest_run_root(cfg.output_dir, spec.name)
                if run_root is not None:
                    plot_best_worst_panels(run_root, dataset_key, ds_conditions)

    if not summaries:
        raise SystemExit("no dataset produced results")
    full = pd.concat(summaries, ignore_index=True)
    out = EXPLORATORY_OUT / f"results_{OUTPUT_PREFIX}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    log.info("wrote %s", out)
    print(full.to_string())


if __name__ == "__main__":
    main()
