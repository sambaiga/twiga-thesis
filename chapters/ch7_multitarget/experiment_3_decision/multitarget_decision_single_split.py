r"""Chapter 6, Experiment 3 variant: one fixed, event-balanced split, two stages.

``multitarget_decision.py``'s own four-fold expanding-window protocol was
confirmed (2026-10-05) to produce a systematic, monotone PICP collapse across
folds for every head and both conformal and native conditions alike (fold 4
PICP 0.02-0.28 against fold 1's 0.20-0.51, on both ``DSS-UK`` and
``DSS-US``) -- the signature of distribution shift between each fold's own
trailing calibration window and its test window, not a per-head or
per-method failure. A single, deliberately chosen split sidesteps that
mechanism entirely (no fold marches forward in calendar time) and, because
both decision datasets have a genuinely rare event (~2% overall rate, from
the same 98th-percentile-of-apparent-power construction
``multitarget_decision.decision_rating`` uses), the split points below were
chosen by scanning real per-week event counts so every portion's own event
rate stays close to the dataset's overall rate -- confirmed directly against
the real record, not assumed from calendar proportions alone.

Two stages, one spec, two condition groups, so both share the identical
``val``/``test`` windows and differ in exactly one respect:

- ``native_only`` group: one ``{head}_native`` condition per head,
  ``reserve_calib_window=False`` (this module's own default) and
  ``conformal_config=None`` -- ``ExperimentEngine``'s own
  ``use_calib = (conformal_config is not None or reserve_calib_window) and
  calib_size > 0`` (``twiga/experiment/engine.py``) then forces
  ``calib_size=0`` for these regardless of ``cv_calib_size``, so they are
  trained on the *combined* train+calib span -- answers "how good is this
  head's own native predictive distribution, before paying any data for
  calibration".
- ``decision`` group: ``multitarget_decision._build_conditions``'s own
  native (``{head}_none``, ``reserve_calib_window=True``, same CV layout as
  its calibrated siblings) plus the full conformal grid -- answers "does
  calibration add value, at the cost of the training data it consumes".

Both groups' own ``{head}_none``/``{head}_native`` conditions share the same
``val``/``test`` windows; the only systematic difference between a
``native_only`` condition and its ``decision``-group ``_none`` sibling is
calibration-window size traded against training-window size, with the test
set -- and therefore every Brier-score/calibration number reported for
either -- identical.

Significance testing is **not** computed here: a single split gives one
value per condition, and ``backbone_conformal_decision.family_significance``
needs several folds to compute a fold-level Diebold-Mariano test. Computing
a defensible test under a single split needs per-timestamp paired losses
within the one test window instead (more observations than four folds would
give, in principle, but a different code path than ``family_significance``
assumes) -- left as a follow-up, flagged explicitly rather than reporting a
p-value from one data point.

Usage::

    uv run python experiment/ch7_multitarget/experiment_3_decision/\
multitarget_decision_single_split.py --decisions DSS-UK
    uv run python experiment/ch7_multitarget/experiment_3_decision/\
multitarget_decision_single_split.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_2_backbone_conformal"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from twiga.core.utils import configure, get_logger, slugify
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec, start_run
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.utils.run_forecast import PipelineConfig

from chapters.ch7_multitarget.experiment_2_backbone_conformal.backbone_conformal_study import DATASETS  # noqa: E402
from chapters.ch7_multitarget.experiment_3_decision.backbone_conformal_decision import DEFAULT_OUTPUT_ROOT  # noqa: E402
from chapters.ch7_multitarget.experiment_3_decision.multitarget_decision import (  # noqa: E402
    HEADS,
    INCLUDE_VOLTAGE,
    SCENARIOS,
    VOLTAGE_BANDS,
    _conformal_variants_for,
    build_condition_scorer,
    decision_rating,
)

log = get_logger(__name__)

# Per-dataset (val, calib, test) windows in weeks, chosen 2026-10-05 by
# scanning real per-week event counts on each record (both ~49-52 weeks
# total, both ~2% overall event rate) for the combination whose train/val/
# calib/test portions all land closest to that overall rate while keeping a
# usable test-set event count. val=4 weeks is shared with
# multitarget_decision.py's own convention; calib/test are dataset-specific.
# fit_size (the actual training window) is not set directly -- it is
# whatever ExperimentEngine's own num_splits=1 derivation
# (combined_span = record_duration - test_size, then val/calib carved from
# its tail -- see twiga.core.backtester.TimeBasedCV._conformal_boundaries)
# leaves after val_size and (for the "decision" group only) calib_size are
# removed, confirmed directly against each record's own date range:
#   DSS-UK (49 weeks): val=4, calib=12, test=16 -> rates 0.021/0.022/0.016
#     (overall 0.020); fit=17 weeks ("decision"), 29 weeks ("native_only").
#   DSS-US (52 weeks): val=4, calib=8,  test=12 -> rates 0.021/0.028/0.016
#     (overall 0.020); fit=28 weeks ("decision"), 36 weeks ("native_only").
SPLIT_WEEKS: dict[str, dict[str, int]] = {
    "DSS-UK": {"val": 4, "calib": 12, "test": 16},
    "DSS-US": {"val": 4, "calib": 8, "test": 12},
}


def _build_native_only_conditions(
    rated: float,
    voltage_band: tuple[float, float],
    include_voltage: bool,
    n_paths: int,
    seed: int,
) -> list[Condition]:
    """One ``{head}_native`` condition per head, no calibration window reserved.

    ``reserve_calib_window`` defaults to ``False`` and ``conformal_config``
    is never set here, so ``ExperimentEngine``'s own ``use_calib`` gate
    (see this module's own docstring) forces ``calib_size=0`` for every one
    of these regardless of the spec's own ``cv_calib_size`` -- the combined
    train+calib span becomes pure training data instead.
    """
    conditions = []
    for head_name, head in HEADS.items():
        conditions.append(
            Condition(
                name=f"{head_name}_native",
                group="native_only",
                description=f"{head_name}'s own native predictive distribution, no calibration window reserved",
                overrides={},
                model_cls=head["model_cls"],
                hpo_model_cls=head["hpo_model_cls"],
                hpo_variant=head["hpo_variant"],
                hpo_parent_model_cls=head["hpo_parent_model_cls"],
                hpo_parent_variant=head["hpo_parent_variant"],
                hpo_fixed_params=dict(head["hpo_fixed_params"]),
                conformal_config=None,
                metric_types=["point"],
                custom_scorer=build_condition_scorer(
                    head_name, "none", rated, voltage_band, include_voltage, n_paths, seed
                ),
            )
        )
    return conditions


def _build_decision_conditions(
    rated: float,
    voltage_band: tuple[float, float],
    include_voltage: bool,
    n_paths: int,
    seed: int,
    hpo_objective: str,
) -> list[Condition]:
    """Native + full conformal grid, same shape as ``multitarget_decision._build_conditions``.

    Reimplemented rather than imported directly so this module owns its own
    ``group="decision"`` label independent of the four-fold script's own
    conditions -- the two scripts' conditions never run in the same
    ``ExperimentEngine.run()`` call, so there is no cache-key collision risk
    either way, but keeping the group name explicit here avoids any
    confusion when both scripts' output trees are inspected side by side.
    """
    conditions = []
    for head_name, head in HEADS.items():
        none_name = f"{head_name}_none"
        for method_name, conformal_config in _conformal_variants_for(head_name).items():
            conditions.append(
                Condition(
                    name=f"{head_name}_{method_name}",
                    group="decision",
                    description=f"{head_name} with conformal={method_name}",
                    overrides={},
                    model_cls=head["model_cls"],
                    hpo_model_cls=head["hpo_model_cls"],
                    hpo_variant=head["hpo_variant"],
                    hpo_parent_model_cls=head["hpo_parent_model_cls"],
                    hpo_parent_variant=head["hpo_parent_variant"],
                    hpo_fixed_params=dict(head["hpo_fixed_params"]),
                    conformal_config=conformal_config,
                    reserve_calib_window=(method_name == "none"),
                    reuse_checkpoint_from=None if method_name == "none" else none_name,
                    hpo_objective_metric=hpo_objective if head["is_quantile_head"] else "winkler_score",
                    custom_scorer=build_condition_scorer(
                        head_name, method_name, rated, voltage_band, include_voltage, n_paths, seed
                    ),
                )
            )
    return conditions


def build_spec(key: str, args) -> ExperimentSpec:
    """One ExperimentSpec per decision, single fixed split, both stages as condition groups."""
    scenario = SCENARIOS[key]
    include_voltage = INCLUDE_VOLTAGE[key]
    voltage_band = VOLTAGE_BANDS[key]
    rated = decision_rating(scenario, args.rated_q)
    weeks = SPLIT_WEEKS[key]
    log.info(
        "%s rating %.1f kVA; split val=%dw calib=%dw test=%dw (num_splits=1 -- see module docstring for the "
        "derived fit-window sizes this implies)",
        key,
        rated,
        weeks["val"],
        weeks["calib"],
        weeks["test"],
    )

    native_only = _build_native_only_conditions(rated, voltage_band, include_voltage, args.paths, args.seed)
    decision = _build_decision_conditions(
        rated, voltage_band, include_voltage, args.paths, args.seed, args.hpo_objective
    )

    return ExperimentSpec(
        name=f"Multitarget Decision Single Split — {key}",
        output_prefix=f"ch6_multitarget_decision_single_split_{key.lower()}",
        condition_cls=MLPFFPQRConfig,
        backbone_cls=MLPFFPQRConfig,
        conditions=native_only + decision,
        datasets={scenario: DATASETS[scenario]},
        cv_split_freq="weeks",
        cv_train_size=26,  # nominal, overridden by cv_folds=1's own num_splits derivation
        cv_test_size=weeks["test"],
        cv_val_size=weeks["val"],
        cv_calib_size=weeks["calib"],
        cv_stride=weeks["test"],
        cv_folds=1,
        cv_window="expanding",
        fixed_overrides={"early_stop_patience": 100, "num_workers": 0},
        hpo_train_months=4,
        hpo_val_months=1,
        reference_conditions={"decision": "laplace_max_norm", "native_only": "laplace_native"},
        save_condition_plots=False,
    )


def run_decision(key: str, args, out_dir: Path, output_root: Path) -> pd.DataFrame:
    spec = build_spec(key, args)
    engine = ExperimentEngine(spec)
    cfg = PipelineConfig(
        dataset_name=SCENARIOS[key],
        data_path="examples/data",
        output_dir=str(output_root / "multitarget_decision_single_split" / key.lower()),
        epochs=5 if args.quick else args.epochs,
        num_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )
    groups = ["native_only", "decision"]
    conditions = spec.conditions
    if args.quick:
        conditions = [c for c in conditions if c.name in (f"{args.heads[0]}_native", f"{args.heads[0]}_none")]
        spec.conditions = conditions
        spec.reference_conditions = {
            "decision": f"{args.heads[0]}_none",
            "native_only": f"{args.heads[0]}_native",
        }

    summary = engine.run(base_cfg=cfg, groups=groups, dataset_keys=[SCENARIOS[key]], skip_hpo=args.skip_hpo)

    experiment_root = Path(cfg.output_dir) / slugify(spec.name)
    # See multitarget_decision.run_decision's own comment: ExperimentEngine's
    # "latest" symlink is best-effort and silently absent on a filesystem
    # that refuses an unprivileged symlink (Windows, by default) -- find the
    # real run directory directly instead.
    runs_dir = experiment_root / "runs"
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime) if runs_dir.exists() else []
    run_root = run_dirs[-1] if run_dirs else experiment_root / "latest"
    parquet_files = list(run_root.rglob("metrics.parquet"))
    if not parquet_files:
        log.warning("%s: no metrics.parquet found under %s — run produced no results", key, run_root)
        return pd.DataFrame()
    full = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    full["decision"] = key

    is_curve = full["tier"] == "value_curve"
    curve_full, full = full[is_curve].copy(), full[~is_curve].copy()
    if not curve_full.empty:
        curve_full.to_csv(out_dir / f"{key.lower()}_value_curve.csv", index=False)

    full.to_csv(out_dir / f"{key.lower()}_per_fold.csv", index=False)

    agg = full.groupby(["metric", "target", "model", "group"])["value"].agg(["mean", "std"]).reset_index()
    agg.to_csv(out_dir / f"{key.lower()}_summary.csv", index=False)
    print(f"\n=== {key} summary (single split; native_only vs decision groups) ===")
    print(
        agg[agg["metric"] == "Brier score"]
        .pivot_table(index="group", columns="model", values="mean")
        .round(4)
        .to_string()
    )
    log.info(
        "%s: significance not computed (single split -- see module docstring); engine summary written under %s",
        key,
        summary,
    )
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument("--rated-q", type=float, default=0.98)
    parser.add_argument("--skip-hpo", dest="skip_hpo", action="store_true", default=True)
    parser.add_argument("--run-hpo", dest="skip_hpo", action="store_false")
    parser.add_argument("--hpo-trials", type=int, default=50)
    parser.add_argument("--hpo-epochs", type=int, default=100)
    parser.add_argument(
        "--hpo-objective",
        default="winkler_score",
        help="HPO objective for fpqr_anchor_outer/qr only, matching multitarget_decision.py's own flag.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.paths = 20
        args.decisions = args.decisions[:1]
        args.heads = args.heads[:1]

    configure(level="INFO")
    run_root = start_run(
        args.output_root / "multitarget_decision_single_split",
        meta={
            "decisions": args.decisions,
            "heads": args.heads,
            "epochs": args.epochs,
            "seed": args.seed,
            "split_weeks": SPLIT_WEEKS,
        },
    )
    out_dir = run_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {key: run_decision(key, args, out_dir, args.output_root) for key in args.decisions}
    (out_dir / "multitarget_decision_single_split_config.json").write_text(
        json.dumps(
            {
                "decisions": args.decisions,
                "heads": args.heads,
                "epochs": args.epochs,
                "seed": args.seed,
                "split_weeks": SPLIT_WEEKS,
                "rows_per_decision": {k: len(v) for k, v in results.items()},
            },
            indent=2,
        )
    )
    log.info("wrote single-split per-decision results under %s", out_dir)


if __name__ == "__main__":
    main()
