r"""Chapter 6, Experiment 3: does calibrated conformal help the decision, MLPF-only, all four heads.

``backbone_conformal_decision.py`` answers the backbone question (MLPF vs
MLPGAM vs NHiTS, each carrying only the FPQR head) with a narrow,
single-score conformal grid (``none``/``max_norm``/CQR). This script answers
the companion head question instead: on a single, fixed MLPF backbone, does
the choice of probabilistic head (BNN, Laplace, QR, or the chapter's own
FPQR configuration) change the operational decision, under the same
conformal methodology ``multitarget_experiment.py``'s own ``benchmark``/
``conformal`` groups and Chapter-6.tex's ``sec:ch6-multitarget-cp``/
``sec:ch6-multitarget-cqr`` have already settled on (max-norm for every
head, plus the full six-variant CQR grid for the two quantile heads).

Deliberately a new file, not an extension of ``backbone_conformal_decision.py``:
that script's own head axis (backbone architecture, FPQR-only) and this
one's (probabilistic head, MLPF-only) are different comparisons that happen
to share machinery, not one script with two purposes. Every reusable piece
(ensemble sampling, interval metrics, DM significance, the hosting-capacity
row builder) is imported directly from ``backbone_conformal_decision.py``
rather than duplicated; only the head set, conformal-variant gating (per
``is_quantile_head``, matching ``multitarget_experiment.py``'s own
``_CONFORMAL_HEADS`` pattern) and the scorer's two new parametric-head
sampling branches (BNN/Laplace have no quantile grid at all, see
``custom_scorer`` below) are new.

Usage::

    uv run python experiment/ch7_multitarget/experiment_3_decision/multitarget_decision.py --decisions DSS-UK
    uv run python experiment/ch7_multitarget/experiment_3_decision/multitarget_decision.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_2_backbone_conformal"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_1_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from twiga.core.config import ConformalConfig
from twiga.core.utils import configure, get_logger, slugify
from twiga.distributions.ml.utils import get_sigma_from_quantile_levels
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec, start_run
from twiga.models.nn import MLPFConfig
from twiga.models.nn.mlpfbnn_model import MLPFBNNConfig
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig
from twiga.models.nn.mlpfqr_model import MLPFQRConfig
from twiga.utils.data_loading import load_dataset
from twiga.utils.run_forecast import PipelineConfig

from chapters.ch7_multitarget.experiment_1_benchmark.multitarget_experiment import (  # noqa: E402
    _FPQR_STABILITY_PINS,
    _REFERENCE_TRAINING_PINS,
    FPQR_ANCHOR_OUTER_HPO_VARIANT,
    SHARED_HPO_VARIANT,
)
from chapters.ch7_multitarget.experiment_2_backbone_conformal.backbone_conformal_study import (  # noqa: E402
    DATASETS,
    DSS_UK,
    DSS_US,
)
from chapters.ch7_multitarget.experiment_3_decision.backbone_conformal_decision import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    INTERVAL_ALPHA,
    family_significance,
    interval_metric_rows,
    sample_calibrated_ensemble,
    sample_quantile_ensemble,
)
from chapters.ch7_multitarget.experiment_3_decision.operational_decision import (  # noqa: E402
    INCLUDE_VOLTAGE,
    VOLTAGE_BANDS,
)
from chapters.ch7_multitarget.shared.forecast_examples import hosting_capacity_rows  # noqa: E402
from twiga_thesis.paths import thesis_data_dir

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
SCENARIOS = {"DSS-UK": DSS_UK, "DSS-US": DSS_US}

# ── Heads: multitarget_experiment.py's own four, MLPF backbone only ───────────
# Mirrors multitarget_experiment.py's _CONFORMAL_HEADS exactly (same pins,
# same hpo_parent wiring) rather than backbone_conformal_decision.py's own
# per-backbone HEADS -- this script holds the backbone fixed at MLPF and
# varies the head instead.
#
# Training-level fields (optimizer_type/lr_scheduler_type/batch_size/
# optimizer_params/scheduler_params/gradient_clip_val) now pinned via
# _REFERENCE_TRAINING_PINS (2026-10-03), aligning with
# multitarget_experiment.py's own current state -- see that constant's own
# comment there for values/rationale/history. fpqr_anchor_outer's own
# per_target_tau/proposal_type/proposal_anchor_outer pins are unaffected by
# that alignment and are kept exactly as before ("keep decision as it is,
# it performs well there" -- this script's own architecture-level pins were
# never part of the "all free" experiment on multitarget_experiment.py's
# side and are not part of this alignment either).
HEADS: dict[str, dict] = {
    "fpqr_anchor_outer": {
        "model_cls": MLPFFPQRConfig,
        "hpo_model_cls": MLPFFPQRConfig,
        "hpo_variant": FPQR_ANCHOR_OUTER_HPO_VARIANT,
        "hpo_parent_model_cls": MLPFConfig,
        "hpo_parent_variant": SHARED_HPO_VARIANT,
        "hpo_fixed_params": {
            **_FPQR_STABILITY_PINS,
            "per_target_tau": True,
            "proposal_type": "softmax",
            "proposal_anchor_outer": True,
            **_REFERENCE_TRAINING_PINS,
        },
        "is_quantile_head": True,
    },
    "qr": {
        "model_cls": MLPFQRConfig,
        "hpo_model_cls": MLPFConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_parent_model_cls": None,
        "hpo_parent_variant": "",
        "hpo_fixed_params": dict(_REFERENCE_TRAINING_PINS),
        "is_quantile_head": True,
    },
    "bnn": {
        "model_cls": MLPFBNNConfig,
        "hpo_model_cls": MLPFConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_parent_model_cls": None,
        "hpo_parent_variant": "",
        "hpo_fixed_params": dict(_REFERENCE_TRAINING_PINS),
        "is_quantile_head": False,
    },
    "laplace": {
        "model_cls": MLPFLaplaceConfig,
        "hpo_model_cls": MLPFConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_parent_model_cls": None,
        "hpo_parent_variant": "",
        "hpo_fixed_params": dict(_REFERENCE_TRAINING_PINS),
        "is_quantile_head": False,
    },
}

# ── Conformal: the chapter's own agreed set (max-norm + full CQR grid), ───────
# gated per head exactly as multitarget_experiment.py's _build_conformal_conditions
# does -- BNN/Laplace have no quantile grid to calibrate a CQR score around,
# so they only ever get "none"/"max_norm".
MULTIVARIATE_CONFORMAL_VARIANTS: dict[str, ConformalConfig] = {
    "max_norm": ConformalConfig(method="multivariate", score_type="max-norm", alpha=0.1),
}
QUANTILE_CONFORMAL_VARIANTS: dict[str, ConformalConfig] = {
    "quantile_scaled_marginal": ConformalConfig(
        method="quantile", score_type="scaled", target_coverage="marginal", alpha=0.1
    ),
    "quantile_scaled_pooled": ConformalConfig(
        method="quantile", score_type="scaled", target_coverage="pooled", alpha=0.1
    ),
    "quantile_scaled_bonferroni": ConformalConfig(
        method="quantile", score_type="scaled", target_coverage="bonferroni", alpha=0.1
    ),
    "quantile_unscaled_marginal": ConformalConfig(
        method="quantile", score_type="unscaled", target_coverage="marginal", alpha=0.1
    ),
    "quantile_unscaled_pooled": ConformalConfig(
        method="quantile", score_type="unscaled", target_coverage="pooled", alpha=0.1
    ),
    "quantile_unscaled_bonferroni": ConformalConfig(
        method="quantile", score_type="unscaled", target_coverage="bonferroni", alpha=0.1
    ),
}


def _conformal_variants_for(head_name: str) -> dict[str, ConformalConfig | None]:
    variants: dict[str, ConformalConfig | None] = {"none": None, **MULTIVARIATE_CONFORMAL_VARIANTS}
    if HEADS[head_name]["is_quantile_head"]:
        variants.update(QUANTILE_CONFORMAL_VARIANTS)
    return variants


def decision_rating(scenario: str, quantile: float) -> float:
    """Apparent-power rating at *quantile* of the whole record's own S, not per fold.

    See backbone_conformal_decision.py's own docstring.
    """
    data, targets, *_ = load_dataset(scenario, data_path=str(thesis_data_dir()))
    p, q = data[targets[0]].to_numpy(), data[targets[1]].to_numpy()
    return float(np.quantile(np.hypot(p, q), quantile))


# ── Scorer: adds the two parametric-head (BNN/Laplace) sampling branches ─────
# backbone_conformal_decision.py's own build_condition_scorer assumes every
# head produces result.quantiles/result.quantile_levels -- true for its own
# FPQR/QR-only head set, but BNN/Laplace never populate those fields at all
# (ForecastResult.scale is the one they populate instead). Both new branches
# below reduce to the same (n_paths, N, H, D) ensemble shape every other
# branch produces, so hosting_capacity_rows/interval_metric_rows downstream
# are untouched.
def build_condition_scorer(
    head_name: str,
    method_name: str,
    rated: float,
    voltage_band: tuple[float, float],
    include_voltage: bool,
    n_paths: int,
    seed: int,
):
    is_quantile_head = HEADS[head_name]["is_quantile_head"]

    def custom_scorer(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        collection = forecaster.forecast(test_df)
        result = next(iter(collection))
        loc = np.asarray(result.loc)
        truth = np.asarray(result.ground_truth)

        if is_quantile_head:
            quantiles = np.asarray(result.quantiles)
            quantile_levels = np.asarray(result.quantile_levels)
            if method_name == "none":
                ensemble = sample_quantile_ensemble(quantiles, quantile_levels, n_paths, rng)
                interval_lower, interval_upper = quantiles[:, 0, :, :], quantiles[:, -1, :, :]
            elif method_name == "max_norm":
                conformal = forecaster.conformal[result.model_name]
                sigma = get_sigma_from_quantile_levels(quantiles, quantile_levels)
                q_hat = np.asarray(conformal.q_hat)
                n_targets = loc.shape[-1]
                identity_corr = np.broadcast_to(np.eye(n_targets), (loc.shape[1], n_targets, n_targets))
                ensemble = sample_calibrated_ensemble(loc, sigma, identity_corr, q_hat, "max-norm", n_paths, rng)
                radius = q_hat[None, :, None] * sigma
                interval_lower, interval_upper = loc - radius, loc + radius
            else:  # CQR variants
                conformal = forecaster.conformal[result.model_name]
                lower, upper = quantiles[:, 0, :, :], quantiles[:, -1, :, :]
                lower_adj, upper_adj = conformal.generate_intervals(lower, upper)
                interval_lower, interval_upper = np.asarray(lower_adj), np.asarray(upper_adj)
                ensemble = sample_quantile_ensemble(
                    quantiles, quantile_levels, n_paths, rng, lower_adj=interval_lower, upper_adj=interval_upper
                )
        else:
            # BNN/Laplace: no quantile grid at all -- ForecastResult.scale is
            # this head's own native scale parameter (BNN's approximate
            # posterior std-dev, Laplace's own scale b), so "none" samples
            # directly from that head's predictive distribution rather than
            # by inverse-transform on a grid that doesn't exist.
            scale = np.asarray(result.scale)
            sampler = rng.laplace if head_name == "laplace" else rng.normal
            if method_name == "none":
                ensemble = loc[None] + sampler(size=(n_paths, *loc.shape)) * scale[None]
                # Nominal (1-INTERVAL_ALPHA) central interval, not a raw
                # +/-1 scale band: a Normal's own 1-sigma band is ~68%
                # coverage and a Laplace's is ~86%, neither matching the
                # alpha=0.1 (90%) convention interval_metric_rows' own
                # PICP/NMPI/Winkler/CWE columns are computed against
                # everywhere else in this project -- the per-distribution
                # multiplier below converts scale into that same nominal
                # interval so bnn_none/laplace_none are comparable to every
                # other head's "none" condition and to
                # multitarget_experiment.py's own benchmark-group reporting.
                # Laplace's own inverse CDF gives multiplier=-ln(alpha)
                # exactly (not an approximation): at p=1-alpha/2,
                # 2*(1-p)=alpha, and the standard Laplace quantile for p>=0.5
                # is -ln(2*(1-p)).
                multiplier = -np.log(INTERVAL_ALPHA) if head_name == "laplace" else norm.ppf(1 - INTERVAL_ALPHA / 2)
                radius = multiplier * scale
                interval_lower, interval_upper = loc - radius, loc + radius
            else:  # "max_norm" -- the only variant is_quantile_head=False heads ever receive
                conformal = forecaster.conformal[result.model_name]
                q_hat = np.asarray(conformal.q_hat)
                n_targets = loc.shape[-1]
                identity_corr = np.broadcast_to(np.eye(n_targets), (loc.shape[1], n_targets, n_targets))
                ensemble = sample_calibrated_ensemble(loc, scale, identity_corr, q_hat, "max-norm", n_paths, rng)
                radius = q_hat[None, :, None] * scale
                interval_lower, interval_upper = loc - radius, loc + radius

        days = np.arange(truth.shape[0])
        preds = {method_name: (None, None, ensemble, None)}
        curve_rows: list[dict] = []
        operational_df = hosting_capacity_rows(
            preds, truth, result.targets, rated, days, voltage_band, include_voltage, curve_sink=curve_rows
        )
        interval_df = pd.DataFrame(
            interval_metric_rows(
                f"{head_name}_{method_name}", result.targets, truth, interval_lower, interval_upper, loc
            )
        )
        # Simultaneous P/Q/V coverage (subsec:ch6-metrics' own calibration-
        # performance criterion for this experiment, never actually computed
        # here before): the fraction of (sample, horizon) instances for which
        # every target's own interval contains its ground truth at once, not
        # just each target's own marginal PICP (already in interval_df above).
        within_interval = (truth >= interval_lower) & (truth <= interval_upper)
        joint_coverage_df = pd.DataFrame(
            [
                {
                    "tier": "operational",
                    "metric": "joint_coverage",
                    "target": "+".join(result.targets),
                    "model": f"{head_name}_{method_name}",
                    "value": float(within_interval.all(axis=-1).mean()),
                }
            ]
        )
        curve_df = pd.DataFrame(curve_rows)
        if not curve_df.empty:
            curve_df["tier"] = "value_curve"
        metric_df = pd.concat([operational_df, interval_df, joint_coverage_df, curve_df], ignore_index=True)
        thermal = np.hypot(ensemble[..., 0], ensemble[..., 1]) > rated
        union_hat = thermal.any(axis=-1).mean(axis=0)
        pred_df = pd.DataFrame({"union_hat": union_hat})
        return pred_df, metric_df

    return custom_scorer


def _build_conditions(
    rated: float,
    voltage_band: tuple[float, float],
    include_voltage: bool,
    n_paths: int,
    seed: int,
    hpo_objective: str = "winkler_score",
):
    """4 heads (fpqr_anchor_outer/qr/bnn/laplace) x their own agreed conformal set, group="decision".

    ``hpo_objective`` only applies to the two quantile-producing heads
    (fpqr_anchor_outer/qr) -- BNN/Laplace stay point-accuracy, matching
    multitarget_experiment.py's own restriction (``cond.model_cls in
    (MLPFFPQRConfig, MLPFQRConfig)``).
    """
    conditions = []
    for head_name, head in HEADS.items():
        # "none" is this head's trained-weights source for every other
        # conformal variant (reuse_checkpoint_from) -- one fit per head per
        # fold, not one per head-x-conformal-method cell. reserve_calib_window
        # carves out the same calibration window its calibrated siblings do
        # (required for reuse_checkpoint_from; see ExperimentEngine's own
        # validation in _resolve_reuse_checkpoint_source).
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
    """One ExperimentSpec per decision (DSS-UK/DSS-US), MLPF backbone, all four heads."""
    scenario = SCENARIOS[key]
    include_voltage = INCLUDE_VOLTAGE[key]
    voltage_band = VOLTAGE_BANDS[key]
    rated = decision_rating(scenario, args.rated_q)
    log.info("%s rating   %.1f kVA at the %.0f%% quantile of recorded apparent power", key, rated, args.rated_q * 100)

    return ExperimentSpec(
        name=f"Multitarget Decision — {key}",
        output_prefix=f"ch6_multitarget_decision_{key.lower()}",
        condition_cls=MLPFFPQRConfig,
        backbone_cls=MLPFFPQRConfig,
        conditions=_build_conditions(rated, voltage_band, include_voltage, args.paths, args.seed, args.hpo_objective),
        datasets={scenario: DATASETS[scenario]},
        cv_split_freq="weeks",
        cv_train_size=26,
        cv_test_size=args.test_size,
        # val=2/calib=8 (was 4/4, 2026-10-04): confirmed directly against the
        # real DSS-UK/DSS-US records that the old 4-week calibration window
        # left some folds with as few as 12-14 thermal/hosting-capacity
        # exceedance events (DSS-UK fold 1/3) to calibrate against, while a
        # 2-week validation window is already ample for early stopping. At
        # val=2/calib=8, every fold's calibration window holds 40-91 events
        # on both datasets -- the fold count, stride, and test window are
        # unchanged, so this costs nothing but a few calibration-window weeks
        # borrowed from validation.
        cv_val_size=2,
        cv_calib_size=8,
        cv_stride=args.step,
        cv_folds=4,
        cv_window="expanding",
        fixed_overrides={"early_stop_patience": 100, "num_workers": 0},
        hpo_train_months=4,
        hpo_val_months=1,
        reference_conditions={"decision": "laplace_max_norm"},
        save_condition_plots=False,
    )


def run_decision(key: str, args, out_dir: Path, output_root: Path) -> pd.DataFrame:
    spec = build_spec(key, args)
    engine = ExperimentEngine(spec)
    cfg = PipelineConfig(
        dataset_name=SCENARIOS[key],
        data_path=str(thesis_data_dir()),
        output_dir=str(output_root / "multitarget_decision" / key.lower()),
        epochs=5 if args.quick else args.epochs,
        num_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )
    groups = ["decision"]
    conditions = spec.conditions
    if args.quick:
        conditions = [c for c in conditions if c.name == f"{args.heads[0]}_none"]
        spec.conditions = conditions
        # The full spec's own reference_conditions (laplace_max_norm) is only
        # guaranteed to survive this filter when --heads is left at its
        # default (every head) -- a single-head --quick smoke test (e.g.
        # --heads qr) otherwise drops that condition entirely, and
        # ExperimentSpec.validate() rejects a reference_conditions entry
        # that names no surviving condition. Point it at whichever single
        # condition --quick actually kept instead.
        spec.reference_conditions = {"decision": conditions[0].name}

    summary = engine.run(base_cfg=cfg, groups=groups, dataset_keys=[SCENARIOS[key]], skip_hpo=args.skip_hpo)

    experiment_root = Path(cfg.output_dir) / slugify(spec.name)
    # ExperimentEngine._update_latest_symlink is best-effort -- it silently
    # no-ops (OSError, caught and logged at debug level only) on a filesystem
    # that refuses an unprivileged symlink, which is the *default* state on
    # Windows (requires Developer Mode or an elevated process). Relying on
    # experiment_root/"latest" here meant this function silently returned an
    # empty DataFrame -- no value_curve/per_fold/summary/significance CSVs at
    # all, just one easy-to-miss warning log -- on exactly that platform.
    # Finding the real run directory directly (newest mtime under runs/) has
    # no symlink dependency at all.
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

    agg = full.groupby(["metric", "target", "model"])["value"].agg(["mean", "std", "min", "max"]).reset_index()
    agg.to_csv(out_dir / f"{key.lower()}_summary.csv", index=False)
    print(f"\n=== {key} summary (mean across {spec.cv_folds} folds) ===")
    print(agg.pivot_table(index="metric", columns="model", values="mean").round(4).to_string())

    dm = family_significance(full)
    if not dm.empty:
        dm.to_csv(out_dir / f"{key.lower()}_significance.csv", index=False)
        print(f"\n=== {key}: head x conformal, Brier score DM table ===")
        print(dm.to_string(index=False))

    log.info("%s: engine summary written under %s", key, summary)
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--test-size", type=int, default=8)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=INTERVAL_ALPHA)
    parser.add_argument("--rated-q", type=float, default=0.98)
    parser.add_argument("--skip-hpo", dest="skip_hpo", action="store_true", default=True)
    parser.add_argument("--run-hpo", dest="skip_hpo", action="store_false")
    parser.add_argument("--hpo-trials", type=int, default=50)
    parser.add_argument("--hpo-epochs", type=int, default=100)
    parser.add_argument(
        "--hpo-objective",
        default="winkler_score",
        help="HPO objective for fpqr_anchor_outer/qr only (BNN/Laplace stay point-accuracy, matching "
        "multitarget_experiment.py's own convention). 'winkler_score' (default) or any name "
        "get_quantile_metrics computes, e.g. 'cwe_stratified'.",
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
        args.output_root / "multitarget_decision",
        meta={"decisions": args.decisions, "heads": args.heads, "epochs": args.epochs, "seed": args.seed},
    )
    out_dir = run_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {key: run_decision(key, args, out_dir, args.output_root) for key in args.decisions}
    (out_dir / "multitarget_decision_config.json").write_text(
        json.dumps(
            {
                "decisions": args.decisions,
                "heads": args.heads,
                "scenarios": {k: SCENARIOS[k] for k in args.decisions},
                "epochs": args.epochs,
                "seed": args.seed,
                "test_size_weeks": args.test_size,
                "step_weeks": args.step,
                "paths": args.paths,
                "alpha": args.alpha,
            },
            indent=2,
        )
        + "\n"
    )
    log.info("wrote per-decision results under %s", out_dir)
    for key, table in results.items():
        log.info("  %s: %d rows", key, len(table))


if __name__ == "__main__":
    main()
