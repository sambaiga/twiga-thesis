r"""Chapter 7 architectural study: backbone, feature gating, gating under noise, horizon-wise calibration.

One :class:`~twiga.experiment.ExperimentSpec`, run through :class:`~twiga.experiment.ExperimentEngine`, covering the
three hypotheses of ``Chapter-7.tex`` (``chap7:exp-two``) under a single configuration (datasets, protocol, HPO), so
the results of all groups are comparable with one another. It supersedes ``architectural_evaluation.py`` (backbone)
and the design ablation's ``gating``/``conformal`` groups for the chapter's architectural evaluation.

Groups
------
``backbone`` (H1)
    MLPF and MLPGAM, each as a point forecaster and wrapped in the same split conformal procedure (unscaled absolute
    residual). Every condition is repeated over ``--seeds`` training seeds (``<name>__s<seed>``) with the
    hyperparameters fixed, so retraining variability is measured instead of confounding the backbone effect. Each
    backbone's HPO budget is proportional to its effective search-space size (``--trials-per-param``), so the larger
    MLPGAM search is not under-tuned relative to MLPF's.
``gating`` (H2)
    The proposed MLPGAM-CRC model with the gate disabled, enabled without penalty or warm-up, at its tuned setting,
    and at fixed penalties ``lambda_gate`` in {1e-4, 1e-3, 1e-2}. Every fold also reports gate faithfulness
    (:mod:`shared.gate_faithfulness`) through a ``custom_scorer``.
``gating_noise`` (H2, stress test)
    The ``gating`` conditions on the ``+noise`` datasets, which add ``NOISE_FEATURES`` independent standard-normal
    exogenous columns. A useful gate keeps accuracy as irrelevant inputs are added and assigns them low gate values
    (``gate_noise_auc``).
``calibration`` (H3)
    Four calibrations of one trained MLPGAM-CRC model per fold (pooled, horizon-wise, horizon-wise with the learned
    scale, and its signed variant), so they differ only in the calibration step; the trained condition also reports
    the model's interpretability (:mod:`shared.mlpgam_interpretability`).

MLPGAM settings
---------------
Every MLPGAM-backed condition uses the gate (``use_gate=True``): earlier runs left it at the config default
``False``, so the backbone comparison and the benchmark evaluated MLPGAM without its gate. The MLPGAM search is run
with the gate on and the target-gate fields pinned (they are inert for a single target), so the search covers only
fields that affect the model. The model configuration is the proposed MLPGAM-CRC of the probabilistic benchmark.

Layout follows Chapter 6's ``multitarget_experiment.py``: conditions and HPO building blocks from
:mod:`shared.conditions`, one spec per dataset key from :func:`shared.protocol.dataset_spec` (with that dataset's
own HPO window), every MLPGAM condition on the shared backbone search (``SHARED_HPO_VARIANT``), and feature selection
left to ``load_dataset``.

Usage::

    uv run python experiment/ch6_conformal/experiment_1_architectural/architectural_study.py --dataset all
    uv run python experiment/ch6_conformal/experiment_1_architectural/architectural_study.py \
        --group backbone --seeds 5 --hpo-cache-policy refresh
    uv run python experiment/ch6_conformal/experiment_1_architectural/architectural_study.py --group gating_noise
    # Smoke test:
    uv run python experiment/ch6_conformal/experiment_1_architectural/architectural_study.py \
        --dataset MLVS-UK --seeds 1 --epochs 2 --hpo-epochs 2 --folds 2 --hpo-trials-floor 1 --trials-per-param 0
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path
import sys

# The chapter directory (for ``shared.*``) and ``experiment/``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twiga.core.utils import configure, get_logger  # noqa: E402
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec  # noqa: E402
from twiga.models.nn import MLPFConfig, MLPGAMConfig, MLPGAMCRCConfig  # noqa: E402
from twiga.utils.run_forecast import setup_environment  # noqa: E402

from chapters.ch6_conformal.shared.conditions import (  # noqa: E402
    DEFAULT_TRIALS_PER_PARAM,
    INTERVAL_METRIC_TYPES,
    MLPGAM_HPO_PINS,
    POINT_METRIC_TYPES,
    PROPOSED,
    PROPOSED_CONFORMAL,
    PROPOSED_CONFORMAL_SIGNED,
    SCP,
    SCP_POOLED,
    SHARED_HPO_VARIANT,
    trial_budget,
)
from chapters.ch6_conformal.shared.gate_faithfulness import make_gate_faithfulness_scorer  # noqa: E402
from chapters.ch6_conformal.shared.mlpgam_interpretability import make_interpretability_scorer  # noqa: E402
from chapters.ch6_conformal.shared.protocol import (  # noqa: E402
    CHAPTER_DATASETS,
    CV_FOLDS,
    DEFAULT_EPOCHS,
    DEFAULT_HPO_EPOCHS,
    DEFAULT_OUTPUT_ROOT,
    NOISY_SUFFIX,
    base_config,
    dataset_spec,
    experiment_output_dir,
    protocol_check,
)

log = get_logger(__name__)

EXPERIMENT = "architectural_study"
SEED_BASE = 42  # ExperimentEngine's own default seed, so seed 0 of a study reproduces a single-seed run
CLEAN_GROUPS = ("backbone", "gating", "calibration")
NOISE_GROUPS = ("gating_noise",)
MLPF_HPO_VARIANT = "mlpf"


def _mlpgam_backbone(trials: int) -> dict:
    """Condition fields placing a condition on the shared, gate-on MLPGAM backbone search (Chapter 6 pattern)."""
    return {
        "hpo_model_cls": MLPGAMConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_fixed_params": MLPGAM_HPO_PINS,
        "hpo_num_trials": trials,
    }


def backbone_conditions(seeds: int, mlpf_trials: int, mlpgam_trials: int) -> list[Condition]:
    """H1: four backbone conditions, each repeated over ``seeds`` training seeds with HPO fixed per backbone."""
    mlpf_backbone = {"hpo_model_cls": MLPFConfig, "hpo_variant": MLPF_HPO_VARIANT, "hpo_num_trials": mlpf_trials}
    base = [
        ("mlpf_point", MLPFConfig, None, {}, mlpf_backbone),
        ("mlpgam_point", MLPGAMConfig, None, {"use_gate": True}, _mlpgam_backbone(mlpgam_trials)),
        ("mlpf_scp", MLPFConfig, SCP, {}, mlpf_backbone),
        ("mlpgam_scp", MLPGAMConfig, SCP, {"use_gate": True}, _mlpgam_backbone(mlpgam_trials)),
    ]
    return [
        Condition(
            name=f"{name}__s{SEED_BASE + k}",
            group="backbone",
            description=f"{name}, training seed {SEED_BASE + k}",
            overrides={**overrides, "seed": SEED_BASE + k},
            model_cls=cls,
            conformal_config=conformal,
            metric_types=INTERVAL_METRIC_TYPES if conformal is not None else POINT_METRIC_TYPES,
            **backbone,
        )
        for name, cls, conformal, overrides, backbone in base
        for k in range(seeds)
    ]


GATE_SETTINGS = {
    "gate_none": ("Gate disabled", {"use_gate": False}),
    "gate_unconstrained": ("Gate enabled, no penalty or warm-up", {"lambda_gate": 0.0, "warmup_epochs": 0}),
    "gate_default": ("Gate at its tuned setting (proposed)", {}),
    "gate_l1e-4": ("Gate with lambda_gate = 1e-4", {"lambda_gate": 1e-4}),
    "gate_l1e-3": ("Gate with lambda_gate = 1e-3", {"lambda_gate": 1e-3}),
    "gate_l1e-2": ("Gate with lambda_gate = 1e-2", {"lambda_gate": 1e-2}),
}


def gating_conditions(group: str, mlpgam_trials: int) -> list[Condition]:
    """H2: the proposed model with only the gate varied; every fold also scores gate faithfulness."""
    scorer = make_gate_faithfulness_scorer()
    return [
        Condition(
            name=name,
            group=group,
            description=description,
            overrides={**PROPOSED, **gate},
            model_cls=MLPGAMCRCConfig,
            conformal_config=PROPOSED_CONFORMAL,
            metric_types=INTERVAL_METRIC_TYPES,
            custom_scorer=scorer,
            **_mlpgam_backbone(mlpgam_trials),
        )
        for name, (description, gate) in GATE_SETTINGS.items()
    ]


def calibration_conditions(mlpgam_trials: int) -> list[Condition]:
    """H3: four calibrations of one trained MLPGAM-CRC model per fold, plus the model's interpretability.

    ``conf_crc`` (the proposed calibration) trains the model; every other condition loads its per-fold checkpoint
    (``reuse_checkpoint_from``) and changes only the calibration step, so differences between the four conditions are
    due to calibration alone and not to run-to-run training variation:

    * ``conf_scp_pooled``: unscaled absolute residual, one quantile pooled over horizons;
    * ``conf_scp``: unscaled absolute residual, one quantile per horizon;
    * ``conf_crc``: absolute residual scaled by the learned sigma, one quantile per horizon (proposed);
    * ``conf_crc_signed``: signed residual scaled by the learned sigma, split quantiles per horizon (asymmetric).

    ``conf_crc`` also runs :func:`shared.mlpgam_interpretability.make_interpretability_scorer`: the exact additive
    branch decomposition of every forecast, permutation importance per feature, and gate faithfulness.
    """
    common = {
        "group": "calibration",
        "overrides": PROPOSED,
        "model_cls": MLPGAMCRCConfig,
        "metric_types": INTERVAL_METRIC_TYPES,
        **_mlpgam_backbone(mlpgam_trials),
    }
    trained = Condition(
        name="conf_crc",
        description="Horizon-wise conformal with the learned scale, absolute score (proposed); trains the model",
        conformal_config=PROPOSED_CONFORMAL,
        custom_scorer=make_interpretability_scorer(),
        **common,
    )
    return [
        trained,
        *(
            Condition(
                name=name,
                description=description,
                conformal_config=conformal,
                reuse_checkpoint_from=trained.name,
                **common,
            )
            for name, description, conformal in (
                ("conf_scp", "Split conformal, one quantile per horizon", SCP),
                ("conf_scp_pooled", "Split conformal, one quantile pooled over horizons", SCP_POOLED),
                (
                    "conf_crc_signed",
                    "Horizon-wise conformal with the learned scale, signed score (asymmetric)",
                    PROPOSED_CONFORMAL_SIGNED,
                ),
            )
        ),
    ]


def build_conditions(seeds: int, trials_per_param: float, floor: int) -> list[Condition]:
    """Every group's conditions; one HPO budget per backbone, proportional to its effective search size."""
    mlpf_trials = trial_budget(MLPFConfig, trials_per_param, floor)
    mlpgam_trials = trial_budget(MLPGAMConfig, trials_per_param, floor, MLPGAM_HPO_PINS)
    log.info("Backbone HPO budget  mlpf=%d trials  mlpgam=%d trials", mlpf_trials, mlpgam_trials)
    return [
        *backbone_conditions(seeds, mlpf_trials, mlpgam_trials),
        *gating_conditions("gating", mlpgam_trials),
        *gating_conditions("gating_noise", mlpgam_trials),
        *calibration_conditions(mlpgam_trials),
    ]


def build_specs(seeds: int, trials_per_param: float, floor: int) -> dict[str, ExperimentSpec]:
    """One spec per dataset key, clean and ``+noise`` (Chapter 6's per-dataset SPEC pattern)."""
    spec_common = {
        "name": "MLPGAM Architectural Study (Chapter 7)",
        "output_prefix": "ch7_architectural_study",
        "condition_cls": MLPGAMCRCConfig,
        "backbone_cls": MLPGAMConfig,
        "conditions": build_conditions(seeds, trials_per_param, floor),
        "controlled_fields": frozenset(),
        "reference_conditions": {
            "backbone": f"mlpgam_scp__s{SEED_BASE}",
            "gating": "gate_default",
            "gating_noise": "gate_default",
            "calibration": "conf_scp",
        },
    }
    keys = [*CHAPTER_DATASETS, *(f"{key}{NOISY_SUFFIX}" for key in CHAPTER_DATASETS)]
    return {key: dataset_spec(key, **spec_common) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", default=["all"], choices=[*CLEAN_GROUPS, *NOISE_GROUPS, "all"])
    ap.add_argument("--dataset", nargs="+", default=["all"], choices=[*CHAPTER_DATASETS, "all"])
    ap.add_argument("--seeds", type=int, default=5, help="training seeds per backbone condition (H1)")
    ap.add_argument(
        "--trials-per-param",
        type=float,
        default=DEFAULT_TRIALS_PER_PARAM,
        help=f"HPO trials per searched hyperparameter (default {DEFAULT_TRIALS_PER_PARAM:.2f}: MLPF keeps 50 trials)",
    )
    ap.add_argument("--hpo-trials-floor", type=int, default=20, help="minimum HPO trials for any backbone")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--hpo-epochs", type=int, default=DEFAULT_HPO_EPOCHS, help="epochs per HPO trial")
    ap.add_argument("--folds", type=int, default=CV_FOLDS)
    ap.add_argument("--skip-hpo", action="store_true", help="require matching fingerprinted HPO caches")
    ap.add_argument(
        "--hpo-cache-policy",
        choices=["reuse", "refresh", "error-if-missing"],
        default="reuse",
        help="refresh is required after changing --trials-per-param (the trial count is not fingerprinted)",
    )
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()

    groups = [*CLEAN_GROUPS, *NOISE_GROUPS] if "all" in args.group else args.group
    datasets = list(CHAPTER_DATASETS) if "all" in args.dataset else args.dataset
    specs = build_specs(args.seeds, args.trials_per_param, args.hpo_trials_floor)
    output_dir = experiment_output_dir(EXPERIMENT, args.output_root)
    cfg = base_config(output_dir, epochs=args.epochs, hpo_epochs=args.hpo_epochs)
    configure(level=cfg.log_level, log_file=cfg.log_file)
    setup_environment(cfg)
    log.info("Backtesting protocol  %s", protocol_check())

    # One engine run per dataset key: the clean groups on the dataset, the noise group on its +noise variant. The
    # analysis reads every run directory together.
    for dataset in datasets:
        for run_groups, dataset_key in (
            ([g for g in groups if g in CLEAN_GROUPS], dataset),
            ([g for g in groups if g in NOISE_GROUPS], f"{dataset}{NOISY_SUFFIX}"),
        ):
            if not run_groups:
                continue
            engine = ExperimentEngine(dc_replace(specs[dataset_key], cv_folds=args.folds))
            engine.run(
                base_cfg=cfg,
                groups=run_groups,
                dataset_keys=[dataset_key],
                skip_hpo=args.skip_hpo,
                hpo_cache_policy=args.hpo_cache_policy,
            )
            log.info("groups=%s dataset=%s written to %s", run_groups, dataset_key, engine.last_run_root)


if __name__ == "__main__":
    main()
