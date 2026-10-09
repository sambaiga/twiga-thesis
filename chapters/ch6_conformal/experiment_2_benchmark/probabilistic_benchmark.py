r"""Chapter 7, Experiment 2: the probabilistic forecasting benchmark.

Runs through :class:`~twiga.experiment.ExperimentEngine` with the layout of Chapter 6's ``multitarget_experiment.py``:
plain :class:`~twiga.experiment.Condition` objects, one :class:`~twiga.experiment.ExperimentSpec` per dataset
(``shared.protocol.dataset_spec``), one ``base_config()``.

The question
------------
``Chapter-7.tex`` (``chap7:exp-three``) compares the complete proposed framework, MLPGAM with adaptive horizon-wise
conformal calibration (``mlpgam_crc``), against quantile regression (``mlpgam_qr``), a Bayesian neural network
(``mlpgam_bnn``), parametric Laplace forecasting (``mlpgam_laplace``), and the FPQR framework of Chapter 5
(``mlpgam_fpqr``), all on the same MLPGAM backbone so that differences reflect the uncertainty method.

Backbone and HPO
----------------
Every head shares one tuned MLPGAM backbone per dataset (``SHARED_HPO_VARIANT``), with the feature gate on and the
inert target-gate fields pinned (``MLPGAM_HPO_PINS``). FPQR, which has head hyperparameters of its own, runs a
head-only search on top of that backbone (``hpo_parent_model_cls``), as in Chapter 6. Backbone capacity is therefore
held equal across heads.

Metrics
-------
QR and FPQR report the quantile evaluator's CRPS and BNN and Laplace the parametric evaluator's CRPS and NLL, next to
the interval metrics. ``mlpgam_crc`` reports point and interval metrics only: the engine's calibrated path supports no
other metric types. The Winkler score is the decisive comparison for every head
(``shared/ch7_analyse_results.py``); CWE, PICP and NMPI are diagnostics.

Usage::

    uv run python experiment/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py --dataset all
    uv run python experiment/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py \
        --dataset MLVS-UK --hpo-cache-policy refresh
    # Smoke test:
    uv run python experiment/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py \
        --dataset MLVS-UK --epochs 2 --hpo-trials 1 --hpo-epochs 2 --folds 2 --output-root /tmp/ch7-smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path
import sys

# The chapter directory (for ``shared.*``) and ``experiment/``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.conditions import (  # noqa: E402
    FPQR_HPO_PINS,
    FPQR_HPO_VARIANT,
    INTERVAL_METRIC_TYPES,
    MLPGAM_HPO_PINS,
    PARAMETRIC_METRIC_TYPES,
    PROPOSED,
    PROPOSED_CONFORMAL,
    QUANTILE_METRIC_TYPES,
    SHARED_HPO_VARIANT,
)
from shared.protocol import (  # noqa: E402
    CHAPTER_DATASETS,
    CV_FOLDS,
    DEFAULT_EPOCHS,
    DEFAULT_HPO_EPOCHS,
    DEFAULT_HPO_TRIALS,
    DEFAULT_OUTPUT_ROOT,
    QR_QUANTILES,
    base_config,
    dataset_spec,
    experiment_output_dir,
    protocol_check,
)
from twiga.core.utils import configure, get_logger  # noqa: E402
from twiga.experiment import Condition, ExperimentEngine  # noqa: E402
from twiga.models.nn import MLPGAMConfig, MLPGAMCRCConfig, MLPGAMFPQRConfig, MLPGAMQRConfig  # noqa: E402
from twiga.models.nn.mlpgambnn_model import MLPGAMBNNConfig  # noqa: E402
from twiga.models.nn.mlpgamlaplace_model import MLPGAMLaplaceConfig  # noqa: E402
from twiga.utils.run_forecast import setup_environment  # noqa: E402

log = get_logger(__name__)
EXPERIMENT = "probabilistic_benchmark"

# Shared by every condition on the shared MLPGAM backbone key.
_SHARED_BACKBONE = {
    "hpo_model_cls": MLPGAMConfig,
    "hpo_variant": SHARED_HPO_VARIANT,
    "hpo_fixed_params": MLPGAM_HPO_PINS,
}

CONDITIONS = [
    Condition(
        name="mlpgam_crc",
        group="benchmark",
        description="MLPGAM-CRC, the proposed framework: adaptive horizon-wise conformal calibration",
        overrides=PROPOSED,
        model_cls=MLPGAMCRCConfig,
        conformal_config=PROPOSED_CONFORMAL,
        metric_types=INTERVAL_METRIC_TYPES,
        **_SHARED_BACKBONE,
    ),
    Condition(
        name="mlpgam_qr",
        group="benchmark",
        description="Quantile regression on the MLPGAM backbone, a fixed-grid baseline",
        overrides={"use_gate": True, "quantiles": QR_QUANTILES},
        model_cls=MLPGAMQRConfig,
        checkpoint_monitor="loss",
        metric_types=QUANTILE_METRIC_TYPES,
        **_SHARED_BACKBONE,
    ),
    Condition(
        name="mlpgam_fpqr",
        group="benchmark",
        description="FPQR (Chapter 5) on the MLPGAM backbone, head hyperparameters tuned on the shared backbone",
        overrides={"use_gate": True, "quantiles": QR_QUANTILES},
        model_cls=MLPGAMFPQRConfig,
        hpo_model_cls=MLPGAMFPQRConfig,
        hpo_variant=FPQR_HPO_VARIANT,
        hpo_parent_model_cls=MLPGAMConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params={**MLPGAM_HPO_PINS, **FPQR_HPO_PINS},
        checkpoint_monitor="loss",
        metric_types=QUANTILE_METRIC_TYPES,
    ),
    Condition(
        name="mlpgam_bnn",
        group="benchmark",
        description="Bayesian neural network (mean-field variational head) on the MLPGAM backbone",
        overrides={"use_gate": True},
        model_cls=MLPGAMBNNConfig,
        checkpoint_monitor="loss",
        metric_types=PARAMETRIC_METRIC_TYPES,
        **_SHARED_BACKBONE,
    ),
    Condition(
        name="mlpgam_laplace",
        group="benchmark",
        description="Parametric Laplace forecasting on the MLPGAM backbone",
        overrides={"use_gate": True},
        model_cls=MLPGAMLaplaceConfig,
        checkpoint_monitor="loss",
        metric_types=PARAMETRIC_METRIC_TYPES,
        **_SHARED_BACKBONE,
    ),
]

_SPEC_COMMON = {
    "name": "Probabilistic Forecasting Benchmark (Chapter 7)",
    "output_prefix": "ch7_benchmark",
    "condition_cls": MLPGAMCRCConfig,
    "backbone_cls": MLPGAMConfig,
    "conditions": CONDITIONS,
    "controlled_fields": frozenset(),
    "reference_conditions": {"benchmark": "mlpgam_crc"},
}
SPECS = {key: dataset_spec(key, **_SPEC_COMMON) for key in CHAPTER_DATASETS}


def main() -> None:
    """CLI entry point: one ExperimentEngine run per requested dataset."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["all"], choices=[*CHAPTER_DATASETS, "all"])
    ap.add_argument("--group", nargs="+", default=["benchmark"], choices=["benchmark"])
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=[c.name for c in CONDITIONS],
        help="run only these conditions (the analysis merges them with earlier runs of the others)",
    )
    ap.add_argument(
        "--experiment-name",
        default=EXPERIMENT,
        help="output sub-folder under --output-root; reuse an earlier run's folder to reuse its HPO caches",
    )
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--hpo-trials", type=int, default=DEFAULT_HPO_TRIALS)
    ap.add_argument("--hpo-epochs", type=int, default=DEFAULT_HPO_EPOCHS, help="epochs per HPO trial")
    ap.add_argument("--folds", type=int, default=CV_FOLDS)
    ap.add_argument("--skip-hpo", action="store_true", help="require matching fingerprinted HPO caches")
    ap.add_argument("--hpo-cache-policy", choices=["reuse", "refresh", "error-if-missing"], default="reuse")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()

    output_dir = experiment_output_dir(args.experiment_name, args.output_root)
    cfg = base_config(output_dir, epochs=args.epochs, hpo_trials=args.hpo_trials, hpo_epochs=args.hpo_epochs)
    configure(level=cfg.log_level, log_file=cfg.log_file)
    setup_environment(cfg)
    log.info("Backtesting protocol  %s", protocol_check())

    for dataset_key in list(CHAPTER_DATASETS) if "all" in args.dataset else args.dataset:
        spec = dc_replace(SPECS[dataset_key], cv_folds=args.folds)
        if args.conditions:
            spec = dc_replace(spec, conditions=[c for c in CONDITIONS if c.name in args.conditions])
        engine = ExperimentEngine(spec)
        engine.run(
            base_cfg=cfg,
            groups=args.group,
            dataset_keys=[dataset_key],
            skip_hpo=args.skip_hpo,
            hpo_cache_policy=args.hpo_cache_policy,
        )
        log.info("dataset=%s written to %s", dataset_key, engine.last_run_root)


if __name__ == "__main__":
    main()
