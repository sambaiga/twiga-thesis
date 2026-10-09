"""Chapter 7's single source of truth: backtesting protocol, datasets, output root, pipeline defaults.

Every Chapter 7 experiment (architectural evaluation, probabilistic benchmark,
design ablation) reads its split numbers, dataset windows and output location
from here, so the three cannot drift apart. The protocol is the one stated in
``Chapter-7.tex`` (``chap:7-expriment-protol``) and is identical to Chapter 6's
MLVS-PT protocol (``Chapter-6.tex``, rolling-origin backtesting paragraph): ten
folds, four-month test window, forecasting origin advanced by one month,
expanding training window of at least twelve months, two-month validation set,
three-month calibration set.

``ExperimentEngine`` derives the first fold's training length from ``CV_FOLDS``
rather than honouring ``cv_train_size`` (see the note in Chapter 6's
``backbone_conformal_study.py``); ``MIN_TRAIN_MONTHS`` is therefore the stated
protocol value, checked by :func:`protocol_check`, not a lever.

The experiment layout follows Chapter 6's ``multitarget_experiment.py``: a shared ``SPEC_COMMON`` dict, one
:class:`~twiga.experiment.ExperimentSpec` per dataset (:func:`dataset_spec`, which also sets that dataset's own HPO
window), and one ``base_config()``. Feature extraction and selection are not configured here: as in Chapter 6,
``load_dataset`` (through ``load_and_preprocess``) selects the weather, calendar and autoregressive features and the
ACF-derived input window (48 samples on both datasets), so every chapter selects features the same way.
"""

from __future__ import annotations

import os
from pathlib import Path

from twiga.experiment import ExperimentSpec
from twiga.utils.run_forecast import PipelineConfig

script_dir = Path(__file__).resolve().parent
CHAPTER_DIR = script_dir.parent

# parents[4]: protocol.py -> shared -> ch6_conformal -> experiment -> twiga-forecast -> projects.
# A sibling of the repository, matching Chapter 6's DEFAULT_OUTPUT_ROOT; override with
# --output-root or TWIGA_EXPERIMENTS_ROOT.
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[3] / "experiments"))) / "ch6_conformal"
)

# ── Nominal coverage ─────────────────────────────────────────────────────────
ALPHA = 0.10
QR_QUANTILES = sorted([ALPHA / 2, 0.25, 0.50, 0.75, 1.0 - ALPHA / 2])

# ── Rolling-origin backtesting protocol (months) ─────────────────────────────
CV_SPLIT_FREQ = "months"
CV_FOLDS = 10
CV_TEST_MONTHS = 4
CV_STRIDE_MONTHS = 1
CV_VAL_MONTHS = 2
CV_CALIB_MONTHS = 3
CV_WINDOW = "expanding"
MIN_TRAIN_MONTHS = 12

# Training budget, as Chapter 6 (multitarget_experiment.py): 100 epochs per condition, 50 HPO trials of 50 epochs,
# early stopping after 40 epochs without improvement (clears reduce_on_plateau's worst-case scheduler patience of
# 0.2 * epochs with margin), and num_workers=0 (forked DataLoader workers hang on MPS and crashed the 2026-09-19
# MLVS-PT benchmark run with "DataLoader worker exited unexpectedly").
DEFAULT_EPOCHS = 100
DEFAULT_HPO_TRIALS = 50  # PipelineConfig's default of 10 never leaves TPESampler's random-startup phase.
DEFAULT_HPO_EPOCHS = 50
FIXED_OVERRIDES = {"early_stop_patience": 40, "num_workers": 0}
# The design ablation (design_ablation/conditions.py) keeps its original 500-epoch / patience-100 budget, so its
# existing results stay reproducible.
EARLY_STOP_PATIENCE = 100

# MLPGAM design-ablation settings that the benchmark inherits from the tuned backbone rather than re-tunes.
CRC_CONTROLLED: frozenset[str] = frozenset(
    {"two_stage", "stage1_epochs", "stage1_epochs_frac", "sigma_loss_fn", "monitor", "lambda_lasso"}
)
NON_TRANSFERABLE: frozenset[str] = frozenset({"opt_lr", "opt_weight_decay", "sch_warmup_epochs", "sch_eta_min"})

# ── Datasets ─────────────────────────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    "MLVS-PT": {
        "dataset_name": "MLVS-PT",
        "data_path": "examples/data",
        "train_start": "2019-03",
        "train_end": "2022-01",
        "test_start": "2021-07",
        "test_end": "2022-01",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
    "MLVS-UK": {
        "dataset_name": "MLVS-UK",
        "data_path": "examples/data",
        "train_start": "2018-01",
        "train_end": "2020-07",
        "test_start": "2019-12",
        "test_end": "2020-07",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
}


# Backbone HPO window per dataset (ExperimentSpec.hpo_train_months / hpo_val_months), counted from train_start and
# ending before the first test fold: MLVS-PT 2019-03 + 15 months = 2020-06 (first test fold 2020-11), MLVS-UK
# 2018-01 + 15 months = 2019-04 (first test fold 2019-06). Chapter 6 uses 12 + 3 on MLVS-PT-3PHASE.
HPO_MONTHS: dict[str, tuple[int, int]] = {"MLVS-PT": (12, 3), "MLVS-UK": (12, 3)}


# Chapter 7's own two datasets; PV-UK/ALBANIA belong to a different paper's benchmark.
CHAPTER_DATASETS = tuple(DATASETS)

# Gate stress test: each dataset with NOISE_FEATURES independent standard-normal exogenous columns added
# (PipelineConfig.noise_features). They carry no information about the target, so a faithful gate should suppress
# them and an ungated model should lose accuracy as they are added. Same seed for every condition.
NOISE_FEATURES = 4
NOISY_SUFFIX = "+noise"
NOISY_DATASETS: dict[str, dict] = {
    f"{key}{NOISY_SUFFIX}": {**entry, "noise_features": NOISE_FEATURES, "noise_seed": 0}
    for key, entry in DATASETS.items()
}


def protocol_check() -> dict[str, int | str]:
    """The protocol as a dict, for logging next to results and for comparing against Chapter 6's stated protocol."""
    return {
        "folds": CV_FOLDS,
        "test_months": CV_TEST_MONTHS,
        "stride_months": CV_STRIDE_MONTHS,
        "val_months": CV_VAL_MONTHS,
        "calib_months": CV_CALIB_MONTHS,
        "min_train_months": MIN_TRAIN_MONTHS,
        "window": CV_WINDOW,
    }


def experiment_output_dir(experiment: str, output_root: Path | None = None) -> Path:
    """``<output_root>/<experiment>``, e.g. ``architectural_evaluation``/``probabilistic_benchmark``."""
    return (output_root or DEFAULT_OUTPUT_ROOT) / experiment


SPEC_COMMON: dict = {
    "cv_split_freq": CV_SPLIT_FREQ,
    "cv_train_size": MIN_TRAIN_MONTHS,
    "cv_test_size": CV_TEST_MONTHS,
    "cv_val_size": CV_VAL_MONTHS,
    "cv_calib_size": CV_CALIB_MONTHS,
    "cv_stride": CV_STRIDE_MONTHS,
    "cv_folds": CV_FOLDS,
    "cv_window": CV_WINDOW,
    "fixed_overrides": FIXED_OVERRIDES,
}


def base_dataset_key(dataset_key: str) -> str:
    """``MLVS-PT+noise`` -> ``MLVS-PT``."""
    return dataset_key.removesuffix(NOISY_SUFFIX)


def dataset_spec(dataset_key: str, **spec_fields) -> ExperimentSpec:
    """One dataset's :class:`ExperimentSpec`: ``SPEC_COMMON`` plus ``spec_fields``, that dataset only, its HPO window.

    Mirrors Chapter 6's ``SPEC_MLVS_PT_3PHASE``: the protocol is shared, the dataset entry and the HPO window are not.
    """
    datasets = {**DATASETS, **NOISY_DATASETS}
    hpo_train, hpo_val = HPO_MONTHS[base_dataset_key(dataset_key)]
    return ExperimentSpec(
        **{**SPEC_COMMON, **spec_fields},
        datasets={dataset_key: datasets[dataset_key]},
        hpo_train_months=hpo_train,
        hpo_val_months=hpo_val,
    )


def base_config(
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    hpo_trials: int = DEFAULT_HPO_TRIALS,
    hpo_epochs: int = DEFAULT_HPO_EPOCHS,
) -> PipelineConfig:
    """Shared :class:`PipelineConfig`; the dataset fields come from each spec's dataset entry (Chapter 6 pattern)."""
    from chapters.ch6_conformal.shared.metrics import INTERVAL_METRICS, POINT_METRICS  # noqa: PLC0415

    return PipelineConfig(
        output_dir=f"{output_dir}/",
        data_path="examples/data",
        out_activation="Identity",
        epochs=epochs,
        num_trials=hpo_trials,
        hpo_epochs=hpo_epochs,
        split_freq=CV_SPLIT_FREQ,
        metrics=POINT_METRICS + INTERVAL_METRICS,
        log_level="INFO",
        log_file=f"{output_dir}/run.log",
        use_latex=False,
        num_threads=1,
    )
