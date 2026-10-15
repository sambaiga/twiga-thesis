"""Chapter 3's single source of truth: backtesting protocols, datasets, pipeline defaults for the protocol comparison.

Mirrors the shape of ``chapters/ch6_conformal/shared/protocol.py``: one ``SPEC_COMMON`` dict per protocol variant,
one :func:`dataset_spec` per dataset, and one :func:`base_config`. Feature extraction and selection are not
configured here: ``load_dataset`` (through ``load_and_preprocess``) selects the weather, calendar and
autoregressive features automatically, same as every other chapter.

Month-equivalents of the legacy notebook's protocol
----------------------------------------------------
The legacy single-dataset notebook this experiment reproduces used ``forecast_len=6`` (a six-month evaluation
window) and ``incremental_len=2`` (the forecasting origin advanced two months between folds). This clone has no
access to that notebook's own commit history to confirm whether those two numbers were themselves chosen to match
Chapter 6/7's own protocol constants (``CV_TEST_MONTHS=4``, ``CV_STRIDE_MONTHS=1``) or were an independent,
earlier choice; ``EXPANDING_TEST_MONTHS``/``EXPANDING_STRIDE_MONTHS`` below are pinned to the legacy notebook's own
values (6 and 2) rather than silently substituted with Chapter 6/7's, since the point of this experiment is to
reproduce the *legacy* expanding-window protocol under a controlled comparison against a single split, not to
invent a new backtesting protocol. ``EXPANDING_VAL_MONTHS``/``MIN_TRAIN_MONTHS`` are carried over from Chapter 6/7
(``ch6_conformal/shared/protocol.py``) unchanged, since the legacy notebook did not carve out a separate
validation window and twelve months is already this thesis's established minimum initial training length.
``EXPANDING_FOLDS=10`` matches Chapter 6/7's own fold count so protocol-agreement tables read on a comparable
number of folds; it is a target, not a guarantee -- the real fold count a dataset yields under these sizes depends
on that dataset's own span (see ``protocol_check``).

The single-split variant uses the same train/test boundary the expanding protocol's own first fold would produce
(``cv_folds=1``): one historical training block, one held-out test block, no re-estimation.
"""

from __future__ import annotations

import os
from pathlib import Path

from twiga.experiment import ExperimentSpec
from twiga.utils.run_forecast import PipelineConfig

from twiga_thesis.paths import thesis_data_dir

script_dir = Path(__file__).resolve().parent
CHAPTER_DIR = script_dir.parent

# A sibling of the repository, matching Chapter 6/7's own DEFAULT_OUTPUT_ROOT convention; override with
# --output-root or TWIGA_EXPERIMENTS_ROOT.
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[2] / "experiments"))) / "ch3_methodology"
)

# ── Protocol constants (months) ──────────────────────────────────────────────
CV_SPLIT_FREQ = "months"
MIN_TRAIN_MONTHS = 12  # Chapter 6/7's own minimum initial training window.
EXPANDING_VAL_MONTHS = 2  # Chapter 6/7's own validation window; the legacy notebook carved out none of its own.
EXPANDING_TEST_MONTHS = 6  # legacy notebook's forecast_len=6.
EXPANDING_STRIDE_MONTHS = 2  # legacy notebook's incremental_len=2.
EXPANDING_FOLDS = 10  # target fold count, matching Chapter 6/7; the realised count depends on each dataset's span.
EXPANDING_WINDOW = "expanding"

# ── Datasets ─────────────────────────────────────────────────────────────────
# Same dataset entries and keys Chapter 6/7 use (dataset_name is the key `load_dataset` resolves); no feature list
# is hardcoded here -- `load_dataset` selects weather/calendar/autoregressive features automatically.
DATASETS: dict[str, dict] = {
    "MLVS-PT": {
        "dataset_name": "MLVS-PT",
        "data_path": str(thesis_data_dir()),
        "train_start": "2019-03",
        "train_end": "2022-01",
        "test_start": "2021-07",
        "test_end": "2022-01",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
}

CHAPTER_DATASETS = tuple(DATASETS)

# HPO is disabled for this experiment: every condition is a simple baseline (statistical or classical ML), and the
# question under test is which backtesting protocol to trust, not which hyperparameters to tune.
HPO_TRAIN_MONTHS = 12
HPO_VAL_MONTHS = 2


def protocol_check(spec_common: dict) -> dict[str, int | str]:
    """``spec_common`` as a dict, for logging next to results and comparing the two protocol variants directly."""
    return {
        "folds": spec_common["cv_folds"],
        "test_months": spec_common["cv_test_size"],
        "stride_months": spec_common["cv_stride"],
        "val_months": spec_common["cv_val_size"],
        "min_train_months": spec_common["cv_train_size"],
        "window": spec_common["cv_window"],
    }


def experiment_output_dir(protocol: str, output_root: Path | None = None) -> Path:
    """``<output_root>/best_practices/<protocol>``, e.g. ``single_split``/``expanding``."""
    return (output_root or DEFAULT_OUTPUT_ROOT) / "best_practices" / protocol


SPEC_SINGLE_SPLIT: dict = {
    "cv_split_freq": CV_SPLIT_FREQ,
    "cv_train_size": MIN_TRAIN_MONTHS,
    "cv_test_size": EXPANDING_TEST_MONTHS,
    "cv_val_size": EXPANDING_VAL_MONTHS,
    "cv_calib_size": 0,
    "cv_stride": EXPANDING_STRIDE_MONTHS,
    "cv_folds": 1,
    "cv_window": EXPANDING_WINDOW,
    "fixed_overrides": {},
    "hpo_enabled": False,
}

SPEC_EXPANDING: dict = {
    "cv_split_freq": CV_SPLIT_FREQ,
    "cv_train_size": MIN_TRAIN_MONTHS,
    "cv_test_size": EXPANDING_TEST_MONTHS,
    "cv_val_size": EXPANDING_VAL_MONTHS,
    "cv_calib_size": 0,
    "cv_stride": EXPANDING_STRIDE_MONTHS,
    "cv_folds": EXPANDING_FOLDS,
    "cv_window": EXPANDING_WINDOW,
    "fixed_overrides": {},
    "hpo_enabled": False,
}


def dataset_spec(dataset_key: str, protocol: str, **spec_fields) -> ExperimentSpec:
    """One dataset's :class:`ExperimentSpec` under the named protocol (``"single_split"`` or ``"expanding"``)."""
    spec_common = {"single_split": SPEC_SINGLE_SPLIT, "expanding": SPEC_EXPANDING}[protocol]
    return ExperimentSpec(
        **{**spec_common, **spec_fields},
        datasets={dataset_key: DATASETS[dataset_key]},
        hpo_train_months=HPO_TRAIN_MONTHS,
        hpo_val_months=HPO_VAL_MONTHS,
    )


def base_config(output_dir: Path, metrics: list[str]) -> PipelineConfig:
    """Shared :class:`PipelineConfig`; the dataset fields come from each spec's dataset entry."""
    return PipelineConfig(
        output_dir=f"{output_dir}/",
        data_path=str(thesis_data_dir()),
        out_activation="Identity",
        epochs=1,  # every condition here is a non-neural baseline; epochs is inert for them.
        num_trials=1,
        split_freq=CV_SPLIT_FREQ,
        metrics=metrics,
        log_level="INFO",
        log_file=f"{output_dir}/run.log",
        use_latex=False,
        num_threads=1,
    )
