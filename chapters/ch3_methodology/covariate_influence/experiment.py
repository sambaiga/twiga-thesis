r"""Chapter 3, RQ2: meteorological covariate ablation -- four conditions on one MLPF backbone, both datasets.

Reuses ``best_practices/protocol.py``'s own expanding-window protocol numbers directly (import, not a second copy):
the backtesting protocol is shared across every Chapter 3 experiment, only the dataset and the conditions change.

Usage::

    uv run python chapters/ch3_methodology/covariate_influence/experiment.py
    uv run python chapters/ch3_methodology/covariate_influence/experiment.py --dataset MLVS-PT --quick
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path

import pandas as pd
from twiga.core.utils import configure, get_logger
from twiga.experiment import ExperimentEngine, ExperimentSpec
from twiga.models.nn import MLPFConfig
from twiga.utils.run_forecast import PipelineConfig

from chapters.ch3_methodology.best_practices.protocol import (
    DEFAULT_OUTPUT_ROOT as BEST_PRACTICES_OUTPUT_ROOT,
    EXPANDING_FOLDS,
    EXPANDING_STRIDE_MONTHS,
    EXPANDING_TEST_MONTHS,
    EXPANDING_VAL_MONTHS,
    EXPANDING_WINDOW,
    HPO_TRAIN_MONTHS,
    HPO_VAL_MONTHS,
    MIN_TRAIN_MONTHS,
)
from chapters.ch3_methodology.covariate_influence.conditions import CONDITIONS

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BEST_PRACTICES_OUTPUT_ROOT.parent / "ch3_methodology" / "covariate_influence"

# Same two datasets the chapter's own README names (MLVS-PT, SPS-UK -- the latter's dataset key is "MLVS-UK", see
# chapters/ch3_methodology/cwe_validation's own real_datasets() and twiga_thesis/datasets.py's own column-naming
# comment for why the display name differs from the key `load_dataset` resolves).
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

SPEC_COMMON: dict = {
    "name": "Covariate Influence Ablation",
    "output_prefix": "covariate_influence",
    "condition_cls": MLPFConfig,
    "backbone_cls": MLPFConfig,
    "conditions": CONDITIONS,
    "controlled_fields": frozenset(),
    "cv_split_freq": "months",
    "cv_train_size": MIN_TRAIN_MONTHS,
    "cv_test_size": EXPANDING_TEST_MONTHS,
    "cv_val_size": EXPANDING_VAL_MONTHS,
    "cv_calib_size": 0,
    "cv_stride": EXPANDING_STRIDE_MONTHS,
    "cv_folds": EXPANDING_FOLDS,
    "cv_window": EXPANDING_WINDOW,
    "fixed_overrides": {"early_stop_patience": 40, "num_workers": 0},
    "reference_conditions": {"covariate_influence": "ghi_temp_time"},
    "hpo_train_months": HPO_TRAIN_MONTHS,
    "hpo_val_months": HPO_VAL_MONTHS,
}


def build_specs() -> dict[str, ExperimentSpec]:
    """One :class:`~twiga.experiment.ExperimentSpec` per dataset, same protocol, same conditions."""
    return {
        dataset_key: ExperimentSpec(**SPEC_COMMON, datasets={dataset_key: dataset_entry})
        for dataset_key, dataset_entry in DATASETS.items()
    }


def base_config(output_dir: Path, epochs: int) -> PipelineConfig:
    """Shared :class:`~twiga.utils.run_forecast.PipelineConfig`."""
    return PipelineConfig(
        output_dir=f"{output_dir}/",
        data_path="examples/data",
        out_activation="Identity",
        epochs=epochs,
        num_trials=50,
        hpo_epochs=50,
        split_freq="months",
        metrics=["nrmse", "mae", "cwe", "cwe_stratified", "picp", "nmpi"],
        log_level="INFO",
        log_file=f"{output_dir}/run.log",
        use_latex=False,
        num_threads=1,
    )


def main() -> None:
    """CLI entry point: run the ablation on the requested dataset(s)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="*", default=list(DATASETS), choices=[*DATASETS, "all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-hpo", dest="skip_hpo", action="store_true", default=True)
    parser.add_argument("--run-hpo", dest="skip_hpo", action="store_false")
    args = parser.parse_args()

    configure(level="INFO")
    specs = build_specs()
    dataset_keys = list(DATASETS) if "all" in args.dataset else args.dataset

    output_dir = args.output_root / "covariate_influence"
    cfg = base_config(output_dir, epochs=5 if args.quick else args.epochs)
    if args.quick:
        cfg = dc_replace(cfg, num_trials=1, hpo_epochs=5)

    summaries = []
    for dataset_key in dataset_keys:
        engine = ExperimentEngine(specs[dataset_key])
        summaries.append(
            engine.run(
                base_cfg=cfg,
                groups=["covariate_influence"],
                dataset_keys=[dataset_key],
                skip_hpo=args.skip_hpo,
                tracking_uri=None,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    out = output_dir / "results_covariate_influence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    log.info("summary written to %s", out)


if __name__ == "__main__":
    main()
