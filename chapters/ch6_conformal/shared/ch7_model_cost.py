r"""Computational cost of the Chapter 7 models: trainable parameters and inference wall time.

Supports the "lightweight" part of RQ3 without retraining anything:

* Parameters: each model is rebuilt from its cached, tuned hyperparameters (the engine's ``backbone_hpo`` caches) on
  the dataset's real ``DataPipelineConfig`` (features and ACF-derived input window from ``load_dataset``), and its
  trainable parameters are counted.
* Inference time: the engine logs, per fold, the wall-clock time to forecast that fold's whole test set (about 120
  day-ahead windows); the mean and standard deviation over the ten folds are reported, averaged over seeds where the
  condition was seeded. Times come from the machine the experiments ran on and are comparable only relative to one
  another.

Usage::

    uv run python experiment/ch6_conformal/shared/ch7_model_cost.py \
        --study-name architectural_study --benchmark-name probabilistic_benchmark_v4
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import json
from pathlib import Path
import sys

import pandas as pd

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir.parents[1]))

from twiga.core.utils import slugify  # noqa: E402
from twiga.models.nn import MLPFConfig, MLPGAMConfig, MLPGAMCRCConfig, MLPGAMQRConfig  # noqa: E402
from twiga.models.nn.mlpf_model import MLPFModel  # noqa: E402
from twiga.models.nn.mlpgam_model import MLPGAMModel  # noqa: E402
from twiga.models.nn.mlpgambnn_model import MLPGAMBNNConfig, MLPGAMBNNModel  # noqa: E402
from twiga.models.nn.mlpgamcrc_model import MLPGAMCRCModel  # noqa: E402
from twiga.models.nn.mlpgamlaplace_model import MLPGAMLaplaceConfig, MLPGAMLaplaceModel  # noqa: E402
from twiga.models.nn.mlpgamqr_model import MLPGAMQRModel  # noqa: E402
from twiga.utils.run_forecast import build_data_config, load_and_preprocess  # noqa: E402

from chapters.ch6_conformal.shared.conditions import PROPOSED  # noqa: E402
from chapters.ch6_conformal.shared.protocol import (  # noqa: E402
    CHAPTER_DATASETS,
    DATASETS,
    DEFAULT_OUTPUT_ROOT,
    QR_QUANTILES,
    base_config,
    experiment_output_dir,
)

# name: (experiment, config class, model class, HPO cache file, overrides, conditions whose inference time is used)
MODELS = {
    "MLPF": ("study", MLPFConfig, MLPFModel, "mlpf_mlpf", {}, "mlpf_point"),
    "MLPGAM": ("study", MLPGAMConfig, MLPGAMModel, "mlpgam_shared_mlpgam", {"use_gate": True}, "mlpgam_point"),
    "MLPGAM-CRC": ("benchmark", MLPGAMCRCConfig, MLPGAMCRCModel, "mlpgam_shared_mlpgam", PROPOSED, "mlpgam_crc"),
    "QR": (
        "benchmark",
        MLPGAMQRConfig,
        MLPGAMQRModel,
        "mlpgam_shared_mlpgam",
        {"use_gate": True, "quantiles": QR_QUANTILES},
        "mlpgam_qr",
    ),
    "BNN": ("benchmark", MLPGAMBNNConfig, MLPGAMBNNModel, "mlpgam_shared_mlpgam", {"use_gate": True}, "mlpgam_bnn"),
    "Laplace": (
        "benchmark",
        MLPGAMLaplaceConfig,
        MLPGAMLaplaceModel,
        "mlpgam_shared_mlpgam",
        {"use_gate": True},
        "mlpgam_laplace",
    ),
}


def data_config(dataset: str, output_dir: Path):
    cfg = dc_replace(base_config(output_dir), **DATASETS[dataset])
    data, target, cal, exog, lags, lat, lon, lookback = load_and_preprocess(cfg)
    return build_data_config(cfg, target, lags, cal, exog, lat, lon, lookback_window_size=lookback)


def parameter_count(config_cls, model_cls, cache: Path, overrides: dict, dc) -> int:
    config = config_cls.from_data_config(dc)
    params = json.loads(cache.read_text())["params"]
    for key, value in {**params, **overrides}.items():
        if hasattr(config, key):
            setattr(config, key, value)
    model = model_cls(model_config=config)
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)


def inference_times(experiment_dir: Path, dataset: str, prefix: str) -> pd.Series:
    """Per-fold test-set inference wall time (s) of ``prefix`` (all seeds of a seeded condition), latest run first."""
    for csv in sorted(experiment_dir.glob("*/runs/*/results/*_full.csv"), reverse=True):
        full = pd.read_csv(csv, usecols=lambda c: c in {"condition", "dataset", "Folds", "inference-time"})
        rows = full[(full["dataset"] == dataset) & full["condition"].str.match(rf"^{prefix}(__s\d+)?$")]
        if not rows.empty:
            return rows.groupby(["condition", "Folds"])["inference-time"].first().groupby("Folds").mean()
    return pd.Series(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--study-name", default="architectural_study")
    ap.add_argument("--benchmark-name", default="probabilistic_benchmark")
    ap.add_argument("--datasets", nargs="+", default=list(CHAPTER_DATASETS))
    args = ap.parse_args()

    dirs = {
        "study": experiment_output_dir(args.study_name, args.output_root),
        "benchmark": experiment_output_dir(args.benchmark_name, args.output_root),
    }
    rows = []
    for dataset in args.datasets:
        dc = data_config(dataset, args.output_root / "analysis" / "model_cost")
        for name, (exp, config_cls, model_cls, cache_name, overrides, condition) in MODELS.items():
            caches = sorted(dirs[exp].glob(f"*/backbone_hpo/{slugify(dataset)}/{cache_name}_best_params.json"))
            n_params = parameter_count(config_cls, model_cls, caches[-1], overrides, dc) if caches else float("nan")
            times = inference_times(dirs[exp], dataset, condition)
            rows.append(
                {
                    "dataset": dataset,
                    "model": name,
                    "parameters": n_params,
                    "inference_ms_mean": 1e3 * times.mean(),
                    "inference_ms_std": 1e3 * times.std(ddof=1),
                    "folds": len(times),
                }
            )
    table = pd.DataFrame(rows)
    out = args.output_root / "analysis" / "model_cost"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "model_cost.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
