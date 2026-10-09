r"""MLPGAM feature-gate interpretability analysis: is the gate faithful, not decorative?

Experiment 2 (`architectural_evaluation.py`) shows MLPGAM forecasts more
accurately than MLPF. It says nothing about the one architectural feature
MLPF has no analogue for at all: the differentiable feature gate. Chapter 7
currently asserts the gate is "retained for the interpretability the
differentiable mask provides" with no supporting evidence -- a claim, not a
finding. This script tests it directly.

The test (rho_faith, following the correlation-based faithfulness check in
`twiga/models/nn/net/ganf/importance.py`, built for MLPGAF's grouped-additive
gating): correlate each feature's post-sigmoid gate value against its
gradient-based local sensitivity, ``||d(latent)/d(feature)||``. A high
correlation means the gate is doing real work -- features it suppresses are
genuinely low-impact on the model's internal representation, not arbitrarily
dropped. A low or negative correlation would mean the gate's 0/1 pattern
is disconnected from what the model actually uses, i.e. decorative.

MLPGAM is not grouped the way MLPGAF is (no `group_modules`/`groups`; three
flat gates instead: `past_feature_gate`, `future_feature_gate`,
`target_gate`), so `importance.py`'s functions do not apply directly and are
reimplemented here for that flat structure. `compute_gate_sensitivity_correlation`
itself is reused unchanged, since it only needs matching Group/Branch/Feature
keys between two record lists.

Data pipeline note: an earlier draft of this analysis (in a separate
worktree) called a `forecaster._get_test_loader` method that does not exist
anywhere in this codebase, silently falling back to a batch of random noise
when that call failed -- which would have made the sensitivity numbers
meaningless. This version uses the real, existing path instead:
`forecaster.data_pipeline.transform_features(df)`, the same windowed,
scaled feature array the model is actually trained and evaluated on
(`twiga/core/data/pipeline.py`'s `DataPipeline.transform_features`).

Run:  uv run python experiment/ch6_conformal/experiment_1_architectural/mlpgam_gate_analysis.py --dataset MLVS-PT
      uv run python experiment/ch6_conformal/experiment_1_architectural/mlpgam_gate_analysis.py --dataset all
      uv run python experiment/ch6_conformal/experiment_1_architectural/mlpgam_gate_analysis.py \
          --dataset MLVS-UK --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiment/ (twiga_thesis.plotting, common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chapter directory (shared.*)
from twiga.core.config import ExperimentConfig  # noqa: E402
from twiga.models.nn import MLPGAMConfig  # noqa: E402
from twiga.utils.run_forecast import (  # noqa: E402
    PipelineConfig,
    build_data_config,
    build_forecaster,
    load_and_preprocess,
    setup_environment,
    split_data,
)

from chapters.ch6_conformal.shared.gate_faithfulness import (  # noqa: E402
    compute_gate_sensitivity_correlation,
    compute_mlpgam_feature_sensitivity,
    extract_mlpgam_gate_importances,
    sample_batch,
    unwrap_network,
)
from chapters.ch6_conformal.shared.protocol import DATASETS, experiment_output_dir  # noqa: E402
from twiga_thesis.plotting.style import configure, format_axes, panel, style  # noqa: E402
from twiga_thesis.plotting.tokens import DANGER, PRIMARY  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUTPUT_DIR = experiment_output_dir("mlpgam_gate_analysis")


def _build_forecaster(dataset_name: str, epochs: int):
    """Fit a single gated MLPGAM on `dataset_name` and return (forecaster, test_df)."""
    ds = DATASETS[dataset_name]
    cfg = PipelineConfig(
        output_dir=str(OUTPUT_DIR / dataset_name),
        data_path=ds["data_path"],
        dataset_name=ds["dataset_name"],
        train_start=ds["train_start"],
        train_end=ds["train_end"],
        test_start=ds["test_start"],
        test_end=ds["test_end"],
        input_window_size=ds["input_window_size"],
        forecast_horizon=ds["forecast_horizon"],
        period=ds["period"],
        out_activation=ds.get("out_activation", "Identity"),
        epochs=epochs,
        split_freq="months",
        metrics=["mae", "rmse", "smape", "corr"],
        log_level="INFO",
        log_file=str(OUTPUT_DIR / dataset_name / "run.log"),
        use_latex=False,
        num_threads=1,
        num_trials=0,  # fixed backbone, no HPO -- this analysis needs one trained gate, not a tuned one
    )
    setup_environment(cfg)

    data, target_series, cal_vars, exog_feats, lags, lat, lon, lookback = load_and_preprocess(cfg)
    data_config = build_data_config(cfg, target_series, lags, cal_vars, exog_feats, lat, lon, lookback)

    model_config = MLPGAMConfig(use_gate=True)
    exp_cfg = ExperimentConfig(
        split_freq=cfg.split_freq,
        test_size=4,
        train_size=1,
        val_size=2,
        calib_size=0,
        num_splits=1,
        project_name=f"{dataset_name}/gate_analysis",
        root_dir=str(OUTPUT_DIR),
        metrics=cfg.metrics,
    )
    forecaster = build_forecaster(data_config, [model_config], exp_cfg)

    train_df, test_df = split_data(data, cfg)
    split_idx = int(len(train_df) * 0.8)
    forecaster.fit(train_df=train_df.iloc[:split_idx], val_df=train_df.iloc[split_idx:])

    return forecaster, test_df


def scatter_figure(merged: pd.DataFrame) -> plt.Figure:
    """Gate importance vs. sensitivity, one point per feature, coloured by branch."""
    with style():
        fig, ax = plt.subplots(figsize=panel(0.6))
        colors = {"past": PRIMARY, "future": DANGER}
        for branch, sub in merged.groupby("Branch"):
            ax.scatter(
                sub["Sensitivity"], sub["Importance"], color=colors.get(branch, "gray"), label=branch, s=18, alpha=0.8
            )
        ax.set_xlabel("Local sensitivity (gradient L2 norm)")
        ax.set_ylabel("Gate importance (post-sigmoid activation)")
        ax.legend(frameon=False)
        format_axes(ax)
    return fig


def run_one(dataset_name: str, epochs: int) -> pd.DataFrame:
    """Fit, analyse, and save output for one dataset; returns its rho_faith table."""
    configure(latex=False)
    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    forecaster, test_df = _build_forecaster(dataset_name, epochs)
    model = unwrap_network(forecaster)
    model.eval()

    sample = sample_batch(forecaster, test_df)

    gate_records = extract_mlpgam_gate_importances(model)
    if not gate_records:
        raise RuntimeError(
            "no gate records extracted -- model has no past_feature_gate/future_feature_gate; "
            "check use_gate=True took effect and _unwrap_network found the right object"
        )
    pd.DataFrame(gate_records).to_csv(out_dir / "gate_importances.csv", index=False)

    sens_records = compute_mlpgam_feature_sensitivity(model, sample)
    pd.DataFrame(sens_records).to_csv(out_dir / "local_sensitivity.csv", index=False)

    rho_df, merged = compute_gate_sensitivity_correlation(gate_records, sens_records)
    rho_df.insert(0, "Dataset", dataset_name)
    rho_df.to_csv(out_dir / "rho_faith.csv", index=False)
    merged.to_csv(out_dir / "gate_vs_sensitivity.csv", index=False)

    print(f"\n{dataset_name}: rho_faith\n{rho_df.to_string(index=False)}")

    fig = scatter_figure(merged)
    fig.savefig(out_dir / "rho_faith_scatter.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "rho_faith_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_dir}")
    return rho_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        nargs="+",
        default=["MLVS-PT"],
        help=f"One or more of {list(DATASETS)}, or 'all' for both.",
    )
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    datasets = list(DATASETS) if "all" in args.dataset else args.dataset
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}; choose from {list(DATASETS)} or 'all'.")

    all_rho = [run_one(dataset_name, args.epochs) for dataset_name in datasets]

    if len(all_rho) > 1:
        combined = pd.concat(all_rho, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "rho_faith_all_datasets.csv", index=False)
        print(f"\nCombined:\n{combined.to_string(index=False)}")
        print(f"\nwrote {OUTPUT_DIR / 'rho_faith_all_datasets.csv'}")


if __name__ == "__main__":
    main()
