r"""PIT histograms and reliability diagrams for the `qvn_type` group's conditions.

`README.md`'s item 10 flags `berkowitz_test`/`pit_uniformity_test` as the one
thing the standard backtesting path cannot cover: the engine's
`predictions.parquet` only carries the reduced `forecast`/`lower`/`upper`
form, not the full learned quantile array (`quantile_hats`/`tau_hats`) a PIT
value needs. This script gets that array the same way `worked_example.py`
already does for its one representative day -- forward hooks on
`FPQRDistribution.quantile_proposal`/`.quantile_output`, present under every
`qvn_type` branch unconditionally (confirmed directly against
`twiga/distributions/nn/fpqr/distribution.py`: `quantile_proposal` is set
once outside the `qvn_type` branching, `quantile_output` is set on every
branch) -- extended across all eight `qvn_type` conditions and a single
train/test split per dataset, rather than one condition on one day.

This is illustrative, not a faithful reproduction of the formal backtest:
one freshly-trained model per condition on `learned_tau.py`'s own
`SPLITS` (a single train/test window), not the 10-fold rolling-origin
protocol `fpqr_experiment.py`'s `qvn_type` group actually ran under. Treat
the PIT/reliability shapes as diagnostic of each `qvn_type` mechanism's
calibration behaviour, not as a substitute for `analyse_results.py`'s
per-fold Winkler-score comparison.

PIT computation: for each (window, horizon-step), the predicted
`(tau_i, q_i)` pairs are sorted by *value* `q_i` (not by `tau_i` -- a
non-monotone head's raw quantile order need not match its tau order) and
linearly interpolated to evaluate the empirical CDF at the realised value,
`u = F_hat(y)`, clipped to `[0, 1]`. Standard practice for PIT-from-quantiles
(the same value-sorted interpolation `crps_from_quantiles`-style estimators
already assume implicitly).

Run:  uv run python experiment/chapter-5-fpqr/qvn_pit.py
      uv run python experiment/chapter-5-fpqr/qvn_pit.py --dataset MLVS-UK --conditions qvn_relu qvn_softplus
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learned_tau import SPLITS, find_head  # noqa: E402
from lets_plot import ggsave  # noqa: E402
from twiga.core.plot import plot_pit_histogram, plot_reliability_diagram  # noqa: E402
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig  # noqa: E402
from twiga.utils.run_forecast import (  # noqa: E402
    PipelineConfig,
    build_data_config,
    build_forecaster,
    build_forecaster_config,
    load_and_preprocess,
    split_data,
)

script_dir = Path(__file__).resolve().parent
OUT = script_dir / "qvn_pit"
EPOCHS = 60
N_DEFAULT = 9
ALPHA = 0.10

# Mirrors fpqr_experiment.py's qvn_type group conditions exactly (same
# overrides, same _fpqr base of n_quantiles=9/conf_level=0.10) -- not
# imported directly, since fpqr_experiment.py's CONDITIONS list builds
# twiga.experiment.Condition objects for ExperimentEngine, not the raw
# kwarg dicts a fresh MLPFFPQRConfig construction here needs.
QVN_CONDITIONS: dict[str, dict] = {
    "qvn_relu": {"qvn_type": "standard", "crossing_mode": "relu", "crossing_penalty": 10.0},
    "qvn_softplus": {
        "qvn_type": "standard",
        "crossing_mode": "softplus",
        "crossing_penalty": 10.0,
        "softplus_beta": 1.0,
    },
    "qvn_monotone": {"qvn_type": "monotone", "crossing_penalty": 0.0},
    "qvn_monotone_cal": {
        "qvn_type": "monotone_cal",
        "crossing_penalty": 0.0,
        "lambda_cal": 0.01,
        "cal_temperature": 1.0,
    },
    "qvn_lattice": {"qvn_type": "lattice", "crossing_penalty": 0.0},
    "qvn_spline": {"qvn_type": "spline", "crossing_penalty": 0.0},
    "qvn_umnn": {"qvn_type": "umnn", "crossing_penalty": 0.0},
    "qvn_attention": {"qvn_type": "attention", "crossing_mode": "relu", "crossing_penalty": 10.0},
}


def build(dataset: str, condition: str, train_start: str, train_end: str, test_start: str, test_end: str):
    """Build a forecaster for one qvn_type condition, mirroring fpqr_experiment.py's own overrides."""
    cfg = PipelineConfig(
        dataset_name=dataset,
        data_path="examples/data",
        output_dir=str(script_dir / "_run"),
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        input_window_size=96,
        forecast_horizon=48,
        window_stride=48,
        period="30min",
        epochs=EPOCHS,
        use_latex=False,
        num_threads=4,
    )
    data, target, cal, exog, lags, lat, lon, lookback = load_and_preprocess(cfg)
    data_config = build_data_config(cfg, target, lags, cal, exog, lat, lon, lookback_window_size=lookback)
    model_cfg = MLPFFPQRConfig(  # type: ignore[call-arg]
        out_activation_function="Identity",
        max_epochs=EPOCHS,
        rich_progress_bar=False,
        n_quantiles=N_DEFAULT,
        conf_level=ALPHA,
        **QVN_CONDITIONS[condition],
    )
    forecaster = build_forecaster(data_config, [model_cfg], build_forecaster_config(cfg))
    train_df, test_df = split_data(data, cfg)
    return forecaster, train_df, test_df


def fitted_target_scaler(data_pipeline):
    """The fitted target-scaling transformer, dug out of the built ColumnTransformer.

    sklearn clones estimators inside a ColumnTransformer, so the fitted
    scaler is the clone living in the built pipeline, not any template held
    on a config object -- identical helper to worked_example.py's own,
    duplicated rather than imported since that script keeps it as a local
    closure, not a module-level export.
    """
    from sklearn.compose import ColumnTransformer

    pipe = getattr(data_pipeline, "data_pipeline", None)
    if pipe is None:
        raise RuntimeError("data pipeline was not retained after fit")
    for _, step in getattr(pipe, "steps", []):
        if isinstance(step, ColumnTransformer):
            named = getattr(step, "named_transformers_", {})
            if "target_scaling" in named:
                return named["target_scaling"]
    raise RuntimeError("no fitted target_scaling transformer found")


# FPQRDistribution._get_quantile_values dispatches to a differently-named
# submodule per qvn_type (confirmed directly against distribution.py); only
# "standard" (relu/softplus) sets `quantile_output` at all. Hooking `head`
# itself doesn't work either: the actual inference entry point,
# `FPQRDistribution.forecast()`, runs under torch.no_grad() and calls
# `self.quantile_proposal(...)`/internal helper methods directly, never
# `self.forward()`/`self(...)` -- confirmed directly (a `head`-level hook
# never fires during `forecaster.predict()`, only genuine nn.Module
# `__call__`s trigger PyTorch's forward hooks). The submodule map below is
# the only thing that actually works for every qvn_type.
_QVN_VALUE_MODULE = {
    "standard": "quantile_output",
    "monotone": "monotone_qvn",
    "monotone_cal": "calibrated_monotone_qvn",
    "attention": "attention_qvn",
    "lattice": "lattice_qvn",
    "spline": "spline_qvn",
    "umnn": "umnn_qvn",
}


def capture(head, forecaster, test_df) -> tuple[np.ndarray, np.ndarray]:
    """Forward-hook capture of (tau, quantile value), shape (W, N, H) each.

    `quantile_proposal` (taus) is set unconditionally in `FPQRDistribution.__init__`,
    outside any `qvn_type` branching, so hooking it directly (as worked_example.py
    does for the single "standard" condition) is safe for every `qvn_type`.
    """
    value_module = getattr(head, _QVN_VALUE_MODULE[head.qvn_type])
    taus, vals = [], []
    h1 = head.quantile_proposal.register_forward_hook(lambda m, a, o: taus.append(o[1].detach().cpu().numpy()))
    h2 = value_module.register_forward_hook(lambda m, a, o: vals.append(o.detach().cpu().numpy()))
    try:
        forecaster.predict(test_df)
    finally:
        h1.remove()
        h2.remove()
    tau = np.concatenate(taus, axis=0)
    q = np.concatenate(vals, axis=0)
    q = q.reshape(q.shape[0], -1, head.horizon, head.num_outputs)
    return tau, q[..., 0]


def pit_values(tau: np.ndarray, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """PIT per (window, horizon-step): interpolate the value-sorted (q, tau) pairs at the realised value.

    Args:
        tau: Quantile levels, shape (W, N, H).
        q: Predicted quantile values, shape (W, N, H).
        actual: Realised values, shape (W, H), original units (same scale as q).

    Returns:
        Flattened PIT values, shape (W * H,), each in [0, 1].
    """
    w, n, h = tau.shape
    out = np.empty((w, h))
    for i in range(w):
        for j in range(h):
            order = np.argsort(q[i, :, j])
            q_sorted, tau_sorted = q[i, order, j], tau[i, order, j]
            out[i, j] = np.interp(actual[i, j], q_sorted, tau_sorted, left=0.0, right=1.0)
    return out.ravel()


def reliability_curve(pit: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Empirical coverage at each nominal level, i.e. the PIT values' own ECDF."""
    return np.array([(pit <= lvl).mean() for lvl in levels])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="MLVS-PT", choices=list(SPLITS))
    parser.add_argument("--conditions", nargs="+", default=list(QVN_CONDITIONS), choices=list(QVN_CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = OUT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b, c, d = SPLITS[args.dataset]

    levels = np.linspace(0.05, 0.95, 19)
    rows = []
    for condition in args.conditions:
        print(f"\n=== {condition} ({args.dataset}) ===", flush=True)
        torch.manual_seed(args.seed)
        forecaster, train_df, test_df = build(args.dataset, condition, a, b, c, d)
        forecaster.fit(train_df)
        obj = forecaster.fitted_models if hasattr(forecaster, "fitted_models") else forecaster
        head = find_head(obj)
        if head is None:
            print(f"  no FPQRDistribution found for {condition}, skipping")
            continue
        tau, q_scaled = capture(head, forecaster, test_df)
        w, n, h = tau.shape

        # Undo the target scaling so PIT is evaluated in the same units as
        # the realised value -- same clone-inside-ColumnTransformer helper
        # worked_example.py uses, reshaped to inverse_transform's expected
        # (n_samples, 1) shape and back.
        scaler = fitted_target_scaler(forecaster.data_pipeline)
        q_unscaled = scaler.inverse_transform(q_scaled.reshape(-1, 1)).reshape(q_scaled.shape)

        # Realised values read directly from the raw parquet, exactly
        # worked_example.py's own alignment (not test_df's own scaled
        # target column, avoiding any scaling/column-name ambiguity):
        # first test window starts input_window_size rows into test_df.
        raw = pd.read_parquet(Path("examples/data") / f"{args.dataset}.parquet")
        net = raw["NetLoad(kW)"].to_numpy()
        stamps_all = pd.to_datetime(raw["timestamp"]).to_numpy()
        first = int(test_df.index[0]) + 96  # input_window_size, matching build()'s own value
        actual = np.stack([net[first + i * h : first + i * h + h] for i in range(w)])
        stamps = np.stack([stamps_all[first + i * h : first + i * h + h] for i in range(w)])

        pit = pit_values(tau, q_unscaled, actual)
        np.save(out_dir / f"{condition}_pit.npy", pit)
        # Cached alongside the PIT values so qvn_forecast_examples.py can draw
        # best/worst-day quantile-fan panels without retraining every
        # condition a second time -- the (tau, q, actual) capture above is
        # the expensive part (one full fit + predict per condition).
        np.savez_compressed(out_dir / f"{condition}_forecast.npz", tau=tau, q=q_unscaled, actual=actual, stamps=stamps)

        fig = plot_pit_histogram(pit, title=f"PIT -- {condition} ({args.dataset})")
        ggsave(fig, f"{condition}_pit.png", path=str(out_dir))

        empirical = reliability_curve(pit, levels)
        rel_fig = plot_reliability_diagram(levels, empirical, title=f"Reliability -- {condition} ({args.dataset})")
        ggsave(rel_fig, f"{condition}_reliability.png", path=str(out_dir))

        rows.append(
            {
                "condition": condition,
                "dataset": args.dataset,
                "n_pit": len(pit),
                "pit_mean": float(pit.mean()),
                "pit_std": float(pit.std()),
                "ks_stat_vs_uniform": float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, len(pit))))),
            }
        )
        print(f"  {len(pit)} PIT values, mean={pit.mean():.3f} (uniform: 0.5), std={pit.std():.3f} (uniform: 0.289)")

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "pit_summary.csv"
    if summary_path.exists():
        # Merge with rows from earlier invocations (e.g. a previous
        # --conditions subset) instead of overwriting them -- keyed on
        # condition, this run's rows win on conflict.
        prior = pd.read_csv(summary_path)
        prior = prior[~prior["condition"].isin(summary["condition"])]
        summary = pd.concat([prior, summary], ignore_index=True)
    summary = summary.sort_values("condition").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
