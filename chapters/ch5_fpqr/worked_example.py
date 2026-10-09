"""Capture every intermediate signal of one MLPQF forecast, for the pipeline figure.

Saves `s563_worked_example.npz` holding, for one chosen forecast day:
    hist        the input window of net load
    ghi         day-ahead irradiance over the forecast horizon
    tau         learned quantile fractions, (N, H)
    quant       quantile values in original units, (N, H)
    actual      realised net load over the horizon, (H,)
    lead        the lead-time index the figure highlights

The day is chosen as the median-CRPS day among the most variable third by
within-day irradiance, so it is a representative hard day rather than a
flattering one.

Run:  uv run python experiment/chapter-5-fpqr/worked_example.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chapters.ch5_fpqr.learned_tau import SPLITS, build, find_head  # noqa: E402

script_dir = Path(__file__).resolve().parent
DATASET = "MLVS-PT"


def capture_both(head, forecaster, test_df):
    """Grab the proposal grid and the quantile values from one prediction pass."""
    taus, vals = [], []
    h1 = head.quantile_proposal.register_forward_hook(lambda m, a, o: taus.append(o[1].detach().cpu().numpy()))
    h2 = head.quantile_output.register_forward_hook(lambda m, a, o: vals.append(o.detach().cpu().numpy()))
    try:
        forecaster.predict(test_df)
    finally:
        h1.remove()
        h2.remove()
    tau = np.concatenate(taus, axis=0)  # (W, N, H)
    q = np.concatenate(vals, axis=0)  # (W, N*H*D) flat
    q = q.reshape(q.shape[0], -1, head.horizon, head.num_outputs)
    return tau, q[..., 0]  # (W, N, H)


def crps_from_quantiles(tau: np.ndarray, q: np.ndarray, y: np.ndarray) -> float:
    """Pinball loss averaged over levels, the discrete CRPS estimator."""
    d = y[None, :] - q
    return float(np.mean(np.maximum(tau[:, None] * d, (tau[:, None] - 1) * d)))


if __name__ == "__main__":
    torch.manual_seed(0)
    a, b, c, d = SPLITS[DATASET]
    cfg, forecaster, train_df, test_df = build(DATASET, a, b, c, d)
    forecaster.fit(train_df)
    obj = forecaster.fitted_models if hasattr(forecaster, "fitted_models") else forecaster
    head = find_head(obj)
    tau, q_scaled = capture_both(head, forecaster, test_df)
    W, N, H = tau.shape
    print("tau", tau.shape, "quant", q_scaled.shape, flush=True)

    # Undo the target scaling so the figure is in kW rather than z-scores.
    # sklearn clones estimators inside a ColumnTransformer, so the fitted scaler
    # is the clone in the built pipeline, not the template on the config object.
    def fitted_target_scaler(dp):
        from sklearn.compose import ColumnTransformer

        pipe = getattr(dp, "data_pipeline", None)
        if pipe is None:
            raise RuntimeError("data pipeline was not retained after fit")
        for _, step in getattr(pipe, "steps", []):
            if isinstance(step, ColumnTransformer):
                named = getattr(step, "named_transformers_", {})
                if "target_scaling" in named:
                    return named["target_scaling"]
        raise RuntimeError("no fitted target_scaling transformer found")

    scaler = fitted_target_scaler(forecaster.data_pipeline)
    flat = q_scaled.reshape(-1, 1)
    quant = scaler.inverse_transform(flat).reshape(q_scaled.shape)
    print(
        f"inverse-scaled with {type(scaler).__name__}: quantile range {quant.min():.2f} to {quant.max():.2f} kW",
        flush=True,
    )

    raw = pd.read_parquet(Path(cfg.data_path) / f"{cfg.dataset_name}.parquet")
    net = raw["NetLoad(kW)"].to_numpy()
    ghi = raw["Ghi"].to_numpy()
    first = int(test_df.index[0]) + cfg.input_window_size

    actual = np.stack([net[first + w * H : first + w * H + H] for w in range(W)])
    hist = np.stack([net[first + w * H - cfg.input_window_size : first + w * H] for w in range(W)])
    ghi_h = np.stack([ghi[first + w * H : first + w * H + H] for w in range(W)])

    cv = np.array(
        [
            float(s[s > 1e-6].std() / s[s > 1e-6].mean()) if (s > 1e-6).any() and s[s > 1e-6].mean() > 0 else 0.0
            for s in ghi_h
        ]
    )
    crps = np.array([crps_from_quantiles(tau[w].mean(axis=1), quant[w], actual[w]) for w in range(W)])

    hard = np.where(cv >= np.quantile(cv, 2 / 3))[0]  # most variable third
    pick = hard[np.argsort(crps[hard])[len(hard) // 2]]  # median CRPS among them
    lead = int(np.argmax(np.abs(quant[pick, -1] - quant[pick, 0])))  # widest interval step

    print(
        f"chosen window {pick}: ghi_cv {cv[pick]:.3f} (rank {np.sum(cv < cv[pick])}/{W}), "
        f"CRPS {crps[pick]:.4f}, highlight lead {lead}",
        flush=True,
    )
    np.savez(
        script_dir / "s563_worked_example.npz",
        hist=hist[pick],
        ghi=ghi_h[pick],
        tau=tau[pick],
        quant=quant[pick],
        actual=actual[pick],
        lead=lead,
        ghi_cv=cv[pick],
        crps=crps[pick],
        n_windows=W,
        input_window=cfg.input_window_size,
    )
    print("wrote s563_worked_example.npz")
