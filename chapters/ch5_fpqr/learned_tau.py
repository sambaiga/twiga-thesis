"""Extract the FPQR quantile grid before and after training on net-load data.

Produces `s563_learned_tau.csv`: the proposal grid at initialisation and after
training, per quantile slot and per forecast step, so the grid can be plotted
against hour of day. This is the measured counterpart to the illustrative
`fig_fpqr_grid_map` panel.

Run:  uv run python experiment/chapter-5-fpqr/learned_tau.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.utils.run_forecast import (
    PipelineConfig,
    build_data_config,
    build_forecaster,
    build_forecaster_config,
    load_and_preprocess,
    split_data,
)

script_dir = Path(__file__).resolve().parent
EPOCHS = 60

# Clean, non-overlapping splits for both datasets.
SPLITS = {
    "MLVS-PT": ("2019-03", "2021-06", "2021-07", "2021-10"),
    "MLVS-UK": ("2018-02", "2019-12", "2020-01", "2020-06"),
}


def build(
    dataset: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    n_quantiles: int = 9,
    entropy_weight: float = 1.0,
    proposal_type: str = "softmax",
):
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
        n_quantiles=n_quantiles,
        entropy_weight=entropy_weight,
        proposal_type=proposal_type,
    )
    forecaster = build_forecaster(data_config, [model_cfg], build_forecaster_config(cfg))
    train_df, test_df = split_data(data, cfg)
    return cfg, forecaster, train_df, test_df


def find_head(obj, depth: int = 0):
    """Locate the FPQRDistribution inside whatever the forecaster holds."""
    from twiga.distributions.nn.fpqr.distribution import FPQRDistribution

    if isinstance(obj, FPQRDistribution):
        return obj
    if depth > 6:
        return None
    children = []
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    elif isinstance(obj, torch.nn.Module):
        children = list(obj.children())
    else:
        children = [getattr(obj, a) for a in vars(obj)] if hasattr(obj, "__dict__") else []
    for c in children:
        found = find_head(c, depth + 1)
        if found is not None:
            return found
    return None


def capture(head, forecaster, test_df) -> np.ndarray:
    """Run a prediction and capture the proposal grid the head actually emitted."""
    grabbed: list[np.ndarray] = []

    def hook(module, args, output):
        # QuantileProposal returns (taus, tau_hats, entropies); tau_hats is (B, N, H)
        grabbed.append(output[1].detach().cpu().numpy())

    handle = head.quantile_proposal.register_forward_hook(hook)
    try:
        forecaster.predict(test_df)
    finally:
        handle.remove()
    if not grabbed:
        raise RuntimeError("proposal hook captured nothing")
    return np.concatenate(grabbed, axis=0)  # (B_total, N, H)


if __name__ == "__main__":
    for DS, (a, b, c, d) in SPLITS.items():
        torch.manual_seed(0)
        print(f"\n===== {DS} =====", flush=True)
        cfg, forecaster, train_df, test_df = build(DS, a, b, c, d)
        print(f"train {train_df.shape}  test {test_df.shape}", flush=True)

        forecaster.fit(train_df)
        obj = forecaster.fitted_models if hasattr(forecaster, "fitted_models") else forecaster
        head = find_head(obj)
        if head is None:
            raise SystemExit("FPQR head not found")

        trained = capture(head, forecaster, test_df)
        print("trained grid:", trained.shape, flush=True)

        # The untrained baseline: the same head with its proposal network returned to
        # initialisation. Everything upstream is unchanged, so the comparison isolates
        # what the proposal network learned rather than what the encoder learned.
        keep = copy.deepcopy(head.quantile_proposal.state_dict())
        qp = head.quantile_proposal
        fresh = type(qp)(
            n_quantiles=qp.n_quantiles,
            z_dim=qp.z_dim,
            dropout=float(qp.dropout.p),
            conf_level=float(qp.conf_level.item()),
            n_outputs=qp.n_outputs,
            crossing_penalty=qp.crossing_penalty,
            stop_gradient=qp.stop_gradient,
        )
        head.quantile_proposal.load_state_dict(fresh.state_dict())
        untrained = capture(head, forecaster, test_df)
        head.quantile_proposal.load_state_dict(keep)
        print("untrained grid:", untrained.shape, flush=True)

        # stride == horizon, so every forecast window starts at the same clock time;
        # step h therefore maps to a fixed hour of day.
        # test_df carries an integer index, so the clock time of the forecast origin
        # must come from the raw timestamp column; stride == horizon means every
        # window starts at the same time of day.
        raw = pd.read_parquet(Path(cfg.data_path) / f"{cfg.dataset_name}.parquet")
        ts = pd.to_datetime(raw["timestamp"])
        first_target = int(test_df.index[0]) + cfg.input_window_size
        origin = ts.iloc[first_target]
        start_hour = origin.hour + origin.minute / 60.0
        H = trained.shape[2]
        hours = (start_hour + 0.5 * np.arange(H)) % 24

        # Keep the full per-window tensors: the batch mean answers "where does the
        # grid sit", but the reviewer's question is about the DISTRIBUTION of the
        # intermediate outputs, which the mean destroys.
        np.save(script_dir / f"s563_tau_trained_{DS}.npy", trained)
        np.save(script_dir / f"s563_tau_untrained_{DS}.npy", untrained)

        # Label each forecast day by its irradiance regime, so the grid can be
        # conditioned on the thing that makes the conditional distribution bimodal.
        ghi = raw["Ghi"].to_numpy()
        n_win = trained.shape[0]
        reg = []
        for w in range(n_win):
            a = first_target + w * H
            seg = ghi[a : a + H]
            day = seg[seg > 1e-6]
            mean_ghi = float(day.mean()) if day.size else 0.0
            cv = float(day.std() / day.mean()) if day.size and day.mean() > 0 else 0.0
            reg.append({"window": w, "mean_ghi": mean_ghi, "ghi_cv": cv})
        pd.DataFrame(reg).to_csv(script_dir / f"s563_regimes_{DS}.csv", index=False)

        rows = []
        for name, arr in (("untrained", untrained), ("trained", trained)):
            mean = arr.mean(axis=0)  # (N, H)
            for n in range(mean.shape[0]):
                for h in range(H):
                    rows.append({"state": name, "slot": n, "step": h, "hour": hours[h], "tau": float(mean[n, h])})
        out = script_dir / f"s563_learned_tau_{DS}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print("wrote", out, flush=True)
