r"""MLPGAM interpretability, computed on a fitted MLPGAM-CRC forecaster.

The proposed model's mean forecast is additive by construction (``MLPGAMNetwork.fuse_projection`` followed by the
identity ``mu_layer`` of ``AdditiveCRCDistribution``)::

    mu_h(x) = W_past^T g_past(x_past * eta_past) + W_fut^T g_fut(x_fut * eta_fut) + b_h

so every forecast splits exactly into a historical-branch term, a known-future-covariate term and a per-horizon offset,
which is the interpretability property of a GAM. This module measures three things a black-box MLP cannot provide in
this exact form:

* Branch decomposition: the two branch terms and the offset for every test window and lead time, in the target's own
  units (the target scaler is affine, so a branch term scales by the scaler's ``scale_`` and the offset absorbs the
  rest). ``decompose`` checks that the three parts sum to the model's forecast.
* Permutation importance: the increase in test MAE when one input feature is shuffled across windows, a model-agnostic
  reference that is independent of the gate.
* Gate-permutation agreement: the Spearman correlation between the post-sigmoid gate value of each feature and its
  permutation importance, a stronger faithfulness test than the gradient sensitivity of ``gate_faithfulness``.

:func:`make_interpretability_scorer` runs all of it as an ``ExperimentEngine`` ``custom_scorer``: fold-level scalars
are added as metric columns, and the per-timestamp branch terms are added to the predictions as ``contrib_past``,
``contrib_future`` and ``contrib_offset`` so that they are saved with ``predictions.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from twiga.core.data.processing import unstack_features
from twiga.core.utils import get_logger

from chapters.ch6_conformal.shared.gate_faithfulness import (
    extract_mlpgam_gate_importances,
    feature_names,
    gate_faithfulness,
    unwrap_network,
)

log = get_logger(__name__)

PERMUTATION_REPEATS = 5
PERMUTATION_SEED = 0
DECOMPOSITION_TOLERANCE = 1e-3  # relative to the target's scale


def _branch_terms(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(past, future, bias)`` terms of ``fuse_projection`` in scaled target space, each ``(B, H * num_targets)``."""
    past, fut = unstack_features(
        x,
        lookback_window_size=model.lookback_window_size,
        forecast_horizon=model.forecast_horizon,
        num_padding=model.num_past_features - model.num_covariate_features,
    )
    if model.use_gating:
        past = past * model.gate_compute_fn(model.past_feature_gate)[None, None, :]
        if model.gate_bias:
            past = past + model.past_gate_bias[None, None, :]
    past_term = model.past_feature_transform(model.encoder(past))
    future_term = torch.zeros_like(past_term)
    if model.num_covariate_features > 0:
        if model.use_gating:
            fut = fut * model.gate_compute_fn(model.future_feature_gate)[None, None, :]
            if model.gate_bias:
                fut = fut + model.future_gate_bias[None, None, :]
        future_term = model.future_feature_transform(model.horizon(fut))
    return past_term, future_term, model.bias.expand_as(past_term)


def _features(forecaster, test_df: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(forecaster.data_pipeline.transform_features(test_df), dtype=torch.float32)


def _to_target_units(forecaster, model: torch.nn.Module, scaled: torch.Tensor, as_scale: bool) -> np.ndarray:
    """``(B, H * K)`` scaled tensor to ``(B, H, K)`` in target units (``as_scale``: multiply by the scale only)."""
    arr = scaled.detach().cpu().numpy().reshape(scaled.shape[0], model.forecast_horizon, model.num_target_feature)
    return forecaster._inverse_scale_predictions(arr, is_scale=as_scale)


@torch.no_grad()
def decompose(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per (timestamp, target) branch terms in target units, and the largest reconstruction error relative to scale.

    Returns:
        ``(frame, max_rel_error)``: ``frame`` has ``timestamp``, ``target``, ``contrib_past``, ``contrib_future``,
        ``contrib_offset`` and ``mu`` (their sum); ``max_rel_error`` compares ``mu`` with the inverse-scaled output of
        the full network, so a value near zero confirms that the decomposition is exact.
    """
    model = unwrap_network(forecaster)
    model.eval()
    x = _features(forecaster, test_df).to(next(model.parameters()).device)
    past, future, bias = _branch_terms(model, x)
    total_scaled = model.output_activation(past + future + bias)
    mu = _to_target_units(forecaster, model, total_scaled, as_scale=False)
    past_u = _to_target_units(forecaster, model, past, as_scale=True)
    future_u = _to_target_units(forecaster, model, future, as_scale=True)
    offset_u = mu - past_u - future_u

    reference = _to_target_units(forecaster, model, model.fuse_projection(x), as_scale=False)
    scale = float(np.nanstd(reference)) or 1.0
    max_rel_error = float(np.nanmax(np.abs(reference - mu)) / scale)

    timestamps, _ = forecaster.data_pipeline.get_ground_truth_sequences(test_df)
    n, h, k = mu.shape
    ts = np.asarray(timestamps).reshape(n, h)  # int64 nanoseconds since the epoch, UTC
    targets = [str(t) for t in np.atleast_1d(forecaster.data_pipeline.target_feature)]
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(ts[:, :, None], k, axis=2).reshape(-1),
            "target": np.tile(targets, n * h),
            "contrib_past": past_u.reshape(-1),
            "contrib_future": future_u.reshape(-1),
            "contrib_offset": offset_u.reshape(-1),
            "mu": mu.reshape(-1),
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame, max_rel_error


@torch.no_grad()
def permutation_importance(
    forecaster, test_df: pd.DataFrame, repeats: int = PERMUTATION_REPEATS, seed: int = PERMUTATION_SEED
) -> pd.DataFrame:
    """Increase in test MAE (target units, and relative to the unpermuted MAE) when one feature is shuffled.

    A feature is shuffled across forecast windows, keeping each window's own temporal profile, in the branch it feeds
    (past features over the lookback window, covariates over the horizon), so the joint distribution of that feature
    with the target is broken while its marginal distribution is kept.
    """
    model = unwrap_network(forecaster)
    model.eval()
    device = next(model.parameters()).device
    x = _features(forecaster, test_df).to(device)
    _, y_scaled = forecaster.data_pipeline.get_ground_truth_sequences(test_df)
    y = forecaster._inverse_scale_predictions(np.asarray(y_scaled), is_scale=False)

    def mae(inputs: torch.Tensor) -> float:
        past, future, bias = _branch_terms(model, inputs)
        mu = _to_target_units(forecaster, model, model.output_activation(past + future + bias), as_scale=False)
        return float(np.nanmean(np.abs(mu - y)))

    base = mae(x)
    lookback, horizon = model.lookback_window_size, model.forecast_horizon
    names = feature_names(forecaster, model)
    rng = np.random.default_rng(seed)
    rows = []
    # Past features occupy the first `lookback` steps; covariates the last `horizon` steps (unstack_features layout).
    for branch, steps, count in (
        ("past", slice(0, lookback), model.num_past_features),
        ("future", slice(lookback, lookback + horizon), model.num_covariate_features),
    ):
        for j in range(count):
            increases = []
            for _ in range(repeats):
                permuted = x.clone()
                order = torch.as_tensor(rng.permutation(x.shape[0]), device=device)
                permuted[:, steps, j] = x[order][:, steps, j]
                increases.append(mae(permuted) - base)
            rows.append(
                {
                    "Branch": branch,
                    "Feature": j,
                    "name": names[branch][j],
                    "perm_mae_increase": float(np.mean(increases)),
                    "perm_rel_increase": float(np.mean(increases) / base) if base > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def interpretability(forecaster, test_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fold-level interpretability scalars and the per-timestamp branch decomposition."""
    model = unwrap_network(forecaster)
    decomposition, max_rel_error = decompose(forecaster, test_df)
    summary: dict = {"decomp_max_rel_error": max_rel_error}
    if max_rel_error > DECOMPOSITION_TOLERANCE:
        log.warning("branch decomposition is not exact (max relative error %.2e)", max_rel_error)

    past_abs, future_abs = decomposition["contrib_past"].abs(), decomposition["contrib_future"].abs()
    summary["branch_share_future"] = float(future_abs.sum() / (past_abs.sum() + future_abs.sum()))

    perm = permutation_importance(forecaster, test_df)
    for row in perm.itertuples(index=False):
        summary[f"perm__{row.Branch}__{row.name}"] = row.perm_rel_increase
    if getattr(model, "use_gating", False):
        gates = pd.DataFrame(extract_mlpgam_gate_importances(model))[["Branch", "Feature", "Importance"]]
        merged = perm.merge(gates, on=["Branch", "Feature"])
        if len(merged) >= 3:
            summary["gate_rho_perm"] = float(spearmanr(merged["Importance"], merged["perm_mae_increase"])[0])
        gate_summary, _ = gate_faithfulness(forecaster, test_df)
        summary.update(gate_summary)
    return summary, decomposition


def make_interpretability_scorer():
    """An ``ExperimentEngine`` ``custom_scorer``: interval evaluation plus interpretability diagnostics.

    Metric rows gain the fold-level scalars (repeated on every row, so fold averaging leaves them unchanged), including
    every column of ``gate_faithfulness``; prediction rows gain the branch terms of their timestamp.
    """

    def scorer(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_df, metric_df = forecaster.evaluate_interval_forecast(test_df=test_df)
        try:
            summary, decomposition = interpretability(forecaster, test_df)
        except Exception:  # a failed diagnostic must not discard the fold's forecast metrics
            log.exception("interpretability failed; forecast metrics kept without interpretability columns")
            return pred_df, metric_df
        # A fitted forecaster carries the lookback history that lets it forecast the first test day; one that only
        # loads a checkpoint (``reuse_checkpoint_from``) does not, and neither does ``transform_features(test_df)``.
        # Keeping only the days the decomposition covers makes this condition score exactly the days of the
        # conditions that reuse its checkpoint.
        covered = set(decomposition["timestamp"])
        keys = ["timestamp", "target"] if "target" in pred_df.columns else ["timestamp"]
        pred_df = pred_df.assign(timestamp=pd.to_datetime(pred_df["timestamp"], utc=True))
        pred_df = pred_df[pred_df["timestamp"].isin(covered)].merge(
            decomposition.drop(columns="mu").drop_duplicates(keys), on=keys, how="left"
        )
        metric_df = metric_df[pd.to_datetime(metric_df["timestamp"], utc=True).isin(covered)]
        return pred_df, metric_df.assign(**{k: np.float64(v) for k, v in summary.items()})

    return scorer
