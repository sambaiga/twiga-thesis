r"""MLPGAM feature-gate faithfulness, computed on a fitted forecaster.

Shared by ``experiment_1_architectural/mlpgam_gate_analysis.py`` (one model) and by the architectural study's gating
conditions, where :func:`make_gate_faithfulness_scorer` runs as an ``ExperimentEngine`` ``custom_scorer`` so every
fold of every gated condition reports gate statistics next to its forecast metrics, at the full training budget.

Two faithfulness measures:

* ``rho_faith``: correlation between each feature's post-sigmoid gate value and its gradient-based local
  sensitivity, ``||d(latent)/d(feature)||`` (the correlation check of ``twiga/models/nn/net/ganf/importance.py``,
  reimplemented for MLPGAM's flat past/future gates).
* Noise suppression: when the data carry injected ``noise_*`` features (``PipelineConfig.noise_features``), which are
  irrelevant by construction, a faithful gate gives them lower values than the real features. ``gate_noise_auc`` is the
  probability that a real feature's gate exceeds a noise feature's gate (0.5 = no separation, 1 = perfect).

Gate index ``i`` of the past gate is ``data_pipeline.feature_columns[i]`` and of the future gate
``data_pipeline.covariate_columns[i]``: the pipeline windows features in exactly those column orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from twiga.core.data.processing import unstack_features
from twiga.core.utils import get_logger

log = get_logger(__name__)

SAMPLE_SIZE = 256
NOISE_PREFIX = "noise_"


def unwrap_network(forecaster) -> torch.nn.Module:
    """The MLPGAM network inside a fitted forecaster, wherever its wrapper nests it.

    A point model nests it as ``forecaster.model.model``; a probabilistic model (MLPGAM-CRC and the other heads) as
    ``forecaster.model.model.backbone``. The network is the first submodule that carries both
    ``lookback_window_size`` and ``use_gating`` (the gate parameters exist only when ``use_gating`` is true).
    """
    node = getattr(forecaster, "model", forecaster)
    while not isinstance(node, torch.nn.Module) and hasattr(node, "model"):
        node = node.model
    if isinstance(node, torch.nn.Module):
        for module in node.modules():
            if hasattr(module, "lookback_window_size") and hasattr(module, "use_gating"):
                return module
    raise AttributeError("no submodule with lookback_window_size and use_gating found in the forecaster's model")


def sample_batch(forecaster, test_df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> torch.Tensor:
    """A real, scaled, windowed batch of model inputs from the forecaster's own fitted data pipeline."""
    features = forecaster.data_pipeline.transform_features(test_df)  # (N, lookback + horizon, num_features)
    return torch.as_tensor(features[: min(sample_size, features.shape[0])], dtype=torch.float32)


def feature_names(forecaster, model: torch.nn.Module) -> dict[str, list[str]]:
    """Gate-index-ordered feature names per branch, falling back to indices if the counts disagree."""
    pipeline = forecaster.data_pipeline
    names = {"past": list(pipeline.feature_columns), "future": list(pipeline.covariate_columns)}
    counts = {"past": model.num_past_features, "future": model.num_covariate_features}
    for branch, expected in counts.items():
        if len(names[branch]) != expected:
            log.warning(
                "%s gate has %d entries but %d named features; using indices", branch, expected, len(names[branch])
            )
            names[branch] = [str(i) for i in range(expected)]
    return names


def extract_mlpgam_gate_importances(model: torch.nn.Module) -> list[dict]:
    """Post-sigmoid gate value per feature, for each of MLPGAM's flat gates."""
    records: list[dict] = []
    for branch, attr in (("past", "past_feature_gate"), ("future", "future_feature_gate")):
        if getattr(model, attr, None) is None:
            continue
        values = model.gate_compute_fn(getattr(model, attr)).detach().cpu().numpy().flatten()
        records.extend(
            {"Group": "MLPGAM", "Branch": branch, "Feature": i, "Importance": float(v)} for i, v in enumerate(values)
        )
    return records


def compute_mlpgam_feature_sensitivity(model: torch.nn.Module, batch: torch.Tensor) -> list[dict]:
    """Gradient-based local sensitivity of the gated latent with respect to each raw input feature.

    Mirrors ``ganf/importance.py``'s ``compute_feature_sensitivity``: unit gradients from the sum of the post-gate,
    pre-encoder tensor back to the (ungated) split input, L2-normed per feature, averaged over batch and time.
    """
    was_training = model.training
    model.train()  # gate behaviour as during training, matching the GANF reference
    x = batch.detach().clone().requires_grad_(True)
    past_feature, covariate_feature = unstack_features(
        x,
        lookback_window_size=model.lookback_window_size,
        forecast_horizon=model.forecast_horizon,
        num_padding=model.num_past_features - model.num_covariate_features,
    )
    records: list[dict] = []
    if getattr(model, "past_feature_gate", None) is not None:
        gated_past = past_feature * model.gate_compute_fn(model.past_feature_gate)[None, None, :]
        (grad,) = torch.autograd.grad(
            model.encoder(gated_past).sum(), past_feature, retain_graph=True, allow_unused=True
        )
        if grad is not None:
            per_feature = torch.sqrt((grad**2).mean(dim=(0, 1)))
            records.extend(
                {"Group": "MLPGAM", "Branch": "past", "Feature": i, "Sensitivity": v.item()}
                for i, v in enumerate(per_feature)
            )
    if model.num_covariate_features > 0 and getattr(model, "future_feature_gate", None) is not None:
        gated_future = covariate_feature * model.gate_compute_fn(model.future_feature_gate)[None, None, :]
        (grad,) = torch.autograd.grad(model.horizon(gated_future).sum(), covariate_feature, allow_unused=True)
        if grad is not None:
            per_feature = torch.sqrt((grad**2).mean(dim=(0, 1)))
            records.extend(
                {"Group": "MLPGAM", "Branch": "future", "Feature": i, "Sensitivity": v.item()}
                for i, v in enumerate(per_feature)
            )
    model.train(was_training)
    return records


def compute_gate_sensitivity_correlation(
    gate_records: list[dict], sens_records: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``rho_faith``: Pearson/Spearman correlation of gate value against sensitivity, overall and per branch."""
    df_gate = pd.DataFrame(gate_records)[["Group", "Branch", "Feature", "Importance"]]
    df_sens = pd.DataFrame(sens_records)[["Group", "Branch", "Feature", "Sensitivity"]]
    merged = df_gate.merge(df_sens, on=["Group", "Branch", "Feature"], how="inner")
    rows = []
    for scope, sub in [("Overall", merged), *merged.groupby("Branch")]:
        if len(sub) >= 3:
            p_r, p_p = pearsonr(sub["Importance"], sub["Sensitivity"])
            s_r, s_p = spearmanr(sub["Importance"], sub["Sensitivity"])
            rows.append(
                {
                    "Scope": scope,
                    "Pearson_r": p_r,
                    "Pearson_p": p_p,
                    "Spearman_rho": s_r,
                    "Spearman_p": s_p,
                    "n": len(sub),
                }
            )
    return pd.DataFrame(rows), merged


def noise_separation_auc(merged: pd.DataFrame) -> float:
    """P(gate of a real feature > gate of a noise feature), ties counted as half; NaN without noise features."""
    noise = merged.loc[merged["is_noise"], "Importance"].to_numpy()
    signal = merged.loc[~merged["is_noise"], "Importance"].to_numpy()
    if noise.size == 0 or signal.size == 0:
        return float("nan")
    diff = signal[:, None] - noise[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def gate_faithfulness(forecaster, test_df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> tuple[dict, pd.DataFrame]:
    """Fold-level gate statistics and the per-feature table for a fitted MLPGAM forecaster.

    Returns:
        ``(summary, merged)``: ``summary`` maps column names to scalars (empty if the gate is disabled);
        ``merged`` has one row per (branch, feature) with gate value, sensitivity and an ``is_noise`` flag.
    """
    model = unwrap_network(forecaster)
    if not getattr(model, "use_gating", False):
        return {}, pd.DataFrame()
    gates = extract_mlpgam_gate_importances(model)
    sens = compute_mlpgam_feature_sensitivity(model, sample_batch(forecaster, test_df, sample_size))
    rho, merged = compute_gate_sensitivity_correlation(gates, sens)
    names = feature_names(forecaster, model)
    merged["name"] = [names[b][f] for b, f in zip(merged["Branch"], merged["Feature"], strict=True)]
    merged["is_noise"] = merged["name"].str.startswith(NOISE_PREFIX)

    overall = rho[rho["Scope"] == "Overall"]
    summary = {
        "gate_rho_spearman": float(overall["Spearman_rho"].iloc[0]) if not overall.empty else float("nan"),
        "gate_rho_pearson": float(overall["Pearson_r"].iloc[0]) if not overall.empty else float("nan"),
        "gate_mean": float(merged["Importance"].mean()),
        "gate_noise_mean": float(merged.loc[merged["is_noise"], "Importance"].mean()),
        "gate_signal_mean": float(merged.loc[~merged["is_noise"], "Importance"].mean()),
        "gate_noise_auc": noise_separation_auc(merged),
    }
    for row in merged.itertuples(index=False):
        summary[f"gate__{row.Branch}__{row.name}"] = row.Importance
        summary[f"sens__{row.Branch}__{row.name}"] = row.Sensitivity
    return summary, merged


def make_gate_faithfulness_scorer(sample_size: int = SAMPLE_SIZE):
    """An ``ExperimentEngine`` ``custom_scorer``: the built-in interval evaluation plus fold-level gate statistics.

    The forecast metrics are exactly those of ``metric_types=["interval"]`` (daily point and interval metrics); the gate
    columns are fold-level scalars repeated on every row of the fold, so fold-level averaging leaves them unchanged.
    """

    def scorer(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pred_df, metric_df = forecaster.evaluate_interval_forecast(test_df=test_df)
        try:
            summary, _ = gate_faithfulness(forecaster, test_df, sample_size)
        except Exception:  # a failed diagnostic must not discard the fold's forecast metrics
            log.exception("gate faithfulness failed; forecast metrics kept without gate columns")
            summary = {}
        return pred_df, metric_df.assign(**{k: np.float64(v) for k, v in summary.items()})

    return scorer
