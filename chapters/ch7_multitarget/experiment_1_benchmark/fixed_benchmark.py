"""Chapter 6, Experiment 1 at fixed parameters: FPQR vs QR, BNN and Laplace, no tuning.

The same backtesting protocol and conformal conditions as
``multitarget_experiment.py``, but every head sits on one fixed MLPF backbone
and FPQR keeps the hand-set configuration ``shared/forecast_examples.py``
evaluated, rather than an HPO result. HPO is switched off at the engine level
(``ExperimentSpec.hpo_enabled=False``), so no cached ``best_params`` file can
leak a tuned value into a field this script does not set.

The question is whether the head ranking ``forecast_examples.py`` found on a
single train/test split holds up under proper cross-validation, with every
head given the same backbone, the same training budget and the same nominal
90% interval.

Conditions
----------
benchmark (every dataset)
  fpqr          FPQR, shared quantile grid, learned (floating) outer levels --
                the exact ``forecast_examples.py`` configuration
  fpqr_anchor   the same, with the outer levels pinned to conf_level/2 and
                1 - conf_level/2 (``proposal_anchor_outer=True``); without the
                pin conf_level is metadata only for FPQR, and its native
                interval is the outermost learned pair at whatever span it
                learns, not 90%
  fpqr_ch5      Chapter 5's MLPQF benchmark head exactly as trained there
                (run 20260917_194347, MLVS-PT benchmark/mlpqf hparams.yaml),
                including its own 256/256/2 MLPF backbone: N=90, TauRoPE
                (num_cosines=0, rope_type="linear"), standard QVN,
                entropy_weight=1, learned outer levels. Unlike the other
                heads it does not share FIXED_BACKBONE, since without that
                backbone it would not be Chapter 5's model. Chapter 5 used a
                96-step lookback; here every head shares --lookback
  qr            fixed-grid quantile regression on its default 0.1..0.9 grid plus
                the alpha/2 and 1 - alpha/2 outer levels (11 levels), as in
                multitarget_experiment.py. QR uses its list exactly as given,
                so without those two its native interval would be 80%
  bnn           Bayesian neural network
  laplace       parametric Laplace

conformal (MLVS-PT-3PHASE only)
  max-norm multivariate conformal on every head, plus the six CQR variants
  (scaled/unscaled x marginal/pooled/bonferroni) on the three quantile heads.
  Each reuses its head's trained weights. Mahalanobis is excluded: it lost to
  max-norm on every axis in an earlier run (see multitarget_experiment.py).

head_tuned_conformal (MLVS-PT-3PHASE only)
  the head-tuned FPQR, which replaces the fixed FPQR in the chapter, refitted
  from its cached tuned settings with no new search, plus the same conformal
  variants reusing the refit's weights. A head_tuned run keeps no checkpoints,
  so its own weights cannot be calibrated afterwards.

Metrics
-------
Every condition reports CRPS alongside Winkler, PICP, NMPI and CWE. Conformal
calibration changes only the interval bounds, never the predictive quantiles
or loc/scale, so a conformal condition's CRPS equals its head's: it is
reported so every row of the table is complete, not as a separate result.
Brier score, the energy score and two-sided stratified CWE are computed
afterwards from the saved predictions by ``fixed_benchmark_analysis.py``.

Protocol
--------
DSS-EU, DSS-UK (weak feeder) and DSS-US (rank-8 transformer), all at 0.4
EV / 0.4 PV: about eleven months of power-flow data each, so the same short
protocol -- every fold fits on 4 months and validates on the next 1, then tests
on 1, rolling, no calibration window, hence no conformal group. The training
window is held fixed (cv_train_size_mode="fixed") and cv_folds only caps the
count, giving 5 folds on DSS-EU and 6 on DSS-UK and DSS-US (their records hold
one more month). Under the default "derived" mode cv_folds would win instead
and each fold would fit on about 30 days.

MLVS-PT-3PHASE: 12-month expanding train, 2-month validation, 3-month test,
3-month calibration, 10 folds; benchmark and conformal groups.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py
    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --quick
    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py \
        --dataset MLVS-PT-3PHASE --group head_tuned_conformal
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import laplace, norm
from twiga.core.config import ConformalConfig
from twiga.core.utils import configure, get_logger
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.experiment.hpo import backbone_params_path, model_name
from twiga.models.nn import MLPFConfig
from twiga.models.nn.mlpfbnn_model import MLPFBNNConfig
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig
from twiga.models.nn.mlpfqr_model import MLPFQRConfig
from twiga.utils.data_loading import load_dataset
from twiga.utils.run_forecast import PipelineConfig

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[3] / "experiments"))) / "ch7_multitarget"
)

ALPHA = 0.1

# The backbone forecast_examples.py ran with (its BACKBONE_FALLBACK, since no
# tune_backbone.py result exists) and its training settings, shared by every
# head so a difference between conditions is attributable to the head alone.
FIXED_BACKBONE: dict = {
    "latent_size": 64,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "optimizer_type": "adamw",
    "optimizer_params": {"lr": 1e-3, "weight_decay": 1e-6},
    "lr_scheduler_type": "multi_step",
    "batch_size": 64,
    "gradient_clip_val": 1.0,
}

# forecast_examples.py's FPQR settings, with conf_level moved to the 90%
# interval every other head is scored at.
FIXED_FPQR: dict = {
    "per_target_tau": False,
    "n_quantiles": 21,
    "entropy_weight": 100.0,
    "num_cosines": 0,
    "rope_type": "tau_pos",
    "qvn_type": "monotone",
    "proposal_type": "softmax",
    "conf_level": ALPHA,
}


# QR's default 0.1..0.9 grid plus the alpha/2 and 1 - alpha/2 outer levels, so
# its native interval is the 90% one every other head is scored at.
QR_GRID: list[float] = [ALPHA / 2, *np.arange(0.1, 1.0, 0.1).round(1).tolist(), 1 - ALPHA / 2]

_QUANTILE_METRICS = ["point", "interval", "quantile"]
_PARAMETRIC_METRICS = ["point", "interval", "parametric"]

# ── head_tuned group: each head's own settings tuned, everything else pinned ──
# Every one of MLPF's 13 backbone search fields at the value the fixed heads
# train with (FIXED_BACKBONE where it sets one, the class default otherwise),
# plus the training budget, so a tuned head differs from its fixed namesake in
# its head settings alone. Pinned through hpo_fixed_params, which removes a
# field from the Optuna search and re-applies it when the model is built.
BACKBONE_PINS: dict = {
    **FIXED_BACKBONE,
    "width_multiplier": 2.0,
    "embedding_size": 28,
    "value_embed_type": None,
    "embedding_type": None,
    "alpha": 0.1,
    "use_norm": True,
    "use_residual": False,
    "activation_function": "SiLU",
    "combination_type": "addition-comb",
    "use_bias": True,
    "out_activation_function": "Identity",
    "scheduler_params": {"milestones": [], "prob_decay_1": 0.5, "prob_decay_2": 0.9, "gamma": 0.1},
    # The search loads data the way the benchmark folds do (fixed_overrides):
    # with worker processes the loader drops the ragged last batch, which on
    # the short HPO windows left one or no batches per epoch.
    "num_workers": 0,
}
# Same objective for every tuned head: CRPS is proper, computed for quantile
# and parametric heads alike, and rewards calibration as well as sharpness.
# Scaled per target by its standard deviation on the tuning data: averaged in
# raw units, P's CRPS (tens of kW) swamps V's (thousandths of a per unit) on
# the simulated records, and the first head-tuned run in effect tuned for P
# alone (FPQR's CRPS rose 34-108% on DSS-EU and DSS-US).
HEAD_TUNING_OBJECTIVE = "crps_scaled"
# The pruner watches the same score per epoch (val_crps, which every tuned head
# now logs), not val_mae (the training monitor) nor val_loss: the loss carries
# the tuned regulariser weights (entropy_weight, crossing_penalty, kl_weight),
# so it would rank trials by regularisation strength. BNN's val_nll was
# rejected too: it scores one weight draw at a time and sat 0.3-5.7 nats above
# the mixture NLL on the fixed runs, a gap that moves with kl_weight.
HEAD_TUNING_PRUNE_MONITOR = "crps"
HEAD_TUNING_TRIALS = 50
# FPQR searches 6 head settings on every trial and 2-5 more conditionally
# (QR 2-3, BNN 3), and TPE spends its first 20 trials sampling at random, so
# FPQR gets twice the budget. The budgets are therefore not equal: report the
# tuned FPQR with that stated, or pass --hpo-trials to equalise them.
HEAD_TUNING_TRIALS_FPQR = 100
# Epochs per HPO trial. The multi_step schedule drops the learning rate
# tenfold at 50% and 90% of a run's own max_epochs, so a trial runs the whole
# schedule compressed to its length rather than the first N epochs of the
# final 60-epoch run (drops at 30 and 54). 50 epochs (drops at 25 and 45)
# keeps 83% of the final run's time at each learning rate, against 67% at 40;
# 60 would match it exactly at 1.2x the cost. The fixed runs' best epochs
# (162 folds) have median 19, BNN's 28-50, which 50 epochs also covers.
HEAD_TUNING_EPOCHS = 50
# Stop a head's search once this many completed trials in a row fail to beat
# its best CRPS by 0.1%, after at least 20 completions; the trial budget above
# stays the ceiling. Stands in for optuna.terminator, deprecated in Optuna 4.9.
HEAD_TUNING_STOP_PATIENCE = 20
# Validation windows per trial. 1 scores each trial on the single HPO split
# with per-epoch Hyperband pruning. K > 1 cuts the validation span into K
# chunks (expanding origin), averages CRPS over them, and prunes with a
# Wilcoxon signed-rank test across windows instead. Each trial then trains K
# times and the test cannot reach p < 0.1 before 4 windows, so it only saves
# compute from K = 5 on; on the short protocol's one-month span that leaves
# chunks of about 6 days. Off by default for cost; pass --hpo-windows.
HEAD_TUNING_WINDOWS = 1
# Optional CWE-aware choice of the final trial: among trials within 1% of the
# best scaled CRPS, keep the one with the highest validation stratified CWE.
# CRPS stays the objective (CWE is not a proper score and is not optimised);
# it only breaks near-ties. Quantile heads only; BNN keeps the usual choice.
HEAD_TUNING_SELECT_ON_CWE = False

_EVALUATORS = {
    "point": "evaluate_point_forecast",
    "interval": "evaluate_interval_forecast",
    "quantile": "evaluate_quantile_forecast",
    "parametric": "evaluate_parametric_forecast",
}

# Sampled paths per forecast window for the joint scores, as forecast_examples.py's --paths 50.
N_PATHS = 50
SEED = 20260904
# Quantiles saved for the forecast-example figures' fan.
FAN_LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def _per_cell_levels(raw) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles and their levels, both (B, N, H, D), sorted along the level axis.

    FPQR reports its learned levels per window and step, (B, N, H), or per
    target too, (B, N, H, D); QR one fixed grid, (N,). Each window's quantile
    values belong to that window's own levels, so they are kept per cell rather
    than averaged into one grid.
    """
    q = np.asarray(raw.quantiles, dtype=float)
    lv = np.asarray(raw.quantile_levels, dtype=float)
    if lv.ndim == 1:
        lv = lv[None, :, None, None]
    elif lv.ndim == 2:
        lv = lv[None, :, None, :]
    elif lv.ndim == 3:
        lv = lv[..., None]
    lv = np.broadcast_to(lv, q.shape)
    order = np.argsort(lv, axis=1)
    return np.take_along_axis(q, order, axis=1), np.take_along_axis(lv, order, axis=1)


def _inverse_cdf(raw, family: str | None, u: np.ndarray) -> np.ndarray:
    """Predictive quantiles at probabilities ``u`` of shape (K, B, H, D), in original units.

    Quantile heads: linear interpolation on each cell's own levels, held flat
    beyond the outermost pair (as np.interp does). Parametric heads: the
    family's quantile function, Laplace on its dispersion b, normal on its sd.
    """
    if raw.quantiles is not None:
        q, lv = _per_cell_levels(raw)
        upper = (lv[None] < u[:, :, None]).sum(axis=2)
        hi = np.clip(upper, 0, q.shape[1] - 1)[:, :, None]
        lo = np.clip(upper - 1, 0, q.shape[1] - 1)[:, :, None]
        full = (u.shape[0], *q.shape)
        q_lo, q_hi = (np.take_along_axis(np.broadcast_to(q[None], full), i, axis=2)[:, :, 0] for i in (lo, hi))
        l_lo, l_hi = (np.take_along_axis(np.broadcast_to(lv[None], full), i, axis=2)[:, :, 0] for i in (lo, hi))
        span = l_hi - l_lo
        weight = np.divide(u - l_lo, span, out=np.zeros_like(u), where=span > 0)
        return q_lo + weight * (q_hi - q_lo)
    loc, scale = np.asarray(raw.loc, dtype=float), np.asarray(raw.scale, dtype=float)
    ppf = laplace.ppf if family == "laplace" else norm.ppf
    return loc[None] + ppf(u) * scale[None]


def _sample_paths(raw, family: str | None, rng: np.random.Generator) -> np.ndarray:
    """Draw (N_PATHS, B, H, D) paths with independent margins, by inverse-CDF sampling."""
    loc = np.asarray(raw.loc)
    return _inverse_cdf(raw, family, rng.uniform(size=(N_PATHS, *loc.shape)))


def _fan_rows(raw, family: str | None, stamps, observed: np.ndarray, targets: list[str]) -> pd.DataFrame:
    """The display quantiles for every window, step and target, for the forecast-example figures.

    The engine saves only the 90% interval, and the figures draw a fan, so the
    fan is saved here, one row per (window, step, target).
    """
    loc = np.asarray(raw.loc)
    fan = _inverse_cdf(raw, family, np.broadcast_to(FAN_LEVELS[:, None, None, None], (len(FAN_LEVELS), *loc.shape)))
    n_b, n_h, n_d = loc.shape
    # Timestamps come back as (B, H) or (B, H, D) depending on the pipeline;
    # every target in a step shares one, so broadcast it across targets.
    stamps = np.broadcast_to(np.asarray(stamps).reshape(n_b, n_h, -1)[..., :1], loc.shape)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps.reshape(-1), unit="ns", utc=True),
            "forecast_origin": pd.to_datetime(np.repeat(stamps[:, 0, 0], n_h * n_d), unit="ns", utc=True),
            "target": np.tile(targets, n_b * n_h),
            "Actual": observed.reshape(-1),
        }
    )
    for k, level in enumerate(FAN_LEVELS):
        frame[f"q{level:.2f}"] = fan[k].reshape(-1)
    return frame.assign(metric_type="fan")


def _crps_per_window(raw, family: str | None, observed: np.ndarray) -> np.ndarray:
    """Exact CRPS per window and target, (B, D), averaged over the horizon.

    Quantile heads: twice the trapezoid integral of the pinball loss over each
    cell's own levels (crps_quantile's construction). BNN: the normal closed
    form, sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)). Laplace: b (|z| + exp(-|z|) - 3/4), z = (y - mu) / b.
    """
    if raw.quantiles is not None:
        q, lv = _per_cell_levels(raw)
        err = observed[:, None] - q
        pinball = np.maximum(lv * err, (lv - 1.0) * err)
        crps = 2.0 * np.trapezoid(pinball, x=lv, axis=1)
    else:
        loc, scale = np.asarray(raw.loc, dtype=float), np.asarray(raw.scale, dtype=float)
        if family == "laplace":
            z = np.abs(observed - loc) / scale
            crps = scale * (z + np.exp(-z) - 0.75)
        else:
            z = (observed - loc) / scale
            crps = scale * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return np.asarray(crps).mean(axis=1)


def _energy_score_per_window(observed: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """Energy score across targets, in raw units, averaged over the horizon: one value per window.

    The pairwise spread term is accumulated one path at a time, so memory stays
    at a single (N_PATHS, B, H, D) array rather than its square.
    """
    to_obs = np.linalg.norm(paths - observed[None], axis=-1).mean(axis=0)
    spread = np.zeros(observed.shape[:-1])
    for path in paths:
        spread += np.linalg.norm(paths - path[None], axis=-1).mean(axis=0)
    return (to_obs - 0.5 * spread / len(paths)).mean(axis=1)


def make_scorer(metric_types: list[str], family: str | None, rated: float, power_targets: tuple[str, str], joint: bool):
    """Per-fold scorer: the built-in evaluators, plus per-window scores for a benchmark head.

    The per-window scores need the full predictive distribution, which the
    engine never persists, so they are computed here while the fold's model is
    live. One row per window, not per fold, so the significance tests have one
    loss per forecast day rather than eight to ten. Conformal conditions get
    only the built-in evaluators: calibration moves the interval bounds and
    nothing else, so their CRPS and joint scores would repeat the head's.

    Rows with ``metric_type="window"`` carry, per target, ``crps``; and, on the
    ``target="all"`` row, the energy score and the thermal event's per-window
    Brier score, predicted probability and realised rate. Brier skill needs the
    pooled base rate, so the analysis step computes it, not each fold.
    """

    def scorer(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        preds, metrics = [], []
        for mtype in metric_types:
            pred_df, metric_df = getattr(forecaster, _EVALUATORS[mtype])(test_df=test_df)
            pred_df["metric_type"] = metric_df["metric_type"] = mtype
            preds.append(pred_df)
            metrics.append(metric_df)
        if joint:
            prepared = forecaster.prepare_test_data(test_df)
            stamps, observed = forecaster._extract_ground_truth(prepared)
            observed = np.asarray(observed, dtype=float)
            prediction, _ = forecaster._predict_core(test_df=prepared, prepare_test_data=False)
            raw = next(iter(prediction.values()))
            targets = list(np.atleast_1d(forecaster.data_pipeline.target_feature))
            p_idx, q_idx = (targets.index(name) for name in power_targets)
            origins = pd.to_datetime(np.asarray(stamps)[:, 0, 0], unit="ns", utc=True)

            crps = _crps_per_window(raw, family, observed)
            rows = [
                {"forecast_origin": origin, "target": target, "crps": float(crps[b, d])}
                for b, origin in enumerate(origins)
                for d, target in enumerate(targets)
            ]
            paths = _sample_paths(raw, family, np.random.default_rng(SEED))
            energy = _energy_score_per_window(observed, paths)
            event = (np.hypot(observed[..., p_idx], observed[..., q_idx]) > rated).astype(float)
            prob = (np.hypot(paths[..., p_idx], paths[..., q_idx]) > rated).mean(axis=0)
            brier = ((prob - event) ** 2).mean(axis=1)
            rows += [
                {
                    "forecast_origin": origin,
                    "target": "all",
                    "energy_score": float(energy[b]),
                    "brier": float(brier[b]),
                    "event_rate_predicted": float(prob[b].mean()),
                    "event_rate_realised": float(event[b].mean()),
                }
                for b, origin in enumerate(origins)
            ]
            metrics.append(pd.DataFrame(rows).assign(metric_type="window"))
            preds.append(_fan_rows(raw, family, stamps, observed, targets))
        return pd.concat(preds, ignore_index=True), pd.concat(metrics, ignore_index=True)

    return scorer


# Chapter 5's MLPQF benchmark head (fpqr_experiment.py's "mlpqf" condition) as
# actually trained, read from its saved hparams.yaml: the condition's own
# overrides (N=90, conf_level, TauRoPE), the tuned backbone fields that
# survived its controlled_fields, and the class defaults controlled_fields
# restored (latent/hidden 256, 2 layers, dropout 0.25, lr 1e-3, wd 1e-6).
# Fields absent from hparams.yaml (batch_size, gradient_clip_val,
# proposal_eps) were class defaults there and are left to the class here.
CH5_MLPQF: dict = {
    "latent_size": 256,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.25,
    "width_multiplier": 3.0,
    "alpha": 0.30413196680293153,
    "embedding_size": 28,
    "embedding_type": "PosEmb",
    "value_embed_type": "ConvEmb",
    "combination_type": "addition-comb",
    "activation_function": "ELU",
    "use_norm": False,
    "use_residual": True,
    "use_bias": True,
    "optimizer_type": "adamw",
    "optimizer_params": {"lr": 1e-3, "weight_decay": 1e-6},
    "lr_scheduler_type": "multi_step",
    "scheduler_params": {"gamma": 0.1, "milestones": [], "prob_decay_1": 0.5, "prob_decay_2": 0.9},
    "n_quantiles": 90,
    "conf_level": ALPHA,
    "num_cosines": 0,
    "rope_type": "linear",
    "qvn_type": "standard",
    "entropy_weight": 1.0,
    "proposal_type": "softmax",
    "proposal_anchor_outer": False,
    "per_target_tau": False,
    "loss_fn": "pinball",
    "kappa": 0.25,
    "crossing_penalty": 10.0,
    "tau_embed_type": "TimeEmb",
    "proposal_loss_fn": "wasserstein",
    "lambda_cal": 0.0,
}

HEADS: dict[str, dict] = {
    "fpqr": {
        "model_cls": MLPFFPQRConfig,
        "overrides": {**FIXED_BACKBONE, **FIXED_FPQR, "proposal_anchor_outer": False},
        "metric_types": _QUANTILE_METRICS,
        "is_quantile_head": True,
        "family": None,
        "description": "FPQR, shared grid, learned outer levels (forecast_examples.py configuration)",
    },
    "fpqr_anchor": {
        "model_cls": MLPFFPQRConfig,
        "overrides": {**FIXED_BACKBONE, **FIXED_FPQR, "proposal_anchor_outer": True},
        "metric_types": _QUANTILE_METRICS,
        "is_quantile_head": True,
        "family": None,
        "description": "FPQR, shared grid, outer levels pinned to the 90% interval boundary",
    },
    "fpqr_ch5": {
        "model_cls": MLPFFPQRConfig,
        "overrides": dict(CH5_MLPQF),
        "metric_types": _QUANTILE_METRICS,
        "is_quantile_head": True,
        "family": None,
        "description": "Chapter 5's MLPQF benchmark head and backbone, as trained there",
    },
    "qr": {
        "model_cls": MLPFQRConfig,
        "overrides": {**FIXED_BACKBONE, "quantiles": QR_GRID, "conf_level": ALPHA},
        "metric_types": _QUANTILE_METRICS,
        "is_quantile_head": True,
        "family": None,
        "description": "Quantile regression on its default grid plus the 90% outer levels",
    },
    "bnn": {
        "model_cls": MLPFBNNConfig,
        "overrides": dict(FIXED_BACKBONE),
        "metric_types": _PARAMETRIC_METRICS,
        "is_quantile_head": False,
        "family": "normal",
        "description": "Bayesian neural network",
    },
    "laplace": {
        "model_cls": MLPFLaplaceConfig,
        "overrides": dict(FIXED_BACKBONE),
        "metric_types": _PARAMETRIC_METRICS,
        "is_quantile_head": False,
        "family": "laplace",
        "description": "Parametric Laplace",
    },
}

# multitarget_experiment.py's conformal set minus Mahalanobis.
MULTIVARIATE_CONFORMAL: dict[str, ConformalConfig] = {
    "max_norm": ConformalConfig(method="multivariate", score_type="max-norm", alpha=ALPHA),
}
QUANTILE_CONFORMAL: dict[str, ConformalConfig] = {
    f"quantile_{score}_{coverage}": ConformalConfig(
        method="quantile", score_type=score, alpha=ALPHA, target_coverage=coverage
    )
    for score in ("scaled", "unscaled")
    for coverage in ("marginal", "pooled", "bonferroni")
}

# Head-tuned heads refitted with their conformal variants (group head_tuned_conformal).
# The tuned FPQR replaces the fixed one in the chapter's benchmark, so its calibrated
# variants are needed beside the fixed QR, BNN and Laplace ones.
TUNED_CONFORMAL_HEADS = ("fpqr_anchor",)
# Heads whose raw forecasts the raw_export group saves.
RAW_EXPORT_HEADS = ("fpqr_anchor", "qr", "bnn", "laplace")


# Per head: what stays pinned beyond the backbone, and any change to the
# class's own search space for this search only. Laplace has no head settings
# and is not tuned.
HEAD_TUNING: dict[str, dict] = {
    "fpqr_anchor": {
        # The interval definition: anchored outer levels at the 90% boundary
        # (anchoring needs the softmax proposal) on one shared grid.
        # Plus the fixed head's TauRoPE path (num_cosines 0, rope_type tau_pos)
        # and every other structural choice at the value the fixed head trains
        # with. The QVN is searched over the three non-crossing-aware variants
        # only (below). Left open, the first head-tuned run spent its ~30
        # completed trials across seven QVN types and picked the lattice QVN on
        # DSS-EU and DSS-US, which underfits (Chapter 5) and lost to the fixed
        # head by 34-108% CRPS on test.
        "pins": {
            "conf_level": ALPHA,
            "proposal_anchor_outer": True,
            "proposal_type": "softmax",
            "per_target_tau": False,
            "num_cosines": FIXED_FPQR["num_cosines"],
            "rope_type": FIXED_FPQR["rope_type"],
            "rope_base": None,
            "loss_fn": "pinball",
            "proposal_shared_horizon": False,
            "proposal_loss_fn": "wasserstein",
            "proposal_stop_gradient": True,
            "gradient_clip_val": 1.0,
        },
        # What stays free: the QVN among standard (crossing penalty, sort,
        # crossing mode), monotone (softplus gate) and monotone_cal (gate plus
        # calibration loss weight and temperature), each with its own
        # conditional settings; n_quantiles (class range 10-100); entropy_weight
        # on a log scale from 0.1 to 100 (the fixed head uses 100, beyond the
        # class range); and proposal_eps (class range).
        "search": {"entropy_weight": (0.1, 100.0), "qvn_type": ["standard", "monotone", "monotone_cal"]},
        # The fixed head is the first trial, so the search keeps it unless a
        # trial beats it on the validation windows.
        "initial": {
            "mlpffpqr_n_quantiles": FIXED_FPQR["n_quantiles"],
            "mlpffpqr_entropy_weight": FIXED_FPQR["entropy_weight"],
            "mlpffpqr_proposal_eps": 1e-3,
            "mlpffpqr_qvn_type": FIXED_FPQR["qvn_type"],
            "mlpffpqr_softplus_beta": 1.0,
        },
        "trials": HEAD_TUNING_TRIALS_FPQR,
        # TPE: the QVN choice makes the space conditional (each variant samples
        # different settings), which TPE's grouped multivariate model handles
        # and the Gaussian process would sample at random.
        "sampler": "tpe",
    },
    # QR and BNN search a few mostly continuous settings, where a Gaussian
    # process makes better use of 50 trials than TPE.
    "qr": {"pins": {"quantiles": QR_GRID, "conf_level": ALPHA}, "search": {}, "sampler": "gp"},
    "bnn": {"pins": {}, "search": {}, "sampler": "gp"},
}


def build_conditions(rated: float, power_targets: tuple[str, str]) -> list[Condition]:
    """The five benchmark heads, then every conformal variant reusing their weights.

    Args:
        rated: apparent-power rating the thermal-event Brier score is scored against.
        power_targets: names of the active and reactive power targets, in that order.
    """
    conditions = [
        Condition(
            name=name,
            group="benchmark",
            description=head["description"],
            model_cls=head["model_cls"],
            overrides=dict(head["overrides"]),
            metric_types=head["metric_types"],
            custom_scorer=make_scorer(head["metric_types"], head["family"], rated, power_targets, joint=True),
            interval_alpha=ALPHA,
            # Same calibration-window reservation as the conformal siblings,
            # or reuse_checkpoint_from is refused (it shifts every fold).
            reserve_calib_window=True,
            # Kept so later analyses can reload the weights instead of training again.
            keep_checkpoints=True,
        )
        for name, head in HEADS.items()
    ]
    for name, head in HEADS.items():
        variants = dict(MULTIVARIATE_CONFORMAL)
        if head["is_quantile_head"]:
            variants.update(QUANTILE_CONFORMAL)
        conditions.extend(
            Condition(
                name=f"{name}_{variant}",
                group="conformal",
                description=f"{name} with conformal={variant}",
                model_cls=head["model_cls"],
                # Identical to the head's, so the reused weights load into the
                # same architecture.
                overrides=dict(head["overrides"]),
                metric_types=head["metric_types"],
                custom_scorer=make_scorer(head["metric_types"], head["family"], rated, power_targets, joint=False),
                interval_alpha=ALPHA,
                conformal_config=config,
                reuse_checkpoint_from=name,
            )
            for variant, config in variants.items()
        )
    for name, tuning in HEAD_TUNING.items():
        head = HEADS[name]
        conditions.append(
            Condition(
                name=f"{name}_tuned",
                group="head_tuned",
                description=f"{name} with its own head settings tuned, backbone and training budget pinned",
                model_cls=head["model_cls"],
                # Nothing in overrides: they would overwrite the tuned values.
                # The pins below are re-applied at build time instead.
                hpo_enabled=True,
                hpo_model_cls=head["model_cls"],
                hpo_variant="head_tuned",
                hpo_fixed_params={**BACKBONE_PINS, **tuning["pins"], "hpo_prune_monitor": HEAD_TUNING_PRUNE_MONITOR},
                hpo_search_space=tuning["search"] or None,
                hpo_objective_metric=HEAD_TUNING_OBJECTIVE,
                hpo_num_trials=tuning.get("trials", HEAD_TUNING_TRIALS),
                hpo_sampler=tuning["sampler"],
                hpo_initial_params=tuning.get("initial"),
                hpo_stop_patience=HEAD_TUNING_STOP_PATIENCE,
                hpo_validation_windows=HEAD_TUNING_WINDOWS,
                hpo_select_on_cwe=HEAD_TUNING_SELECT_ON_CWE,
                metric_types=head["metric_types"],
                custom_scorer=make_scorer(head["metric_types"], head["family"], rated, power_targets, joint=True),
                interval_alpha=ALPHA,
                # Same folds as the fixed heads.
                reserve_calib_window=True,
            )
        )
    # raw_export: each fixed head again, saving its raw calibration- and test-window forecasts so
    # any conformal method can be calibrated offline (conformal_study.py). With --reuse-run it loads
    # the earlier run's per-fold weights instead of training.
    conditions.extend(
        Condition(
            name=f"{name}_raw",
            group="raw_export",
            description=f"{name}, raw calibration and test forecasts saved",
            model_cls=head["model_cls"],
            overrides=dict(head["overrides"]),
            metric_types=head["metric_types"],
            custom_scorer=make_scorer(head["metric_types"], head["family"], rated, power_targets, joint=True),
            interval_alpha=ALPHA,
            reserve_calib_window=True,
            save_raw_predictions=True,
            keep_checkpoints=True,
        )
        for name, head in HEADS.items()
        if name in RAW_EXPORT_HEADS
    )
    for name in TUNED_CONFORMAL_HEADS:
        head = HEADS[name]
        # The pins only: main() adds the cached tuned values beneath them, in the
        # order the engine builds a tuned head (tuned values, then the pins).
        pins = {**BACKBONE_PINS, **HEAD_TUNING[name]["pins"]}
        refit = f"{name}_tuned_refit"
        conditions.append(
            Condition(
                name=refit,
                group="head_tuned_conformal",
                description=f"{name}_tuned refitted from its cached tuned settings, no search",
                model_cls=head["model_cls"],
                overrides=dict(pins),
                metric_types=head["metric_types"],
                custom_scorer=make_scorer(head["metric_types"], head["family"], rated, power_targets, joint=True),
                interval_alpha=ALPHA,
                reserve_calib_window=True,
            )
        )
        variants = dict(MULTIVARIATE_CONFORMAL)
        if head["is_quantile_head"]:
            variants.update(QUANTILE_CONFORMAL)
        conditions.extend(
            Condition(
                name=f"{name}_tuned_{variant}",
                group="head_tuned_conformal",
                description=f"{name}_tuned with conformal={variant}",
                model_cls=head["model_cls"],
                overrides=dict(pins),
                metric_types=head["metric_types"],
                custom_scorer=make_scorer(head["metric_types"], head["family"], rated, power_targets, joint=False),
                interval_alpha=ALPHA,
                conformal_config=config,
                reuse_checkpoint_from=refit,
            )
            for variant, config in variants.items()
        )
    return conditions


def tuned_overrides(name: str, dataset_key: str, hpo_root: Path) -> dict:
    """A head-tuned head's full settings: its cached tuned values beneath its pins.

    The same layering the engine applies when it builds the tuned head (cached
    values, then ``hpo_fixed_params``), so a condition given these as plain
    overrides, with HPO off, trains the tuned head without searching again.

    Args:
        name: head name in :data:`HEAD_TUNING`.
        dataset_key: the dataset the settings were tuned on.
        hpo_root: the ``backbone_hpo`` folder of the head-tuned run.

    Returns:
        Model overrides for the head's configuration class.

    Raises:
        FileNotFoundError: if no tuned settings are cached for this head and dataset.
    """
    path = backbone_params_path(dataset_key, hpo_root, model_name(HEADS[name]["model_cls"], "head_tuned"))
    if not path.exists():
        raise FileNotFoundError(f"no tuned settings for {name} on {dataset_key}: {path}")
    params = json.loads(path.read_text())["params"]
    log.info("tuned %s settings for %s: %s", name, dataset_key, params)
    return {**params, **BACKBONE_PINS, **HEAD_TUNING[name]["pins"]}


def apply_tuned_settings(spec: ExperimentSpec, dataset_key: str, hpo_root: Path) -> None:
    """Give each refitted head (group head_tuned_conformal) its tuned settings, in place.

    Args:
        spec: the dataset's spec.
        dataset_key: the dataset the settings were tuned on.
        hpo_root: the ``backbone_hpo`` folder of the head-tuned run.
    """
    for name in TUNED_CONFORMAL_HEADS:
        overrides = tuned_overrides(name, dataset_key, hpo_root)
        for condition in spec.conditions:
            if condition.group == "head_tuned_conformal" and condition.name.startswith(f"{name}_tuned"):
                condition.overrides = dict(overrides)


DSS_EU = "DSS-EU:default:0.4:0.4"
# The weak feeder (the one the hosting-capacity study binds on) and DSS-US
# rank 8, each at mid penetration like DSS-EU. Rank 8 is the most loaded
# SMART-DS transformer whose load responds steadily to EV and PV; rank 0 does
# not (SMART-DS re-draws DER customers at each level), and the decision sweep
# uses rank 8 too, so both experiments forecast the same asset.
DSS_UK = "DSS-UK:weak:0.4:0.4"
DSS_US = "DSS-US:8:0.4:0.4"
MLVS_PT_3PHASE = "MLVS-PT-3PHASE"

# Active and reactive power, for the apparent-power thermal event; looked up by
# name in the forecaster's target order rather than assumed to come first.
POWER_TARGETS: dict[str, tuple[str, str]] = {
    DSS_EU: ("p_kw", "q_kvar"),
    DSS_UK: ("p_kw", "q_kvar"),
    DSS_US: ("p_kw", "q_kvar"),
    MLVS_PT_3PHASE: ("LP", "LQ"),
}

# The three power-flow records each span about eleven months, too short for a
# calibration window, so they share DSS-EU's protocol and run the benchmark
# group only. MLVS-PT-3PHASE is a 37.5-month measured record with room for one.
CV_PROTOCOLS: dict[str, dict] = {
    "short": {
        # The training window holds validation, so 5 months is 4 to fit on
        # plus 1 to validate. "fixed" keeps it: in the default "derived"
        # mode cv_folds wins and every fold fitted on about 30 days.
        "cv_train_size": 5,
        "cv_train_size_mode": "fixed",
        "cv_val_size": 1,
        "cv_test_size": 1,
        "cv_calib_size": 0,
        "cv_folds": 10,
        "cv_stride": 1,
        "cv_window": "rolling",
        # HPO split (head_tuned only): four months from January, then May as
        # validation, ending before the first test month (late June on DSS-EU,
        # June on DSS-UK/DSS-US). DSS-EU's record starts on 29 January, so three
        # months gave about 60 daily samples, fewer than one 64-sample batch.
        "hpo_train_months": 4,
        "hpo_val_months": 1,
    },
    "long": {
        "cv_train_size": 12,
        "cv_val_size": 2,
        "cv_test_size": 3,
        "cv_calib_size": 3,
        "cv_folds": 10,
        "cv_stride": 1,
        "cv_window": "expanding",
        # HPO split (head_tuned only): 2019-03 to 2020-11, before the first
        # test window (2020-12).
        "hpo_train_months": 14,
        "hpo_val_months": 6,
    },
}
PROTOCOL: dict[str, str] = {DSS_EU: "short", DSS_UK: "short", DSS_US: "short", MLVS_PT_3PHASE: "long"}

_DATASET_COMMON = {"data_path": "examples/data", "forecast_horizon": 48, "window_stride": 48, "period": "30min"}
# The 2019 records mirror DSS-EU's 2021 calendar: September to December as the test period.
_DSS_2019 = {"train_start": "2019-01", "train_end": "2019-12", "test_start": "2019-09", "test_end": "2019-12"}
DATASETS: dict[str, dict] = {
    DSS_EU: {
        **_DATASET_COMMON,
        "dataset_name": DSS_EU,
        "train_start": "2021-01",
        "train_end": "2021-12",
        "test_start": "2021-09",
        "test_end": "2021-12",
    },
    DSS_UK: {**_DATASET_COMMON, **_DSS_2019, "dataset_name": DSS_UK},
    DSS_US: {**_DATASET_COMMON, **_DSS_2019, "dataset_name": DSS_US},
    MLVS_PT_3PHASE: {
        **_DATASET_COMMON,
        "dataset_name": MLVS_PT_3PHASE,
        "train_start": "2019-03",
        "train_end": "2022-01",
        "test_start": "2021-07",
        "test_end": "2022-01",
    },
}

# The engine's folder for this experiment under <output-root>/fixed_benchmark.
EXPERIMENT_SLUG = "multitarget_fixed_benchmark"
_SPEC_COMMON = {
    "name": "Multitarget Fixed Benchmark",
    "output_prefix": "ch6_fixed_benchmark",
    "condition_cls": MLPFFPQRConfig,
    "backbone_cls": MLPFConfig,
    "hpo_enabled": False,
    # num_workers=0: forked DataLoader workers hang after the MPS backend has
    # initialised (see multitarget_experiment.py's _SPEC_COMMON).
    # early_stop_patience=100 and 60 epochs (below) match Chapter 5's MLPQF
    # benchmark budget, so fpqr_ch5 trains as it did there and every other
    # head gets the same budget.
    "fixed_overrides": {"num_workers": 0, "early_stop_patience": 100},
    "reference_conditions": {"benchmark": "fpqr", "conformal": "laplace_max_norm"},
}


def thermal_rating(dataset_key: str) -> float:
    """98th percentile of apparent power over the dataset's test period, as forecast_examples.py rates it.

    Taken from the record so the event is one this feeder actually reaches. One
    value per dataset, not per fold, so every fold scores the same event.
    """
    cfg = DATASETS[dataset_key]
    data, *_ = load_dataset(dataset_key, data_path=cfg["data_path"])
    p_name, q_name = POWER_TARGETS[dataset_key]
    stamps = pd.to_datetime(data["timestamp"])
    end = pd.Timestamp(cfg["test_end"]) + pd.offsets.MonthEnd(1)
    window = data.loc[((stamps >= cfg["test_start"]) & (stamps <= end)).to_numpy()]
    return float(np.quantile(np.hypot(window[p_name], window[q_name]), 0.98))


def make_specs(input_window_size: int, dataset_keys: list[str]) -> dict[str, ExperimentSpec]:
    """One spec per dataset: their CV protocols and thermal ratings differ (see the module docstring)."""
    datasets = {key: {**cfg, "input_window_size": input_window_size} for key, cfg in DATASETS.items()}
    ratings = {key: thermal_rating(key) for key in dataset_keys}
    for key, rated in ratings.items():
        log.info("thermal rating for %s: %.3f (98th percentile of apparent power over the test period)", key, rated)
    return {
        key: ExperimentSpec(
            **_SPEC_COMMON,
            **CV_PROTOCOLS[PROTOCOL[key]],
            conditions=build_conditions(ratings[key], POWER_TARGETS[key]),
            datasets={key: datasets[key]},
        )
        for key in dataset_keys
    }


def base_config(
    epochs: int, output_root: Path, hpo_trials: int = HEAD_TUNING_TRIALS, hpo_epochs: int = HEAD_TUNING_EPOCHS
) -> PipelineConfig:
    """PipelineConfig shared by every dataset; dataset fields are applied per spec."""
    return PipelineConfig(
        dataset_name=DSS_EU,
        data_path="examples/data",
        output_dir=str(output_root / "fixed_benchmark"),
        epochs=epochs,
        num_trials=hpo_trials,
        hpo_epochs=hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )


def main() -> None:
    """Run the benchmark on every selected dataset and the conformal group where there is a calibration window."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument(
        "--group",
        nargs="*",
        default=["benchmark", "conformal"],
        choices=["benchmark", "conformal", "head_tuned", "head_tuned_conformal", "raw_export"],
        help=(
            "head_tuned tunes FPQR/QR/BNN's own settings on the same folds; run it on its own. "
            "head_tuned_conformal refits the tuned FPQR from its cached settings and calibrates it"
        ),
    )
    ap.add_argument(
        "--reuse-run",
        type=Path,
        default=None,
        help=(
            "raw_export only: an earlier run's dataset folder (runs/<id>/conditions/<dataset>) whose "
            "benchmark heads kept their per-fold checkpoints; loads them instead of training"
        ),
    )
    ap.add_argument(
        "--tuned-hpo-root",
        type=Path,
        default=None,
        help="backbone_hpo folder holding the tuned settings (default: this experiment's own)",
    )
    ap.add_argument("--epochs", type=int, default=60, help="Chapter 5's MLPQF benchmark trained for 60")
    ap.add_argument("--lookback", type=int, default=48, help="forecast_examples.py used 48 steps (one day)")
    ap.add_argument("--quick", action="store_true", help="2 folds, 2 epochs, 2 HPO trials: a smoke test")
    ap.add_argument(
        "--hpo-trials",
        type=int,
        default=None,
        help=(
            f"trials for every tuned head; default {HEAD_TUNING_TRIALS_FPQR} for FPQR "
            f"and {HEAD_TUNING_TRIALS} for QR and BNN"
        ),
    )
    ap.add_argument("--hpo-epochs", type=int, default=HEAD_TUNING_EPOCHS, help="epochs per HPO trial")
    ap.add_argument(
        "--hpo-windows",
        type=int,
        default=HEAD_TUNING_WINDOWS,
        help="validation windows per HPO trial; >1 averages CRPS and prunes with a Wilcoxon test (useful from 5)",
    )
    ap.add_argument(
        "--hpo-select-on-cwe",
        action=argparse.BooleanOptionalAction,
        default=HEAD_TUNING_SELECT_ON_CWE,
        help="among trials within 1%% of the best CRPS, keep the highest validation CWE (quantile heads)",
    )
    ap.add_argument(
        "--hpo-stop-patience",
        type=int,
        default=HEAD_TUNING_STOP_PATIENCE,
        help="stop a search after this many completed trials without improvement; 0 runs every trial",
    )
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()

    configure(level="INFO")
    cfg = base_config(
        2 if args.quick else args.epochs,
        args.output_root,
        hpo_trials=2 if args.quick else (args.hpo_trials or HEAD_TUNING_TRIALS),
        hpo_epochs=2 if args.quick else args.hpo_epochs,
    )
    dataset_keys = args.dataset
    specs = make_specs(args.lookback, dataset_keys)

    summaries = []
    for dataset_key in dataset_keys:
        spec = specs[dataset_key]
        if args.quick:
            spec = dc_replace(spec, cv_folds=2)
        # Per-head budgets (build_conditions) unless --quick or --hpo-trials asks for one for all.
        trials = 2 if args.quick else args.hpo_trials
        for condition in spec.conditions:
            if condition.group != "head_tuned":
                continue
            if trials is not None:
                condition.hpo_num_trials = trials
            condition.hpo_validation_windows = args.hpo_windows
            condition.hpo_stop_patience = args.hpo_stop_patience or None
            condition.hpo_select_on_cwe = args.hpo_select_on_cwe
        groups = list(args.group)
        for group in ("conformal", "head_tuned_conformal"):
            if spec.cv_calib_size == 0 and group in groups:
                groups.remove(group)
                log.info("Skipping group=%r on %s: no calibration window (cv_calib_size=0)", group, dataset_key)
        if "raw_export" in groups and args.reuse_run is not None:
            for condition in spec.conditions:
                if condition.group == "raw_export":
                    head = condition.name.removesuffix("_raw")
                    model_type = HEADS[head]["model_cls"].__name__.lower().removesuffix("config")
                    source = args.reuse_run / "benchmark" / head / "checkpoints" / head / model_type
                    condition.reuse_checkpoint_dir = str(source)
        if "head_tuned_conformal" in groups:
            hpo_root = args.tuned_hpo_root or args.output_root / "fixed_benchmark" / EXPERIMENT_SLUG / "backbone_hpo"
            apply_tuned_settings(spec, dataset_key, hpo_root)
        if "conformal" in groups and "benchmark" not in groups:
            # Conformal conditions reuse the benchmark heads' checkpoints from
            # the same run, so the heads have to train first.
            groups.insert(0, "benchmark")
        if not groups:
            continue
        summaries.append(
            ExperimentEngine(spec).run(base_cfg=cfg, groups=groups, dataset_keys=[dataset_key], tracking_uri=None)
        )

    if not summaries:
        raise SystemExit("nothing to run: no group left for the selected datasets")
    summary = pd.concat(summaries, ignore_index=True)
    out = script_dir / ("results_fixed_benchmark_quick.csv" if args.quick else "results_fixed_benchmark.csv")
    summary.to_csv(out, index=False)
    log.info("summary written to %s", out)
    print(summary.to_string())


if __name__ == "__main__":
    main()
