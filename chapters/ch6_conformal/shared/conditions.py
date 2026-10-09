"""Condition building blocks shared by Chapter 7's engine experiments.

Follows Chapter 6's ``multitarget_experiment.py``: conditions are plain :class:`~twiga.experiment.Condition` objects,
every head shares one tuned backbone through a named ``hpo_variant`` (``SHARED_HPO_VARIANT``), a head with its own
hyperparameters (FPQR) runs a head-only search on top of that backbone through ``hpo_parent_model_cls``, and each
condition reports a proper scoring rule through its ``metric_types`` wherever the engine allows it.
"""

from __future__ import annotations

from shared.protocol import ALPHA, DEFAULT_HPO_TRIALS
from twiga.core.config import ConformalConfig
from twiga.models.nn import MLPFConfig, MLPGAMConfig

# ── Metric types ─────────────────────────────────────────────────────────────
# "quantile"/"parametric" add CRPS (and the pinball loss / NLL), as in Chapter 6. The engine's calibrated (conformal)
# path supports only point and interval metrics, so conformalised conditions use INTERVAL_METRIC_TYPES.
POINT_METRIC_TYPES = ["point"]
INTERVAL_METRIC_TYPES = ["point", "interval"]
QUANTILE_METRIC_TYPES = ["point", "interval", "quantile"]
PARAMETRIC_METRIC_TYPES = ["point", "interval", "parametric"]

# ── Backbone HPO ─────────────────────────────────────────────────────────────
# One MLPGAM backbone search per dataset, shared by every MLPGAM-backed condition (Chapter 6's SHARED_HPO_VARIANT).
SHARED_HPO_VARIANT = "shared_mlpgam"
# FPQR's own head-only search on top of the shared backbone (Chapter 6's FPQR_HPO_VARIANT).
FPQR_HPO_VARIANT = "fpqr_head"

# Pinned inside the MLPGAM search (Condition.hpo_fixed_params): the gate on, and the target-gate fields, which act only
# when use_target_gate=True (multi-target), at their defaults so they take no share of the trial budget. Every
# condition on the shared key must carry the same pins (the engine merges them per HPO cache key).
_TARGET_GATE_FIELDS = ("lambda_target_gate", "target_gate_delta", "target_gate_warmup_epochs")
MLPGAM_HPO_PINS = {
    "use_gate": True,
    "use_target_gate": False,
    **{name: MLPGAMConfig.model_fields[name].default for name in _TARGET_GATE_FIELDS},
}

# FPQR stability pins (Chapter 6's _FPQR_STABILITY_PINS, with the default quantile-value network): the proposal family
# and the QVN architecture are fixed rather than searched, because free draws over them converge to inconsistent
# configurations run to run (Chapter 6 observed PICP between 0.75 and 0.91 on identical reruns; Chapter 7's first
# benchmark drew a Kumaraswamy proposal with a spline QVN, which failed in the five data-scarce MLVS-PT folds).
# MLPGAMFPQRConfig leaves gradient clipping off by default, so it is pinned to MLPFFPQRConfig's 1.0.
FPQR_HPO_PINS = {
    "proposal_type": "softmax",
    "qvn_type": "standard",
    "proposal_loss_fn": "wasserstein",
    "proposal_eps": 1e-4,
    "gradient_clip_val": 1.0,
}

# ── Proposed model and conformal procedures ──────────────────────────────────
# The proposed MLPGAM-CRC configuration (Chapter-7.tex chap-7:mlpgam), with the feature gate enabled.
PROPOSED = {"two_stage": True, "sigma_loss_fn": "laplace", "monitor": "loss", "lambda_lasso": 1e-6, "use_gate": True}
PROPOSED_CONFORMAL = ConformalConfig(method="residual-fitting", score_type="res", alpha=ALPHA)
# Signed variant of the proposed score: split quantiles at alpha/2 and 1 - alpha/2 (asymmetric interval).
PROPOSED_CONFORMAL_SIGNED = ConformalConfig(method="residual-fitting", score_type="sign-res", alpha=ALPHA)
SCP = ConformalConfig(method="residual", score_type="res", alpha=ALPHA)
SCP_POOLED = ConformalConfig(method="residual", score_type="res", alpha=ALPHA, pooled=True)


def search_size(config_cls: type, pinned: dict | None = None) -> int:
    """Number of hyperparameters ``config_cls``'s Optuna search samples once ``pinned`` fields are fixed."""
    space = config_cls.model_fields["search_space"].default_factory().model_dump(exclude_none=True)
    return len(set(space) - set(pinned or {}))


def trial_budget(config_cls: type, trials_per_param: float, floor: int, pinned: dict | None = None) -> int:
    """HPO trials proportional to the effective search size, never below ``floor``."""
    return max(floor, round(trials_per_param * search_size(config_cls, pinned)))


# MLPF keeps the chapter's 50 trials; every other backbone gets the same number of trials per searched parameter.
DEFAULT_TRIALS_PER_PARAM = DEFAULT_HPO_TRIALS / search_size(MLPFConfig)
