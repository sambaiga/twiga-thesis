"""Design-ablation conditions, spec and engine for Chapter 7's MLPGAM design evaluation.

Split out of the former ``crc_experiment.py``. Dataset, protocol and output
settings come from :mod:`shared.protocol`.
"""

from __future__ import annotations

from shared.protocol import (
    CV_CALIB_MONTHS,
    CV_FOLDS,
    CV_STRIDE_MONTHS,
    CV_TEST_MONTHS,
    CV_VAL_MONTHS,
    CV_WINDOW,
    DATASETS,
    EARLY_STOP_PATIENCE,
)
from twiga.core.config import ConformalConfig
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.models.nn import MLPGAMConfig, MLPGAMCRCConfig


def make_conformal_config(
    method: str = "residual-fitting", score_type: str = "res", alpha: float = 0.10, pooled: bool = False
) -> ConformalConfig:
    return ConformalConfig(method=method, score_type=score_type, alpha=alpha, pooled=pooled)


def make_condition(
    name: str,
    group: str,
    description: str,
    *,
    two_stage: bool = True,
    stage1_epochs_frac: float | None = None,
    sigma_loss_fn: str = "hybrid_sqrt",
    monitor: str | None = "sigma_loss",
    lambda_lasso: float = 1e-6,
    method: str = "residual-fitting",
    score_type: str = "res",
    alpha: float = 0.10,
    pooled: bool = False,
) -> Condition:
    return Condition(
        name=name,
        group=group,
        description=description,
        overrides={
            "two_stage": two_stage,
            "sigma_loss_fn": sigma_loss_fn,
            "monitor": monitor,
            "lambda_lasso": lambda_lasso,
        },
        conformal_config=make_conformal_config(method=method, score_type=score_type, alpha=alpha, pooled=pooled),
        stage1_epochs_frac=stage1_epochs_frac,
        metric_types=["point", "interval"],
    )


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = [
    # ── training_mode ────────────────────────────────────────────────────────
    make_condition("joint", "training_mode", "Joint training — single optimizer", two_stage=False),
    make_condition("two_stage_batch", "training_mode", "Two-stage per-batch alternation (reference)", two_stage=True),
    # stage1_epochs_frac previously swept 0.30/0.50/0.75 of max_epochs=150
    # (45/75/112 epochs). A completed run's checkpoint logs showed the best
    # epoch per fold landing in the 0-29 range under early_stop_patience=10,
    # so all three conditions early-stopped before ever reaching the
    # stage1/stage2 boundary and were bit-identical to each other and to
    # "joint" as a result. Rescoped to land inside that observed range.
    make_condition(
        "two_stage_ep_05",
        "training_mode",
        "Sequential: 5% backbone (~8 epochs), 95% sigma",
        two_stage=True,
        stage1_epochs_frac=0.05,
    ),
    make_condition(
        "two_stage_ep_10",
        "training_mode",
        "Sequential: 10% backbone (~15 epochs), 90% sigma",
        two_stage=True,
        stage1_epochs_frac=0.10,
    ),
    make_condition(
        "two_stage_ep_20",
        "training_mode",
        "Sequential: 20% backbone (~30 epochs), 80% sigma",
        two_stage=True,
        stage1_epochs_frac=0.20,
    ),
    # ── sigma_loss ───────────────────────────────────────────────────────────
    make_condition("sigma_mse", "sigma_loss", "MSE — penalises large sigma errors heavily", sigma_loss_fn="mse"),
    make_condition("sigma_hybrid", "sigma_loss", "Hybrid MSE+MAE — balanced penalty", sigma_loss_fn="hybrid"),
    make_condition(
        "sigma_hybrid_sqrt",
        "sigma_loss",
        "Hybrid-sqrt (reference) — dampens large residuals",
        sigma_loss_fn="hybrid_sqrt",
    ),
    make_condition("sigma_gaussian", "sigma_loss", "Gaussian NLL", sigma_loss_fn="gaussian"),
    make_condition("sigma_laplace", "sigma_loss", "Laplace NLL — heavier-tail assumption", sigma_loss_fn="laplace"),
    make_condition("sigma_log", "sigma_loss", "Log loss — encourages tighter intervals", sigma_loss_fn="log"),
    # ── monitor ──────────────────────────────────────────────────────────────
    make_condition("monitor_mae", "monitor", "Checkpoint on best val_mae", monitor="mae"),
    make_condition("monitor_loss", "monitor", "Checkpoint on combined val_loss", monitor="loss"),
    make_condition("monitor_sigma", "monitor", "Checkpoint on val_sigma_loss (reference)", monitor="sigma_loss"),
    # ── lasso ────────────────────────────────────────────────────────────────
    # Uniform log-decade sweep (each step x10) anchored at the default 1e-6,
    # rather than the earlier version's asymmetric skip over 1e-5 and 1e-3.
    make_condition("lasso_0", "lasso", "No GAM regularisation", lambda_lasso=0.0),
    make_condition("lasso_1e7", "lasso", "Very weak GAM regularisation", lambda_lasso=1e-7),
    make_condition("lasso_1e6", "lasso", "Default GAM regularisation (reference)", lambda_lasso=1e-6),
    make_condition("lasso_1e5", "lasso", "Weak-moderate GAM regularisation", lambda_lasso=1e-5),
    make_condition("lasso_1e4", "lasso", "Strong GAM regularisation", lambda_lasso=1e-4),
    make_condition("lasso_1e3", "lasso", "Stronger GAM regularisation", lambda_lasso=1e-3),
    make_condition("lasso_1e2", "lasso", "Very strong GAM regularisation", lambda_lasso=1e-2),
    make_condition("lasso_1e1", "lasso", "Extreme GAM regularisation", lambda_lasso=1e-1),
    # ── conformal ────────────────────────────────────────────────────────────
    make_condition(
        "conf_scp",
        "conformal",
        "Split conformal (SCP) — unscaled residual",
        method="residual",
        score_type="res",
        alpha=0.10,
    ),
    # Same as conf_scp (unscaled residual, no learned sigma), but with a
    # single quantile pooled across every calibration (sample, horizon) pair
    # instead of one quantile per horizon. conf_scp itself already calibrates
    # per horizon by default (twiga.distributions.conformal.base.calibrate's
    # axis=0), so it is not actually the "one global non-conformity
    # distribution" baseline Chapter-7.tex's adaptive-CP section contrasts
    # horizon-wise calibration against -- this condition is that baseline.
    make_condition(
        "conf_scp_pooled",
        "conformal",
        "Split conformal (SCP), pooled across horizon — single global quantile",
        method="residual",
        score_type="res",
        alpha=0.10,
        pooled=True,
    ),
    make_condition(
        "conf_90_res",
        "conformal",
        "CRC (reference) — 90% coverage, sigma-scaled residual",
        method="residual-fitting",
        score_type="res",
        alpha=0.10,
    ),
    make_condition(
        "conf_90_sign",
        "conformal",
        "CRC — 90% coverage, sigma-scaled signed residual",
        method="residual-fitting",
        score_type="sign-res",
        alpha=0.10,
    ),
    # ── calib_isolation ──────────────────────────────────────────────────────
    make_condition("ci_joint", "calib_isolation", "Calib-isolation: joint training", two_stage=False),
    make_condition("ci_batch", "calib_isolation", "Calib-isolation: batch alternation (reference)", two_stage=True),
    # Rescoped alongside training_mode's two_stage_ep_* — see that group's
    # comment for why 0.30/0.75 never actually diverged under early stopping.
    make_condition(
        "ci_ep_05", "calib_isolation", "Calib-isolation: sequential 5% epochs", two_stage=True, stage1_epochs_frac=0.05
    ),
    make_condition(
        "ci_ep_20", "calib_isolation", "Calib-isolation: sequential 20% epochs", two_stage=True, stage1_epochs_frac=0.20
    ),
    # ── best_config ───────────────────────────────────────────────────────────
    make_condition(
        "best_config",
        "best_config",
        "Optimal: two_stage_batch + hybrid_sqrt + sigma_loss monitor + lasso=1e-6 + CRC-90-sign",
        two_stage=True,
        sigma_loss_fn="hybrid_sqrt",
        monitor="sigma_loss",
        lambda_lasso=1e-6,
        method="residual-fitting",
        score_type="sign-res",
        alpha=0.10,
    ),
    make_condition(
        "best_config_v1",
        "best_config",
        "Original optimal (v1): batch + MSE + MAE monitor + lasso=1e-4 + CRC-90-sign",
        two_stage=True,
        sigma_loss_fn="mse",
        monitor="mae",
        lambda_lasso=1e-4,
        method="residual-fitting",
        score_type="sign-res",
        alpha=0.10,
    ),
    # The pairing the per-group sweeps actually jointly recommend (MSE scale
    # loss, sigma_loss monitor -- see the scale_loss and monitor groups
    # above), never evaluated end-to-end until now: `best_config` uses
    # hybrid_sqrt+sigma_loss, `best_config_v1` uses MSE+MAE, and neither is
    # this combination. Otherwise identical to `best_config` (lasso=1e-6,
    # CRC-90-sign) so only the scale-loss/monitor pairing varies.
    make_condition(
        "best_config_v2",
        "best_config",
        "Joint recommendation: batch + MSE + sigma_loss monitor + lasso=1e-6 + CRC-90-sign",
        two_stage=True,
        sigma_loss_fn="mse",
        monitor="sigma_loss",
        lambda_lasso=1e-6,
        method="residual-fitting",
        score_type="sign-res",
        alpha=0.10,
    ),
    # ── alpha_level ──────────────────────────────────────────────────────────
    # Two fixed nominal-coverage levels, 0.10 (the chapter's usual 90%
    # coverage, the reference) and 0.05 (95%).
    #
    # This group was previously named "adaptive_alpha" with four conditions
    # ("adaptive_90_sign"/"fixed_90_sign"/"adaptive_95_sign"/"fixed_95_sign")
    # framed as fixed-vs-adaptive CRC pairs at each level. That framing was
    # never implemented: make_condition/ConformalConfig have no adaptive
    # (online alpha-update) mechanism, so the "adaptive_*" and "fixed_*"
    # conditions at a given level were configured identically and differed
    # only by name. Confirmed empirically in a completed run: the alpha=0.05
    # pair produced bit-identical per-fold predictions (same MAE, same
    # Model/timestamp, down to the row), and the alpha=0.10 pair's small
    # difference was consistent with GPU/cuDNN warm-up nondeterminism on the
    # first-trained condition in the process, not a real adaptive effect.
    # Collapsed to the two conditions this group actually measures until an
    # adaptive (e.g. ACI-style online alpha update) mechanism is implemented.
    make_condition(
        "alpha_90_sign",
        "alpha_level",
        "Target 90% coverage, CRC signed-residual (reference)",
        two_stage=True,
        sigma_loss_fn="hybrid_sqrt",
        monitor="sigma_loss",
        lambda_lasso=1e-6,
        method="residual-fitting",
        score_type="sign-res",
        alpha=0.10,
    ),
    make_condition(
        "alpha_95_sign",
        "alpha_level",
        "Target 95% coverage, CRC signed-residual",
        two_stage=True,
        sigma_loss_fn="hybrid_sqrt",
        monitor="sigma_loss",
        lambda_lasso=1e-6,
        method="residual-fitting",
        score_type="sign-res",
        alpha=0.05,
    ),
]

# ---------------------------------------------------------------------------
# Factorial: training_mode × conformal_score (resolves additivity failure)
# ---------------------------------------------------------------------------

FACTORIAL_TRAINING_MODES: dict[str, dict] = {
    "joint": {"two_stage": False},
    "two_stage_batch": {"two_stage": True},
    "ep_05": {"two_stage": True, "stage1_epochs_frac": 0.05},
    "ep_10": {"two_stage": True, "stage1_epochs_frac": 0.10},
    "ep_20": {"two_stage": True, "stage1_epochs_frac": 0.20},
}

FACTORIAL_CONFORMAL_SCORES: dict[str, dict] = {
    "res": {"method": "residual-fitting", "score_type": "res"},
    "sign": {"method": "residual-fitting", "score_type": "sign-res"},
    "scp": {"method": "residual", "score_type": "res"},
}

FACTORIAL_CONDITIONS: list[Condition] = [
    make_condition(
        f"{tm_name}_x_{cs_name}",
        "training_x_conformal",
        f"Factorial: {tm_name} training + {cs_name} conformal score",
        sigma_loss_fn="hybrid_sqrt",
        monitor="sigma_loss",
        lambda_lasso=1e-6,
        alpha=0.10,
        **tm_kwargs,
        **cs_kwargs,
    )
    for tm_name, tm_kwargs in FACTORIAL_TRAINING_MODES.items()
    for cs_name, cs_kwargs in FACTORIAL_CONFORMAL_SCORES.items()
]

ABLATION_CONDITIONS = [*ABLATION_CONDITIONS, *FACTORIAL_CONDITIONS]

# ---------------------------------------------------------------------------
# Ablation spec + engine
# ---------------------------------------------------------------------------

ABLATION_SPEC = ExperimentSpec(
    name="MLPGAM-CRC Design Ablation",
    output_prefix="crc_ablation",
    condition_cls=MLPGAMCRCConfig,
    backbone_cls=MLPGAMConfig,
    conditions=ABLATION_CONDITIONS,
    datasets=DATASETS,
    # early_stop_patience=20, not the NeuralModelConfig default of 10
    # (twiga/core/config/neural.py:117), scoped to this chapter's own spec
    # rather than the shared library default. reduce_on_plateau's scheduler
    # patience is int(sch_prob_patience * max_epochs) (twiga/models/nn/core/
    # base_model.py:453); sch_prob_patience's HPO search range is (0.05, 0.2)
    # (twiga/core/config/neural.py:252). At the original max_epochs=150 the
    # scheduler's patience could land anywhere from 7.5 to 30 epochs --
    # indistinguishable from, or larger than, the old early_stop_patience=10,
    # the mechanism behind every fold's "best epoch" landing within the
    # first 0-8 epochs on both datasets (design_ablation_figures.py's
    # training-schedule mechanism figure). 2026-09-18: max_epochs raised to
    # 500 (mlpgam_design_ablation.py's --epochs default) specifically to let
    # training run well past that early-saturation point rather than
    # stopping at or before it, so early_stop_patience is raised to 100
    # alongside it -- the scheduler's worst case is now int(0.2*500)=100,
    # exactly meeting rather than safely clearing 100; a trial sampling
    # sch_prob_patience near the top of its range can still tie, but the
    # typical case (~0.125) gives scheduler patience ~62, well clear.
    # Pinning lr_scheduler_type away from reduce_on_plateau entirely would
    # remove the race outright but was not made here since it also changes
    # which scheduler hyperparameters need retuning, out of scope for this
    # fix.
    fixed_overrides={"early_stop_patience": EARLY_STOP_PATIENCE},
    controlled_fields=frozenset(
        {
            "two_stage",
            "stage1_epochs",
            "sigma_loss_fn",
            "monitor",
            "lambda_lasso",
            "opt_lr",
            "opt_weight_decay",
            "sch_warmup_epochs",
            "sch_eta_min",
        }
    ),
    cv_test_size=CV_TEST_MONTHS,
    cv_val_size=CV_VAL_MONTHS,
    cv_calib_size=CV_CALIB_MONTHS,
    cv_stride=CV_STRIDE_MONTHS,
    cv_folds=CV_FOLDS,
    cv_window=CV_WINDOW,
    reference_conditions={
        "training_mode": "two_stage_batch",
        "sigma_loss": "sigma_hybrid_sqrt",
        "monitor": "monitor_sigma",
        "lasso": "lasso_1e6",
        "conformal": "conf_90_res",
        "calib_isolation": "ci_batch",
        "best_config": "best_config",
        "alpha_level": "alpha_90_sign",
        "training_x_conformal": "two_stage_batch_x_sign",
    },
)

ABLATION_ENGINE = ExperimentEngine(ABLATION_SPEC)
