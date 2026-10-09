r"""Chapter 6, follow-on: backbone architecture x conformal calibration bake-off.

Runs through :class:`~twiga.experiment.ExperimentEngine`, the same machinery as
``fpqr_gating_ablation.py``/``multitarget_experiment.py`` -- this script reuses
that pattern rather than hand-rolling a backtest loop.

The question
------------
``fpqr_gating_ablation.py`` concluded the FPQR mechanism ablation with a
simple configuration (shared grid, softmax proposal, standard quantile-value
network -- all ``BaseFPQRConfig``'s own defaults, so no override is needed for
either); ``copula_variants.py`` concluded the dependence ablation with no
copula beating plain FPQR on Brier score, on any dataset. Two questions are
still open once those two conclusions are taken as given:

1. Does a different backbone architecture (MLPF, MLPGAM, NHiTS) carrying that
   same simple FPQR configuration change the marginal picture?
2. Does conformal calibration -- joint/multivariate
   (``"max_norm"``/``"max_abs"``/``"mahalanobis"``,
   :class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`)
   or CQR (``"quantile"``,
   :class:`~twiga.distributions.conformal.cqr.ConformalQuantileRegressor`)
   -- change the calibrated interval picture for any of the three FPQR
   backbones or for QR's own fixed grid, and does CQR's own joint variants
   (``target_coverage="pooled"`` / ``"bonferroni"``) close the gap against
   the multivariate methods, or does the score itself (quantile-interval
   vs. location/scale residual) matter independently of whether it pools
   across targets? Further, does the correlation estimator
   (``dependence_method``) matter for the ``"mahalanobis"`` score?

One group, ``"backbone"``: 4 heads (MLPF/MLPGAM/NHiTS FPQR plus QR's
fixed-grid baseline, added 2026-09-30) x up to 13 conformal conditions --
the uncalibrated ``none`` baseline, two location/scale joint scores
(``max_norm``, ``max_abs``), four correlation estimators for
``mahalanobis`` (``pearson``, ``spearman``, ``kendall``, ``gaussian_copula``),
and six CQR variants (2 score types x 3 ``target_coverage`` modes:
``marginal``, ``pooled``, ``bonferroni``). The CQR family uses
:class:`~twiga.distributions.conformal.cqr.ConformalQuantileRegressor`
with ``score_type="scaled"`` or ``"unscaled"``; the three
``target_coverage`` modes are the marginal, pooled, and Bonferroni
variants of the same score, covering the full space of cross-target
corrections the class offers. QR reuses a dedicated, backbone-only MLPF
HPO search (``QR_HPO_VARIANT``) rather than its own full joint study, since
it has no FPQR-specific hyperparameters to tune -- see ``_build_conditions``'s
own docstring for why.

``max_norm``/``mahalanobis`` on a quantile-only (FPQR) backbone
------------------------------------------------------------------
``multitarget_experiment.py``'s own ``conformal`` group only applies
``ConformalConfig(method="multivariate", ...)`` to its two parametric
conditions (Laplace, BNN), with an explicit note that FPQR/QR "have no scale
and are not repeated here" -- at the time that was written, ``core.py``'s
``calibrate()`` required a native ``RawPrediction.scale`` field for
``score_type in ("max-norm", "mahalanobis")``, which no quantile-only head
populates. That gap is now closed: ``twiga/forecaster/core.py`` derives an
adaptive scale from the quantile spread (``get_sigma_from_quantile_levels``)
whenever a QUANTILE-kind ``RawPrediction`` has quantiles but no native scale,
so all three FPQR backbones here can run every conformal method the engine
supports, not only ``"quantile"``.

Datasets and the conformal calibration window
------------------------------------------------
Four datasets, all now engine-supported: ``DSS-EU``, ``MLVS-PT-3PHASE``
(``ExperimentSpec.cv_split_freq="months"``, the engine's long-standing
default) and ``DSS-UK``, ``DSS-US`` (``cv_split_freq="weeks"``, only
reachable now that ``ExperimentSpec``/``ExperimentEngine`` expose that field
rather than hardcoding ``"months"`` -- previously why DSS-UK/DSS-US could
only be reached through ``copula_variants.py``/``operational_decision.py``'s
own hand-rolled ``build_backtest_folds``). ``DSS-UK``/``DSS-US`` reuse
``copula_variants.py``'s/``operational_decision.py``'s own verified
weeks-based protocol numbers (train=26/test=4/stride=4 weeks) rather than
inventing new ones, with ``cv_calib_size=4`` weeks so the conformal-bearing
conditions get a real calibration window each fold.

``DSS-EU`` keeps ``cv_calib_size=0`` (``multitarget_experiment.py``'s own
documented reason: DSS-EU's ~11-month span cannot absorb any calibration
reservation above ~1 month at all, swept directly there) -- this script's
``main()`` skips the conformal-bearing condition variants on DSS-EU the same
way ``multitarget_experiment.py`` skips its whole ``conformal`` group there,
running only ``*_none`` on DSS-EU.

Output layout
-------------
The experiment's output directory is resolved via ``--output-root`` (or the
``TWIGA_EXPERIMENTS_ROOT`` environment variable, if set), matching
``multitarget_experiment.py``'s convention. This lets the same script write
into different trees on server runs (scratch disk vs. thesis archive) without
editing the file, and lets chunked sweeps of distinct conformal-variant
families land in distinct output roots rather than colliding.

Usage::

    uv run python experiment/ch7_multitarget/backbone_conformal_study.py --dataset MLVS-PT-3PHASE
    uv run python experiment/ch7_multitarget/backbone_conformal_study.py --quick
    # Focused sweep of one family:
    uv run python experiment/ch7_multitarget/backbone_conformal_study.py \\
        --variants mahalanobis_pearson mahalanobis_spearman mahalanobis_kendall mahalanobis_gaussian_copula
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import json
import os
from pathlib import Path

import pandas as pd
from twiga.core.config import ConformalConfig
from twiga.core.utils import configure, get_logger
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.models.nn import MLPFConfig
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.models.nn.mlpfqr_model import MLPFQRConfig
from twiga.models.nn.mlpgamfpqr_model import MLPGAMFPQRConfig
from twiga.models.nn.nhitsfpqr_model import NHITSFPQRConfig
from twiga.utils.run_forecast import PipelineConfig

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent

# parents[3]: backbone_conformal_study.py -> ch7_multitarget ->
# experiment -> twiga-forecast (repo root) -> projects -- matches
# multitarget_experiment.py's own output_dir convention, a sibling of the
# repo, not inside it. Overridable via --output-root or the
# TWIGA_EXPERIMENTS_ROOT environment variable.
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[3] / "experiments"))) / "ch7_multitarget"
)

# ── Datasets ──────────────────────────────────────────────────────────────────
DSS_EU = "DSS-EU:default:0.4:0.4"
MLVS_PT_3PHASE = "MLVS-PT-3PHASE"
# DSS-UK/DSS-US scenario strings, identical to operational_decision.SCENARIOS
# (weak feeder / rank-0 transformer, both at 0.4/0.4 penetration) -- not
# imported directly since that module has no engine-facing DATASETS entry to
# reuse, only the scenario strings themselves, which are duplicated here.
DSS_UK = "DSS-UK:weak:0.4:0.4"
DSS_US = "DSS-US:0:0.4:0.4"

DATASETS: dict[str, dict] = {
    DSS_EU: {
        "dataset_name": DSS_EU,
        "data_path": "examples/data",
        "train_start": "2021-01",
        "train_end": "2021-12",
        "test_start": "2021-09",
        "test_end": "2021-12",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
    MLVS_PT_3PHASE: {
        "dataset_name": MLVS_PT_3PHASE,
        "data_path": "examples/data",
        "train_start": "2019-03",
        "train_end": "2022-01",
        "test_start": "2021-07",
        "test_end": "2022-01",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
    # DSS-UK/DSS-US: one full calendar year (2019-01-01..2019-12-31,
    # confirmed against the parquet files), same relative train/test split as
    # DSS-EU's own entry above (test window the last quarter of the year).
    DSS_UK: {
        "dataset_name": DSS_UK,
        "data_path": "examples/data",
        "train_start": "2019-01",
        "train_end": "2019-12",
        "test_start": "2019-09",
        "test_end": "2019-12",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
    DSS_US: {
        "dataset_name": DSS_US,
        "data_path": "examples/data",
        "train_start": "2019-01",
        "train_end": "2019-12",
        "test_start": "2019-09",
        "test_end": "2019-12",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
}

# Datasets reached only via cv_split_freq="weeks" (see module docstring) --
# DSS-EU/MLVS-PT-3PHASE keep the engine's long-standing months-based default.
WEEKS_BASED_DATASETS = (DSS_UK, DSS_US)

_BASE_METRICS = ["point", "interval", "quantile"]
_CONFORMAL_METRICS = ["point", "interval"]

# ── Backbones, each carrying the FPQR mechanism ablation's simple config ──────
# fpqr_gating_ablation.py's ablation kept shared grid (per_target_tau=False),
# softmax proposal and standard quantile-value network -- both BaseFPQRConfig's
# own defaults. Only proposal_type/qvn_type are restored to the ablation's
# winning values here (see below); every other FPQR hyperparameter --
# n_quantiles, num_cosines, crossing_penalty, rope_type/tau_embed_type,
# sort_output, crossing_mode, softplus_beta, qvn_num_heads -- is left to
# run_backbone_hpo's own Optuna search (BaseFPQRConfig.BASE_FPQR_SEARCH /
# get_optuna_params), since the ablation only tested proposal_type/qvn_type
# in isolation and per-dataset/per-backbone HPO is better placed to pick the
# rest than a single fixed value copied across every condition (2026-09-27).
# MLPF's own config class locally defaults crossing_penalty/rope_type away
# from BaseFPQRConfig's own defaults (10.0/"linear" instead of 1.0/"tau_pos"
# -- see MLPFFPQRConfig's own docstring); this is no longer restored by an
# explicit override, so HPO's own search (which samples from
# BaseFPQRConfig's ablation-informed range, not MLPFFPQRConfig's class
# default) determines the final value, same as mlpgam/nhits already do.
# MLPGAM's feature/target gates default off already (MLPGAMConfig.use_gate/
# use_target_gate both default False) but are passed explicitly since they
# are architectural, not FPQR-head, hyperparameters and are out of scope for
# this tuning pass.
#
# proposal_type="softmax" and qvn_type="standard" are passed explicitly too
# (discovered 2026-09-26, not present before): the comment above used to
# claim no override was needed for either, on the reasoning that
# BaseFPQRConfig's own defaults already are "softmax"/"standard" -- true for
# a run with HPO skipped, but every backtesting condition here is built by
# layering condition.overrides (this dict) on top of run_backbone_hpo's own
# best_params (engine.py's "condition overrides always win" merge order),
# and run_backbone_hpo's Optuna search always samples both fields from its
# full default search space regardless of these overrides -- so without an
# explicit override here, the FINAL config's proposal_type/qvn_type is
# whatever HPO happened to search into, not BaseFPQRConfig's default.
# Confirmed directly: NHiTS's own HPO search for DSS-UK landed on
# proposal_type="horizon" and qvn_type="attention", both flagged by
# twiga/core/config/neural.py's own field docs as fpqr_gating_ablation.py-
# confirmed non-winners ("softmax wins; horizon worst"; "standard: best
# MAE") -- and every NHiTS condition on that dataset (none/mahalanobis/
# max_norm) showed a fully collapsed predicted quantile grid as a result
# (lower == upper, exactly, on every one of 24,192 rows), which in turn
# gave PICP=0.0 and cascaded into Experiment 3's decision Brier scores
# (nhits_max_norm/mahalanobis both hit Brier=1.0, the worst possible, on
# DSS-UK hosting-capacity). HPO's own objective is point-accuracy-only, so
# it never penalises the resulting quantile-spread collapse.
#
# NHiTS's per_target_tau already defaults to False, passed explicitly too,
# for symmetry across all three entries.
#
# NHiTS was briefly excluded (2026-09-22) after a live crash
# ("max_pool1d: Expected 2D or 3D... got [92, 1, 48, 3]") traced to
# twiga.models.nn.net.nhits.NHITSNetwork never actually supporting
# num_target_feature > 1 -- four inconsistent shape assumptions across the
# class. Fixed the same day: target channels now flow through as a real
# (B, L, D)/(B, H, D) axis end to end, pooled independently per channel the
# same way hist_exog/futr_exog already were (see nhits.py's own module
# history) -- verified against every existing single-target NHiTS test
# (unaffected, D=1 is just the general D case) plus a new multi-target
# forward/encode/backward/gradient-flow check.
#
# per_target_tau=True: P, Q, and V are physically distinct quantities (skewed
# active power, near-symmetric voltage, reactive power with a much smaller
# natural spread than either), so each target gets its own quantile grid
# rather than sharing one. A shared grid lets the pooled pinball loss collapse
# a low-variance target's spread toward a point estimate, which then poisons
# every pooled/simultaneous conformal condition: "scaled" CQR and max-norm's
# sigma-normalised score both divide by the target's own predicted
# width/scale, so a near-zero denominator on one target dominates the shared
# max-pooled threshold applied to every target's interval.
BACKBONES: dict[str, tuple[type, dict]] = {
    "mlpf": (
        MLPFFPQRConfig,
        {
            "per_target_tau": True,
            "proposal_type": "softmax",
            "qvn_type": "standard",
        },
    ),
    "mlpgam": (
        MLPGAMFPQRConfig,
        {
            "per_target_tau": True,
            "use_gate": False,
            "use_target_gate": False,
            "proposal_type": "softmax",
            "qvn_type": "standard",
        },
    ),
    "nhits": (
        NHITSFPQRConfig,
        {
            "per_target_tau": True,
            "proposal_type": "softmax",
            "qvn_type": "standard",
        },
    ),
}

# QR: a fixed-grid baseline head sharing the MLPF backbone, not a separate
# backbone architecture -- added alongside BACKBONES (2026-09-30) so this
# study's own conformal comparison covers a quantile-only head without
# FPQR's adaptive proposal, the same baseline role QR plays in
# multitarget_experiment.py's own Experiment 1. Kept out of BACKBONES itself
# since it needs different HPO wiring (see QR_HPO_VARIANT below), not
# because it is evaluated any differently once built: it still gets every
# entry in CONFORMAL_VARIANTS, same as mlpf/mlpgam/nhits.
QR_HEAD: dict[str, tuple[type, dict]] = {
    "qr": (MLPFQRConfig, {}),
}
# All heads this study builds conditions for -- BACKBONES plus QR.
HEADS: dict[str, tuple[type, dict]] = {**BACKBONES, **QR_HEAD}

# QR has no FPQR-specific hyperparameters at all (n_quantiles/proposal_type/
# qvn_type/per_target_tau don't exist on MLPFQRConfig), so giving it its own
# full joint search over MLPFFPQRConfig-style machinery would be pure waste
# -- there would be nothing FPQR-specific in that space for it to use. It
# needs only a well-tuned MLPF backbone, so it reuses a dedicated,
# backbone-only MLPFConfig search instead: the same SHARED_HPO_VARIANT
# pattern multitarget_experiment.py already uses for qr/bnn/laplace there,
# not a byte-for-byte reuse of mlpf's own FPQR joint search (that search's
# best_params carry FPQR-only fields MLPFQRConfig would reject outright).
QR_HPO_VARIANT = "shared_mlp_backbone"

# ── Conformal methods ────────────────────────────────────────────────────────
# "none" is the uncalibrated baseline.
# "max_norm" / "max_abs" are the two MultivariateConformal scores that do not
# use a correlation matrix: isotropic (sigma-normalised) and raw absolute
# error respectively.
# "mahalanobis_*" are the four dependence_method variants of the
# correlation-whitened MultivariateConformal score. All use the default
# dependence_shrink_lambda=0.05, which is the estimator's smoothing knob, not
# a modelling axis.
# "quantile_{scaled,unscaled}_{marginal,pooled,bonferroni}" is CQR in its
# 2 x 3 grid: score type x target_coverage mode. This covers the full
# ConformalQuantileRegressor surface for a quantile-only backbone.
#
# max_abs/mahalanobis_* dropped (2026-10-02), matching
# multitarget_experiment.py's own trim (commit 96b8602) and Chapter-6.tex's
# sec:ch6-mahalanobis: Mahalanobis (all four dependence_method estimators)
# gave a Winkler score 17-37% worse than max-norm across every probabilistic
# head on this chapter's own MLVS-PT-3PHASE run, with more severe
# over-coverage, and the four estimators were themselves near-identical to
# each other -- a fixed, calibration-set-estimated correlation does not
# reliably improve on max-norm's own implicit handling of dependence for
# this problem. max_abs was never separately justified; kept historically
# alongside mahalanobis as the other MultivariateConformal score that
# doesn't use a correlation matrix, not because its own numbers were
# distinguished from max_norm's. Only max_norm (the one multivariate score
# this chapter's analysis actually keeps) and the full CQR grid remain.
CONFORMAL_VARIANTS: dict[str, ConformalConfig | None] = {
    "none": None,
    "max_norm": ConformalConfig(method="multivariate", score_type="max-norm", alpha=0.1),
    # CQR, three target_coverage modes x two score types
    "quantile_scaled_marginal": ConformalConfig(
        method="quantile",
        score_type="scaled",
        alpha=0.1,
        target_coverage="marginal",
    ),
    "quantile_scaled_pooled": ConformalConfig(
        method="quantile",
        score_type="scaled",
        alpha=0.1,
        target_coverage="pooled",
    ),
    "quantile_scaled_bonferroni": ConformalConfig(
        method="quantile",
        score_type="scaled",
        alpha=0.1,
        target_coverage="bonferroni",
    ),
    "quantile_unscaled_marginal": ConformalConfig(
        method="quantile",
        score_type="unscaled",
        alpha=0.1,
        target_coverage="marginal",
    ),
    "quantile_unscaled_pooled": ConformalConfig(
        method="quantile",
        score_type="unscaled",
        alpha=0.1,
        target_coverage="pooled",
    ),
    "quantile_unscaled_bonferroni": ConformalConfig(
        method="quantile",
        score_type="unscaled",
        alpha=0.1,
        target_coverage="bonferroni",
    ),
}
# Conformal-bearing conditions, skipped on DSS-EU (cv_calib_size=0 there --
# see module docstring), run everywhere else. Used as a suffix tuple by the
# DSS-EU filter in main().
CALIBRATED_CONFORMAL_VARIANTS = tuple(name for name in CONFORMAL_VARIANTS if name != "none")


def _build_conditions(variants: tuple[str, ...] | None = None) -> list[Condition]:
    """3 FPQR backbones + QR x (up to 13) conformal variants, group="backbone".

    ``variants`` restricts the conformal-variant axis to a subset of
    ``CONFORMAL_VARIANTS``'s keys. ``None`` (the default) uses every variant,
    the full 4 x 13 sweep.

    ``hpo_fixed_params`` pins ``proposal_type``/``qvn_type``/``per_target_tau``
    inside the Optuna search itself (not just on the final config via
    ``overrides``), so HPO's own conditional crossing-group params
    (crossing_penalty, sort_output, crossing_mode, softplus_beta) get sampled
    under the "standard" branch that will actually be used, rather than under
    whatever ``qvn_type`` a free, unconstrained draw happened to land on
    (2026-09-27). ``per_target_tau`` added to this pinned set 2026-09-29,
    after ``MLPFFPQRConfig.search_space`` was extended (chapter-6-
    multitarget's own Experiment 1 fix) to include ``qvn_type``/
    ``per_target_tau``/``gradient_clip_val`` and widen ``n_quantiles``/
    ``num_cosines`` -- without this, per_target_tau would now be sampled
    freely during HPO (every other backbone hyperparameter tuned for
    whatever draw HPO happened to make) and then silently forced to the
    True this dict's own ``overrides`` sets afterward, so the tuned
    backbone would not actually match the per_target_tau value the model
    is really built with. ``hpo_fixed`` is empty for ``qr`` (none of those
    three fields exist on ``MLPFQRConfig``), which is harmless -- an empty
    dict pins nothing.

    QR (2026-09-30) gets its own ``hpo_model_cls``/``hpo_variant`` rather
    than the implicit default (``cond.model_cls``, i.e. ``MLPFQRConfig``,
    under ``hpo_variant=""``) every FPQR backbone here relies on: see
    ``QR_HPO_VARIANT``'s own comment for why it shares a dedicated
    backbone-only search instead of paying for a full joint one.
    """
    allowed = set(variants) if variants is not None else set(CONFORMAL_VARIANTS)
    conditions = []
    for backbone_name, (model_cls, overrides) in HEADS.items():
        hpo_fixed = {k: overrides[k] for k in ("proposal_type", "qvn_type", "per_target_tau") if k in overrides}
        if backbone_name == "qr":
            hpo_model_cls, hpo_variant = MLPFConfig, QR_HPO_VARIANT
            description_prefix = "QR (fixed-grid baseline)"
        else:
            hpo_model_cls, hpo_variant = model_cls, ""
            description_prefix = f"{backbone_name.upper()} FPQR (simple config)"
        for conformal_name, conformal_config in CONFORMAL_VARIANTS.items():
            if conformal_name not in allowed:
                continue
            conditions.append(
                Condition(
                    name=f"{backbone_name}_{conformal_name}",
                    group="backbone",
                    description=f"{description_prefix}, conformal={conformal_name}",
                    overrides=overrides,
                    model_cls=model_cls,
                    hpo_model_cls=hpo_model_cls,
                    hpo_variant=hpo_variant,
                    hpo_fixed_params=hpo_fixed,
                    metric_types=_BASE_METRICS if conformal_config is None else _CONFORMAL_METRICS,
                    conformal_config=conformal_config,
                )
            )
    return conditions


# Full sweep, for the module-level constants and the reference_conditions
# spec. main() rebuilds the condition list per run to honour --variants.
CONDITIONS: list[Condition] = _build_conditions()

_SPEC_COMMON: dict = {
    "name": "Backbone Conformal Study",
    "output_prefix": "ch6_backbone_conformal",
    "condition_cls": MLPGAMFPQRConfig,
    "backbone_cls": MLPGAMFPQRConfig,
    "conditions": CONDITIONS,
    "cv_train_size": 12,
    "cv_test_size": 4,
    "cv_val_size": 2,
    "cv_calib_size": 0,
    "cv_folds": 10,
    # num_workers=0: the default of 8 forks DataLoader workers after the
    # MPS/Metal backend is already initialised -- observed hanging
    # indefinitely (71+ min, near-zero CPU progress, unresponsive to
    # `timeout`) during verification, the same class of issue fixed in
    # backbone_conformal_decision.py's fixed_overrides for the same reason.
    "fixed_overrides": {"early_stop_patience": 100, "num_workers": 0},
    "reference_conditions": {"backbone": "mlpgam_none"},
}

# DSS-EU: identical CV numbers to multitarget_experiment.py's own
# SPEC_DSS_EU -- train=5/val=1/test=1 months, rolling, stride 1,
# cv_train_size_mode="fixed": 5 folds, each fitting on 4 months (2026-10-06;
# previously train=4 under the default "derived" mode, 8 folds fitting on
# about 30 days). cv_calib_size stays 0 (same reasoning), so its conformal
# conditions are skipped entirely by main() rather than run uncalibrated.
SPEC_DSS_EU = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_train_size": 5, "cv_val_size": 1, "cv_test_size": 1, "cv_calib_size": 0},
    datasets={DSS_EU: DATASETS[DSS_EU]},
    cv_stride=1,
    cv_window="rolling",
    cv_train_size_mode="fixed",
    hpo_train_months=8,
    hpo_val_months=2,
)
# IMPORTANT, discovered 2026-09-24: ExperimentEngine always builds this
# spec's TimeBasedCV with num_splits=spec.cv_folds set (engine.py's own
# ExperimentConfig(..., num_splits=spec.cv_folds, ...), both call sites,
# unconditionally). Whenever num_splits is set, TimeBasedCV.set_split_scheme
# **recomputes train_size from scratch** --
# train_size = total_duration - test_size - gap - (num_splits - 1) * stride
# -- and discards whatever cv_train_size says entirely (confirmed by reading
# set_split_scheme's own source, then verified against the real files: a
# spec with cv_train_size=18 but cv_folds=10 actually ran at a derived
# train_size of 10, not 18). So cv_train_size below is documentation-only,
# not a real lever, whenever cv_folds is also set -- which it always is.
# The two real levers for the training-window size are cv_folds (fold
# count) and cv_test_size/cv_stride (via the formula above); every comment
# below states the *derived* train_size and verifies it directly against
# TimeBasedCV.split(num_splits=cv_folds), not against cv_train_size's own
# value. An earlier version of these comments (now corrected) reasoned
# entirely in terms of cv_train_size as if it were honored, which was wrong.
# Update 2026-10-06: ExperimentSpec.cv_train_size_mode="fixed" now makes
# cv_train_size a real lever (kept as given, cv_folds only caps the count);
# DSS-EU and DSS-UK/DSS-US use it. MLVS-PT-3PHASE keeps "derived": its
# folds fit on 15-24 months either way, identical in both modes.
#
# MLVS-PT-3PHASE: the protocol stated in Chapter-6.tex (rolling-origin paragraph) and
# shared with Chapter 7 (experiment/ch6_conformal/shared/protocol.py): cv_folds=10,
# cv_test_size=4, cv_stride=1, cv_val_size=2, cv_calib_size=3, expanding window. Derived
# train_size = total - test - (folds - 1) * stride, so the first fold fits on at least twelve
# months after the validation and calibration carve-outs. This replaces an earlier
# test=3/stride=2/calib=2 configuration that contradicted the chapter text. The record is
# truncated at 2021-12-31 by load_dataset, the same end date as Chapter 7's MLVS-PT.
SPEC_MLVS_PT_3PHASE = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_train_size": 12, "cv_test_size": 4, "cv_val_size": 2, "cv_calib_size": 3},
    datasets={MLVS_PT_3PHASE: DATASETS[MLVS_PT_3PHASE]},
    cv_stride=1,
    cv_window="expanding",
    hpo_train_months=12,
    # hpo_val_months raised 2->6 (2026-09-29), same fix and same reasoning as
    # multitarget_experiment.py's own SPEC_MLVS_PT_3PHASE: cwe_stratified
    # needs ~50 samples in its largest lead-time bucket (n_strata_bins=10
    # over a 48-step horizon), which the 2-month default wasn't reliably
    # clearing during HPO trials on this same dataset. 12+6=18 months from
    # train_start=2019-03 (ending ~2020-09) still sits comfortably before
    # this script's own test_start; MLVS-PT-3PHASE's 37.5-month record has
    # the same headroom here it had for Experiment 1.
    hpo_val_months=6,
)

# DSS-UK/DSS-US: cv_folds=6 (not _SPEC_COMMON's inherited 10 -- an explicit
# override here is required, since cv_folds is the real lever, see the note
# above), cv_test_size=cv_stride=4 weeks (zero overlap) -- derived train_size
# = 50 - 4 - 5*4 = 26 weeks, giving a real fit window of 26 - val4 - calib4 =
# 18 weeks (~4.1 months, this dataset's own train-size ceiling) on fold 0
# (confirmed: 125 real days on both files). fold 0's test window starts
# 2019-07; genuine Q2 test start and a ~4-month real train window are not
# jointly achievable on this ~50-week record once val/calib carve-out is
# accounted for correctly. cv_window="expanding" matches Chapter 7's protocol:
# train_start is fixed at the record start, so the real fit window grows fold
# over fold (18 weeks on fold 0, up to 18 + 5*4 = 38 weeks on fold 5) rather
# than a fixed 18 weeks throughout. This trades away the season/training-size
# isolation a fixed-width window would give (noted as a limitation) in favour
# of matching Chapter 6's own stated protocol. hpo_train_months=4/
# hpo_val_months=1:
# both records run 2019-01-01..2019-12-31 (~12 months, confirmed against
# the parquet files), so run_backbone_hpo's own fixed calendar split from
# train_start='2019-01' needs enough margin within that span; 4/1 (5
# months) fits comfortably, unaffected by the cv_folds fix above (hpo's own
# split is independent of TimeBasedCV entirely).
_DSS_WEEKS_COMMON = {
    **_SPEC_COMMON,
    "cv_split_freq": "weeks",
    "cv_train_size": 26,
    "cv_test_size": 4,
    "cv_val_size": 4,
    "cv_calib_size": 4,
    "cv_folds": 6,
    # "fixed" keeps the 26-week window as specified. Under "derived", DSS-UK
    # (whose record starts 2019-01-29) got 24 weeks, fold 0 fitting on 111
    # days, not the 125 stated above; it now gets 5 folds of the stated
    # window. DSS-US is identical in both modes (6 folds, 125 days on fold 0).
    "cv_train_size_mode": "fixed",
    "hpo_train_months": 4,
    "hpo_val_months": 1,
}
SPEC_DSS_UK = ExperimentSpec(
    **_DSS_WEEKS_COMMON,
    datasets={DSS_UK: DATASETS[DSS_UK]},
    cv_stride=4,
    cv_window="expanding",
)
SPEC_DSS_US = ExperimentSpec(
    **_DSS_WEEKS_COMMON,
    datasets={DSS_US: DATASETS[DSS_US]},
    cv_stride=4,
    cv_window="expanding",
)

# Specs are kept in a dict rather than pre-built engines -- main() builds a
# fresh engine per (dataset, variant-restriction) pair so a filtered
# condition list never mutates the module-level singleton (see main()).
SPECS: dict[str, ExperimentSpec] = {
    DSS_EU: SPEC_DSS_EU,
    MLVS_PT_3PHASE: SPEC_MLVS_PT_3PHASE,
    DSS_UK: SPEC_DSS_UK,
    DSS_US: SPEC_DSS_US,
}


def base_config(
    epochs: int,
    hpo_trials: int = 50,
    hpo_epochs: int = 100,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PipelineConfig:
    """Shared PipelineConfig defaults for this script's specs; dataset overridden per run.

    Args:
        epochs: Training budget for the final (already-tuned) conditions.
        hpo_trials: Optuna trials per HPO run.
        hpo_epochs: Training budget per Optuna trial, distinct from ``epochs``
            (the final, already-tuned conditions' budget); matches
            ``backbone_conformal_decision.py``'s own ``--hpo-epochs`` convention.
        output_root: Directory under which this run's artifacts and HPO
            caches are written. Defaults to ``DEFAULT_OUTPUT_ROOT``, which
            itself respects ``TWIGA_EXPERIMENTS_ROOT`` when set.

    Returns:
        A :class:`PipelineConfig` with this script's defaults.
    """
    return PipelineConfig(
        dataset_name=DSS_EU,
        data_path="examples/data",
        output_dir=str(output_root / "backbone_conformal_study"),
        epochs=epochs,
        num_trials=hpo_trials,
        hpo_epochs=hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )


# ── Run discovery ─────────────────────────────────────────────────────────────
# Mirrors fpqr_gating_ablation.py's discover_runs()/latest_run_for_group() unchanged.
RESULTS_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "backbone_conformal_study"


def discover_runs() -> list[dict]:
    """Every synced run under ``RESULTS_ROOT``, read from its own ``meta.json``."""
    runs = []
    for meta_path in sorted(RESULTS_ROOT.glob("*/runs/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        runs.append(
            {
                "run_dir": meta_path.parent,
                "timestamp": meta["timestamp"],
                "groups": meta["groups"],
                "datasets": meta["datasets"],
            }
        )
    return runs


def latest_run_for_group(runs: list[dict], group: str) -> dict | None:
    """The most recently timestamped run (of ``runs``) that covers ``group``, or ``None``."""
    candidates = [run for run in runs if group in run["groups"]]
    return max(candidates, key=lambda run: run["timestamp"]) if candidates else None


def significance_report(group: str = "backbone", alpha: float = 0.05) -> None:
    """Print a Winkler-score DM/MCS significance table for one condition group.

    Reuses ``experiment.common.significance`` per that chapter's own rule:
    CWE is a diagnostic and never decides a comparison, only the Winkler
    score (a proper scoring rule) does. Mirrors ``fpqr_gating_ablation.py``'s
    own ``significance_report`` exactly.

    Reads the *latest* synced run that covers ``group`` (see run-discovery
    section above), not a fixed path -- a re-run of one dataset alone is
    picked up automatically.
    """
    from experiment.common.significance import mcs_ranking, pairwise_dm_table  # noqa: PLC0415

    run = latest_run_for_group(discover_runs(), group)
    if run is None:
        log.warning("No synced run under %s covers group=%s", RESULTS_ROOT, group)
        return
    full_csv = run["run_dir"] / "results" / "ch6_backbone_conformal_full.csv"
    if not full_csv.exists():
        log.warning("group=%s: %s not found", group, full_csv)
        return
    log.info("[run] group=%s <- %s  (%s)", group, run["run_dir"], run["timestamp"])

    df = pd.read_csv(full_csv)
    group_df = df[df["group"] == group] if "group" in df.columns else df
    if "winkler_score" not in group_df.columns:
        log.warning("No winkler_score column in %s — skipping significance report", full_csv)
        return

    for h in sorted(group_df["h"].unique()) if "h" in group_df.columns else [None]:
        h_df = group_df if h is None else group_df[group_df["h"] == h]
        losses = h_df.pivot_table(index="Folds", columns="condition", values="winkler_score")
        print(f"\n=== group={group} h={h} — DM table (Winkler score) ===")
        print(pairwise_dm_table(losses.to_dict(orient="series"), alpha=alpha))
        print(f"\n=== group={group} h={h} — Model Confidence Set (Winkler score) ===")
        print(mcs_ranking(losses.to_dict(orient="series"), alpha=alpha))


def main() -> None:
    """CLI entry point: parse args, run each requested dataset, write the combined summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="*", default=["backbone"], choices=["backbone"])
    ap.add_argument(
        "--dataset",
        nargs="*",
        default=[MLVS_PT_3PHASE],
        choices=[*DATASETS, "all"],
        help="dataset keys, or 'all' for every dataset in DATASETS",
    )
    ap.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=list(CONFORMAL_VARIANTS),
        help=(
            "restrict to a subset of conformal variants (default: all). Useful for chunked "
            "sweeps of one family, e.g. --variants mahalanobis_pearson mahalanobis_spearman "
            "mahalanobis_kendall mahalanobis_gaussian_copula. The uncalibrated baseline "
            "'none' is never skipped implicitly: include it in the list if a family needs a "
            "reference, e.g. --variants none mahalanobis_gaussian_copula."
        ),
    )
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="short run for a smoke test")
    ap.add_argument(
        "--skip-hpo",
        dest="skip_hpo",
        action="store_true",
        default=True,
        help="reuse cached backbone params instead of searching (default: True, unlike this "
        "script's own behaviour before 2026-09-24 -- unified with multitarget_experiment.py's/"
        "backbone_conformal_decision.py's convention, skip by default, override with --run-hpo).",
    )
    ap.add_argument("--run-hpo", dest="skip_hpo", action="store_false", help="run real backbone HPO")
    ap.add_argument("--hpo-trials", type=int, default=50)
    ap.add_argument("--hpo-epochs", type=int, default=100, help="epochs per HPO trial, not per final condition")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root directory for experiment artifacts and HPO caches",
    )
    ap.add_argument(
        "--hpo-objective",
        default="winkler_score",
        help="HPO objective for every condition here (all FPQR-family, quantile-producing). "
        "'winkler_score' (default) or any name get_quantile_metrics computes, e.g. the "
        "diagnostic 'cwe_stratified' -- both derived from the outermost predicted quantile "
        "pair, no calibration needed.",
    )
    ap.add_argument("--report", metavar="GROUP", help="print a significance report for one group and exit")
    args = ap.parse_args()

    if args.report:
        significance_report(args.report)
        return

    # Rebuild the condition list for this run's variant filter, then assign the
    # objective metric. The spec's own "conditions" field is not mutated --
    # main() constructs a fresh ExperimentEngine per (dataset, variant) pair
    # below, so the module-level SPECS dict stays untouched.
    run_conditions = _build_conditions(variants=tuple(args.variants) if args.variants is not None else None)
    for cond in run_conditions:
        cond.hpo_objective_metric = args.hpo_objective

    configure(level="INFO")
    groups = args.group
    dataset_keys = [MLVS_PT_3PHASE] if args.quick else (list(DATASETS) if "all" in args.dataset else args.dataset)
    cfg = base_config(
        5 if args.quick else args.epochs,
        hpo_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        output_root=args.output_root,
    )
    if args.quick:
        cfg = dc_replace(cfg, num_trials=1, hpo_epochs=5)

    summaries = []
    for dataset_key in dataset_keys:
        conditions = run_conditions
        if dataset_key == DSS_EU:
            # cv_calib_size=0 on DSS-EU (see module docstring) -- the
            # conformal-bearing variants would run with calib_idx=None every
            # fold, calibrating against nothing, same skip pattern
            # multitarget_experiment.py already uses for its own conformal
            # group there.
            conditions = [c for c in conditions if not c.name.endswith(CALIBRATED_CONFORMAL_VARIANTS)]
            log.info(
                "Skipping conformal-bearing conditions on %s -- see SPEC_DSS_EU's own cv_calib_size=0",
                dataset_key,
            )
        # Fresh engine per dataset: build from the module-level spec but with
        # this run's filtered condition list, so no shared state leaks between
        # datasets in the same process.
        spec = dc_replace(SPECS[dataset_key], conditions=conditions)
        engine = ExperimentEngine(spec)
        summaries.append(
            engine.run(
                base_cfg=cfg,
                groups=groups,
                dataset_keys=[dataset_key],
                skip_hpo=args.skip_hpo or args.quick,
                tracking_uri=None,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    out = script_dir / "results_backbone_conformal_study.csv"
    summary.to_csv(out, index=False)
    log.info("summary written to %s", out)
    print(summary.to_string())


if __name__ == "__main__":
    main()
