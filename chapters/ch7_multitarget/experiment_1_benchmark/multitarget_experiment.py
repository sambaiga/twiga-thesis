"""Chapter 6, Experiment 1: the multi-target probabilistic forecasting benchmark.

Runs through :class:`~twiga.experiment.ExperimentEngine`, the same machinery as
``experiment/ch6_conformal/crc_experiment.py``, so the backtesting protocol, controlled fields
and reference conditions are shared with the rest of the thesis rather than
hand-rolled.

The question
------------
``Chapter-6.tex`` (``subsec:ch6-protocol``, ``chap6:exp-one``) names this
experiment directly: "a multi-target probabilistic forecasting benchmark by
comparing FPQR against QR, BNN, and Laplace forecasting models using a common
MLPF backbone... conducted using the MLVS-PT and DSS-EU datasets." The
``benchmark`` group below is exactly that comparison, no more and no less.
Every condition sits on the same ``MLPFConfig`` backbone and fits each of
active power, reactive power and busbar voltage as an independent margin
(none of QR, BNN or Laplace has a native multi-target extension), so the
comparison is about the per-target distributional method, not about
dependence modelling -- copula-based dependence modelling is dropped from
this chapter entirely (see ``README.md``'s Objective); Experiment 2
(``backbone_conformal_study.py``) is where the dependence-free FPQR is
evaluated across backbones and conformal calibration methods instead.

A second group, ``grid``, is a supporting mechanism check, not part of the
chapter's named Experiment 1: whether FPQR's quantile grid should be shared
across targets or learned independently per target. It stays here because it
answers a question *about* the FPQR condition already running in this
script, on the same folds, rather than because the chapter names it as one
of the three studies.

A third group, ``conformal``, demonstrates ``Chapter-6.tex``'s
(``subsec:ch6-evaluation``) multivariate adaptive conformal calibration
(``sec:ch6-conformal``), which pools a non-conformity score across the
target axis for simultaneous coverage rather than per-target coverage
(:class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`, now
reachable via ``ConformalConfig(method="multivariate", ...)``). It is not
applied to the ``benchmark`` group above: ``ExperimentEngine``'s calibrated
path only supports ``metric_types <= {"point", "interval"}``
(the same constraint ``crc_experiment.py``'s own conformal conditions
accept), which would drop the proper-score reporting the benchmark
comparison needs. Laplace and BNN are the two benchmark methods with a
native scale head, so ``score_type="max-norm"`` applies to both.

Metrics
-------
Every condition reports a proper scoring rule alongside the diagnostics, because
Chapter 3 argues CWE is a diagnostic rather than a proper score and so cannot on
its own support a claim that one model beats another.

  CRPS            proper score; from the parametric evaluator for the Gaussian/
                  Laplace/BNN heads and the quantile evaluator for QR/FPQR
  Winkler score   proper score for a central interval
  CWE             marginal coverage-width-error, the Chapter 3 diagnostic
  CWE stratified  the same diagnostic scored *within* lead-time strata and
                  averaged afterwards. Marginal CWE compares aggregate width
                  against the reference, so a forecast with one constant width
                  can match both marginal coverage and marginal width while
                  over-covering at short lead times and under-covering at long
                  ones. Stratifying first stops those errors cancelling, which
                  is the failure mode a horizon forecast is most prone to.
  PICP, nMPI      coverage and normalised width, reported as the two components
                  the CWE scores combine, so a change in CWE can be attributed

Higher is better for CWE, stratified CWE and PICP; lower is better for CRPS and
the Winkler score. Significance between conditions rests on the Winkler score
via ``twiga.core.metrics``'s ``pairwise_dm_table``/``mcs_ranking``,
never on CWE, per that module's own rule and ``Chapter-6.tex``'s
(``subsec:ch6-metrics``) explicit adoption of it.

Protocol
--------
``Chapter-6.tex`` (``subsec:ch6-evaluation``) states this "reuses Chapter 7's...
design unchanged": ten folds, an expanding training set of at least twelve
months in the first fold, a four-month test window, a two-month validation
set, and a three-month calibration set (Chapter 6 applies multivariate
conformal calibration on top of every backbone). ``ExperimentSpec``'s
defaults already give the expanding window; the sizes are set explicitly in
``_SPEC_COMMON`` below to match.

Stride is no longer shared between the two datasets (split 2026-09-17):
DSS-EU (~11 months of synthetic power-flow data) is too short for a literal
one-month stride at this protocol's other sizes and keeps its own,
still-open gap (2 folds -- see ``SPEC_DSS_EU``'s own comment for why).
MLVS-PT-3PHASE (a real 37.5-month record) uses the chapter's literal
``cv_stride=1`` directly (restored 2026-09-27, after a since-reverted detour
through ``cv_stride=3``/``cv_window="rolling"`` -- see ``SPEC_MLVS_PT_3PHASE``'s
own comment): with ``cv_folds=10`` set, ``TimeBasedCV`` derives ``train_size``
from the leftover budget rather than sliding the full 37.5-month span one
month at a time, so this gives exactly 10 folds (dry-run confirmed against
the real file), each a genuine 4-month, one-month-apart, expanding-window
fold -- matching ``Chapter-6.tex``'s stated protocol literally, not only in
spirit.

Usage::

    uv run python experiment/ch7_multitarget/multitarget_experiment.py --group benchmark
    uv run python experiment/ch7_multitarget/multitarget_experiment.py --quick
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import os
from pathlib import Path

import numpy as np
import pandas as pd
from twiga.core.config import ConformalConfig
from twiga.core.utils import configure, get_logger
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.experiment.hpo import HPOCachePolicy
from twiga.models.nn import MLPFConfig
from twiga.models.nn.mlpfbnn_model import MLPFBNNConfig
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig
from twiga.models.nn.mlpfqr_model import MLPFQRConfig
from twiga.utils.run_forecast import PipelineConfig

from twiga_thesis.paths import thesis_data_dir

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
# parents[3]: experiment_1_benchmark -> ch7_multitarget -> experiment ->
# twiga-forecast (repo root) -> projects -- matches every other Chapter 6
# script's own output_dir convention (e.g. backbone_conformal_study.py's
# script_dir.parent.parent.parent.parent), a sibling of the repo, not inside it.
# parents[2] here previously landed one level too shallow (inside the repo),
# so every run since --output-root/TWIGA_EXPERIMENTS_ROOT was added wrote to
# a location none of the sync/analysis tooling or other scripts ever read
# from (2026-09-28 fix).
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[3] / "experiments"))) / "ch7_multitarget"
)
SHARED_HPO_VARIANT = "shared_mlp"
# The "grid" group's two conditions exist specifically to compare
# per_target_tau=False vs True in a controlled way, so each gets its own,
# separate joint-search cache (distinct hpo_variant) with per_target_tau
# pinned via hpo_fixed_params -- pinning it *inside* the Optuna search itself
# (Condition.hpo_fixed_params's documented purpose), not layered on
# afterward via overrides, so every other FPQR field is genuinely tuned under
# that fixed choice rather than tuned for an unpinned default and then
# relabelled.
FPQR_GRID_SHARED_HPO_VARIANT = "fpqr_grid_shared"
FPQR_GRID_PER_TARGET_HPO_VARIANT = "fpqr_grid_per_target"
# Same reasoning as the two above: "grid" group condition, own joint-search
# cache, with proposal_anchor_outer=True pinned via hpo_fixed_params so every
# other field is tuned under that fixed choice. proposal_type="softmax" is
# pinned alongside it because FPQRDistribution rejects
# proposal_anchor_outer=True under any other proposal_type (twiga/distributions
# /nn/fpqr/distribution.py) -- leaving proposal_type free to search would let
# Optuna sample "kumaraswamy"/"horizon" on a trial that still carries
# anchor_outer=True from this pin, crashing that trial outright rather than
# just wasting its budget. per_target_tau=True matches the main "fpqr_anchor_outer"
# condition's own pin, so this varies only the anchoring mechanism relative
# to that condition, not also the grid-sharing axis the "grid" group's other
# two conditions are already isolating (2026-09-30).
FPQR_ANCHOR_OUTER_HPO_VARIANT = "fpqr_anchor_outer"
# Own cache key for "fpqr_anchor_outer_v3_fixed" below -- deliberately NOT
# FPQR_ANCHOR_OUTER_HPO_VARIANT, since that condition's own hpo_fixed_params
# (fully free head search) would conflict with this one's (every head field
# pinned): Condition.hpo_fixed_params is merged per (model_cls, hpo_variant)
# key, last condition sharing a key wins, so two conditions that disagree
# cannot share one. Still shares hpo_parent_model_cls=MLPFConfig/
# hpo_parent_variant=SHARED_HPO_VARIANT with every other benchmark-group
# head, so a backbone-only HPO rerun (--run-hpo) affects this condition too.
FPQR_ANCHOR_OUTER_V3_FIXED_HPO_VARIANT = "fpqr_anchor_outer_v3_fixed"
# "grid" group condition, own joint-search cache, with data_overrides pinning
# LV to be forecast as its own first difference (reconstructed to level at
# evaluation -- twiga.core.data.pipeline.DataPipeline.target_differencing /
# twiga.forecaster.base.BaseForecaster._undifference_predictions, 2026-10-01).
# LV's own SignalCharacteriser verdict on MLVS-PT-3PHASE is fractional
# integration (d=1): its monthly mean drifts by ~3.5V against a ~1.8V total
# std over the record, comparable to the whole signal's own variance, which
# every benchmark condition's raw-level LV forecast shows as a shared,
# architecture-independent bias (see Chapter-6.tex's own voltage-difficulty
# discussion). data_overrides (not overrides, which only reaches the model
# config) is required here since target_differencing lives on
# DataPipelineConfig, a separate object built fresh per condition.
FPQR_LV_DIFFERENCED_HPO_VARIANT = "fpqr_lv_differenced"

# Stability pins, restored 2026-10-02: briefly emptied on the mistaken belief
# that the 96b860 run (the "v4" baseline, whose own first HPO trial scored
# cwe_stratified=0.5 against a post-pin rerun's 0.17-0.19 on a comparable
# first draw) predated these pins. It did not -- 96b8602 is a descendant of
# 967119f (the commit that introduced this dict), so v4's own run already had
# qvn_type/proposal_loss_fn/proposal_eps pinned exactly as below; emptying
# the dict was a real divergence from v4, not a restoration of it. Restored
# to v4's actual values. The genuine, still-untested differences between v4
# and the current search space are the entropy_weight floor (0.1 -> 0.25),
# the loss_fn restriction (winkler dropped), and the val_mae/val_loss monitor
# fix in hpo.py -- none of which this dict touches.
_FPQR_STABILITY_PINS: dict = {
    "qvn_type": "monotone_cal",
    "proposal_loss_fn": "wasserstein",
    "proposal_eps": 1e-4,
}

# Training-level pins for the shared MLPF backbone and fpqr_anchor_outer
# (re-added 2026-10-03), sourced from PhD-Thesis-Anthony's own reference
# single-target FPQR (src/utils/hparams.py's PT_params/UK_params/hparams
# dicts, all agreeing; src/net/model.py's configure_optimizers hardcodes
# Adam + MultiStepLR unconditionally, no other optimizer/scheduler ever
# used there). Requires the hpo_frozen_fields mechanism
# (twiga/core/config/neural.py, NeuralModelConfig.sample_training_params)
# to actually take effect -- optimizer_params/scheduler_params are only
# genuinely skipped from sampling when named in hpo_frozen_fields, which
# hpo_fixed_params' own keys populate automatically. Applied to
# fpqr_anchor_outer and to qr/bnn/laplace -- those three are the only
# benchmark-group conditions whose own (hpo_model_cls, hpo_variant) key
# *is* (MLPFConfig, SHARED_HPO_VARIANT), so pinning here fixes the shared
# MLPF backbone's own HPO phase, not just the heads that later borrow its
# cached result via hpo_parent_model_cls. Backbone architecture dims
# (hidden_dim, num_layers, embedding_size, ...) stay genuinely free --
# nothing here touches model_config.search_space, only the training-level
# fields sample_training_params controls.
#   - optimizer_type/batch_size/optimizer_params: reference's own
#     learning_rate=1e-3, weight_decay=1e-6, batch_size=64, Adam only.
#   - lr_scheduler_type/scheduler_params: reference's own
#     MultiStepLR(milestones=[0.75,0.9]*max_epochs, gamma=0.1) --
#     twiga's own "multi_step" space parameterises the same two
#     milestones as prob_decay_1/prob_decay_2 fractions of max_epochs,
#     so 0.75/0.9 transfer directly; "milestones" itself stays empty
#     (twiga derives it from prob_decay_1/prob_decay_2, not a fixed list).
#   - gradient_clip_val: NOT taken from the reference (0.0507, untested
#     here and considerably more aggressive than twiga's own class
#     default) -- pinned to 1.0 instead, matching MLPFFPQRConfig's own
#     documented "stabilise FPQR training" default.
#
# Note this was briefly removed on 2026-10-03 (commit 02453de) on the
# theory that the two best benchmark-group FPQR results on record
# (commits ed3c1b/52cfe3e, WS~19.9/CWE~0.80) predate 0434f50
# ("hpo_fixed_params never reached the built model config") and so had
# no training-level pin actually applied -- re-added the same day per
# explicit instruction to fix these hyperparameters for both the FPQR
# head and the shared MLPF backbone regardless of that historical-match
# consideration.
_REFERENCE_TRAINING_PINS: dict = {
    "optimizer_type": "adam",
    "lr_scheduler_type": "multi_step",
    "batch_size": 64,
    "optimizer_params": {"lr": 1e-3, "weight_decay": 1e-6},
    "scheduler_params": {"gamma": 0.1, "prob_decay_1": 0.75, "prob_decay_2": 0.9, "milestones": []},
    "gradient_clip_val": 1.0,
}

# fpqr_anchor_outer's OWN training-level pins (2026-10-04), from its actual
# HPO-found best_params.json at commit 904e94 ("v3") --
# mlpffpqr_fpqr_anchor_outer_best_params.json, fingerprint
# b4e2f5e8da33937bf63bb3a4bb171d61fe83282280e5c9fc1b509b1681dc7a63 -- not
# _REFERENCE_TRAINING_PINS, which fpqr_anchor_outer never actually trained
# under: v3 predates that constant entirely, and its own training config
# (warmup_cosine, not multi_step; opt_lr~9.7e-4, not 1e-3) was itself a
# free HPO search result, not a hand-set reference value. Pinning
# _REFERENCE_TRAINING_PINS here instead would not reproduce v3's own
# WS=18.61/PICP=0.920 result, since it forces a scheduler v3 never used.
# fpqr_anchor_outer has its own (hpo_model_cls, hpo_variant) cache key
# (FPQR_ANCHOR_OUTER_HPO_VARIANT), separate from SHARED_HPO_VARIANT, so
# this does not affect qr/bnn/laplace's own shared-backbone training pins.
_FPQR_ANCHOR_OUTER_TRAINING_PINS: dict = {
    "optimizer_type": "adam",
    "lr_scheduler_type": "warmup_cosine",
    "batch_size": 64,
    "optimizer_params": {"lr": 9.718319944817398e-4, "weight_decay": 3.6999724314638066e-06},
    "scheduler_params": {"warmup_epochs": 3, "eta_min": 1.1557352816269861e-07},
    "gradient_clip_val": 0.560088399988946,
}

# fpqr_anchor_outer's own head-level hyperparameters (2026-10-04), also from
# v3's mlpffpqr_fpqr_anchor_outer_best_params.json -- the six fields that
# stayed genuinely free under "fpqr_anchor_outer"'s own hpo_fixed_params
# (architecture + training pins only). Used exclusively by
# "fpqr_anchor_outer_v3_fixed" below, which pins every one of FPQR's own
# fields to v3's exact result so that re-running --run-hpo only re-tunes the
# shared MLPF backbone (hpo_parent_model_cls) -- the head itself is frozen,
# not re-searched. "fpqr_anchor_outer" itself is untouched by this and keeps
# searching these six fields fresh.
_FPQR_ANCHOR_OUTER_V3_HEAD_PINS: dict = {
    "qvn_type": "monotone_cal",
    "n_quantiles": 16,
    "loss_fn": "pinball",
    "num_cosines": 8,
    "entropy_weight": 1.3780336485751996,
    "proposal_eps": 1.5843250684388674e-05,
}

# Best cwe_stratified trial on record for fpqr_anchor_outer (0.68847,
# shared v3/v4 Optuna journal's own trial 21 -- current best is 0.595,
# commit dcece6d's trial 0; this one is better and verified directly via
# optuna.load_study against that journal file, 2026-10-03). Raw Optuna
# trial.params, enqueued as hpo_initial_params (a seed, not a pin -- see
# Condition.hpo_initial_params's own docstring). proposal_type/
# proposal_anchor_outer/per_target_tau/proposal_loss_fn are absent because
# they were pinned (not searched) in whatever code state produced this
# trial, so this seeded trial samples them fresh under the current, fully
# free search space; nothing here constrains any other trial.
_FPQR_ANCHOR_OUTER_SEED: dict = {
    "mlpffpqr_use_bias": True,
    "mlpffpqr_out_activation_function": "Identity",
    "mlpffpqr_n_quantiles": 88,
    "mlpffpqr_loss_fn": "pinball",
    "mlpffpqr_num_cosines": 92,
    "mlpffpqr_tau_embed_type": "ConvEmb",
    "mlpffpqr_entropy_weight": 0.2785940931049672,
    "mlpffpqr_proposal_eps": 0.00478642127162763,
    "mlpffpqr_qvn_type": "umnn",
    "mlpffpqr_gradient_clip_val": 2.8725931347135414,
    "optimizer_type": "adam",
    "lr_scheduler_type": "multi_step",
    "batch_size": 64,
    "opt_lr": 0.0011955832199332366,
    "opt_weight_decay": 8.18453548177985e-06,
    "sch_prob_decay_1": 0.4746057872587867,
    "sch_prob_decay_2": 0.8223235451679188,
    "sch_gamma": 0.2,
}

# The scenario grid is encoded in the dataset key, which the DSS-EU loader parses
# as feeder:ev_pen:pv_pen ("feeder" is always "default" -- this record has no
# per-feeder dimension, see load_dataset()'s DSS-EU branch). Mid-grid is the
# reference scenario for the benchmark.
DSS_EU = "DSS-EU:default:0.4:0.4"
# The measured (not power-flow-simulated) P/Q/V record at the Madeira substation,
# the other dataset Chapter-6.tex names for this experiment -- same substation as
# Chapter 5's single-target MLVS-PT, summed/averaged across the three metered
# phases (see load_dataset()'s MLVS-PT-3PHASE branch, ported from
# PhD-Thesis-Anthony's load_three_phase_PT_substation_data()).
MLVS_PT_3PHASE = "MLVS-PT-3PHASE"

DATASETS: dict[str, dict] = {
    DSS_EU: {
        "dataset_name": DSS_EU,
        "data_path": str(thesis_data_dir()),
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
        "data_path": str(thesis_data_dir()),
        "train_start": "2019-03",
        "train_end": "2022-01",
        "test_start": "2021-07",
        "test_end": "2022-01",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
}

# "quantile"/"parametric" add CRPS/the pinball loss/NLL. Chapter 3 argues CWE is
# a diagnostic rather than a proper scoring rule, so this benchmark needs a
# proper score to rest on for every condition, not only the interval metrics.
_QUANTILE_METRICS = ["point", "interval", "quantile"]

# Every head is scored at the conformal conditions' own 90% interval.
ALPHA = 0.1
# QR uses its quantile list exactly as given (no conf_level boundary levels are
# inserted), so its default 0.1..0.9 grid gave an 80% native interval. Keep the
# default grid and add the alpha/2 and 1 - alpha/2 outer levels: 11 levels.
QR_GRID: list[float] = [ALPHA / 2, *np.arange(0.1, 1.0, 0.1).round(1).tolist(), 1 - ALPHA / 2]
_QR_OVERRIDES: dict = {"quantiles": QR_GRID, "conf_level": ALPHA}
# proposal_anchor_outer pins FPQR's outer levels to conf_level/2 and
# 1 - conf_level/2, so conf_level sets its interval: 0.1 anchors it at 90%,
# not the class default 0.05 (95%). Set through overrides, not
# hpo_fixed_params, so the existing tuned cache (fingerprinted with its fixed
# params) still loads under --skip-hpo; those head params were tuned at 0.05.
# Move this into hpo_fixed_params when re-tuning with --run-hpo.
_FPQR_ANCHOR_OVERRIDES: dict = {"conf_level": ALPHA}
_PARAMETRIC_METRICS = ["point", "interval", "parametric"]

# ── conformal group: full variant coverage, generated ───────────────────────
# Mirrors backbone_conformal_study.py's own CONFORMAL_VARIANTS exactly (kept
# as a second copy, not a cross-module import -- these experiment scripts
# are each standalone, run directly via `python <path>.py`, with no shared
# package `__init__.py` between experiment_1_benchmark/ and
# experiment_2_backbone_conformal/). Keep both in sync by hand if either
# changes. "none" is deliberately excluded here: it is already covered by
# this file's own "benchmark" group (fpqr/qr/bnn/laplace with
# conformal_config=None), so repeating it here would just be a duplicate,
# unconformal copy of an existing condition.
_MULTIVARIATE_CONFORMAL_VARIANTS: dict[str, ConformalConfig] = {
    # Joint, location/scale -- no correlation matrix. The only multivariate
    # variant kept (2026-10-02): a real run on MLVS-PT-3PHASE
    # (run_id=20261002_07253650b7d21a6dc24d5fb8748d1fdb480710_f25312) showed
    # max_abs and all four mahalanobis_* dependence variants losing to
    # max_norm on every axis -- higher Winkler score, worse (more severe)
    # overcoverage, worse CWE -- for all three heads that carry a native or
    # quantile-derived scale (qr/bnn/laplace). mahalanobis's own four
    # dependence_method choices were additionally near-indistinguishable from
    # each other (kendall and gaussian_copula identical to 4 decimal places,
    # pearson/spearman within <1%), so the complexity wasn't even buying a
    # meaningfully different joint region, let alone a better one. This
    # empirically confirms MultivariateConformal's own documented caveat that
    # a *fixed*, calibration-set-estimated correlation doesn't reliably beat
    # max-norm's own implicit handling of it -- a real negative result, not
    # just unexplored territory; dropped rather than carried forward as dead
    # weight in every future run. (A locally-adaptive radius, per Xu/Jiang/Xie
    # ICML 2024, might behave differently -- a static one clearly doesn't
    # help here; worth a footnote/future-work mention, not a kept condition.)
    "max_norm": ConformalConfig(method="multivariate", score_type="max-norm", alpha=0.1),
}
# CQR, 2 score types x 3 target_coverage modes -- only valid for quantile-
# output heads (fpqr/qr): method="quantile" requires a RawPrediction with its
# own quantiles field set (TwigaForecaster.calibrate), which BNN/Laplace
# (parametric loc/scale heads) never produce.
_QUANTILE_CONFORMAL_VARIANTS: dict[str, ConformalConfig] = {
    "quantile_scaled_marginal": ConformalConfig(
        method="quantile", score_type="scaled", alpha=0.1, target_coverage="marginal"
    ),
    "quantile_scaled_pooled": ConformalConfig(
        method="quantile", score_type="scaled", alpha=0.1, target_coverage="pooled"
    ),
    "quantile_scaled_bonferroni": ConformalConfig(
        method="quantile", score_type="scaled", alpha=0.1, target_coverage="bonferroni"
    ),
    "quantile_unscaled_marginal": ConformalConfig(
        method="quantile", score_type="unscaled", alpha=0.1, target_coverage="marginal"
    ),
    "quantile_unscaled_pooled": ConformalConfig(
        method="quantile", score_type="unscaled", alpha=0.1, target_coverage="pooled"
    ),
    "quantile_unscaled_bonferroni": ConformalConfig(
        method="quantile", score_type="unscaled", alpha=0.1, target_coverage="bonferroni"
    ),
}

# Per-head wiring, one entry per "benchmark" condition above this generates a
# conformal sibling of. Mirrors each benchmark Condition's own
# model_cls/hpo_model_cls/hpo_variant/hpo_parent_*/hpo_fixed_params/
# metric_types exactly -- reuse_checkpoint_from validates this at run time,
# so a drift here surfaces as a clear ValueError, not a silent mismatch.
_CONFORMAL_HEADS: dict[str, dict] = {
    "fpqr_anchor_outer": {
        "model_cls": MLPFFPQRConfig,
        # Must match the benchmark condition's own overrides, which the
        # reused checkpoint was trained under.
        "overrides": _FPQR_ANCHOR_OVERRIDES,
        "hpo_model_cls": MLPFFPQRConfig,
        "hpo_variant": FPQR_ANCHOR_OUTER_HPO_VARIANT,
        "hpo_parent_model_cls": MLPFConfig,
        "hpo_parent_variant": SHARED_HPO_VARIANT,
        # Training-level fields pinned via _FPQR_ANCHOR_OUTER_TRAINING_PINS
        # (2026-10-04, see that constant's own comment for values/rationale
        # -- NOT _REFERENCE_TRAINING_PINS, which this condition never
        # actually trained under). per_target_tau/proposal_type/
        # proposal_anchor_outer re-pinned 2026-10-04, matching commit
        # 904e94 ("v3")'s own config -- confirmed directly against the real
        # grid-group data that this is the best-calibrated fpqr_anchor_outer
        # on record (WS=18.61, PICP=0.920, cwe_stratified=0.810), clearly
        # ahead of every other version's own PICP (v2=0.647, v4=0.633,
        # 2026-10-03's "all free" config=0.602) at a modest WS cost. qvn_type/
        # proposal_loss_fn stay free.
        "hpo_fixed_params": {
            "per_target_tau": True,
            "proposal_type": "softmax",
            "proposal_anchor_outer": True,
            **_FPQR_ANCHOR_OUTER_TRAINING_PINS,
        },
        # Seeds (not pins) the first HPO trial with the best cwe_stratified
        # trial on record (0.68847, the shared v3/v4 Optuna journal's own
        # trial 21) -- proposal_type/proposal_anchor_outer/per_target_tau
        # are absent from this dict since they were pinned (not searched)
        # at the time that trial ran, so this seeded trial still samples
        # them fresh; every later trial is completely unconstrained by
        # this, same as any other free search.
        "hpo_initial_params": _FPQR_ANCHOR_OUTER_SEED,
        "metric_types": _QUANTILE_METRICS,
        "is_quantile_head": True,
    },
    "qr": {
        "model_cls": MLPFQRConfig,
        # Must match the "qr" benchmark condition: the reused checkpoint's
        # output layer is sized by the quantile count.
        "overrides": _QR_OVERRIDES,
        "hpo_model_cls": MLPFConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_parent_model_cls": None,
        "hpo_parent_variant": "",
        "hpo_fixed_params": dict(_REFERENCE_TRAINING_PINS),
        "metric_types": _QUANTILE_METRICS,
        "is_quantile_head": True,
    },
    "bnn": {
        "model_cls": MLPFBNNConfig,
        "hpo_model_cls": MLPFConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_parent_model_cls": None,
        "hpo_parent_variant": "",
        "hpo_fixed_params": dict(_REFERENCE_TRAINING_PINS),
        "metric_types": _PARAMETRIC_METRICS,
        "is_quantile_head": False,
    },
    "laplace": {
        "model_cls": MLPFLaplaceConfig,
        "hpo_model_cls": MLPFConfig,
        "hpo_variant": SHARED_HPO_VARIANT,
        "hpo_parent_model_cls": None,
        "hpo_parent_variant": "",
        "hpo_fixed_params": dict(_REFERENCE_TRAINING_PINS),
        "metric_types": _PARAMETRIC_METRICS,
        "is_quantile_head": False,
    },
}


def _build_conformal_conditions() -> list[Condition]:
    """6 multivariate variants x 4 heads, plus 6 CQR variants x 2 quantile heads = 36 conditions.

    Every one reuses its head's "benchmark"-group namesake's already-trained
    per-fold weights (reuse_checkpoint_from) rather than retraining -- valid
    because every "benchmark" condition below sets reserve_calib_window=True,
    matching the calibration window every condition here reserves (see
    Condition.reserve_calib_window's own docstring).
    """
    conditions = []
    for head_name, head in _CONFORMAL_HEADS.items():
        variants = dict(_MULTIVARIATE_CONFORMAL_VARIANTS)
        if head["is_quantile_head"]:
            variants.update(_QUANTILE_CONFORMAL_VARIANTS)
        for variant_name, conformal_config in variants.items():
            conditions.append(
                Condition(
                    name=f"{head_name}_{variant_name}",
                    group="conformal",
                    description=f"{head_name.upper()} with conformal={variant_name}",
                    overrides=dict(head.get("overrides", {})),
                    model_cls=head["model_cls"],
                    hpo_model_cls=head["hpo_model_cls"],
                    hpo_variant=head["hpo_variant"],
                    hpo_parent_model_cls=head["hpo_parent_model_cls"],
                    hpo_parent_variant=head["hpo_parent_variant"],
                    hpo_fixed_params=dict(head["hpo_fixed_params"]),
                    metric_types=head["metric_types"],
                    conformal_config=conformal_config,
                    reuse_checkpoint_from=head_name,
                )
            )
    return conditions


CONDITIONS = [
    Condition(
        name="fpqr_anchor_outer",
        group="benchmark",
        description=(
            "FPQR, the proposed framework's margin, per-target quantile grid with outer "
            "quantile levels anchored to the nominal interval boundary"
        ),
        overrides=dict(_FPQR_ANCHOR_OVERRIDES),
        model_cls=MLPFFPQRConfig,
        # Shares its backbone with qr/bnn/laplace (hpo_parent_model_cls=MLPFConfig,
        # hpo_parent_variant=SHARED_HPO_VARIANT) rather than running its own joint
        # backbone+head search (2026-10-01; reverses the 2026-09-28 decision --
        # that joint search's own 50-trial budget was diluted across ~15 backbone
        # dims it didn't need its own copy of, on top of ~15 FPQR-head dims, and
        # the resulting backbone asymmetry versus qr/bnn/laplace made the
        # benchmark's head-to-head comparison confounded by backbone capacity,
        # not just the head. load_backbone_params merges SHARED_HPO_VARIANT's
        # already-tuned backbone fields into fixed_search_params -- same
        # mechanism as hpo_fixed_params below, confirmed (2026-10-01) to now
        # hold for every trial, not just the first, after the MLPFFPQRModel.update()
        # fix (self.model_config = self.model_config.model_copy(update=params),
        # not a fresh Config(**params) that silently reverted the stripped
        # search_space back to its unstripped class default each trial).
        #
        # Points at the "grid" group's own fpqr_anchor_outer cache
        # (hpo_variant=FPQR_ANCHOR_OUTER_HPO_VARIANT, same hpo_fixed_params),
        # not a separate FPQR_HPO_VARIANT cache of its own (2026-10-01): the
        # "grid" group's own DM/MCS comparison found fpqr_anchor_outer
        # significantly beats fpqr_shared_grid on winkler_score and is never
        # significantly worse than fpqr_per_target_grid, so it is promoted
        # here from "one grid-ablation arm" to the chapter's actual FPQR
        # configuration -- reusing that exact cache key means a
        # --skip-hpo run picks up the already-tuned params directly, no
        # fresh search needed. per_target_tau/proposal_type/
        # proposal_anchor_outer are pinned to match that condition's own
        # pins exactly, not just per_target_tau alone as before -- a
        # mismatched hpo_fixed_params dict on a shared cache key is merged
        # (last condition sharing the key wins), so this condition and
        # "fpqr_anchor_outer" below must agree completely.
        hpo_model_cls=MLPFFPQRConfig,
        hpo_variant=FPQR_ANCHOR_OUTER_HPO_VARIANT,
        hpo_parent_model_cls=MLPFConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        # Training-level fields pinned via _FPQR_ANCHOR_OUTER_TRAINING_PINS
        # (2026-10-04), plus per_target_tau/proposal_type/
        # proposal_anchor_outer re-pinned the same day to match commit
        # 904e94 ("v3")'s own config -- see
        # _CONFORMAL_HEADS["fpqr_anchor_outer"]'s own comment above for the
        # real-data justification (WS=18.61, PICP=0.920 on this exact cache
        # key, the best-calibrated fpqr_anchor_outer on record).
        hpo_fixed_params={
            "per_target_tau": True,
            "proposal_type": "softmax",
            "proposal_anchor_outer": True,
            **_FPQR_ANCHOR_OUTER_TRAINING_PINS,
        },
        hpo_initial_params=_FPQR_ANCHOR_OUTER_SEED,  # see that dict's own comment for why
        metric_types=_QUANTILE_METRICS,
        # Reserves the same calibration window fpqr_conformal (group=
        # "conformal") carves out, even though this condition itself never
        # calibrates -- required for fpqr_conformal's reuse_checkpoint_from
        # ("fpqr_anchor_outer") to be valid: ExperimentEngine refuses a checkpoint reuse
        # whenever the two conditions' calibration-reservation status
        # disagrees, since that alone shifts every fold's train/test boundary.
        # See Condition.reserve_calib_window's own docstring.
        reserve_calib_window=True,
    ),
    Condition(
        name="qr",
        group="benchmark",
        description="Quantile regression, a fixed-grid baseline",
        overrides=dict(_QR_OVERRIDES),
        model_cls=MLPFQRConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        # Training-level fields pinned via _REFERENCE_TRAINING_PINS
        # (re-added 2026-10-03) -- fixes the shared MLPF backbone's own
        # HPO phase itself, see that constant's own comment for why.
        # Architecture dims (hidden_dim, num_layers, ...) stay free.
        hpo_fixed_params=dict(_REFERENCE_TRAINING_PINS),
        metric_types=_QUANTILE_METRICS,
        reserve_calib_window=True,  # see "fpqr_anchor_outer"'s own comment above
    ),
    Condition(
        name="bnn",
        group="benchmark",
        description="Bayesian neural network, a widely used probabilistic baseline",
        overrides={},
        model_cls=MLPFBNNConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params=dict(_REFERENCE_TRAINING_PINS),  # see "qr"'s own comment above
        metric_types=_PARAMETRIC_METRICS,
        reserve_calib_window=True,  # see "fpqr_anchor_outer"'s own comment above
    ),
    Condition(
        name="laplace",
        group="benchmark",
        description="Parametric Laplace forecasting, the margin Chapter 5 selects",
        overrides={},
        model_cls=MLPFLaplaceConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params=dict(_REFERENCE_TRAINING_PINS),  # see "qr"'s own comment above
        reserve_calib_window=True,  # see "fpqr_anchor_outer"'s own comment above
        metric_types=_PARAMETRIC_METRICS,
    ),
    # ── grid: supporting mechanism check, not part of the chapter's named
    #    Experiment 1 -- see module docstring. ────────────────────────────────
    Condition(
        name="fpqr_shared_grid",
        group="grid",
        description="FPQR with one quantile grid shared across P, Q and V (reference)",
        overrides={},
        model_cls=MLPFFPQRConfig,
        # Own joint search, separate cache from "fpqr_anchor_outer" and from
        # fpqr_per_target_grid below, so the per_target_tau=False pin here
        # doesn't collide with either's own differently-pinned search
        # (Condition.hpo_fixed_params merges per (model_cls, hpo_variant)
        # cache key -- same key with disagreeing pins would silently let the
        # last condition sharing it win).
        hpo_model_cls=MLPFFPQRConfig,
        hpo_variant=FPQR_GRID_SHARED_HPO_VARIANT,
        hpo_parent_model_cls=MLPFConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params={**_FPQR_STABILITY_PINS, "per_target_tau": False},
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="fpqr_per_target_grid",
        group="grid",
        description="FPQR with an independently learned quantile grid per target",
        overrides={},
        model_cls=MLPFFPQRConfig,
        hpo_model_cls=MLPFFPQRConfig,
        hpo_variant=FPQR_GRID_PER_TARGET_HPO_VARIANT,
        hpo_parent_model_cls=MLPFConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params={**_FPQR_STABILITY_PINS, "per_target_tau": True},
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="fpqr_anchor_outer",
        group="grid",
        description=(
            "FPQR with its two outermost quantile levels pinned to conf_level/2 and "
            "1 - conf_level/2 (the nominal interval boundary a fixed-grid model like QR gets "
            "for free); interior quantiles stay adaptive"
        ),
        overrides=dict(_FPQR_ANCHOR_OVERRIDES),
        model_cls=MLPFFPQRConfig,
        hpo_model_cls=MLPFFPQRConfig,
        hpo_variant=FPQR_ANCHOR_OUTER_HPO_VARIANT,
        hpo_parent_model_cls=MLPFConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        # Training-level fields pinned via _FPQR_ANCHOR_OUTER_TRAINING_PINS,
        # plus per_target_tau/proposal_type/proposal_anchor_outer re-pinned
        # 2026-10-04 to match commit 904e94 ("v3") -- see the matching
        # conformal-head dict's own comment above
        # (_CONFORMAL_HEADS["fpqr_anchor_outer"]) for the real-data
        # justification.
        hpo_fixed_params={
            "per_target_tau": True,
            "proposal_type": "softmax",
            "proposal_anchor_outer": True,
            **_FPQR_ANCHOR_OUTER_TRAINING_PINS,
        },
        hpo_initial_params=_FPQR_ANCHOR_OUTER_SEED,  # see that dict's own comment for why
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="fpqr_anchor_outer_v3_fixed",
        group="grid",
        description=(
            "fpqr_anchor_outer with every one of FPQR's own fields (architecture, "
            "qvn_type/n_quantiles/loss_fn/num_cosines/entropy_weight/proposal_eps, and "
            "training config) pinned to commit 904e94's ('v3') exact trained result -- "
            "isolates whether a fresh backbone-only HPO rerun (--run-hpo re-tunes "
            "the shared MLPF backbone this condition still points at via "
            "hpo_parent_model_cls) changes the outcome with the head itself held fixed, "
            "rather than the head and backbone both moving at once as "
            "'fpqr_anchor_outer' above allows"
        ),
        overrides=dict(_FPQR_ANCHOR_OVERRIDES),
        model_cls=MLPFFPQRConfig,
        hpo_model_cls=MLPFFPQRConfig,
        hpo_variant=FPQR_ANCHOR_OUTER_V3_FIXED_HPO_VARIANT,
        hpo_parent_model_cls=MLPFConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params={
            "per_target_tau": True,
            "proposal_type": "softmax",
            "proposal_anchor_outer": True,
            **_FPQR_ANCHOR_OUTER_V3_HEAD_PINS,
            **_FPQR_ANCHOR_OUTER_TRAINING_PINS,
        },
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="fpqr_lv_differenced",
        group="grid",
        description=(
            "FPQR with LV (voltage) forecast as its own first difference rather than its raw "
            "level, reconstructed back to level scale at evaluation; LP and LQ are unaffected "
            "(LV's own SignalCharacteriser verdict is fractional integration, LP/LQ's are not)"
        ),
        overrides={},
        data_overrides={"target_differencing": {"LV": 1}},
        model_cls=MLPFFPQRConfig,
        hpo_model_cls=MLPFFPQRConfig,
        hpo_variant=FPQR_LV_DIFFERENCED_HPO_VARIANT,
        hpo_parent_model_cls=MLPFConfig,
        hpo_parent_variant=SHARED_HPO_VARIANT,
        hpo_fixed_params={**_FPQR_STABILITY_PINS, "per_target_tau": True},
        metric_types=_QUANTILE_METRICS,
    ),
    # ── conformal: generated below (see _build_conformal_conditions) --
    #    every conformal variant backbone_conformal_study.py's own
    #    CONFORMAL_VARIANTS defines, applied to all four benchmark heads,
    #    each reusing its "benchmark"-group namesake's already-trained
    #    weights via reuse_checkpoint_from instead of retraining. ──────────
    *_build_conformal_conditions(),
]

_SPEC_COMMON = {
    "name": "Multitarget Forecasting Benchmark",
    "output_prefix": "ch6_benchmark",
    "condition_cls": MLPFFPQRConfig,
    "backbone_cls": MLPFConfig,
    "conditions": CONDITIONS,
    # Backbone capacity is held equal across qr/bnn/laplace *and* every FPQR
    # condition (fpqr/fpqr_shared_grid/fpqr_per_target_grid/
    # fpqr_anchor_outer) via the shared hpo_model_cls=MLPFConfig/
    # hpo_variant=SHARED_HPO_VARIANT key -- one MLPFConfig backbone search
    # runs once per dataset and every condition reuses that same tuned
    # backbone through hpo_parent_model_cls/hpo_parent_variant (restored
    # 2026-10-01; an intervening 2026-09-28 design gave each FPQR condition
    # its own full joint backbone+head search instead, which diluted that
    # search's own trial budget across backbone dims it didn't need its own
    # copy of and made the benchmark's head-to-head comparison confounded by
    # backbone capacity, not just the head -- see "fpqr_anchor_outer"'s own condition
    # comment above). Each FPQR condition still tunes its own head-only
    # fields independently (hpo_model_cls=MLPFFPQRConfig, its own
    # hpo_variant, backbone fields merged in from the shared parent
    # search). controlled_fields stays unused (frozenset()) either way -- it
    # is not the mechanism at play here.
    "controlled_fields": frozenset(),
    "cv_train_size": 12,
    "cv_test_size": 4,
    "cv_val_size": 2,
    "cv_calib_size": 3,
    "cv_folds": 10,
    # early_stop_patience=40, not the NeuralModelConfig default of 10
    # (twiga/core/config/neural.py:117), and lowered from 100 (2026-10-01;
    # was raised there 2026-09-18 for the reduce_on_plateau scheduler-patience
    # race -- EarlyStopping must never fire before the LR scheduler's own
    # patience gets a chance to act, or training stops right as a plateau
    # recovery was about to happen). 40 still clears that race with real
    # margin: reduce_on_plateau's own prob_patience search space
    # (NeuralModelConfig.SCHEDULER_PARAM_SEARCH, (0.05, 0.2)) is a
    # fraction of max_epochs, so even at this spec's own epochs=150 ceiling
    # the scheduler's worst case is 0.2*150=30 epochs -- 40 clears that with
    # headroom while cutting the wasted-compute tail the old 100 produced.
    # Confirmed directly from TensorBoard val_mae curves (not assumed): 9 of
    # FPQR's 10 CV folds in both this run and the one before it show the true
    # best-val epoch between 4 and 72, train_mae still improving while
    # val_mae degrades past it (standard overfitting, not noise) -- patience
    # =100 let every fold train 80-130+ needless epochs past its own genuine
    # optimum before stopping, the same waste on every one of 50 HPO trials.
    # Checkpointing itself was never wrong (ModelCheckpoint already saves the
    # true best-val epoch regardless of patience) -- this is a compute-budget
    # fix, not a correctness fix.
    # Applies to the actual backtesting conditions' training regardless of
    # --skip-hpo/--run-hpo (added 2026-09-18, see main()'s own args -- skip_hpo
    # was hardcoded True before that, this spec's own backbone HPO was
    # unreachable at all), and 2026-09-18's own epochs=25 -> 500 change
    # (base_config()) would otherwise still stop those runs at the old
    # patience=10 regardless of the new ceiling -- defeating the point of
    # raising it in the first place.
    # num_workers=0: the default of 8 forks DataLoader workers after the
    # MPS/Metal backend is already initialised -- observed hanging
    # indefinitely (71+ min, near-zero CPU progress, unresponsive to
    # `timeout`) in backbone_conformal_study.py's own --quick smoke test
    # under the same engine machinery; applied here preemptively for the
    # same reason rather than waiting to reproduce it again.
    "fixed_overrides": {"early_stop_patience": 40, "num_workers": 0},
    "reference_conditions": {
        "benchmark": "fpqr_anchor_outer",
        "grid": "fpqr_shared_grid",
        "conformal": "laplace_max_norm",
    },
}

# Two specs, not one, as of 2026-09-17: DSS-EU and MLVS-PT-3PHASE cannot share
# one cv_stride safely. 2026-09-18: also no longer share cv_calib_size, train,
# test, or window -- swept empirically against the real files, same protocol
# as fpqr_gating_ablation.py's own SPEC_DSS_EU/SPEC_MLVS_PT_3PHASE arrives at
# (that file's comment has the full sweep methodology; not repeated here).
#   - DSS-EU is ~11 months of synthetic power-flow data (confirmed against
#     the file). On MLVS-PT-3PHASE, every "benchmark" condition now reserves
#     a calibration window too (reserve_calib_window=True, added so their
#     "conformal"-group siblings can reuse their trained weights via
#     reuse_checkpoint_from) in addition to the "conformal" group's own 36
#     conditions -- but on DSS-EU this spec's own cv_calib_size=0 override
#     below makes every one of those a no-op (use_calib's own calib_size>0
#     gate is false regardless of conformal_config/reserve_calib_window), so
#     the DSS-EU filter below (stripping "conformal" from groups) is what
#     actually matters here, not reserve_calib_window. cv_calib_size applies
#     to every group's folds alike, since TimeBasedCV's split scheme is built
#     once per dataset, not per group. DSS-EU cannot absorb any calib
#     reservation above 1 month at
#     all: swept directly, cv_calib_size=2 or the shared default of 3 yields
#     zero folds outright, regardless of train_size (tried 5/6/7). Given
#     MLVS-PT-3PHASE below already gives the conformal group a proper
#     10-fold evaluation with real calibration data, DSS-EU's calib is set to
#     0 rather than the degraded 6-fold/1-month-calib alternative, and
#     main()'s own dataset loop below restricts the "conformal" group to
#     MLVS-PT-3PHASE only -- running it on DSS-EU with calib_idx=None every
#     fold would not calibrate anything meaningful. train=5/val=1/test=1,
#     cv_stride=1, cv_window="rolling", cv_train_size_mode="fixed" -> 5 folds,
#     each fitting on 4 months and validating on the next (the training
#     window holds validation, hence 5), test months late June to November
#     (confirmed non-overlapping: each 1-month test window sits back-to-back
#     with its neighbours'). Fixed 2026-10-06: under the default "derived"
#     mode, TimeBasedCV recomputed train_size from cv_folds as leftover budget
#     after test/val/gap, so this spec's train=4 gave 8 folds that each
#     fitted on about 30 days. "fixed" keeps the window and caps the fold
#     count instead (see TimeBasedCV's train_size_mode).
#   - MLVS-PT-3PHASE is a real 37.5-month measured record -- plenty of data,
#     keeps cv_calib_size=3 from _SPEC_COMMON unchanged (confirmed: still 10
#     folds with it). Fixed 2026-09-27: this spec previously shrank
#     test_size to 3 and set cv_stride=3/cv_window="rolling" to keep test
#     windows non-overlapping at 10 folds -- a self-imposed constraint that
#     does not appear anywhere in Chapter-6.tex's own stated protocol
#     (subsec:ch6-evaluation: "a four-month test window, with the
#     forecasting origin advanced by one month" -- a 4-month test window
#     with a 1-month stride *necessarily* overlaps consecutive test windows
#     by 3 months; the text describes this directly, so non-overlap was
#     never a real requirement to engineer around). Reverted to
#     _SPEC_COMMON's own train=12/val=2/test=4/calib=3, cv_stride=1,
#     cv_window="expanding" -- dry-run confirmed directly against the real
#     file (``TimeBasedCV.set_split_scheme`` on MLVS-PT-3PHASE's actual
#     timestamp range) still gives exactly 10 folds, with the training
#     window expanding fold to fold as the text states (first-fold train
#     span ~18 months, comfortably above the stated "at least twelve months").
# hpo_train_months/hpo_val_months (run_backbone_hpo's own fixed split,
# independent of cv_stride above): defaults 14/2 need >=16 months of
# history and crash ("'val_df' is empty (0 rows)") on DSS-EU's ~11-month
# span -- confirmed on the same 14/2 default in fpqr_gating_ablation.py's
# SPEC. Reachable now via --run-hpo (2026-09-18, see main()'s own args --
# skip_hpo was hardcoded True before that, so this only mattered
# theoretically until now). 8/2 (10 months from train_start=2021-01) fits
# DSS-EU's 2021-01..2021-12 span with a month of margin; MLVS-PT-3PHASE
# (37.5 months) has ample room for the unchanged 14/2 default.
SPEC_DSS_EU = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_train_size": 5, "cv_val_size": 1, "cv_test_size": 1, "cv_calib_size": 0},
    datasets={DSS_EU: DATASETS[DSS_EU]},
    cv_stride=1,
    cv_window="rolling",
    cv_train_size_mode="fixed",
    hpo_train_months=8,
    hpo_val_months=2,
)
SPEC_MLVS_PT_3PHASE = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_test_size": 3},
    datasets={MLVS_PT_3PHASE: DATASETS[MLVS_PT_3PHASE]},
    cv_stride=1,
    cv_window="expanding",
    # Raised back to 14/6 (2026-10-03, matching v2/v4's own values) from the
    # 12/3 dcece6d had settled on -- MLVS-PT-3PHASE's 37.5-month span
    # comfortably absorbs hpo_train_months=14 + hpo_val_months=6 = 20 months.
    hpo_train_months=14,
    hpo_val_months=6,
)

ENGINE_DSS_EU = ExperimentEngine(SPEC_DSS_EU)
ENGINE_MLVS_PT_3PHASE = ExperimentEngine(SPEC_MLVS_PT_3PHASE)
ENGINES: dict[str, ExperimentEngine] = {DSS_EU: ENGINE_DSS_EU, MLVS_PT_3PHASE: ENGINE_MLVS_PT_3PHASE}


def base_config(
    epochs: int,
    hpo_trials: int = 100,
    hpo_epochs: int = 50,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PipelineConfig:
    """Shared PipelineConfig defaults for this script's ENGINES; dataset/output_dir overridden per run."""
    return PipelineConfig(
        dataset_name=DSS_EU,
        data_path=str(thesis_data_dir()),
        output_dir=str(output_root / "multitarget_experiment"),
        epochs=epochs,
        # Reachable now via --run-hpo (2026-09-18, see main()'s own args --
        # skip_hpo was hardcoded True before that, making this inert
        # regardless of its value). Raised from 1 to 50, matching the fix
        # applied to fpqr_gating_ablation.py's base_config() and Chapter
        # 5/7's own PipelineConfig defaults for the same reason: 1 trial
        # cannot leave TPESampler's n_startup_trials=20
        # (twiga/forecaster/base.py:663) random-startup phase.
        num_trials=hpo_trials,
        # Distinct from `epochs` (the final, already-tuned conditions'
        # budget): epochs per Optuna trial, matching backbone_conformal_decision.py's
        # own --hpo-epochs convention.
        hpo_epochs=hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )


def main() -> None:
    """CLI entry point: parse args, run each requested dataset's ExperimentEngine, write the combined summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group",
        nargs="*",
        default=["benchmark", "grid", "conformal"],
        help="condition groups to run; all three by default for an unattended server run",
    )
    ap.add_argument(
        "--dataset",
        nargs="*",
        default=[DSS_EU, MLVS_PT_3PHASE],
        choices=[*DATASETS, "all"],
        help="dataset keys, or 'all' for every dataset in DATASETS",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="short run on the reference scenario")
    ap.add_argument(
        "--skip-hpo",
        dest="skip_hpo",
        action="store_true",
        default=True,
        help="reuse cached backbone params instead of searching (default: True, unlike "
        "fpqr_gating_ablation.py -- this script's HPO was unreachable at all before 2026-09-18, "
        "hardcoded regardless of any flag; default kept True here to preserve every prior run's "
        "behavior, override with --run-hpo).",
    )
    ap.add_argument("--run-hpo", dest="skip_hpo", action="store_false", help="run real backbone HPO")
    ap.add_argument(
        "--hpo-cache-policy",
        choices=["reuse", "refresh"],
        default="reuse",
        help="only used with --run-hpo (ignored under the default --skip-hpo, which always forces "
        "'error-if-missing'). 'reuse' (default) accepts a cached file whose fingerprint still "
        "matches and only searches when no match exists; 'refresh' ignores any existing cache and "
        "always runs a fresh search, overwriting the cache file -- use this to force a real rerun "
        "after a code change that doesn't happen to change the fingerprint, or when you just want "
        "a new search regardless.",
    )
    ap.add_argument("--hpo-trials", type=int, default=100)
    ap.add_argument("--hpo-epochs", type=int, default=50, help="epochs per HPO trial, not per final condition")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root directory for experiment artifacts and HPO caches",
    )
    ap.add_argument(
        "--hpo-objective",
        default="winkler_score",
        help="HPO objective for FPQR/QR conditions only (BNN/Laplace stay point-accuracy). "
        "'winkler_score' (default) or any name get_quantile_metrics computes, e.g. the "
        "diagnostic 'cwe_stratified' -- both derived from the outermost predicted quantile "
        "pair, no calibration needed.",
    )
    args = ap.parse_args()
    if args.skip_hpo and args.hpo_cache_policy != "reuse":
        raise SystemExit("--hpo-cache-policy=refresh requires --run-hpo (skip_hpo=True always forces error-if-missing)")
    hpo_cache_policy: HPOCachePolicy = args.hpo_cache_policy

    configure(level="INFO")
    cfg = base_config(
        5 if args.quick else args.epochs,
        hpo_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        output_root=args.output_root,
    )
    if args.quick:
        cfg = dc_replace(cfg, num_trials=1, hpo_epochs=5)

    for cond in CONDITIONS:
        if cond.model_cls in (MLPFFPQRConfig, MLPFQRConfig):
            cond.hpo_objective_metric = args.hpo_objective

    dataset_keys = [DSS_EU] if args.quick else (list(DATASETS) if "all" in args.dataset else args.dataset)

    # One engine call per dataset, not one call across both: DSS-EU and
    # MLVS-PT-3PHASE need different cv_test_size/cv_val_size/cv_calib_size
    # (see SPEC_DSS_EU/SPEC_MLVS_PT_3PHASE's own comments above), so a single
    # shared ExperimentSpec can no longer cover both.
    summaries = []
    for dataset_key in dataset_keys:
        groups = args.group
        if dataset_key == DSS_EU and "conformal" in groups:
            # 2026-09-18, explicit decision: SPEC_DSS_EU's cv_calib_size=0
            # (see its own comment above -- DSS-EU cannot absorb any real
            # calibration reservation) means every "conformal"-group
            # condition would run here with calib_idx=None every fold,
            # calibrating against nothing -- and, since forecaster.conformal_params
            # is still set regardless, predict_interval's own guard
            # ("Conformal model is not calibrated") would raise outright, not
            # just degrade silently. MLVS-PT-3PHASE keeps cv_calib_size=3 and
            # gives the conformal group a proper 10-fold evaluation, so it
            # runs there only.
            groups = [g for g in groups if g != "conformal"]
            log.info("Skipping group='conformal' on %s -- see SPEC_DSS_EU's own comment (cv_calib_size=0)", dataset_key)
        summaries.append(
            ENGINES[dataset_key].run(
                base_cfg=cfg,
                groups=groups,
                dataset_keys=[dataset_key],
                skip_hpo=args.skip_hpo,
                hpo_cache_policy=hpo_cache_policy,
                tracking_uri=None,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    out = script_dir / "results_multitarget.csv"
    summary.to_csv(out, index=False)
    log.info("summary written to %s", out)
    print(summary.to_string())


if __name__ == "__main__":
    main()
