"""Multi-target FPQR + gating ablation for Chapter 6, on the MLPGAM backbone.

Runs through :class:`~twiga.experiment.ExperimentEngine`, the same machinery as
``experiment/ch6_conformal/mlpgam_design_ablation.py`` and
``experiment/ch7_multitarget/multitarget_experiment.py`` — this script
reuses both patterns rather than re-deriving them. Nothing here re-implements
condition/backbone/CV wiring; it only adds the multi-target-specific pieces
Chapter 6 needs that neither existing script has:

  - a one-factor-at-a-time sweep of the FPQR fields that affect the adaptive
    quantile grid, quantile loss, and value network, in addition to a complete
    conditional Optuna search for each FPQR condition;
  - the "grid" group already in ``multitarget_experiment.py`` (shared vs.
    per-target quantile grid), reused unchanged but retargeted at MLPGAM+FPQR
    so it sits alongside the gating and mechanism groups under one CV protocol;
  - "multigate": the target-level output gate built this session
    (``MLPGAMConfig.use_target_gate`` — see ``twiga/models/nn/net/mlpfgam.py``),
    on/off/combined with the existing feature gate, following the same
    ``gate_none``/``gate_unconstrained``/``gate_default`` pattern
    ``experiment/ch6_conformal/mlpgam_design_ablation.py`` already reuses
    for Chapter 7's own (feature-)gating group;
  - "head": the joint parametric head (:class:`MLPGAMMVNormalConfig`) as a
    fourth comparison condition alongside FPQR (uncoupled), per the user's
    explicit instruction — Chapter-6.tex's prose does not (and will not) claim
    this head belongs in the chapter; it is here purely so the FPQR+copula+
    gating design is judged against a real alternative rather than assumed to
    win.

Per this session's plan (``~/.claude/plans/staged-dancing-seal.md``, Part B):
none of this script's results are to be written into Chapter-6.tex's §3/§4
mechanism-level prose until it has actually been run and read. Running it is
what this script is for.

Deferred, not implemented here
-------------------------------
Copula-coupling the FPQR margins (``twiga/distributions/nn/copula_sampling.py``
— ``fit_gaussian_copula``/``coupled_samples``, a numpy post-hoc pathway, not
wired into any model config yet) is not part of the ``ExperimentEngine`` loop
below. Coupling only makes sense on top of *already-trained* per-target FPQR
quantile predictions, evaluated with a joint proper score (energy or variogram
score, per ``CHAPTER-6-REWRITE-PLAN.md``) rather than the marginal metrics the
engine reports — that is a separate post-processing pass over this script's
output, not a `Condition`. Scope it once the "grid" group here has picked a
per-target-tau setting worth coupling.

Dataset allocation
-------------------
MLVS-PT and DSS-EU, per the user's explicit decision this session — as
``MLVS-PT-3PHASE``, the measured (not power-flow-simulated) P/Q/V record at
the same Madeira substation as Chapter 5's single-target ``MLVS-PT``.
``twiga/utils/data_loading.py::load_dataset()`` now has a branch for it,
ported from PhD-Thesis-Anthony's ``load_three_phase_PT_substation_data()``
(``src/utils/load_data.py``) — that loader was the missing piece, not a
missing dataset; it already exists and sums/averages the three metered phases
into LP/LQ/LV. What's still missing is the *file*: the loader reads raw
per-year substation CSVs available only on the project server, and
``MLVS-PT-3PHASE.parquet`` has not yet been produced from them (via
``PhD-Thesis-Anthony/notebook/Prepare-data.ipynb``) or copied into
``examples/data/``. Until it is, ``--dataset MLVS-PT-3PHASE`` fails with a
plain ``FileNotFoundError`` (confirmed) rather than silently running on the
wrong target — this script's own default still runs DSS-EU only, so a
plain invocation is unaffected. DSS-UK/DSS-US are reserved for Chapter 6's
Experiment 3 (operational decision-making) and not needed here regardless.

Usage::

    uv run python experiment/ch7_multitarget/fpqr_gating_ablation.py --group grid
    uv run python experiment/ch7_multitarget/fpqr_gating_ablation.py --group multigate
    uv run python experiment/ch7_multitarget/fpqr_gating_ablation.py --group proposal_design --run-hpo
    uv run python experiment/ch7_multitarget/fpqr_gating_ablation.py --quick
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import json
import os
from pathlib import Path

import pandas as pd
from twiga.core.utils import configure, get_logger
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.experiment.hpo import HPOCachePolicy
from twiga.models.nn.mlpgam_model import MLPGAMConfig
from twiga.models.nn.mlpgamfpqr_model import MLPGAMFPQRConfig
from twiga.models.nn.mlpgammvnormal_model import MLPGAMMVNormalConfig
from twiga.utils.run_forecast import PipelineConfig

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[3] / "experiments"))) / "ch7_multitarget"
)
SHARED_MLPGAM_HPO_VARIANT = "shared_mlpgam"

# ── Datasets ──────────────────────────────────────────────────────────────────
# DSS-EU: the one synthetic record already used in the published Experiment 1
# baseline (multitarget_experiment.py's REFERENCE). Confirmed via
# load_dataset()'s own DSS-EU branch: target_series=["p_kw","q_kvar","v_lv_pu"],
# scenario selected by "DSS-EU:<feeder>:<ev_pen>:<pv_pen>" suffix.
DSS_EU_REFERENCE = "DSS-EU:default:0.4:0.4"

# MLVS-PT-3PHASE: the measured P/Q/V record, now wired (see module docstring)
# but not yet runnable — MLVS-PT-3PHASE.parquet does not exist locally.
# date range mirrors the single-target MLVS-PT branch elsewhere in this
# repo (twiga/utils/data_loading.py), since it is the same substation.
MLVS_PT_3PHASE = "MLVS-PT-3PHASE"

DATASETS: dict[str, dict] = {
    DSS_EU_REFERENCE: {
        "dataset_name": DSS_EU_REFERENCE,
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
}

# ── FPQR condition factory and parameter coverage ──────────────────────────────
# Every FPQR condition gets its own head search, seeded from the same
# MLPGAMConfig backbone search. Its tested values are pinned *inside* HPO so
# the conditional search space matches the condition that is later evaluated.
# This follows multitarget_experiment.py's shared-backbone / condition-specific
# head-search pattern rather than tuning one unconstrained head and overriding
# it after the fact.
_BASE_METRICS = ["point", "interval", "quantile"]


def _fpqr_condition(
    name: str,
    group: str,
    description: str,
    overrides: dict,
    *,
    metric_types: list[str] | None = None,
) -> Condition:
    """Create a condition with a shared MLPGAM parent and a pinned FPQR HPO search."""
    return Condition(
        name=name,
        group=group,
        description=description,
        overrides=overrides,
        model_cls=MLPGAMFPQRConfig,
        hpo_model_cls=MLPGAMFPQRConfig,
        hpo_variant=f"fpqr_{name}",
        hpo_parent_model_cls=MLPGAMConfig,
        hpo_parent_variant=SHARED_MLPGAM_HPO_VARIANT,
        hpo_fixed_params=dict(overrides),
        metric_types=metric_types or _BASE_METRICS,
    )


def _build_fpqr_parameter_conditions() -> list[Condition]:
    """Build one-factor conditions for FPQR settings outside the original gating sweep."""
    conditions: list[Condition] = []

    def add(
        name: str,
        group: str,
        description: str,
        baseline: dict,
        **axis: object,
    ) -> None:
        conditions.append(_fpqr_condition(name, group, description, {**baseline, **axis}))

    target_grid = {"per_target_tau": True}
    add("loss_pinball", "loss", "Pinball quantile loss (reference)", target_grid, loss_fn="pinball")
    add("loss_huber", "loss", "Huber-smoothed pinball quantile loss", target_grid, loss_fn="huber-pinball")
    add("loss_winkler", "loss", "Winkler interval loss", target_grid, loss_fn="winkler")
    for kappa in (0.01, 0.25, 1.0):
        add(
            f"kappa_{str(kappa).replace('.', '_')}",
            "kappa",
            f"Huber-pinball loss with kappa={kappa:g}",
            {"per_target_tau": True, "loss_fn": "huber-pinball"},
            kappa=kappa,
        )

    add(
        "tau_cosine_timeemb",
        "tau_embedding",
        "Cosine tau embedding with TimeEmb (reference)",
        target_grid,
        num_cosines=32,
        tau_embed_type="TimeEmb",
    )
    for embed_type in ("LinearEmb", "ConvEmb"):
        add(
            f"tau_cosine_{embed_type.lower()}",
            "tau_embedding",
            f"Cosine tau embedding with {embed_type}",
            target_grid,
            num_cosines=32,
            tau_embed_type=embed_type,
        )
    for rope_type in ("tau_pos", "linear", "factored", "direct"):
        add(
            f"tau_rope_{rope_type}",
            "tau_embedding",
            f"RoPE tau embedding with {rope_type}",
            target_grid,
            num_cosines=0,
            rope_type=rope_type,
        )
    for rope_base in (2.0, 100.0, 10000.0):
        add(
            f"tau_rope_base_{int(rope_base)}",
            "tau_embedding",
            f"RoPE base frequency {rope_base:g}",
            {"per_target_tau": True, "num_cosines": 0, "rope_type": "tau_pos"},
            rope_base=rope_base,
        )

    proposal_baseline = {
        "per_target_tau": True,
        "proposal_type": "softmax",
        "proposal_shared_horizon": False,
        "proposal_anchor_outer": False,
        "proposal_stop_gradient": True,
        "proposal_loss_fn": "wasserstein",
        "entropy_weight": 1.0,
        "proposal_eps": 1e-3,
        "sqda_weight": 0.1,
    }
    add(
        "proposal_design_reference",
        "proposal_design",
        "Unshared, unanchored softmax proposal (reference)",
        proposal_baseline,
    )
    add(
        "proposal_shared_horizon",
        "proposal_design",
        "Share the softmax proposal across forecast horizons",
        proposal_baseline,
        proposal_shared_horizon=True,
    )
    add(
        "proposal_anchor_outer",
        "proposal_design",
        "Anchor the outer softmax quantiles at nominal boundaries",
        proposal_baseline,
        proposal_anchor_outer=True,
    )
    add(
        "proposal_stop_gradient_off",
        "proposal_design",
        "Backpropagate pinball gradients into proposal parameters",
        proposal_baseline,
        proposal_stop_gradient=False,
    )
    for loss_name in ("sqda", "sqda_kl", "combined"):
        add(
            f"proposal_loss_{loss_name}",
            "proposal_design",
            f"Proposal objective: {loss_name}",
            proposal_baseline,
            proposal_loss_fn=loss_name,
        )
    for sqda_weight in (0.01, 0.1, 1.0):
        add(
            f"proposal_sqda_weight_{str(sqda_weight).replace('.', '_')}",
            "proposal_design",
            f"Combined proposal objective with SQDA weight={sqda_weight:g}",
            {**proposal_baseline, "proposal_loss_fn": "combined"},
            sqda_weight=sqda_weight,
        )
    for entropy_weight in (0.25, 1.0, 3.0):
        add(
            f"proposal_entropy_{str(entropy_weight).replace('.', '_')}",
            "proposal_design",
            f"Proposal entropy weight={entropy_weight:g}",
            proposal_baseline,
            entropy_weight=entropy_weight,
        )
    for proposal_eps in (1e-5, 1e-3, 5e-3):
        add(
            f"proposal_eps_{proposal_eps:g}".replace(".", "_"),
            "proposal_design",
            f"Proposal probability floor={proposal_eps:g}",
            proposal_baseline,
            proposal_eps=proposal_eps,
        )

    crossing_baseline = {
        "per_target_tau": True,
        "qvn_type": "standard",
        "crossing_mode": "relu",
        "crossing_penalty": 1.0,
        "softplus_beta": 1.0,
        "sort_output": False,
    }
    add(
        "crossing_reference",
        "crossing",
        "ReLU crossing penalty, weight 1, unsorted outputs (reference)",
        crossing_baseline,
    )
    add("crossing_penalty_zero", "crossing", "No crossing penalty", crossing_baseline, crossing_penalty=0.0)
    add("crossing_penalty_ten", "crossing", "Crossing penalty weight 10", crossing_baseline, crossing_penalty=10.0)
    add(
        "crossing_softplus",
        "crossing",
        "Softplus crossing penalty",
        crossing_baseline,
        crossing_mode="softplus",
    )
    add(
        "crossing_softplus_beta_five",
        "crossing",
        "Softplus crossing penalty with beta 5",
        {**crossing_baseline, "crossing_mode": "softplus"},
        softplus_beta=5.0,
    )
    add(
        "crossing_sorted_output",
        "crossing",
        "Sort predicted quantiles after the value network",
        crossing_baseline,
        sort_output=True,
    )

    for n_quantiles in (9, 15, 32):
        add(
            f"quantiles_{n_quantiles}",
            "quantiles",
            f"Adaptive grid with {n_quantiles} quantile proposals",
            target_grid,
            n_quantiles=n_quantiles,
        )

    for heads in (2, 4, 8):
        add(
            f"qvn_attention_heads_{heads}",
            "qvn_attention",
            f"Attention QVN with {heads} attention heads",
            {"per_target_tau": True, "qvn_type": "attention"},
            qvn_num_heads=heads,
        )
    for points in (4, 8, 16, 32):
        add(
            f"qvn_umnn_points_{points}",
            "qvn_umnn",
            f"UMNN QVN with {points} quadrature points",
            {"per_target_tau": True, "qvn_type": "umnn"},
            umnn_quad_points=points,
        )
    for weight in (1e-3, 0.01, 0.1):
        add(
            f"qvn_cal_lambda_{str(weight).replace('.', '_')}",
            "qvn_monotone_cal",
            f"Monotone-cal QVN with calibration loss weight={weight:g}",
            {"per_target_tau": True, "qvn_type": "monotone_cal", "cal_temperature": 1.0},
            lambda_cal=weight,
        )
    for temperature in (0.1, 1.0, 5.0):
        add(
            f"qvn_cal_temperature_{str(temperature).replace('.', '_')}",
            "qvn_monotone_cal",
            f"Monotone-cal QVN with calibration temperature={temperature:g}",
            {"per_target_tau": True, "qvn_type": "monotone_cal", "lambda_cal": 0.01},
            cal_temperature=temperature,
        )

    for clip_value in (0.1, 1.0, 10.0):
        add(
            f"gradient_clip_{str(clip_value).replace('.', '_')}",
            "gradient_clip",
            f"FPQR gradient clip norm={clip_value:g}",
            target_grid,
            gradient_clip_val=clip_value,
        )

    return conditions


CONDITIONS: list[Condition] = [
    # The proposal and QVN mechanism sweeps include every supported categorical
    # architecture choice; continuous and branch-specific settings are also
    # represented by _build_fpqr_parameter_conditions() and the expanded HPO.
    *[
        _fpqr_condition(
            name,
            "proposal",
            description,
            {"proposal_type": proposal_type, "per_target_tau": True},
        )
        for name, description, proposal_type in (
            ("proposal_softmax", "Softmax proposal (reference)", "softmax"),
            ("proposal_kumaraswamy", "Kumaraswamy tail-adaptive proposal", "kumaraswamy"),
            ("proposal_horizon", "Horizon-conditioned proposal", "horizon"),
        )
    ],
    *[
        _fpqr_condition(
            f"qvn_{qvn_type}",
            "qvn",
            description,
            {"qvn_type": qvn_type, "per_target_tau": True},
        )
        for qvn_type, description in (
            ("standard", "Standard value network with crossing penalty (reference)"),
            ("monotone", "Monotone value network"),
            ("monotone_cal", "Calibration-regularised monotone value network"),
            ("attention", "Cross-attention value network"),
            ("lattice", "Lattice-monotone value network"),
            ("spline", "Spline-monotone value network"),
            ("umnn", "Unconstrained monotone neural network with quadrature"),
        )
    ],
    _fpqr_condition("grid_shared", "grid", "One quantile grid shared across targets", {"per_target_tau": False}),
    _fpqr_condition(
        "grid_per_target",
        "grid",
        "Independently learned quantile grid per target",
        {"per_target_tau": True},
    ),
    *[
        _fpqr_condition(name, "multigate", description, overrides)
        for name, description, overrides in (
            (
                "gate_none",
                "No feature or target gating",
                {"per_target_tau": True, "use_gate": False, "use_target_gate": False},
            ),
            (
                "gate_feature_only",
                "Input feature-group gating only",
                {"per_target_tau": True, "use_gate": True, "use_target_gate": False},
            ),
            (
                "gate_target_only",
                "Target-level output gating only",
                {"per_target_tau": True, "use_gate": False, "use_target_gate": True},
            ),
            (
                "gate_both",
                "Feature-group and target-level gating",
                {"per_target_tau": True, "use_gate": True, "use_target_gate": True},
            ),
        )
    ],
    _fpqr_condition(
        "head_fpqr",
        "head",
        "Multi-target FPQR, per-target grid, uncoupled margins",
        {"per_target_tau": True},
    ),
    Condition(
        name="head_joint_parametric",
        group="head",
        description="Joint parametric multivariate-Normal head with target covariance",
        overrides={},
        model_cls=MLPGAMMVNormalConfig,
        metric_types=["point", "interval", "parametric"],
    ),
    *_build_fpqr_parameter_conditions(),
]

_SPEC_COMMON: dict = {
    "name": "FPQR Gating Ablation",
    "output_prefix": "ch6_fpqr_gating",
    "condition_cls": MLPGAMFPQRConfig,
    "backbone_cls": MLPGAMFPQRConfig,
    "conditions": CONDITIONS,
    # Every FPQR condition uses its own head search, pinned to its tested
    # setting, while sharing the same MLPGAM parent search.
    "controlled_fields": frozenset({"latent_size", "hidden_dim", "num_layers", "dropout", "opt_lr"}),
    # Chapter 7's adapted-from-Chapter-5 protocol, minus the calibration split:
    # 12-month min initial train, 4-month test, 2-month val, 10 folds.
    # cv_calib_size=0, not Chapter 7's 3 -- that reservation exists there for
    # the conformal-calibrated CRC conditions; grepping this file confirms no
    # condition here builds a ConformalConfig or runs a calibration step at
    # all (proposal/qvn/grid/multigate/head are all plain FPQR mechanism
    # comparisons), so the 3-month calib window this spec inherited from
    # _SPEC_COMMON's Chapter-7-derived starting point was dead weight, never
    # used by anything downstream -- and a real cost on DSS-EU specifically,
    # where every month matters (see SPEC_DSS_EU's own comment below: cutting
    # this reduces the true per-fold minimum from 21 months to 18, still more
    # than DSS-EU's ~10-month span, but a real reduction in the shortfall).
    # cv_stride is the one field that must differ by dataset -- see the two
    # specs below, same reasoning as multitarget_experiment.py's own
    # SPEC_DSS_EU/SPEC_MLVS_PT_3PHASE split (2026-09-17).
    "cv_train_size": 12,
    "cv_test_size": 4,
    "cv_val_size": 2,
    "cv_calib_size": 0,
    "cv_folds": 10,
    # Independent of the CV protocol above: backbone HPO's own fixed
    # train/val split (run_backbone_hpo, hpo.py) defaults to 14/2 months,
    # which needs >=16 months of history and crashes ("'val_df' is empty (0
    # rows)") on DSS-EU (~11 months, confirmed against the file) -- this
    # spec's other dataset, MLVS-PT-3PHASE (37.5 months), has ample margin
    # either way, so shrinking the shared split costs it nothing. 8/2
    # (10 months from train_start=2021-01) fits DSS-EU's 2021-01..2021-12
    # span with a month of margin to spare.
    # early_stop_patience=40 matches multitarget_experiment.py: it clears the
    # reduce_on_plateau scheduler's worst-case patience (0.2 * max_epochs) at the
    # 100-epoch default while avoiding wasted post-optimum training.
    "fixed_overrides": {"early_stop_patience": 40, "num_workers": 0},
    "reference_conditions": {
        "proposal": "proposal_softmax",
        "qvn": "qvn_standard",
        "grid": "grid_shared",
        "multigate": "gate_none",
        "head": "head_fpqr",
        "loss": "loss_pinball",
        "kappa": "kappa_0_25",
        "tau_embedding": "tau_cosine_timeemb",
        "proposal_design": "proposal_design_reference",
        "crossing": "crossing_reference",
        "quantiles": "quantiles_9",
        "qvn_attention": "qvn_attention_heads_4",
        "qvn_umnn": "qvn_umnn_points_8",
        "qvn_monotone_cal": "qvn_cal_lambda_0_01",
        "gradient_clip": "gradient_clip_1_0",
    },
}

# Two specs, not one, mirroring multitarget_experiment.py's own
# SPEC_DSS_EU/SPEC_MLVS_PT_3PHASE split (2026-09-17) and the same reasoning:
# DSS-EU and MLVS-PT-3PHASE cannot share one cv_stride safely. Both below
# were swept empirically against the real files (train/val/test/stride grid
# search via TimeBasedCV directly, not guessed) to land on the current
# values, 2026-09-18, explicitly requested: 8 folds on DSS-EU, 10 on
# MLVS-PT-3PHASE, both with stride == test_size so folds are non-overlapping
# (an earlier pass used stride < test_size on MLVS-PT-3PHASE for a higher
# fold count, confirmed to produce 2-month-overlapping test windows between
# every adjacent pair -- rejected once a non-overlapping alternative that
# still hits 10 folds was found, at a smaller test_size instead of a smaller
# stride).
#   - DSS-EU: train=5/val=1/test=1 months (not _SPEC_COMMON's 12/2/4),
#     cv_stride=1, cv_window="rolling", cv_train_size_mode="fixed" -> 5
#     folds, each fitting on 4 months and validating on the next (the
#     training window holds validation, hence 5), test months 2021-06-29 to
#     2021-11-29, confirmed non-overlapping (each 1-month test window sits
#     back-to-back with its neighbours').
#   - MLVS-PT-3PHASE: 12/2/3, cv_stride=1, cv_window="expanding",
#     HPO split 14/6 -- identical to multitarget_experiment.py except
#     cv_calib_size=0 (no conformal calibration window in this experiment).
# Training windows, re-checked 2026-10-06 against the real files: under the
# default cv_train_size_mode="derived", TimeBasedCV recomputes train_size
# from cv_folds as leftover budget after test/val/gap, so DSS-EU's earlier
# train=4 gave 8 folds that each fitted on about 30 days. DSS-EU now uses
# "fixed", which keeps the window and caps the fold count instead.
# MLVS-PT-3PHASE is unaffected: its expanding folds fit on 19-28 months
# (the 37.5-month record leaves ample budget), identical in both modes, so
# it keeps the default. An earlier version of this comment said those folds
# trained on ~5 months; that was wrong.
SPEC_DSS_EU = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_train_size": 5, "cv_val_size": 1, "cv_test_size": 1},
    datasets={DSS_EU_REFERENCE: DATASETS[DSS_EU_REFERENCE]},
    hpo_train_months=8,
    hpo_val_months=2,
    cv_stride=1,
    cv_window="rolling",
    cv_train_size_mode="fixed",
)
SPEC_MLVS_PT_3PHASE = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_test_size": 3},
    datasets={MLVS_PT_3PHASE: DATASETS[MLVS_PT_3PHASE]},
    # Matches multitarget_experiment.py's SPEC_MLVS_PT_3PHASE (12/2/3, stride 1,
    # expanding, HPO 14/6, early_stop_patience=40); only cv_calib_size=0 differs.
    cv_stride=1,
    cv_window="expanding",
    hpo_train_months=14,
    hpo_val_months=6,
)

ENGINE_DSS_EU = ExperimentEngine(SPEC_DSS_EU)
ENGINE_MLVS_PT_3PHASE = ExperimentEngine(SPEC_MLVS_PT_3PHASE)
ENGINES: dict[str, ExperimentEngine] = {DSS_EU_REFERENCE: ENGINE_DSS_EU, MLVS_PT_3PHASE: ENGINE_MLVS_PT_3PHASE}


def base_config(
    epochs: int,
    hpo_trials: int = 100,
    hpo_epochs: int = 50,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PipelineConfig:
    """Shared PipelineConfig defaults for this script's engines."""
    return PipelineConfig(
        dataset_name=DSS_EU_REFERENCE,
        data_path="examples/data",
        output_dir=str(output_root / "fpqr_gating_ablation"),
        epochs=epochs,
        num_trials=hpo_trials,
        hpo_epochs=hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )


# ── Run discovery ─────────────────────────────────────────────────────────────
# Mirrors chapter-5-fpqr/analyse_results.py's discover_runs()/latest_run_for_group()
# unchanged in spirit: runs are synced into this repo under RESULTS_ROOT (from the
# server's `../experiments/...` output_dir, per this script's own base_config()),
# and a run's meta.json records which condition groups it actually covers, so the
# per-group re-runs Step 1a/1c of this session's design-decision plan produce
# (e.g. a `proposal`-only or `qvn`-only re-run) resolve to their own latest run
# instead of a stale, hand-maintained --full-csv path. The previous flat
# `--full-csv` default (script_dir.parent.parent.parent.parent / "experiments" / ...) never
# matched what ExperimentEngine.run() actually writes -- it omitted the
# `<experiment_slug>/runs/<run_id>` path segments entirely -- so `--report`
# could not have worked against it; this replaces that default rather than
# patching it.
RESULTS_ROOT = script_dir.parent / "results" / "ch7_multitarget" / "fpqr_gating_ablation"


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


def significance_report(group: str, alpha: float = 0.05) -> None:
    """Print a Winkler-score DM/MCS significance table for one condition group.

    Reuses ``experiment.common.significance`` per
    that chapter's own rule: CWE is a diagnostic and never decides a
    comparison, only the Winkler score (a proper scoring rule) does.

    Reads the *latest* synced run that covers ``group`` (see run-discovery
    section above), not a fixed path -- a re-run of one group alone (per this
    session's design-decision plan) is picked up automatically.
    """
    import sys

    from experiment.common.significance import mcs_ranking, pairwise_dm_table  # noqa: PLC0415

    run = latest_run_for_group(discover_runs(), group)
    if run is None:
        log.warning("No synced run under %s covers group=%s", RESULTS_ROOT, group)
        return
    full_csv = run["run_dir"] / "results" / "ch6_fpqr_gating_full.csv"
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
        # mean-aggregates per (Folds, condition) via pivot_table's default aggfunc,
        # matching chapter-5-fpqr/analyse_results.py's fold_level() intent: one
        # Winkler-score observation per condition-fold, not per timestamp.
        losses = h_df.pivot_table(index="Folds", columns="condition", values="winkler_score")
        print(f"\n=== group={group} h={h} — DM table (Winkler score) ===")
        # version="multivariate" (pairwise_dm_table's own default, matching
        # chapter-5-fpqr/analyse_results.py's dm_and_mcs() call exactly) --
        # "two-sided" was never a valid value for diebold_mariano_from_losses's
        # `version` (only "univariate"/"multivariate" are), so this call could
        # never have succeeded; caught while fixing this function's run-discovery
        # bug, not previously reachable since --full-csv's old default never
        # resolved to a real file.
        print(pairwise_dm_table(losses.to_dict(orient="series"), alpha=alpha))
        print(f"\n=== group={group} h={h} — Model Confidence Set (Winkler score) ===")
        print(mcs_ranking(losses.to_dict(orient="series"), alpha=alpha))


def main() -> None:
    """Run selected condition groups and datasets, then write the combined summary."""
    ap = argparse.ArgumentParser()
    groups_available = sorted({condition.group for condition in CONDITIONS})
    ap.add_argument(
        "--group",
        nargs="*",
        default=["all"],
        choices=["all", *groups_available],
        help="condition groups to run",
    )
    ap.add_argument(
        "--dataset",
        nargs="*",
        default=[DSS_EU_REFERENCE],
        choices=[*DATASETS, "all"],
        help="dataset keys, or 'all' for every dataset in DATASETS",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="short run for a smoke test")
    ap.add_argument(
        "--skip-hpo",
        dest="skip_hpo",
        action="store_true",
        default=True,
        help="reuse matching cached backbone params instead of searching (default: True; use --run-hpo to tune)",
    )
    ap.add_argument("--run-hpo", dest="skip_hpo", action="store_false", help="run HPO for all active FPQR conditions")
    ap.add_argument(
        "--hpo-cache-policy",
        choices=["reuse", "refresh"],
        default="reuse",
        help="only used with --run-hpo; reuse matching caches or always run a fresh search",
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
        help="HPO objective for quantile-producing conditions; e.g. winkler_score or cwe_stratified",
    )
    ap.add_argument("--report", metavar="GROUP", help="print a significance report for one group and exit")
    args = ap.parse_args()

    if args.skip_hpo and args.hpo_cache_policy != "reuse":
        raise SystemExit("--hpo-cache-policy=refresh requires --run-hpo (skip_hpo=True always forces error-if-missing)")

    if args.report:
        significance_report(args.report)
        return

    configure(level="INFO")
    groups = None if "all" in args.group else args.group
    dataset_keys = list(DATASETS) if "all" in args.dataset else args.dataset
    cfg = base_config(
        5 if args.quick else args.epochs,
        hpo_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        output_root=args.output_root,
    )
    if args.quick:
        cfg = dc_replace(cfg, num_trials=1, hpo_epochs=5)

    for condition in CONDITIONS:
        if condition.model_cls is MLPGAMFPQRConfig:
            condition.hpo_objective_metric = args.hpo_objective
    hpo_cache_policy: HPOCachePolicy = args.hpo_cache_policy

    # One engine call per dataset, not one call across both: DSS-EU and
    # MLVS-PT-3PHASE have different cv_stride (SPEC_DSS_EU/SPEC_MLVS_PT_3PHASE
    # above), so a single shared ExperimentSpec can no longer cover both --
    # same reasoning and pattern as multitarget_experiment.py's own per-dataset
    # engine dispatch.
    summaries = []
    for dataset_key in dataset_keys:
        summaries.append(
            ENGINES[dataset_key].run(
                base_cfg=cfg,
                groups=groups,
                dataset_keys=[dataset_key],
                skip_hpo=args.skip_hpo or args.quick,
                hpo_cache_policy=hpo_cache_policy,
                tracking_uri=None,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    out = script_dir / "results_fpqr_gating.csv"
    summary.to_csv(out, index=False)
    log.info("summary written to %s", out)
    print(summary.to_string())


if __name__ == "__main__":
    main()
