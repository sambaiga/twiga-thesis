"""Chapter 5 FPQR experiments, on the twiga ExperimentEngine.

Replaces the hand-rolled loops in `learned_tau.py` and `tau_matrix.py` for
everything metric-based. Those scripts use a single train/test split, which is
the design Chapter 3 argues against; the engine runs the ten-fold
expanding-window protocol instead (12-month train, 4-month test, 2-month
val, stride 2 -- confirmed against the thesis text directly: `chatter-5-new.tex`
states "the sliding window was incremented by two months"), so every number
arrives with fold-level variability. `cv_stride` is set explicitly below
rather than left to `ExperimentSpec`'s default (`1`, Chapter 6/7's stride) --
leaving it unset was a real bug in this file: every existing group ran at
stride 1 rather than the chapter's own stride 2 until this fix.

Groups:
    grid            FPQR against a fixed grid on the same backbone (A18)
    n_quantiles     how N is chosen: 9/14/23/36/57/90, geometrically spaced
                    (ratio (90/9)**(1/5) ≈ 1.585) from the chapter's
                    illustrative N=9 to N=90, the value actually used for the
                    thesis's real Experiment 3 numbers (see `benchmark` below)
    loss_terms      whether each term of the objective matters at all (on/off)
    proposal        proposal_type: softmax / horizon / kumaraswamy
    tau_embed       cosine-embedding projection: TimeEmb / LinearEmb / ConvEmb
    num_cosines     cosine embedding capacity: 0 (TauRoPE) / 16 / 32 / 64
    rope_type       TauRoPE variant (num_cosines=0): linear / tau_pos / factored / direct
    crossing        crossing_penalty value: 0 (off) then 1/3.16/10/31.6/100,
                    a half-decade log sweep anchored at the default (10)
    sort_output     differentiable sort: False / True / True+penalty=0
    qvn_type        value network: relu / softplus / monotone / monotone_cal
    loss_fn         pinball (reference) / huber-pinball (kappa sweep) / winkler
    lambda_cal      CALCE weight for monotone_cal: 0.001 / 0.005 / 0.01 / 0.05 / 0.1
    prop_nq         2-D cell {softmax,kumaraswamy} x {9,14,23,36,57,90} (same
                    geometric N sweep as n_quantiles)
    stop_gradient   2x2 {softmax,kuma} x {sg_on,sg_off}
    prop_loss       proposal training loss: wasserstein / sqda / sqda_kl / combined
    horizon_sharing BxNxH (actual reference code) vs BxNx1 (the chapter's
                    stated design) -- settles the discrepancy this session
                    found between Chapter 6's prose and the real
                    implementation
    best_config     assembles the winner from every group above
    benchmark       Experiment 3 (sec:mlpfq-exp-setup): MLPQF against QR,
                    parametric Laplace, and BNN on the MLPF backbone
    benchmark_nhits Experiment 3, repeated on the N-HiTS backbone as a
                    robustness check that the head ranking holds across
                    architectures

`tau_embed` through `best_config` mirror
`experiment/ch6_conformal/mlpgamfpqr_ablation.py`'s fourteen-group
design ablation, adapted for Chapter 5: MLPF backbone only (not MLPGAM),
single-target (no per_target_tau), and this chapter's own two real datasets
(MLVS-PT, MLVS-UK) rather than the four (incl. two synthetic) that chapter
uses. `loss_terms`/`crossing`/`loss_fn` test different granularities on
purpose -- `loss_terms` asks whether each term matters at all (on/off),
`crossing` and the (folded-into-`loss_fn`) Huber-kappa sweep ask what the best
*value* is -- so they coexist rather than duplicate.

`benchmark`'s MLPQF condition sets `num_cosines=0, rope_type="linear"`
explicitly: the thesis states MLPQF embeds learned fractions via RoPE, not a
cosine basis ("this is what distinguishes MLPQF from FPSe2Q"), and the actual
formula rotates by the slot's integer rank -- `rope_type="linear"` in twiga's
terms, not twiga's own later-discovered-better `"tau_pos"` default. Everything
else uses twiga's current defaults (plain pinball loss, standard QVN, relu
crossing penalty) rather than literally reproducing Chapter 5's original
Huber-pinball + squared-margin-hinge objective, which twiga's own later
ablation (Chapter 7) found degrades calibration relative to the modern
defaults -- a deliberate choice, not an oversight.

The legacy FPSe2Q-MLP paper's own architecture is *not* reproduced here as a
`benchmark` condition -- it is a separate, pre-twiga codebase
(`PhD-Thesis-Anthony/src/fpq/FPQ.py`) run as-is on the same data split, and its
output is quoted as an external reference row rather than ported into this
spec.

Run:
    uv run python experiment/chapter-5-fpqr/fpqr_experiment.py --help
    uv run python experiment/chapter-5-fpqr/fpqr_experiment.py --group grid
    uv run python experiment/chapter-5-fpqr/fpqr_experiment.py --group benchmark benchmark_nhits
"""

from __future__ import annotations

from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.models.nn.mlpf_model import MLPFConfig
from twiga.models.nn.mlpfbnn_model import MLPFBNNConfig
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig
from twiga.models.nn.mlpfqr_model import MLPFQRConfig
from twiga.models.nn.nhitsbnn_model import NHITSBNNConfig
from twiga.models.nn.nhitslaplace_model import NHITSLaplaceConfig
from twiga.models.nn.nhitsqr_model import NHITSFPQRConfig, NHITSQRConfig
from twiga.utils.run_forecast import PipelineConfig

from twiga_thesis.paths import thesis_data_dir

ALPHA = 0.10
N_DEFAULT = 9
# The `grid`/`n_quantiles`/... ablation groups sweep/reference N=9, matching
# the thesis's illustrative figure ("N=9 quantile trajectories",
# Chapter-5.tex line 134). The `benchmark`/`benchmark_nhits` groups instead
# need the value actually used to produce the thesis's real Experiment 3
# numbers, confirmed against PhD-Thesis-Anthony/src/experiment.py's
# Optuna-tuned params dict (non-round decimals -- clipping_value, margin --
# are the signature of a real tuned config, not a placeholder): N=90, not 9.
N_MLPQF_BENCHMARK = 90
METRICS = ["point", "interval", "quantile"]
# "parametric" adds NLL for the Laplace/BNN conditions in the benchmark group;
# CRPS still comes via "quantile" for the FPQR/QR conditions, so every
# condition in the group rests on a proper score, not only interval metrics.
_QUANTILE_METRICS = ["point", "interval", "quantile"]
_PARAMETRIC_METRICS = ["point", "interval", "parametric"]

# The fixed grid the ablation compares against: the same number of levels as
# the FPQR default, uniformly spaced, so only the placement differs.
FIXED_GRID = [round((i + 0.5) / N_DEFAULT, 4) for i in range(N_DEFAULT)]

DATASETS: dict[str, dict] = {
    "MLVS-PT": {
        "dataset_name": "MLVS-PT",
        "data_path": str(thesis_data_dir()),
        "train_start": "2019-03",
        "train_end": "2021-12",
        "test_start": "2021-07",
        "test_end": "2021-12",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
    "MLVS-UK": {
        "dataset_name": "MLVS-UK",
        "data_path": str(thesis_data_dir()),
        "train_start": "2018-01",
        "train_end": "2020-07",
        "test_start": "2019-12",
        "test_end": "2020-07",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
}


def _fpqr(name: str, group: str, description: str, **overrides) -> Condition:
    """An FPQR condition; unset keys fall back to the model defaults."""
    return Condition(
        name=name,
        group=group,
        description=description,
        overrides={"n_quantiles": N_DEFAULT, "conf_level": ALPHA, **overrides},
        metric_types=METRICS,
    )


CONDITIONS = [
    # ── grid: the control the chapter's central claim requires ───────────────
    _fpqr("fpqr", "grid", "Learned quantile grid (proposed)"),
    Condition(
        name="qr_fixed",
        group="grid",
        description="Same backbone and budget, uniformly spaced fixed grid",
        overrides={"quantiles": FIXED_GRID, "conf_level": ALPHA},
        model_cls=MLPFQRConfig,
        metric_types=METRICS,
    ),
    # ── n_quantiles: answers "how was N chosen?" with evidence ───────────────
    # Geometrically spaced from 9 to 90 (ratio = (90/9)**(1/5) ≈ 1.585, i.e.
    # ×1.6 per step): 9 is the chapter's illustrative figure value, 90 is what
    # actually produced the thesis's real Experiment 3 numbers (see
    # `benchmark`'s `mlpqf` condition) -- this sweep is what tests whether
    # that choice is justified, spaced evenly on a log scale across the whole
    # gap rather than clustered at the low end.
    *[_fpqr(f"nq_{n}", "n_quantiles", f"{n} learned quantile levels", n_quantiles=n) for n in (9, 14, 23, 36, 57, 90)],
    # ── loss_terms: what each part of the objective buys ─────────────────────
    _fpqr("loss_full", "loss_terms", "All four terms (reference)"),
    _fpqr("loss_no_entropy", "loss_terms", "Entropy regulariser removed", entropy_weight=0.0),
    _fpqr("loss_no_crossing", "loss_terms", "Non-crossing penalty removed", crossing_penalty=0.0),
    _fpqr(
        "loss_huber",
        "loss_terms",
        "Huber-pinball as described in the chapter, against the pinball default",
        loss_fn="huber-pinball",
        kappa=0.5,
    ),
    # ── proposal: is a single linear map the binding constraint? ─────────────
    _fpqr("prop_softmax", "proposal", "Softmax proposal (reference)"),
    _fpqr("prop_horizon", "proposal", "Horizon-conditioned proposal", proposal_type="horizon"),
    _fpqr("prop_kumaraswamy", "proposal", "Kumaraswamy tail-adaptive proposal", proposal_type="kumaraswamy"),
    # ── tau_embed: does the cosine-embedding projection matter? ───────────────
    _fpqr("time_emb", "tau_embed", "TimeEmb projection (Linear+ReLU) — default", tau_embed_type="TimeEmb"),
    _fpqr("linear_emb", "tau_embed", "LinearEmb projection (pure linear)", tau_embed_type="LinearEmb"),
    _fpqr("conv_emb", "tau_embed", "ConvEmb projection (Conv1d over quantile axis)", tau_embed_type="ConvEmb"),
    # ── num_cosines: cosine-basis capacity vs the RoPE alternative ───────────
    _fpqr("cos_rope", "num_cosines", "TauRoPE only (num_cosines=0)", num_cosines=0),
    _fpqr("cos_16", "num_cosines", "16 cosine basis functions — low capacity", num_cosines=16),
    _fpqr("cos_32", "num_cosines", "32 cosine basis functions — default", num_cosines=32),
    _fpqr("cos_64", "num_cosines", "64 cosine basis functions — high capacity", num_cosines=64),
    # ── rope_type: which RoPE variant, given num_cosines=0 ───────────────────
    # MLPQF's own stated design (Chapter 5, sec:MLPFPQR-paper2): rotates by the
    # slot's integer rank, matching rope_linear -- kept as this group's
    # reference condition rather than twiga's later, ablation-improved
    # rope_tau_pos default (see `benchmark` group's `mlpqf` condition).
    _fpqr(
        "rope_linear",
        "rope_type",
        "TauRoPE linear — integer slot rank as position (MLPQF's stated design)",
        num_cosines=0,
        rope_type="linear",
    ),
    _fpqr(
        "rope_tau_pos",
        "rope_type",
        "TauRoPE tau_pos — per-slot mean tau as float position",
        num_cosines=0,
        rope_type="tau_pos",
    ),
    _fpqr(
        "rope_factored",
        "rope_type",
        "TauRoPE factored — split z_dim between quantile and horizon",
        num_cosines=0,
        rope_type="factored",
    ),
    _fpqr(
        "rope_direct",
        "rope_type",
        "TauRoPE direct — backbone z rotated by tau*N_q",
        num_cosines=0,
        rope_type="direct",
    ),
    # ── crossing: penalty strength, finer-grained than loss_terms' on/off ────
    # Half-decade log sweep (ratio sqrt(10) ≈ 3.16), anchored at 10 -- the
    # default crossing_penalty every other group in this file assumes
    # (qvn_relu, qvn_softplus, sort_false/sort_true all hardcode it). Replaces
    # an earlier ad hoc 0/1/10/50/100 (×10, ×5, ×2 -- inconsistent ratios)
    # with a consistent one; 0 stays a separate categorical "off" case since
    # geometric spacing has no zero.
    _fpqr("cross_0", "crossing", "No crossing penalty", crossing_penalty=0.0),
    _fpqr("cross_1", "crossing", "Weak penalty (λ=1)", crossing_penalty=1.0),
    _fpqr("cross_3", "crossing", "λ=3.16", crossing_penalty=3.16),
    _fpqr("cross_10", "crossing", "Default penalty (λ=10)", crossing_penalty=10.0),
    _fpqr("cross_32", "crossing", "λ=31.6", crossing_penalty=31.6),
    _fpqr("cross_100", "crossing", "Very strong penalty (λ=100)", crossing_penalty=100.0),
    # ── sort_output: does a differentiable sort help beyond the penalty? ─────
    _fpqr("sort_false", "sort_output", "Penalty-only (default)", crossing_penalty=10.0, sort_output=False),
    _fpqr("sort_true", "sort_output", "Differentiable sort + penalty", crossing_penalty=10.0, sort_output=True),
    _fpqr("sort_only", "sort_output", "Sort only — no crossing penalty", crossing_penalty=0.0, sort_output=True),
    # ── qvn_type: value-network architecture ─────────────────────────────────
    _fpqr(
        "qvn_relu",
        "qvn_type",
        "Standard QVN + ReLU hinge (MLPQF's stated design)",
        qvn_type="standard",
        crossing_mode="relu",
        crossing_penalty=10.0,
    ),
    _fpqr(
        "qvn_softplus",
        "qvn_type",
        "Standard QVN + softplus hinge",
        qvn_type="standard",
        crossing_mode="softplus",
        crossing_penalty=10.0,
        softplus_beta=1.0,
    ),
    _fpqr(
        "qvn_monotone",
        "qvn_type",
        "Monotone QVN — non-crossing by construction (uncalibrated)",
        qvn_type="monotone",
        crossing_penalty=0.0,
    ),
    _fpqr(
        "qvn_monotone_cal",
        "qvn_type",
        "Calibrated Monotone QVN + CALCE loss",
        qvn_type="monotone_cal",
        crossing_penalty=0.0,
        lambda_cal=0.01,
        cal_temperature=1.0,
    ),
    # Added after qvn_monotone's under-fit, undercovering result (PICP below
    # qvn_relu's, NMPI narrower not wider): three alternative monotonicity
    # constructions, each answering the failure differently. See
    # twiga/distributions/nn/fpqr/qvn.py's module docstring for the theory.
    _fpqr(
        "qvn_lattice",
        "qvn_type",
        "Lattice QVN — non-negative weight reparameterisation (Cannon, 2018)",
        qvn_type="lattice",
        crossing_penalty=0.0,
    ),
    _fpqr(
        "qvn_spline",
        "qvn_type",
        "Spline QVN — harmonic-mean-smoothed rate over adaptive tau knots (Fritsch & Carlson, 1980)",
        qvn_type="spline",
        crossing_penalty=0.0,
    ),
    _fpqr(
        "qvn_umnn",
        "qvn_type",
        "UMNN QVN — integrates an unconstrained derivative via quadrature (Wehenkel & Louppe, 2019)",
        qvn_type="umnn",
        crossing_penalty=0.0,
    ),
    # Reproduces PhD-Thesis-Anthony's FPQSeq2Network.QVN *fusion mechanism*
    # (attention between tau embedding and backbone context) on the MLPF
    # backbone -- not the legacy's own two-stream architecture, which twiga's
    # single-latent backbone contract cannot express (see AttentionQVN's own
    # docstring). Same soft crossing penalty as qvn_relu/qvn_softplus, since
    # AttentionQVN has no architectural non-crossing guarantee.
    _fpqr(
        "qvn_attention",
        "qvn_type",
        "Attention QVN — cross-attention tau/context fusion (legacy FPSe2Q-MLP mechanism)",
        qvn_type="attention",
        crossing_mode="relu",
        crossing_penalty=10.0,
    ),
    # ── loss_fn: pinball vs Huber-pinball (kappa sweep) vs Winkler ───────────
    _fpqr("loss_pinball", "loss_fn", "Standard pinball loss (twiga default, reference)", loss_fn="pinball"),
    _fpqr("loss_huber_k01", "loss_fn", "Huber-pinball κ=0.1", loss_fn="huber-pinball", kappa=0.1),
    _fpqr(
        "loss_huber_k025",
        "loss_fn",
        "Huber-pinball κ=0.25 (Chapter 5's stated design)",
        loss_fn="huber-pinball",
        kappa=0.25,
    ),
    _fpqr("loss_huber_k1", "loss_fn", "Huber-pinball κ=1.0", loss_fn="huber-pinball", kappa=1.0),
    _fpqr("loss_winkler", "loss_fn", "Winkler interval score on outer pair + pinball on inner", loss_fn="winkler"),
    # ── lambda_cal: CALCE weight sweep (monotone_cal only) ───────────────────
    *[
        _fpqr(
            f"lambda_cal_{v!s}".replace(".", ""),
            "lambda_cal",
            f"Calibrated monotone QVN — λ_cal={v}",
            qvn_type="monotone_cal",
            crossing_penalty=0.0,
            lambda_cal=v,
            cal_temperature=1.0,
        )
        for v in (0.001, 0.005, 0.01, 0.05, 0.1)
    ],
    # ── prop_nq: proposal type × grid size 2-D cell ──────────────────────────
    # Same geometric N sweep as `n_quantiles` above, for consistency.
    *[
        _fpqr(f"softmax_nq{n}", "prop_nq", f"Softmax × N={n}", proposal_type="softmax", n_quantiles=n)
        for n in (9, 14, 23, 36, 57, 90)
    ],
    *[
        _fpqr(f"kuma_nq{n}", "prop_nq", f"Kumaraswamy × N={n}", proposal_type="kumaraswamy", n_quantiles=n)
        for n in (9, 14, 23, 36, 57, 90)
    ],
    # ── stop_gradient: does the pinball gradient reaching the proposal help? ─
    _fpqr(
        "softmax_sg_on",
        "stop_gradient",
        "Softmax + stop_gradient=True (MLPQF's stated design)",
        proposal_type="softmax",
        proposal_stop_gradient=True,
    ),
    _fpqr(
        "softmax_sg_off",
        "stop_gradient",
        "Softmax + stop_gradient=False (pinball gradient flows to proposal)",
        proposal_type="softmax",
        proposal_stop_gradient=False,
    ),
    _fpqr(
        "kuma_sg_on",
        "stop_gradient",
        "Kumaraswamy + stop_gradient=True",
        proposal_type="kumaraswamy",
        proposal_stop_gradient=True,
    ),
    _fpqr(
        "kuma_sg_off",
        "stop_gradient",
        "Kumaraswamy + stop_gradient=False",
        proposal_type="kumaraswamy",
        proposal_stop_gradient=False,
    ),
    # ── prop_loss: proposal training signal ──────────────────────────────────
    _fpqr("prop_loss_w1", "prop_loss", "Wasserstein-1 (FQF-style, reference)", proposal_loss_fn="wasserstein"),
    _fpqr("prop_loss_sqda", "prop_loss", "SQDA squared — q^{-1} target widths", proposal_loss_fn="sqda"),
    _fpqr("prop_loss_sqda_kl", "prop_loss", "SQDA with KL divergence", proposal_loss_fn="sqda_kl"),
    _fpqr("prop_loss_combined", "prop_loss", "0.5×W1 + 0.5×SQDA", proposal_loss_fn="combined"),
    # ── horizon_sharing: BxNxH (actual code) vs BxNx1 (the chapter's stated
    #    design) -- settles the discrepancy this session found between
    #    Chapter 6's prose ("a single quantile grid... applied uniformly
    #    across all forecast horizons") and the real reference implementation
    #    (PhD-Thesis-Anthony/src/net/quantile_block.py::MLPFPQNetwork, which
    #    instantiates NQuantileProposal(M=horizon) and softmaxes independently
    #    per horizon column -- genuinely BxNxH, confirmed numerically). Every
    #    other group and the `benchmark` group's `mlpqf` both use the
    #    reference (BxNxH) condition as their baseline; this group is the one
    #    place BxNx1 is actually run and measured against it.
    _fpqr(
        "horizon_varying",
        "horizon_sharing",
        "BxNxH -- independent grid per horizon step (the actual reference code, reference condition)",
        proposal_shared_horizon=False,
    ),
    _fpqr(
        "horizon_shared",
        "horizon_sharing",
        "BxNx1 -- one grid broadcast across all horizons (the chapter's stated design)",
        proposal_shared_horizon=True,
    ),
    # ── best_config: assembles the winner from every group above ────────────
    _fpqr(
        "default_config",
        "best_config",
        "Factory defaults — reference baseline",
        proposal_type="softmax",
        tau_embed_type="TimeEmb",
        n_quantiles=9,
        num_cosines=32,
        crossing_penalty=10.0,
        sort_output=False,
        loss_fn="pinball",
    ),
    _fpqr(
        "best_config",
        "best_config",
        "Best settings confirmed on both MLVS-PT and MLVS-UK: n_quantiles=23 (prop_nq), "
        "crossing_penalty=100 (crossing), crossing_mode=softplus (qvn_type). proposal_type, "
        "rope_type, and n_quantiles beyond 23 disagreed by dataset and are left at the "
        "reference rather than tuned to one dataset; proposal_loss_fn/proposal_stop_gradient/ "
        "proposal_shared_horizon are left at their defaults since the prop_loss/stop_gradient/ "
        "horizon_sharing groups were confirmed silently no-op (MLPFFPQRConfig did not expose "
        "those fields; fixed in twiga/models/nn/mlpffpqr_model.py) and need a re-run before "
        "their groups' answers can be trusted.",
        proposal_type="softmax",
        tau_embed_type="TimeEmb",
        n_quantiles=23,
        num_cosines=32,
        crossing_penalty=100.0,
        crossing_mode="softplus",
        sort_output=False,
        loss_fn="pinball",
    ),
    # ── benchmark: Experiment 3 (sec:mlpfq-exp-setup) ─────────────────────────
    # MLPQF against QR, parametric Laplace, and BNN, all on the MLPF backbone.
    Condition(
        name="mlpqf",
        group="benchmark",
        description="MLPQF: FPQR + RoPE tau embedding on the MLPF backbone (proposed)",
        overrides={
            "n_quantiles": N_MLPQF_BENCHMARK,
            "conf_level": ALPHA,
            "num_cosines": 0,
            "rope_type": "linear",
        },
        model_cls=MLPFFPQRConfig,
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="qr",
        group="benchmark",
        description="Quantile regression, a fixed-grid baseline",
        overrides={},
        model_cls=MLPFQRConfig,
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="laplace",
        group="benchmark",
        description="Parametric Laplace forecasting",
        overrides={},
        model_cls=MLPFLaplaceConfig,
        metric_types=_PARAMETRIC_METRICS,
    ),
    Condition(
        name="bnn",
        group="benchmark",
        description="Bayesian neural network, a widely used probabilistic baseline",
        overrides={},
        model_cls=MLPFBNNConfig,
        metric_types=_PARAMETRIC_METRICS,
    ),
    # ── benchmark_nhits: Experiment 3 repeated on N-HiTS ──────────────────────
    # Robustness check -- does the head ranking found on MLPF hold on a second
    # backbone? Not part of the chapter's own named experiment; a supporting
    # check the same way `proposal`/`grid` are.
    Condition(
        name="nhits_mlpqf",
        group="benchmark_nhits",
        description="MLPQF head (FPQR + RoPE) on the N-HiTS backbone",
        overrides={
            "n_quantiles": N_MLPQF_BENCHMARK,
            "conf_level": ALPHA,
            "num_cosines": 0,
            "rope_type": "linear",
        },
        model_cls=NHITSFPQRConfig,
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="nhits_qr",
        group="benchmark_nhits",
        description="Quantile regression on the N-HiTS backbone",
        overrides={},
        model_cls=NHITSQRConfig,
        metric_types=_QUANTILE_METRICS,
    ),
    Condition(
        name="nhits_laplace",
        group="benchmark_nhits",
        description="Parametric Laplace forecasting on the N-HiTS backbone",
        overrides={},
        model_cls=NHITSLaplaceConfig,
        metric_types=_PARAMETRIC_METRICS,
    ),
    Condition(
        name="nhits_bnn",
        group="benchmark_nhits",
        description="Bayesian neural network on the N-HiTS backbone",
        overrides={},
        model_cls=NHITSBNNConfig,
        metric_types=_PARAMETRIC_METRICS,
    ),
]

SPEC = ExperimentSpec(
    name="Chapter 5 FPQR",
    output_prefix="ch5_fpqr",
    condition_cls=MLPFFPQRConfig,
    backbone_cls=MLPFConfig,
    conditions=CONDITIONS,
    datasets=DATASETS,
    # Explicit rather than left to ExperimentSpec's defaults: the thesis states
    # this protocol directly ("initial historical period set at a minimum of
    # 12 months... fixed future window of 4 months... sliding window was
    # incremented by two months"). Leaving cv_stride unset previously meant
    # every group in this file silently ran at stride 1 (the ExperimentSpec
    # default, and Chapter 6/7's own stride) rather than the chapter's actual
    # stride 2 -- fixed here. This file's ExperimentSpec is independent of
    # every Chapter 6/7 spec (separate objects in separate files), so this
    # fix cannot affect those experiments.
    cv_train_size=12,
    cv_test_size=4,
    cv_val_size=2,
    cv_stride=2,
    cv_folds=10,
    # early_stop_patience=100, not the NeuralModelConfig default of 10
    # (twiga/core/config/neural.py:117). reduce_on_plateau's scheduler
    # patience is int(sch_prob_patience * max_epochs) (twiga/models/nn/core/
    # base_model.py:453); sch_prob_patience's HPO search range (0.05, 0.2)
    # (twiga/core/config/neural.py:252). 2026-09-18: BASE_CFG's epochs raised
    # 60 -> 500 to let training run well past the early-saturation point
    # (best epoch landing in the first 0-8 epochs, twiga-wide, per Chapter
    # 7's mechanism figure) rather than stopping at or before it, so
    # early_stop_patience is raised to match -- the scheduler's worst case
    # is now int(0.2*500)=100, exactly meeting rather than safely clearing
    # 100 (a trial sampling sch_prob_patience near the top of its range can
    # still tie; the typical case, ~0.125, gives scheduler patience ~62,
    # well clear). Same collision confirmed and fixed for Chapter 7's
    # MLPGAM backbone (experiment/ch6_conformal/README.md's 2026-09-18
    # entries).
    fixed_overrides={"early_stop_patience": 100},
    # Only the knob under test may vary; the engine holds these equal so an
    # apparent effect cannot come from a differing backbone. FPQR-specific
    # fields (num_cosines, crossing_penalty, etc.) are silently skipped by the
    # engine for conditions whose model_cls doesn't have them (QR/Laplace/BNN),
    # so listing them here is safe for the mixed-model-class benchmark groups
    # too -- confirmed via twiga/experiment/engine.py's hasattr-guarded apply.
    controlled_fields=frozenset(
        {
            "n_quantiles",
            "conf_level",
            "loss_fn",
            "kappa",
            "entropy_weight",
            "crossing_penalty",
            "proposal_type",
            "num_cosines",
            "tau_embed_type",
            "rope_type",
            "sort_output",
            "qvn_type",
            "crossing_mode",
            "softplus_beta",
            "lambda_cal",
            "cal_temperature",
            "proposal_stop_gradient",
            "proposal_loss_fn",
            "proposal_shared_horizon",
            "opt_lr",
            "opt_weight_decay",
            "latent_size",
            "hidden_dim",
            "num_layers",
            "dropout",
        }
    ),
    reference_conditions={
        "grid": "fpqr",
        "n_quantiles": "nq_9",
        "loss_terms": "loss_full",
        "proposal": "prop_softmax",
        "tau_embed": "time_emb",
        "num_cosines": "cos_32",
        "rope_type": "rope_linear",
        "crossing": "cross_10",
        "sort_output": "sort_false",
        "qvn_type": "qvn_relu",
        "loss_fn": "loss_pinball",
        "lambda_cal": "lambda_cal_001",
        "prop_nq": "softmax_nq9",
        "stop_gradient": "softmax_sg_on",
        "prop_loss": "prop_loss_w1",
        "horizon_sharing": "horizon_varying",
        "best_config": "default_config",
        "benchmark": "mlpqf",
        "benchmark_nhits": "nhits_mlpqf",
    },
)

ENGINE = ExperimentEngine(SPEC)

BASE_CFG = PipelineConfig(
    dataset_name="MLVS-PT",
    data_path=str(thesis_data_dir()),
    output_dir="../experiments/chapter-5-fpqr/",
    input_window_size=96,
    forecast_horizon=48,
    window_stride=48,
    period="30min",
    epochs=500,
    use_latex=False,
    num_threads=4,
    # PipelineConfig's own default (10) never exceeds TPESampler's
    # n_startup_trials=patience*2=20 (twiga/forecaster/base.py:663, tune()'s
    # own default patience=10), so every trial stays in the sampler's
    # pure-random-startup phase and the search never becomes data-adaptive.
    # Confirmed directly in this chapter's own completed runs: every
    # backbone_hpo/<dataset>/<model>_best_params.json's Optuna journal log
    # (results/chapter_5_fpqr/backbone_hpo/*/logs/*/*.log) shows exactly 10
    # trials. 50 leaves ~30 trials of genuine TPE-guided search once backbone
    # HPO is rerun (existing cached best_params.json files must be deleted
    # first -- run_backbone_hpo reuses whatever is cached regardless of this
    # value). See experiment/ch6_conformal/README.md's 2026-09-18 entry
    # for the full trace; this repeats the same bug found there.
    num_trials=50,
)


if __name__ == "__main__":
    ENGINE.cli_main(BASE_CFG)
