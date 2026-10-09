"""MLPGAM-QR Design Ablation.

Two-phase experiment that isolates each QR design choice while controlling
for backbone hyperparameters (Phase 1 Optuna HPO with plain MLPGAMConfig).

Six experiment groups:
    crossing      — crossing_penalty: 0 / 0.5 / 1 / 5 / 10
    crossing_mode — penalty mode (relu / softplus) at fixed λ=1
    sort_output   — sort quantiles at inference: False / True / True+λ=0
    monotone      — cumulative-softplus parameterisation: False / True
    loss_fn       — pinball / huber-pinball (κ sweep) / winkler
    best_config   — combines winners from all groups

Usage::

    python experiment/ch6_conformal/exploratory/mlpgamqr_ablation.py               # all groups
    python experiment/ch6_conformal/exploratory/mlpgamqr_ablation.py --group crossing loss_fn
    python experiment/ch6_conformal/exploratory/mlpgamqr_ablation.py --skip-hpo
"""

from __future__ import annotations

from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.models.nn import MLPGAMConfig, MLPGAMQRConfig
from twiga.utils.run_forecast import PipelineConfig

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "MLVS-PT": {
        "dataset_name": "MLVS-PT",
        "data_path": "examples/data",
        "train_start": "2019-03",
        "train_end": "2022-01",
        "input_window_size": 96,
        "forecast_horizon": 48,
        "window_stride": 48,
        "period": "30min",
    },
    "MLVS-UK": {
        "dataset_name": "MLVS-UK",
        "data_path": "examples/data",
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

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DEFAULT_METRIC_TYPES = ["point", "interval", "quantile"]

# Default values for all QR controlled fields.
BASE_QR_OVERRIDES = {
    "conf_level": 0.05,
    "loss_fn": "pinball",
    "kappa": 0.25,
    "crossing_penalty": 10.0,
    "crossing_mode": "relu",
    "softplus_beta": 1.0,
    "sort_output": False,
    "monotone_output": False,
}


def _make_condition(name: str, group: str, desc: str, **overrides) -> Condition:
    return Condition(name, group, desc, overrides={**BASE_QR_OVERRIDES, **overrides}, metric_types=DEFAULT_METRIC_TYPES)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = [
    # ── Group 1: crossing_penalty ─────────────────────────────────────────────
    _make_condition("cross_0", "crossing", "No crossing penalty — raw unconstrained output", crossing_penalty=0.0),
    _make_condition("cross_05", "crossing", "Very weak crossing penalty (λ=0.5)", crossing_penalty=0.5),
    _make_condition("cross_1", "crossing", "Weak crossing penalty (λ=1) — expected sweet spot", crossing_penalty=1.0),
    _make_condition("cross_5", "crossing", "Moderate crossing penalty (λ=5)", crossing_penalty=5.0),
    _make_condition(
        "cross_10", "crossing", "Default crossing penalty (λ=10) — known to cause PICP→1", crossing_penalty=10.0
    ),
    # ── Group 2: crossing_mode ────────────────────────────────────────────────
    _make_condition(
        "relu_1",
        "crossing_mode",
        "ReLU hinge (λ=1) — zero gradient for ordered pairs",
        crossing_penalty=1.0,
        crossing_mode="relu",
    ),
    _make_condition(
        "softplus_beta1",
        "crossing_mode",
        "Softplus hinge (λ=1, β=1) — smooth gradient",
        crossing_penalty=1.0,
        crossing_mode="softplus",
        softplus_beta=1.0,
    ),
    _make_condition(
        "softplus_beta10",
        "crossing_mode",
        "Softplus hinge (λ=1, β=10) — closer to ReLU",
        crossing_penalty=1.0,
        crossing_mode="softplus",
        softplus_beta=10.0,
    ),
    # ── Group 3: sort_output ──────────────────────────────────────────────────
    _make_condition(
        "no_sort",
        "sort_output",
        "No sort — penalty-only non-crossing (λ=1, relu)",
        crossing_penalty=1.0,
        sort_output=False,
    ),
    _make_condition(
        "sort_with_penalty",
        "sort_output",
        "Sort at inference + crossing penalty (λ=1)",
        crossing_penalty=1.0,
        sort_output=True,
    ),
    _make_condition(
        "sort_no_penalty", "sort_output", "Sort only — no crossing penalty", crossing_penalty=0.0, sort_output=True
    ),
    # ── Group 4: monotone_output ──────────────────────────────────────────────
    _make_condition(
        "raw",
        "monotone",
        "Raw linear output + crossing penalty (λ=1, baseline)",
        crossing_penalty=1.0,
        monotone_output=False,
    ),
    _make_condition(
        "monotone",
        "monotone",
        "Cumulative-softplus — non-crossing guaranteed by construction",
        crossing_penalty=0.0,
        monotone_output=True,
    ),
    # ── Group 5: loss_fn ──────────────────────────────────────────────────────
    _make_condition("pinball", "loss_fn", "Standard pinball loss (reference)", loss_fn="pinball", crossing_penalty=1.0),
    _make_condition(
        "huber_kappa_025", "loss_fn", "Huber-pinball κ=0.25", loss_fn="huber-pinball", kappa=0.25, crossing_penalty=1.0
    ),
    _make_condition(
        "huber_kappa_1", "loss_fn", "Huber-pinball κ=1.0", loss_fn="huber-pinball", kappa=1.0, crossing_penalty=1.0
    ),
    _make_condition(
        "winkler", "loss_fn", "Winkler interval score on outer pair + pinball", loss_fn="winkler", crossing_penalty=1.0
    ),
    # ── Group 6: best_config ──────────────────────────────────────────────────
    _make_condition(
        "default_config",
        "best_config",
        "Factory defaults — reference (λ=10, relu, pinball, no sort, raw)",
        crossing_penalty=10.0,
        crossing_mode="relu",
        loss_fn="pinball",
        sort_output=False,
        monotone_output=False,
    ),
    _make_condition(
        "best_config",
        "best_config",
        "Best settings from all groups — update after ablation",
        crossing_penalty=1.0,
        crossing_mode="relu",
        loss_fn="pinball",
        sort_output=False,
        monotone_output=False,
    ),
    _make_condition(
        "best_config_monotone",
        "best_config",
        "Best settings with monotone parameterisation",
        crossing_penalty=0.0,
        monotone_output=True,
        loss_fn="pinball",
    ),
]

# ---------------------------------------------------------------------------
# Spec + Engine
# ---------------------------------------------------------------------------

SPEC = ExperimentSpec(
    name="MLPGAM-QR Design Ablation",
    output_prefix="qr_ablation",
    condition_cls=MLPGAMQRConfig,
    backbone_cls=MLPGAMConfig,
    conditions=CONDITIONS,
    datasets=DATASETS,
    controlled_fields=frozenset(
        {
            # QR-specific fields — always set by condition, never from backbone HPO.
            "loss_fn",
            "kappa",
            "crossing_penalty",
            "sort_output",
            "monotone_output",
            # Backbone optimizer/scheduler fields absent on MLPGAMQRConfig.
            "opt_lr",
            "opt_weight_decay",
            "sch_warmup_epochs",
            "sch_eta_min",
        }
    ),
    reference_conditions={
        "crossing": "cross_0",
        "crossing_mode": "relu_1",
        "sort_output": "no_sort",
        "monotone": "raw",
        "loss_fn": "pinball",
        "best_config": "default_config",
    },
)

ENGINE = ExperimentEngine(SPEC)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_cfg = PipelineConfig(
        output_dir="../experiments/mlpgamqr_ablation/",
        out_activation="Identity",
        epochs=100,
        num_trials=50,
        split_freq="months",
        metrics=["mae", "rmse", "smape", "corr", "cwe", "picp", "nmpi", "winkler_score", "pinball", "crps"],
        figure_size=(9, 3),
        plot_dpi=150,
        log_level="INFO",
        log_file="results/mlpgamqr_ablation/run.log",
        use_latex=False,
        num_threads=1,
    )
    ENGINE.cli_main(base_cfg)
