"""Embedding ablation experiment for MLPGAM and MLPGAF.

Assesses whether value and positional embeddings improve MLP-based
time series forecasting by running a controlled ablation over all
embedding combinations while holding every other hyperparameter fixed.
Both MLPGAM and MLPGAF are evaluated under identical conditions so
results are directly comparable across architectures.

Example::

    python experiment/ch6_conformal/exploratory/data_embedding.py
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from twiga.core.utils import LogContext, configure, get_logger, log_section
from twiga.models.nn import MLPGAFConfig, MLPGAMConfig
from twiga.utils.run_forecast import (
    PipelineConfig,
    _resolve_output_dirs,
    build_data_config,
    build_forecaster,
    build_forecaster_config,
    cleanup_training_artifacts,
    evaluate,
    load_and_preprocess,
    save_metrics,
    save_predictions,
    setup_environment,
    split_data,
    visualize_results,
)

log = get_logger(__name__)

try:
    from twiga.tracking import TwigaTracker as _TwigaTracker

    _MLFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001
    _MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Embedding condition definition
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCondition:
    """A single ablation condition fixing the embedding configuration.

    Attributes:
        name: Short identifier used in filenames and log messages.
        value_embed_type: Value embedding type passed to the model config.
        embedding_type: Positional embedding type passed to the model config.
        description: Human-readable explanation of what the condition tests.
    """

    name: str
    value_embed_type: Literal["LinearEmb", "ConvEmb"] | None
    embedding_type: Literal["PosEmb", "LearnPosEmb"] | None
    description: str


# All conditions — every other hyperparameter is inherited from the base config.
# The same conditions are applied to both MLPGAM and MLPGAF so results are
# directly comparable across architectures.
ABLATION_CONDITIONS: list[EmbeddingCondition] = [
    EmbeddingCondition(
        name="baseline",
        value_embed_type=None,
        embedding_type=None,
        description="No embedding — raw features fed directly to MLP",
    ),
    EmbeddingCondition(
        name="pos_only",
        value_embed_type=None,
        embedding_type="LearnPosEmb",
        description="Learnable positional embedding only — no value projection",
    ),
    EmbeddingCondition(
        name="value_linear",
        value_embed_type="LinearEmb",
        embedding_type=None,
        description="Linear value projection only — no positional encoding",
    ),
    EmbeddingCondition(
        name="value_conv",
        value_embed_type="ConvEmb",
        embedding_type=None,
        description="Convolutional value projection only — no positional encoding",
    ),
    EmbeddingCondition(
        name="full_linear_learnable",
        value_embed_type="LinearEmb",
        embedding_type="LearnPosEmb",
        description="Full embedding: linear value + learnable positional",
    ),
    EmbeddingCondition(
        name="full_conv_learnable",
        value_embed_type="ConvEmb",
        embedding_type="LearnPosEmb",
        description="Full embedding: conv value + learnable positional",
    ),
    EmbeddingCondition(
        name="full_linear_fixed",
        value_embed_type="LinearEmb",
        embedding_type="PosEmb",
        description="Full embedding: linear value + fixed sinusoidal positional",
    ),
]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_EXPERIMENT_PREFIX = "embedding_ablation"
_EXPERIMENT_NAME = "Embedding Ablation"
_REFERENCE_CONDITION: dict[str, str] = {
    "mlpgam": "baseline",
    "mlpgaf": "baseline",
}

# Maps a short model tag to (ConfigClass, field names that accept embedding).
# Both configs share the same field names so a single builder handles both.
_MODEL_REGISTRY: dict[str, type] = {
    "mlpgam": MLPGAMConfig,
    "mlpgaf": MLPGAFConfig,
}

# Paths to pre-tuned best hyperparameter files produced by a prior Optuna run.
# These are applied to every condition so that only the embedding fields vary.
_BEST_PARAMS_PATHS: dict[str, Path] = {
    "mlpgam": Path("results/MLVS-PT/mlpgam/best_params.json"),
    "mlpgaf": Path("results/MLVS-PT/mlpgaf/best_params.json"),
}

# Fields that are intentionally NOT loaded from best_params — these are
# controlled by the ablation condition and must never be overwritten.
_EMBEDDING_FIELDS: frozenset[str] = frozenset({"value_embed_type", "embedding_type"})


def _save_plots(root: Path) -> None:
    """Save ablation figures after summarise_ablation completes."""
    try:
        from twiga.core.plot.evaluation import save_ablation_plots  # noqa: PLC0415

        save_ablation_plots(root, prefix=_EXPERIMENT_PREFIX, name=_EXPERIMENT_NAME, reference=_REFERENCE_CONDITION)
        log.info("Ablation figures saved  dir=%s", root.parent / "figures")
    except Exception:
        log.warning("Figure export skipped (non-critical)", exc_info=True)


def _add_delta(summary: pd.DataFrame, col: str, refs: dict[str, str], *, lower_is_better: bool) -> None:
    """Add a Δ-vs-reference column in-place. Positive Δ = better than reference."""
    if col not in summary.columns:
        return
    sign = -1.0 if lower_is_better else 1.0
    delta_col = f"Δ {col} vs ref (%, +ve=better)"
    for model, ref_cond in refs.items():
        idx = (model, ref_cond)
        if idx not in summary.index:
            continue
        ref_val = float(summary.loc[idx, col])
        mask = summary.index.get_level_values("model") == model
        summary.loc[mask, delta_col] = (sign * (summary.loc[mask, col] - ref_val) / (abs(ref_val) + 1e-9) * 100).round(
            2
        )


def load_best_params(model_tag: str) -> dict:
    """Load pre-tuned hyperparameters from a saved ``.json`` file.

    Embedding fields are stripped from the result so they cannot
    accidentally overwrite the ablation condition.

    Args:
        model_tag: One of ``"mlpgam"`` or ``"mlpgaf"``.

    Returns:
        Dictionary of hyperparameters with all embedding fields removed.
        Returns an empty dict and logs a warning when the file is missing.
    """
    path = _BEST_PARAMS_PATHS[model_tag]
    if not path.exists():
        log.warning(
            "best_params not found for %s — using config defaults  path=%s",
            model_tag,
            path,
        )
        return {}

    params: dict = json.loads(path.read_text())
    prefix = f"{model_tag}_"
    params = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in params.items()}

    # Drop embedding fields — these are set by the ablation condition
    stripped = {k: v for k, v in params.items() if k not in _EMBEDDING_FIELDS}
    log.info(
        "Loaded best_params for %s  n_params=%d  stripped=%s  path=%s",
        model_tag,
        len(stripped),
        sorted(params.keys() & _EMBEDDING_FIELDS),
        path,
    )
    return stripped


# ---------------------------------------------------------------------------
# Per-condition, per-model runner
# ---------------------------------------------------------------------------


def _build_model_config(
    model_tag: str,
    data_config,
    condition: EmbeddingCondition,
    cfg: PipelineConfig,
):
    """Instantiate a model config seeded with pre-tuned hyperparameters.

    The config is built in three layers of precedence (lowest → highest):

    1. ``from_data_config`` defaults — infers input/output dimensions.
    2. ``best_params`` loaded from the saved ``.json`` file — replaces all
       tunable fields except the embedding fields.
    3. Ablation condition — sets ``value_embed_type`` and ``embedding_type``
       last so they always reflect the current experimental condition.

    Args:
        model_tag: One of ``"mlpgam"`` or ``"mlpgaf"``.
        data_config: Populated :class:`~twiga.core.config.DataPipelineConfig`.
        condition: Embedding condition to apply.
        cfg: Base pipeline configuration.

    Returns:
        A fully configured model config instance.

    Raises:
        KeyError: If *model_tag* is not registered in :data:`_MODEL_REGISTRY`.
    """
    config_cls = _MODEL_REGISTRY[model_tag]
    model_config = config_cls.from_data_config(data_config)

    # Layer 2 — apply pre-tuned params (embedding fields already stripped)
    best_params = load_best_params(model_tag)
    for key, value in best_params.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        else:
            log.warning("best_params key '%s' not found on %s config — skipping", key, model_tag)

    # Layer 3 — pipeline-level overrides
    model_config.out_activation_function = cfg.out_activation
    model_config.max_epochs = cfg.epochs
    model_config.rich_progress_bar = False

    # Layer 4 — ablation condition (always last, always wins)
    model_config.value_embed_type = condition.value_embed_type
    model_config.embedding_type = condition.embedding_type

    return model_config


def run_condition(
    model_tag: str,
    condition: EmbeddingCondition,
    cfg: PipelineConfig,
    data: pd.DataFrame,
    target_series: str,
    calendar_variables: list,
    exogenous_features: list,
    lags: list,
    latitude: float,
    longitude: float,
    lookback_window_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate one model under one embedding condition.

    Args:
        model_tag: Model identifier — ``"mlpgam"`` or ``"mlpgaf"``.
        condition: Embedding condition to evaluate.
        cfg: Base pipeline configuration shared across all runs.
        data: Full preprocessed dataset ``DataFrame``.
        target_series: Name of the target column.
        calendar_variables: Calendar feature column names.
        exogenous_features: Exogenous feature column names.
        lags: Lag feature column names.
        latitude: Geographic latitude for solar features.
        longitude: Geographic longitude for solar features.
        lookback_window_size: ACF-derived recommendation from `load_and_preprocess()`'s
            eighth return value. `None` (the default) keeps `cfg.input_window_size`.

    Returns:
        A 2-tuple of ``(pred, metric)`` ``DataFrame`` objects, each
        tagged with ``"model"`` and ``"condition"`` columns.
    """
    # Build a run-local config copy so the shared cfg is never mutated.
    # project_name is formatted as "<dataset>/<model_tag>/<condition>" which
    # gives every ForecasterConfig a unique, human-readable identifier.
    run_project_name = f"{cfg.dataset_name}/{model_tag}/{condition.name}"
    run_cfg = dataclasses.replace(cfg, project_name=run_project_name)

    with LogContext(log, model=model_tag, condition=condition.name):
        log.info("Starting  project='%s'  desc='%s'", run_project_name, condition.description)

        # Isolated output directory per model × condition
        condition_output = Path(run_cfg.output_dir) / model_tag / condition.name
        output_dirs = _resolve_output_dirs(condition_output)

        data_config = build_data_config(
            run_cfg,
            target_series,
            lags,
            calendar_variables,
            exogenous_features,
            latitude,
            longitude,
            lookback_window_size=lookback_window_size,
        )

        model_config = _build_model_config(model_tag, data_config, condition, run_cfg)
        forecaster_config = build_forecaster_config(run_cfg)
        forecaster = build_forecaster(data_config, [model_config], forecaster_config)

        train_df, test_df = split_data(data, run_cfg)
        split_idx = int(len(train_df) * run_cfg.train_val_split)
        fit_df = train_df.iloc[:split_idx]
        val_df = train_df.iloc[split_idx:]

        # Only tune if best_params were not found — if they were loaded the
        # model config is already fully populated and tuning would be wasteful.
        best_params_available = bool(load_best_params(model_tag))
        if best_params_available:
            log.info(
                "Skipping tuning — best_params loaded for %s  fit_n=%d  val_n=%d",
                model_tag,
                len(fit_df),
                len(val_df),
            )
        else:
            log.info("Tuning  trials=%d  fit_n=%d  val_n=%d", run_cfg.num_trials, len(fit_df), len(val_df))
            forecaster.tune(train_df=fit_df, val_df=val_df, num_trials=run_cfg.num_trials)

        log.info("Fitting  n=%d", len(fit_df))
        forecaster.fit(train_df=fit_df, val_df=val_df)

        pred, metric = evaluate(forecaster, test_df, run_cfg)

        # Tag results for downstream aggregation
        pred["model"] = model_tag
        pred["condition"] = condition.name
        metric["model"] = model_tag
        metric["condition"] = condition.name

        save_predictions(pred, output_dirs, run_cfg)
        save_metrics(metric, output_dirs, run_cfg)
        visualize_results(pred, output_dirs, run_cfg)

        if _MLFLOW_AVAILABLE:
            with _TwigaTracker(
                experiment="twiga/embedding_ablation",
                run_name=f"{model_tag}/{condition.name}",
                tags={"model": model_tag, "dataset": run_cfg.dataset_name},
                tracking_uri=str(output_dirs["mlruns"].resolve()),
            ) as tracker:
                tracker.log_forecaster_metadata(forecaster)
                tracker.log_evaluation(metric, pred)

        log.info(
            "Condition complete  mae=%.4f  rmse=%.4f  corr=%.4f",
            metric["mae"].mean(),
            metric["rmse"].mean(),
            metric["corr"].mean(),
        )

    del forecaster
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        log.debug("torch.cuda.empty_cache skipped: %s", exc)
    cleanup_training_artifacts(run_cfg.output_dir, project_name=run_cfg.project_name)

    return pred, metric


# ---------------------------------------------------------------------------
# Aggregation & summary
# ---------------------------------------------------------------------------


def summarise_ablation(
    all_metrics: list[pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Aggregate per-condition metrics into a single ranked summary table.

    Results are grouped by ``(model, condition)`` and sorted by MAE
    ascending within each model so the best-performing embedding
    configuration appears first per architecture.

    Both a full per-fold breakdown (``ablation_full.csv``) and the mean
    summary (``ablation_summary.csv``) are written to *output_dir*.

    Args:
        all_metrics: List of metric ``DataFrame`` objects, one per
            ``(model, condition)`` pair.
        output_dir: Root output directory for the experiment.

    Returns:
        Summary ``DataFrame`` indexed by ``(model, condition)`` with
        mean MAE, RMSE, and Corr columns.
    """
    combined = pd.concat(all_metrics, axis=0)

    # Full per-fold breakdown
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    full_path = results_dir / "ablation_full.csv"
    combined.to_csv(full_path, index=False)
    log.info("Full results saved  path=%s", full_path)

    # Mean summary grouped by model × condition, sorted by MAE within model
    summary = (
        combined.groupby(["model", "condition"])[["mae", "rmse", "corr"]]
        .mean()
        .round(4)
        .rename(columns={"mae": "MAE", "rmse": "RMSE", "corr": "Corr"})
    )
    summary = summary.groupby("model", group_keys=False).apply(lambda g: g.sort_values("MAE"))

    _add_delta(summary, "MAE", _REFERENCE_CONDITION, lower_is_better=True)

    summary_path = results_dir / "ablation_summary.csv"
    summary.reset_index().to_csv(summary_path, index=False)
    log.info("Ablation summary saved  path=%s\n%s", summary_path, summary.to_string())
    _save_plots(results_dir)
    return summary


# ---------------------------------------------------------------------------
# Top-level experiment runner
# ---------------------------------------------------------------------------


def run_ablation(cfg: PipelineConfig | None = None) -> pd.DataFrame:
    """Run the full embedding ablation experiment across both architectures.

    Iterates over the Cartesian product of :data:`_MODEL_REGISTRY` ×
    :data:`ABLATION_CONDITIONS`, trains and evaluates a separate model for
    each pair, then aggregates results into a ranked summary table.

    Args:
        cfg: Base pipeline configuration shared across all runs.  Defaults
            to :class:`~run_forecast.PipelineConfig` with all default
            values when ``None``.

    Returns:
        Summary ``DataFrame`` with mean MAE, RMSE, and Corr per
        ``(model, condition)`` pair, sorted by MAE ascending within
        each model.
    """
    if cfg is None:
        cfg = PipelineConfig()

    configure(level=cfg.log_level, log_file=cfg.log_file)

    log_section(log, "Embedding Ablation Experiment")
    log.info(
        "Models=%s  conditions=%d  total_runs=%d",
        list(_MODEL_REGISTRY),
        len(ABLATION_CONDITIONS),
        len(_MODEL_REGISTRY) * len(ABLATION_CONDITIONS),
    )

    setup_environment(cfg)

    log_section(log, "Data Loading")
    (
        data,
        target_series,
        calendar_variables,
        exogenous_features,
        lags,
        latitude,
        longitude,
        lookback,
    ) = load_and_preprocess(cfg)

    all_metrics: list[pd.DataFrame] = []

    for model_tag in _MODEL_REGISTRY:
        log_section(log, f"Model: {model_tag.upper()}")

        for condition in ABLATION_CONDITIONS:
            log_section(log, f"{model_tag.upper()} — {condition.name}")

            _, metric = run_condition(
                model_tag=model_tag,
                condition=condition,
                cfg=cfg,
                data=data,
                target_series=target_series,
                calendar_variables=calendar_variables,
                exogenous_features=exogenous_features,
                lags=lags,
                latitude=latitude,
                longitude=longitude,
                lookback_window_size=lookback,
            )
            all_metrics.append(metric)

    log_section(log, "Ablation Summary")
    return summarise_ablation(all_metrics, cfg.output_dir)


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ablation_cfg = PipelineConfig(
        dataset_name="MLVS-PT",
        data_path="examples/data",
        output_dir="../experiments/embedding_ablation/",
        train_start="2019-03",
        train_end="2020-02",
        test_start="2020-03",
        test_end="2020-06",
        input_window_size=48,
        forecast_horizon=48,
        period="30min",
        out_activation="Identity",
        epochs=500,
        train_val_split=0.7,
        # Was 10: never leaves TPESampler's n_startup_trials=patience*2=20
        # (twiga/forecaster/base.py:663) random-startup phase, so every trial
        # is a seed-determined draw independent of the observed objective --
        # same bug found and fixed in crc_experiment.py/mlpgam_design_ablation.py
        # (see that README's 2026-09-18 entry) and fpqr_experiment.py.
        num_trials=50,
        split_freq="months",
        cv_train_size=3,
        cv_test_size=1,
        metrics=["rmse", "mae", "corr"],
        figure_size=(9, 3),
        plot_dpi=150,
        log_level="INFO",
        log_file="../experiments/embedding_ablation/logs/run.log",
        use_latex=False,
        num_threads=1,
    )

    summary = run_ablation(ablation_cfg)
    print(summary.to_string())
