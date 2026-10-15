"""Backtesting benchmark — point forecasting with expanding window.

Models compared
---------------
Neural networks     : N-HiTS, MLPF, MLPGAM, MLPGAF
Tree-based ML       : LightGBM, XGBoost, CatBoost
Linear baseline     : LinearRegression
Statistical baseline: SeasonalNaive, Naive

RQ4's benchmark also compares against MSTL in the legacy notebooks this script
descends from; MSTL has no twiga-native equivalent (it is a
statsforecast/darts-specific seasonal-trend decomposition), so it is dropped
here rather than reimplemented, and SeasonalNaive/Naive stand in as the
statistical floor RQ4's own "least training time"/"inference time comparable
to S-Naive" findings are checked against.

CV strategy
-----------
- Window     : expanding (train set grows with each fold)
- Min train  : 12 months
- Test step  : 3 months per fold
- Dataset    : MLVS-PT (30-min net-load, 2019-02 → 2021-12)

HPO strategy
------------
Hyperparameters are tuned once on fold 0 (train 80 % / val 20 %).
The best params are then frozen and reused across all folds so that
every fold uses identical architecture settings and only refits weights.

Usage
-----
    uv run python experiment/common/benchmark_backtesting.py
"""

from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from twiga.core.config import DataPipelineConfig, ExperimentConfig
from twiga.core.plot.matplot_theme import configure_matlib_style
from twiga.core.utils.logger import configure, get_logger, log_section
from twiga.forecaster.core import TwigaForecaster
from twiga.models.baseline import NAIVEConfig, SEASONALNAIVEConfig
from twiga.models.ml import CATBOOSTConfig, LIGHTGBMConfig, LINEAREGConfig, XGBOOSTConfig
from twiga.models.nn.mlpf_model import MLPFConfig
from twiga.models.nn.mlpgaf_model import MLPGAFConfig
from twiga.models.nn.mlpgam_model import MLPGAMConfig
from twiga.models.nn.nhits_model import NHITSConfig
from twiga.utils.data_loading import load_dataset

from twiga_thesis.paths import thesis_data_dir

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET = "MLVS-PT"
DATA_PATH = thesis_data_dir()
OUTPUT_DIR = Path("../experiments/benchmark_backtesting")
PROJECT_NAME = "benchmark-backtesting"

# 30-min resolution
PERIOD = "30min"
FORECAST_HORIZON = 48  # 1 day ahead  (48 × 30 min)
LOOKBACK = 336  # 1-week lookback (7 × 48)

# Expanding CV: ≥ 12 months training, 3-month test folds
TRAIN_SIZE_MONTHS = 12
TEST_SIZE_MONTHS = 3

# HPO: tune on fold-0 only
NUM_TRIALS = 20  # Optuna trials per model on fold 0
TUNE_VAL_SPLIT = 0.8  # fraction of fold-0 train used for fitting; rest for val

# NN training budget
MAX_EPOCHS = 30

METRICS = ["rmse", "mae", "corr", "wmape", "smape"]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup() -> None:
    """Initialize environment and create output directories.

    Disables OpenMP threading, suppresses warnings, configures matplotlib style,
    and creates output subdirectories for figures, metrics, predictions, and logs.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    configure_matlib_style(latex=False)
    for sub in ("figures", "results/metrics", "results/predictions", "logs"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, str, list, list, list, float, float, int | None]:
    """Load and preprocess dataset.

    Applies median filter to target variable and forward-fills missing values.

    Returns:
        A tuple of (data, target, calendar_features, exogenous_features, lags, latitude, longitude,
        lookback_window_size).
    """
    log.info("Loading dataset '%s' from '%s'", DATASET, DATA_PATH)
    data, target, cal, exog, lags, lat, lon, lookback = load_dataset(DATASET, data_path=str(DATA_PATH))
    data[target] = signal.medfilt(data[target].values, 3)
    data.ffill(inplace=True)
    log.info(
        "Dataset  rows=%d  target='%s'  span=[%s → %s]",
        len(data),
        target,
        data["timestamp"].min().strftime("%Y-%m"),
        data["timestamp"].max().strftime("%Y-%m"),
    )
    return data, target, cal, exog, lags, lat, lon, lookback


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def build_data_config(
    target: str,
    lags: list,
    cal: list,
    exog: list,
    lat: float,
    lon: float,
    lookback_window_size: int | None = None,
) -> DataPipelineConfig:
    """Build data pipeline configuration.

    Args:
        target: Target feature name.
        lags: Historical lags to include.
        cal: Calendar features.
        exog: Exogenous features.
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        lookback_window_size: ACF-derived recommendation from `load_dataset()`'s
            eighth return value, used in place of the module-level `LOOKBACK`
            constant when given. `None` (the default) keeps `LOOKBACK` exactly
            as before.

    Returns:
        Configured DataPipelineConfig instance.
    """
    return DataPipelineConfig(
        target_feature=target,
        historical_features=lags,
        period=PERIOD,
        latitude=lat,
        longitude=lon,
        calendar_features=cal,
        exogenous_features=exog,
        forecast_horizon=FORECAST_HORIZON,
        lookback_window_size=lookback_window_size if lookback_window_size is not None else LOOKBACK,
        input_scaler=StandardScaler(),
        target_scaler=MinMaxScaler(feature_range=(-1, 1)),
    )


def build_model_configs(cal: list, exog: list, lags: list) -> list:
    """10 models: 4 neural-network + 3 tree-based + 1 linear + 2 statistical baselines."""
    nn_kwargs = {
        "max_epochs": MAX_EPOCHS,
        "rich_progress_bar": False,
        "num_target_feature": 1,
        "forecast_horizon": FORECAST_HORIZON,
        "lookback_window_size": LOOKBACK,
        "num_calendar_features": len(cal),
        "num_exogenous_features": len(exog),
        "num_historical_features": len(lags),
        "num_future_covariates": 0,
    }
    return [
        # ── Neural networks ──────────────────────────────────────────────
        NHITSConfig(**nn_kwargs),
        MLPFConfig(out_activation_function="Identity", **nn_kwargs),
        MLPGAMConfig(out_activation_function="Identity", **nn_kwargs),
        MLPGAFConfig(out_activation_function="Identity", **nn_kwargs),
        # ── Tree-based ML ────────────────────────────────────────────────
        LIGHTGBMConfig(),
        XGBOOSTConfig(),
        CATBOOSTConfig(verbose=False),
        # ── Linear baseline ─────────────────────────────────────────────
        LINEAREGConfig(),
        # ── Statistical baselines ────────────────────────────────────────
        # Daily seasonality at 30-min resolution: one day = 48 samples.
        SEASONALNAIVEConfig(period="1D", freq=PERIOD),
        NAIVEConfig(strategy="window_last"),
    ]


def build_forecaster_config() -> ExperimentConfig:
    return ExperimentConfig(
        split_freq="months",
        window="expanding",
        train_size=TRAIN_SIZE_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        stride=TEST_SIZE_MONTHS,  # non-overlapping folds
        metrics=METRICS,
        project_name=PROJECT_NAME,
    )


# ---------------------------------------------------------------------------
# Fold-0 HPO
# ---------------------------------------------------------------------------


def tune_on_first_fold(
    forecaster: TwigaForecaster,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Tune all models on fold-0 data, freeze best params into forecaster.models.

    Steps
    -----
    1. Call ``forecaster.split(data)`` once to (a) initialise the cached
       ``_split_scheme`` for all folds and (b) retrieve fold-0's train slice.
    2. Split that train slice 80 / 20 → tune_train / tune_val.
    3. Call ``forecaster.tune(tune_train, tune_val)`` which updates
       ``forecaster.models`` in-place with each model's best Optuna params.
    4. Log the winning hyperparameter set for each model.

    Because ``_split_scheme`` is already set, the subsequent
    ``forecaster.backtesting(data)`` call reuses the exact same folds
    without re-computing them, and uses the tuned model configs for every fold.

    Returns:
        Fold-0's full training slice (before the 80/20 tune split), so callers
        (e.g. ``measure_train_times``) can reuse the same window without
        re-deriving the split scheme.
    """
    log.info("Initialising split scheme and extracting fold-0 train data …")

    # Advancing the generator once initialises _split_scheme for ALL folds
    # (set_split_scheme runs before the first yield) and gives us fold 0.
    # forecaster.split() yields twiga.core.backtester.SplitBundle, a 6-field
    # namedtuple (fit_df, val_df, calib_df, test_df, scheme, split_key) — this
    # unpack previously read `fold0_train, _, scheme0, _ = next(split_gen)` (4
    # names against a 6-tuple), which raises "too many values to unpack" the
    # instant this function is called. Found during the re-audit this
    # script's own README names as an open gap; fixed here.
    split_gen = forecaster.split(data)
    fold0_train, _val_df, _calib_df, _fold0_test, scheme0, _split_key = next(split_gen)

    log.info(
        "Fold 0 train window: %s → %s  (%d rows)",
        scheme0["train_period"][0],
        scheme0["train_period"][1],
        len(fold0_train),
    )

    # 80 / 20 split within fold-0 training data
    split_idx = int(len(fold0_train) * TUNE_VAL_SPLIT)
    tune_train = fold0_train.iloc[:split_idx].reset_index(drop=True)
    tune_val = fold0_train.iloc[split_idx:].reset_index(drop=True)

    log.info(
        "Tuning on fold-0  tune_train=%d rows  tune_val=%d rows  trials=%d",
        len(tune_train),
        len(tune_val),
        NUM_TRIALS,
    )

    forecaster.tune(train_df=tune_train, val_df=tune_val, num_trials=NUM_TRIALS)

    # Log best params per model
    log.info("─── Best hyperparameters (fold-0 tuning) ───")
    for model in forecaster.models:
        cfg = model.model_config
        name = cfg.name
        # Show only non-default / tunable fields
        params = {
            k: v
            for k, v in cfg.model_dump().items()
            if k not in {"name", "domain", "search_space", "initial_params"} and v is not None
        }
        log.info("  %-18s  %s", name, params)

    return fold0_train


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


def run_backtesting(
    forecaster: TwigaForecaster,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window backtesting with already-tuned model configs."""
    model_names = [type(m).__name__.replace("Model", "") for m in forecaster.models]
    log.info(
        "Backtesting  models=%s  train_min=%d mo  test=%d mo",
        model_names,
        TRAIN_SIZE_MONTHS,
        TEST_SIZE_MONTHS,
    )
    return forecaster.backtesting(data=data, verbose=True)


def measure_train_times(
    data_config: DataPipelineConfig,
    model_configs: list,
    fold0_train: pd.DataFrame,
) -> pd.DataFrame:
    """Time one fit of each model, on the same fold-0 training window, for the cost comparison.

    ``TwigaForecaster``'s multi-model ``backtesting()``/``evaluate_point_forecast()`` path only
    persists *inference* wall-time into its metrics frame (the ``inference-time`` column set in
    ``BaseForecaster._evaluate_core``). The training wall-time
    ``BaseForecaster._fit`` records (``self.train_walltime``) is set on the shared
    ``TwigaForecaster`` instance itself — overwritten on every model/fold iteration of its own
    internal loop — not on the individual items in ``forecaster.models``, so it cannot be read
    back per model after the fact (confirmed by reading ``TwigaForecaster.fit``/``_fit`` directly;
    an earlier version of this function assumed the opposite and always returned ``NaN``). This
    times one standalone fit per model directly instead, each on the same fold-0 train window
    `tune_on_first_fold` already extracted, so every model's figure is comparable.

    Args:
        data_config: The shared `DataPipelineConfig` (`build_data_config()`'s output).
        model_configs: The model configs under comparison (`build_model_configs()`'s output).
        fold0_train: Fold-0's training slice (same window `tune_on_first_fold` uses).

    Returns:
        One row per model: `Model` (matching the metrics frame's own `Model` column) and
        `train_time_s` (seconds, a single representative fit, not averaged across folds).
    """
    import time

    forecaster_config = build_forecaster_config()
    rows = []
    for model_config in model_configs:
        solo = TwigaForecaster(data_params=data_config, model_params=[model_config], train_params=forecaster_config)
        start = time.perf_counter()
        solo.fit(train_df=fold0_train)
        elapsed = time.perf_counter() - start
        rows.append({"Model": str(model_config.name).upper(), "train_time_s": elapsed})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rich_table(df: pd.DataFrame, title: str) -> str:
    display_cols = [c for c in ["RMSE", "MAE", "Corr", "WMAPE", "SMAPE"] if c in df.columns]
    minimize = {"RMSE", "MAE", "WMAPE", "SMAPE"}
    maximize = {"Corr"}

    best: dict = {}
    for col in display_cols:
        if col in minimize:
            best[col] = df[col].min()
        elif col in maximize:
            best[col] = df[col].max()

    table = Table(title=title, header_style="bold cyan", show_lines=True)
    table.add_column("Model", style="bold white", min_width=18)
    for col in display_cols:
        table.add_column(col, justify="right", min_width=9)

    for _, row in df.iterrows():
        cells = []
        for col in display_cols:
            val = f"{row[col]:.4f}"
            if col in best and row[col] == best[col]:
                val = f"[bold green]{val}[/bold green]"
            cells.append(val)
        table.add_row(str(row["Model"]), *cells)

    buf = StringIO()
    Console(file=buf, highlight=False, width=130).print(table)
    return buf.getvalue()


def report(metrics: pd.DataFrame) -> pd.DataFrame:
    """Generate and log performance report.

    Aggregates metrics by fold and model, logs per-fold and overall performance,
    and returns overall summary.

    Args:
        metrics: DataFrame with columns [Model, Folds, mae, rmse, corr, wmape, smape].

    Returns:
        Overall performance summary grouped by model.
    """
    rename = {"rmse": "RMSE", "mae": "MAE", "corr": "Corr", "wmape": "WMAPE", "smape": "SMAPE"}
    agg_cols = [c for c in rename if c in metrics.columns]

    per_fold = metrics.groupby(["Model", "Folds"])[agg_cols].mean().round(4).reset_index().rename(columns=rename)
    overall = metrics.groupby("Model")[agg_cols].mean().round(4).reset_index().rename(columns=rename)

    for fold_id in sorted(per_fold["Folds"].unique()):
        fold_df = per_fold[per_fold["Folds"] == fold_id].drop(columns="Folds")
        log.info("Fold %d\n%s", fold_id, _rich_table(fold_df, f"Fold {fold_id} Performance"))

    log.info("Overall (mean across folds)\n%s", _rich_table(overall, "Overall Performance"))
    return overall


# ---------------------------------------------------------------------------
# Save & plot
# ---------------------------------------------------------------------------


def save_results(predictions: pd.DataFrame, metrics: pd.DataFrame, train_times: pd.DataFrame | None = None) -> None:
    """Save predictions, metrics, and (if given) per-model train times to CSV.

    Args:
        predictions: DataFrame with model predictions.
        metrics: DataFrame with evaluation metrics.
        train_times: Optional per-model training wall-time, from
            :func:`capture_train_times`; read by ``analysis.py`` for the
            accuracy-vs-cost trade-off.
    """
    metrics.to_csv(OUTPUT_DIR / "results" / "metrics" / "backtest_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "results" / "predictions" / "backtest_predictions.csv", index=False)
    log.info("Saved metrics     → %s", OUTPUT_DIR / "results" / "metrics" / "backtest_metrics.csv")
    log.info("Saved predictions → %s", OUTPUT_DIR / "results" / "predictions" / "backtest_predictions.csv")
    if train_times is not None:
        path = OUTPUT_DIR / "results" / "metrics" / "backtest_train_times.csv"
        train_times.to_csv(path, index=False)
        log.info("Saved train times → %s", path)


def plot_results(predictions: pd.DataFrame, overall: pd.DataFrame) -> None:
    """Generate and save performance visualizations.

    Creates per-fold sample plots and comparison plots across models.

    Args:
        predictions: DataFrame with model predictions.
        overall: Overall performance summary by model.
    """
    configure_matlib_style(latex=False)
    models = list(predictions["Model"].unique())
    folds = sorted(predictions["Folds"].unique())
    n = len(models)

    # ── Per-fold sample plot (1 week of predictions for each model) ───────
    for fold in folds:
        fp = predictions[predictions["Folds"] == fold]
        ylim = (fp["Actual"].min() * 0.95, fp["Actual"].max() * 1.05)

        fig, axs = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)
        axs = np.atleast_1d(axs).ravel()

        for i, model in enumerate(models):
            df = fp[fp["Model"] == model].iloc[: 7 * FORECAST_HORIZON]
            axs[i].plot(df["Actual"].values, color="#1c2429", lw=1.2, label="Actual")
            axs[i].plot(df["forecast"].values, color="#069fac", lw=1.0, ls="--", label="Forecast")
            axs[i].set_ylabel("P (kW)", fontsize=8)
            axs[i].set_ylim(ylim)
            axs[i].set_title(model, fontsize=9, fontweight="bold")
            if i == 0:
                axs[i].legend(fontsize=7, loc="upper right")

        axs[-1].set_xlabel("Samples (30-min steps)", fontsize=8)
        fig.suptitle(f"Fold {fold} — one-week sample", fontsize=10, fontweight="bold")
        fig.tight_layout()
        path = OUTPUT_DIR / "figures" / f"fold_{fold}_sample.pdf"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved fold plot → %s", path)

    # ── Overall RMSE bar chart ────────────────────────────────────────────
    if "RMSE" in overall.columns:
        srt = overall.sort_values("RMSE")
        best_rmse = srt["RMSE"].min()
        colors = ["#069fac" if v == best_rmse else "#455a64" for v in srt["RMSE"]]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(srt["Model"], srt["RMSE"], color=colors)
        ax.set_xlabel("RMSE (mean across folds)")
        ax.set_title("Model Comparison — Expanding Backtesting")
        ax.invert_yaxis()
        fig.tight_layout()
        path = OUTPUT_DIR / "figures" / "rmse_comparison.pdf"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved RMSE chart → %s", path)

    # ── Per-fold metric heatmap ───────────────────────────────────────────
    if "RMSE" in overall.columns:
        fold_rmse = (
            predictions.groupby(["Model", "Folds"])
            .apply(lambda g: np.sqrt(((g["forecast"] - g["Actual"]) ** 2).mean()))
            .reset_index(name="RMSE")
            .pivot(index="Model", columns="Folds", values="RMSE")
        )
        fig, ax = plt.subplots(figsize=(max(6, len(folds) * 1.5), max(4, n * 0.6)))
        im = ax.imshow(fold_rmse.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(fold_rmse.index.tolist(), fontsize=8)
        plt.colorbar(im, ax=ax, label="RMSE")
        ax.set_title("RMSE per Model × Fold", fontsize=10)
        fig.tight_layout()
        path = OUTPUT_DIR / "figures" / "rmse_heatmap.pdf"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved RMSE heatmap → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure(level="INFO", log_file=str(OUTPUT_DIR / "logs" / "backtest.log"))

    log_section(log, "Setup")
    setup()

    log_section(log, "Data Loading")
    data, target, cal, exog, lags, lat, lon, lookback = load_data()

    log_section(log, "Configuration")
    data_config = build_data_config(target, lags, cal, exog, lat, lon, lookback_window_size=lookback)
    model_configs = build_model_configs(cal, exog, lags)
    forecaster_config = build_forecaster_config()

    log.info(
        "Models : %s",
        [type(m).__name__.replace("Config", "") for m in model_configs],
    )
    log.info(
        "CV     : expanding window, min_train=%d mo, test=%d mo",
        TRAIN_SIZE_MONTHS,
        TEST_SIZE_MONTHS,
    )

    log_section(log, "Forecaster Init")
    forecaster = TwigaForecaster(
        data_params=data_config,
        model_params=model_configs,
        train_params=forecaster_config,
    )

    log_section(log, "Fold-0 Hyperparameter Tuning")
    fold0_train = tune_on_first_fold(forecaster, data)

    log_section(log, "Expanding-Window Backtesting")
    predictions, metrics = run_backtesting(forecaster, data)

    log_section(log, "Results")
    overall = report(metrics)

    log_section(log, "Training-Time Measurement")
    train_times = measure_train_times(data_config, model_configs, fold0_train)

    log_section(log, "Saving")
    save_results(predictions, metrics, train_times)
    plot_results(predictions, overall)

    log.info("Done. All results in %s", OUTPUT_DIR)
