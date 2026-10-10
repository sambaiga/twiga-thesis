"""Chapter 4 RQ3 — autoregressive-iterative (ARI) robustness of MLPF.

Trains MLPF once via the standard expanding-window backtest protocol
(reusing `benchmark.py`'s own CV numbers directly -- `TRAIN_SIZE_MONTHS`,
`TEST_SIZE_MONTHS`, `FORECAST_HORIZON`, `LOOKBACK`, `DATASET`, `DATA_PATH` --
imported rather than redefined, so this script and `benchmark.py` cannot
drift onto different protocol numbers), then evaluates k-day-ahead forecasts
produced by iteratively feeding the model's own prior-day output back in as
input, for k = 2..7.

ARI mechanism: `twiga.utils.recursive_forecast.recursive_forecast` already
implements exactly this ("rolls a fitted forecaster forward beyond its native
forecast_horizon by repeatedly calling predict() on a sliding raw-data window
and splicing each step's prediction back into the target column as if it
were observed history") -- confirmed by reading its full source before
writing this module, so no manual feedback loop is written here.

Scope note on CWE (flagged, not silently decided): the plan's own
`subfig:ARI_nrsme`/`ARI_ciwe` wants both NRMSE and CWE against k. NRMSE needs
only a point forecast, which plain `MLPFConfig` (as used throughout
`benchmark.py`) already produces, and `recursive_forecast`'s own
implementation explicitly operates on that point-array shape
(`pred_dict[name][-1]`, shape `(forecast_horizon, n_targets)`). CWE needs a
prediction *interval*, which means swapping in one of `twiga`'s
distributional MLPF variants (e.g. `MLPFLaplaceConfig`/`MLPFNormalConfig`);
whether that variant's `predict()` output is shaped the same way
`recursive_forecast` assumes could not be confirmed against real data in
this clone (no MLVS-PT/SPS-UK parquet is present locally -- see the data
README), and `recursive_forecast`'s own docstring claim ("works unmodified
for any registered model") does not itself establish the quantile/
distribution output shape matches the point-array indexing the function
performs. Rather than wire that up unverified and risk a silently wrong
result, this module implements the NRMSE-vs-k half only, and leaves a
`predict_interval`-based CWE-vs-k extension as a documented follow-up
(`cwe_vs_k` below is the integration point, currently `NotImplementedError`).

Usage::

    uv run python chapters/ch4_mlpf/ari_robustness.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from twiga.core.config import ExperimentConfig
from twiga.forecaster.core import TwigaForecaster
from twiga.models.nn.mlpf_model import MLPFConfig
from twiga.utils.recursive_forecast import recursive_forecast

from chapters.ch4_mlpf.benchmark import (
    FORECAST_HORIZON,
    LOOKBACK,
    TEST_SIZE_MONTHS,
    TRAIN_SIZE_MONTHS,
    build_data_config,
    load_data,
    setup,
)
from twiga_thesis.plotting import line_style, tokens

OUTPUT_DIR = Path("../experiments/ch4_mlpf/ari_robustness")
K_DAYS = range(2, 8)  # k = 2..7 day-ahead, per RQ3
MAX_EPOCHS = 30


def build_mlpf_config(cal: list, exog: list, lags: list) -> MLPFConfig:
    """MLPF's own default architecture (benchmark.py's reference config), point forecast only."""
    return MLPFConfig(
        out_activation_function="Identity",
        max_epochs=MAX_EPOCHS,
        rich_progress_bar=False,
        num_target_feature=1,
        forecast_horizon=FORECAST_HORIZON,
        lookback_window_size=LOOKBACK,
        num_calendar_features=len(cal),
        num_exogenous_features=len(exog),
        num_historical_features=len(lags),
        num_future_covariates=0,
    )


def fit_backbone(
    data: pd.DataFrame, data_config, model_config: MLPFConfig
) -> tuple[TwigaForecaster, pd.DataFrame, pd.DataFrame]:
    """Fit MLPF once on the first expanding-window fold (benchmark.py's own CV numbers).

    Args:
        data: Full preprocessed dataset (`benchmark.load_data()`'s first return value).
        data_config: `DataPipelineConfig` (`benchmark.build_data_config()`'s output).
        model_config: `build_mlpf_config()`'s output.

    Returns:
        `(forecaster, fold0_train, fold0_test)` -- the fitted forecaster and the
        first fold's own train/test split, so `fold0_test` can seed the ARI
        rollout's `future_df` (it has real covariates and ground truth to score against).
    """
    forecaster_config = ExperimentConfig(
        split_freq="months",
        window="expanding",
        train_size=TRAIN_SIZE_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        stride=TEST_SIZE_MONTHS,
        metrics=["rmse", "mae", "corr", "wmape", "smape"],
        project_name="ari-robustness",
    )
    forecaster = TwigaForecaster(data_params=data_config, model_params=[model_config], train_params=forecaster_config)

    # forecaster.split() yields twiga.core.backtester.SplitBundle, a 6-field namedtuple
    # (fit_df, val_df, calib_df, test_df, scheme, split_key) -- see benchmark.py's
    # tune_on_first_fold, where the same unpack shape was fixed during this pass's re-audit.
    split_gen = forecaster.split(data)
    fold0_train, _val_df, _calib_df, fold0_test, _scheme0, _split_key = next(split_gen)

    # forecaster.fit() fits the data pipeline itself when it is not already fitted
    # (TwigaForecaster.fit, twiga/forecaster/core.py), so no separate pipeline-fit call is needed.
    forecaster.fit(train_df=fold0_train)
    return forecaster, fold0_train, fold0_test


def ari_nrmse_vs_k(
    forecaster: TwigaForecaster,
    fold0_train: pd.DataFrame,
    fold0_test: pd.DataFrame,
    target: str,
    k_days: range = K_DAYS,
) -> pd.DataFrame:
    """NRMSE of the k-day-ahead ARI forecast, for every k in `k_days`.

    Args:
        forecaster: The fitted forecaster from `fit_backbone`.
        fold0_train: First fold's train slice (the ARI rollout's `history_df`).
        fold0_test: First fold's test slice (its timestamps/covariates seed
            `future_df`; its own target values are the ground truth scored against).
        target: Target column name.
        k_days: Which k (in days) to evaluate; each day is `FORECAST_HORIZON` samples.

    Returns:
        Columns `k` (days) and `nrmse`.

    Raises:
        ValueError: If `fold0_test` has fewer than `max(k_days) * FORECAST_HORIZON` rows.
    """
    from twiga.core.metrics.point import nrmse  # noqa: PLC0415 (avoid import cost when unused)

    max_k = max(k_days)
    required_rows = max_k * FORECAST_HORIZON
    if len(fold0_test) < required_rows:
        raise ValueError(f"fold0_test has {len(fold0_test)} rows; need at least {required_rows} for k up to {max_k}")

    # `_predict_core` (twiga/forecaster/base.py) keys its prediction dict by
    # `model.model_config.name` (e.g. "mlpf"), not by any `model_type` attribute on the model
    # object itself -- the latter is only ever set transiently on the *shared* forecaster
    # instance during its internal fit/predict loop, not persisted per model (the same fact
    # that made `benchmark.py`'s original `capture_train_times` attempt read back `NaN`; fixed
    # there via `measure_train_times` instead).
    model_name = forecaster.models[0].model_config.name
    rows = []
    for k in k_days:
        k_steps = k * FORECAST_HORIZON
        rolled = recursive_forecast(
            forecaster,
            history_df=fold0_train,
            future_df=fold0_test.iloc[:k_steps],
            k_steps=k_steps,
            model_name=model_name,
        )
        truth = fold0_test.iloc[:k_steps][target].to_numpy()
        forecast = rolled[target].to_numpy()
        rows.append({"k": k, "nrmse": float(nrmse(truth, forecast, axis=None))})
    return pd.DataFrame(rows)


def cwe_vs_k(*args: object, **kwargs: object) -> pd.DataFrame:
    """CWE of the k-day-ahead ARI forecast -- not implemented in this pass.

    See this module's docstring: wiring a distributional MLPF variant
    (`MLPFLaplaceConfig`/`MLPFNormalConfig`) through `recursive_forecast`
    could not be verified against real data in this clone, so it is left as
    an explicit follow-up rather than guessed at.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "CWE-vs-k needs a distributional MLPF variant whose predict() output shape through "
        "recursive_forecast has not been verified against real data; see this module's docstring."
    )


def plot_nrmse_vs_k(table: pd.DataFrame):
    """NRMSE against k (days ahead), matching `subfig:ARI_nrsme`."""
    line_style.configure(latex=False)
    fig, ax = line_style.new_panel()
    ordered = table.sort_values("k")
    ax.plot(ordered["k"], ordered["nrmse"], "-o", color=tokens.PALETTE[0], lw=line_style.LINE_WIDTH)
    ax.set_xlabel("Forecast horizon k (days)")
    ax.set_ylabel("NRMSE")
    line_style.set_y_axis(ax, ordered["nrmse"])
    line_style.finish(ax)
    return fig


def main() -> None:
    setup()
    data, target, cal, exog, lags, lat, lon, lookback = load_data()
    data_config = build_data_config(target, lags, cal, exog, lat, lon, lookback_window_size=lookback)
    model_config = build_mlpf_config(cal, exog, lags)

    forecaster, fold0_train, fold0_test = fit_backbone(data, data_config, model_config)
    table = ari_nrmse_vs_k(forecaster, fold0_train, fold0_test, target)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "ari_nrmse_vs_k.csv", index=False)
    print(table.to_string(index=False))

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    line_style.save(plot_nrmse_vs_k(table), figures_dir / "ari_nrmse_vs_k")


if __name__ == "__main__":
    main()
