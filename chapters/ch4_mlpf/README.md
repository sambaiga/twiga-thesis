# Chapter 4: MLPF (RQ1, RQ2, RQ3, RQ4)

Answers RQ2: how a lightweight DNN can be designed to improve both computational efficiency and
forecasting performance in LV net-load forecasting, plus the MLPF-specific design-choice (RQ1),
autoregressive-iterative robustness (RQ3), and benchmark (RQ4) questions the thesis's MLPF chapter
asks alongside it.

| Script | Produces |
|---|---|
| `benchmark.py` | Point-forecasting benchmark (RQ4): MLPF against N-HiTS/MLPGAM/MLPGAF (neural), LightGBM/XGBoost/CatBoost (tree-based), LinearRegression, and SeasonalNaive/Naive (statistical baselines), via expanding-window backtesting on MLVS-PT. |
| `analysis.py` | RQ4 analysis: per-model fold-then-mean accuracy table, DM/MCS significance, inference-time table, accuracy-vs-cost trade-off table, and seasonal NRMSE breakdown, from `benchmark.py`'s own saved CSVs. |
| `figures.py` | RQ4 figures: accuracy-vs-inference-time scatter and seasonal NRMSE line plot, from `analysis.py`'s tables. |
| `design_choices.py` | RQ1: MLPF design-choice ablations on SPS-UK — forecast-horizon sweep (1h-7d), context-window sweep (1d-7d), positional-embedding choice (`None`/`PosEmb`/`LearnPosEmb`), and feature-combination choice (`addition-comb`/`attn-comb`/`weighted-comb`), on `twiga.experiment.ExperimentEngine`. |
| `ari_robustness.py` | RQ3: trains MLPF once via the standard expanding-window backtest, then evaluates autoregressive-iterative k-day-ahead forecasts (k=2..7, via `twiga.utils.recursive_forecast.recursive_forecast`) and reports NRMSE vs. k. |

```bash
uv run python chapters/ch4_mlpf/benchmark.py
uv run python chapters/ch4_mlpf/design_choices.py
uv run python chapters/ch4_mlpf/ari_robustness.py
```

**Re-audit note (previously an open gap in this README):** `benchmark.py`'s imports/paths from its
current location were checked line-by-line; they all resolve correctly. The re-audit did surface
one real bug, now fixed: `tune_on_first_fold`'s `forecaster.split(data)` unpack assumed a 4-tuple
(`fold0_train, _, scheme0, _ = next(split_gen)`), but `BaseForecaster.split` actually yields
`twiga.core.backtester.SplitBundle`, a 6-field namedtuple (`fit_df, val_df, calib_df, test_df,
scheme, split_key`) — the old unpack would have raised `ValueError: too many values to unpack` the
first time the function ran. `ari_robustness.py`'s own use of the same `split()` call follows the
corrected 6-field unpack from the start.

MSTL (one of RQ4's legacy baselines) has no twiga-native equivalent (a statsforecast/darts-specific
seasonal-trend decomposition) and is not included; SeasonalNaive/Naive stand in as the statistical
floor instead.
