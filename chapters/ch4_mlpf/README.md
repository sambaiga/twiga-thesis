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
| `design_choices.py` | RQ1: MLPF design-choice ablations on SPS-UK, covering a forecast-horizon sweep (1h-7d), a context-window sweep (1d-7d), positional-embedding choice (`None`/`PosEmb`/`LearnPosEmb`), and feature-combination choice (`addition-comb`/`attn-comb`/`weighted-comb`), on `twiga.experiment.ExperimentEngine`. |
| `ari_robustness.py` | RQ3: trains MLPF once via the standard expanding-window backtest, then evaluates autoregressive-iterative k-day-ahead forecasts (k=2..7, via `twiga.utils.recursive_forecast.recursive_forecast`) and reports NRMSE vs. k. |

```bash
uv run python chapters/ch4_mlpf/benchmark.py
uv run python chapters/ch4_mlpf/design_choices.py
uv run python chapters/ch4_mlpf/ari_robustness.py
```

MSTL, one of RQ4's baselines in the original study, has no twiga-native equivalent and is therefore
not included; SeasonalNaive and Naive provide the statistical floor instead.
