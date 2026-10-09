# Chapter 4: MLPF (RQ2)

Answers RQ2: how a lightweight DNN can be designed to improve both computational efficiency and
forecasting performance in LV net-load forecasting.

| Script | Produces |
|---|---|
| `benchmark.py` | Point-forecasting benchmark: MLPF against N-HiTS/MLPGAM/MLPGAF (neural), LightGBM/XGBoost/CatBoost (tree-based), and a linear baseline, via expanding-window backtesting on MLVS-PT. |

```bash
uv run python chapters/ch4_mlpf/benchmark.py
```

**Known gap:** this script was originally a shared utility (`experiment/common/benchmark_backtesting.py`)
rather than living in its own chapter directory; it has not yet been re-audited line-by-line to
confirm every internal import and path is correct in its new location.
