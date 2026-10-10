# Covariate influence (RQ2)

Answers RQ2: the influence of meteorological covariates on net-load forecasting accuracy. Four conditions on one
MLPF backbone, varying only the exogenous (known-future) feature set: GHI + time, temperature + time, GHI +
temperature + time, and time only (no meteorological covariate).

| File | Produces |
|---|---|
| `conditions.py` | The four `Condition`s, each overriding `known_future_features` explicitly via `data_overrides` -- see its own docstring for why this bypasses `load_dataset`'s automatic weather-feature selection deliberately. |
| `experiment.py` | Runs the ablation on MLVS-PT and SPS-UK, reusing `best_practices/protocol.py`'s CV numbers. |
| `analysis.py` | NRMSE/CWE/PICP/nMPI per condition per dataset (fold-then-mean), DM + GW significance. |
| `figures.py` | Four-condition NRMSE/CWE bars. |

```bash
uv run python chapters/ch3_methodology/covariate_influence/experiment.py
uv run python chapters/ch3_methodology/covariate_influence/analysis.py
uv run python -c "from chapters.ch3_methodology.covariate_influence.figures import render_all; render_all()"
```
