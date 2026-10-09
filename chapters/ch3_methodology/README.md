# Chapter 3: Methodology (RQ1)

Answers RQ1: how ML forecasting methods for net-load at secondary substations can be developed and
evaluated, considering the influence of meteorological variables on accuracy.

| Directory/script | Produces |
|---|---|
| `cwe_validation/` | The CWE metric's definition, design-choice ablations, and soundness validation (own [README](cwe_validation/README.md)). |
| `figures/feature_evidence.py` | Rank-fusion consensus evidence: ACF-derived lookback/lag split, per-metric ranker disagreement on the reactive-power target. |
| `figures/figures.py` | General Chapter 3 line-style figures. |

Datasets: MLVS-PT, SPS-UK (see `data/README.md`).

```bash
uv run python chapters/ch3_methodology/cwe_validation/run_all.py
uv run python -c "from chapters.ch3_methodology.figures.feature_evidence import render_all; render_all()"
```
