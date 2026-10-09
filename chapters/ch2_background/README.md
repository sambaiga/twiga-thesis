# Chapter 2: Background

No experiments; concept illustrations only, using synthetic data generated inline by each script
(no external dataset needed).

| Script | Produces |
|---|---|
| `figures/uq_paradigms.py` | Four-panel uncertainty-quantification taxonomy (BNN, parametric, quantile, conformal). |
| `figures/interval_validity.py` | Three-panel prediction-interval validity/efficiency illustration (under/well/over-confident coverage). |
| `figures/calibration_diagnostics.py` | Reliability diagram and PIT histograms for three forecaster dispersions. |

```bash
uv run python -c "from chapters.ch2_background.figures.uq_paradigms import render; render()"
uv run python -c "from chapters.ch2_background.figures.interval_validity import render; render()"
uv run python -c "from chapters.ch2_background.figures.calibration_diagnostics import render; render()"
```
