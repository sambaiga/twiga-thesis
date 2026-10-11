# Chapter 3, Section 3.5: validation of the CWE metric

Every experiment here supports one subsection of `Chapters/section-3-5-cwe.tex`.
Result files carry the subsection number so that a number in the thesis can be
traced to the run that produced it.

## Scripts

| File | Purpose |
|---|---|
| `metric_definitions.py` | CWC, the closed-form Gaussian CRPS, and the sweep helpers the other scripts share. |
| `cwe_validation.py` | The experiments themselves, one function per subsection. |
| `design_choices.py` | Supporting study behind the definition: gated versus graded coverage, and the choice of ratio. |
| `run_all.py` | Runs everything and writes the CSVs below. |

## Experiments and where they appear

| Result file | Function | Subsection |
|---|---|---|
| `s3521_soundness_gaussian.csv` | `gaussian_sweep` | 3.5.2.1–2, the ideal-forecaster protocol and soundness |
| `s3522_cwc_eta_sensitivity.csv` | `cwc_eta_sensitivity` | 3.5.2.2, how far CWC's optimum moves with its free parameter |
| `s3523_netload_factorial.csv` | `factorial_reference_stratification` | 3.5.2.3, reference × stratification on synthetic net-load |
| `s3523_reference_scales.csv` | `reference_scales` | 3.5.2.3, the three width scales the text quotes |
| `s3524_ranking_measured.csv` | `real_datasets` | 3.5.2.4, ranking agreement on MLVS-PT and SPS-UK |
| `s3524_width_sweep_*.csv` | `real_datasets` | 3.5.2.4, the width-multiplier sweep per dataset |

## Protocol

Measured-data experiments follow the protocol this chapter prescribes rather than
a single train-test split: ten-fold back-testing cross-validation with an
expanding window, a twelve-month initial history, a four-month evaluation window
and a two-month step, with held-out forecasts pooled across folds. Ten folds are
available for MLVS-PT and eight for SPS-UK.

Forecasts come from a quantile-regression CatBoost on the quantile grid twiga
fits by default, `np.arange(0.1, 1, 0.1)`, augmented with 0.05 and 0.95 because
the nine defaults cannot form the 90 % interval this section reports.

## Figures

Figures are not produced here. `chapters/ch3_methodology/figures/figures.py` reads these CSVs and
renders them to `chapters/ch3_methodology/figures/output/`.

```bash
uv run python chapters/ch3_methodology/cwe_validation/run_all.py
uv run python -c "from chapters.ch3_methodology.figures.figures import render; render()"
```
