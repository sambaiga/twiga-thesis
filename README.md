# twiga-thesis

Per-chapter reproducibility code for the PhD thesis *Efficient and Robust Machine Learning for
Net-Load Forecasting in Low-Voltage Distribution Systems*. This repository holds the experiment,
analysis, and figure-generation scripts for each thesis chapter; the forecasting library itself
is [`twiga`](https://github.com/sambaiga/twiga-forecast), installed here as an ordinary external
dependency (via PyPI), not vendored.

## Install

```bash
uv sync --extra nn --dev
```

`--extra nn` pulls in PyTorch and Lightning, needed for Chapters 5-7 (the neural-network
architectures). Omit it if you only need Chapter 3's metric validation or Chapter 4's tree-based
baselines.

To develop against an unreleased `twiga` change instead of the published PyPI version, point uv at
a local checkout:

```bash
uv add --editable ../twiga-forecast
```

## Chapters

Each row links to that chapter's own `README.md`, which gives the exact reproduction command,
expected runtime, and which thesis table/figure number each output corresponds to.

| Chapter | Research question | Directory | Reproduces |
|---|---|---|---|
| 1. Introduction | N/A | [`chapters/ch1_introduction/`](chapters/ch1_introduction/) | Thesis roadmap illustration |
| 2. Background | N/A | [`chapters/ch2_background/`](chapters/ch2_background/) | UQ taxonomy, calibration/interval-validity illustrations |
| 3. Methodology | **RQ1**: How can ML algorithms for net-load forecasting at secondary substations be developed and evaluated, considering the influence of meteorological variables on forecasting accuracy? | [`chapters/ch3_methodology/`](chapters/ch3_methodology/) | Rank-fusion feature-selection evidence, the CWE metric and its validation |
| 4. MLPF | **RQ2**: How can a lightweight DNN be designed to improve both computational efficiency and forecasting performance in LV net-load forecasting? | [`chapters/ch4_mlpf/`](chapters/ch4_mlpf/) | MLPF benchmark against ML and deep-learning baselines |
| 5. FPQR | **RQ3** (part 1): How can lightweight DNN models be developed to produce well-calibrated forecasts while balancing accuracy and predictive-uncertainty performance? | [`chapters/ch5_fpqr/`](chapters/ch5_fpqr/) | FPQR experiments 1-4, quantile-fraction placement diagnostics |
| 6. Conformal prediction | **RQ3** (part 2): the same question, answered via MLPGAM plus horizon-wise adaptive conformal calibration | [`chapters/ch6_conformal/`](chapters/ch6_conformal/) | Architectural study, design ablation, probabilistic benchmark |
| 7. Multi-target forecasting | **RQ4**: How can lightweight DNN-based models be designed to generate well-calibrated multi-target probabilistic forecasts of LV network variables for uncertainty-aware network decision-making? | [`chapters/ch7_multitarget/`](chapters/ch7_multitarget/) | Joint P/Q/V benchmark, dependence-cost analysis, operational decision experiment |
| 10. Conclusion | N/A | [`chapters/ch10_conclusion/`](chapters/ch10_conclusion/) | (no separate figures at present) |

RQ3's thesis text covers both FPQR (Chapter 5) and MLPGAM/conformal calibration (Chapter 6)
together, so both chapters are tagged RQ3 rather than inventing a split the thesis itself doesn't
make.

## Data

See [`data/README.md`](data/README.md). In short: MLVS-PT is unpublished and available from the
author on request; SPS-UK is public and partially auto-fetched (see that file for the one open
gap); Chapter 7's power-flow records are resolved via the `EXTERNAL_DATA_DIR` environment
variable, documented in the same file.

## Project structure and tooling

This repository mirrors `twiga-forecast`'s own conventions: Ruff for linting and formatting
(Google-style docstrings), `ty` for type checking, pre-commit hooks (Ruff, Bandit, detect-secrets,
Commitizen-style commits), and a GitHub Actions CI workflow running the same checks plus `pytest`.

```bash
uv run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
uv run ruff check .
uv run ruff format --check .
uv run ty check twiga_thesis chapters
uv run pytest
```

## Resolved during initial cleanup

- All `chapters/` internal imports (`sys.path.insert`-style and bare sibling-module imports
  inherited from the original monorepo's flat `experiment/` layout) have been converted to
  absolute package imports. `uv run ty check twiga_thesis chapters` is fully clean (0 diagnostics).
  Along the way this also caught and fixed a real bug: two output-artifact directories
  (`ch5_fpqr/qvn_pit/`, `ch5_fpqr/qvn_forecast_examples/`) had been given `__init__.py` files during
  the reorganization, which made Python's import system treat them as packages shadowing the
  real same-named `.py` modules.
- SPS-UK now has a working, documented raw-to-parquet pipeline (`twiga_thesis.datasets.
  fetch_sps_uk_raw` + `build_sps_uk_parquet`), verified end-to-end against the real Zenodo files.
  One assumption (averaging six weather stations into single Ghi/Temperature columns) is not
  specified in the thesis text and could not be checked against the original parquet file, which
  could not be located in this project, see `data/README.md` for the precise caveat.
