<div align="center">

<img src="assets/banner.png" alt="twiga-thesis" width="100%">

[![CI](https://github.com/sambaiga/twiga-thesis/actions/workflows/ci.yaml/badge.svg)](https://github.com/sambaiga/twiga-thesis/actions/workflows/ci.yaml)
[![Security](https://github.com/sambaiga/twiga-thesis/actions/workflows/security.yaml/badge.svg)](https://github.com/sambaiga/twiga-thesis/actions/workflows/security.yaml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![twiga](https://img.shields.io/pypi/v/twiga?label=twiga)](https://pypi.org/project/twiga/)

Per-chapter reproducibility code for the thesis. Depends on
[`twiga`](https://github.com/sambaiga/twiga-forecast) from PyPI, not vendored.

</div>

<br>

<img src="assets/roadmap.png" alt="Thesis roadmap: chapters and research questions" width="100%">

## Chapters

| # | Chapter | RQ | Reproduces |
|---|---|---|---|
| 1 | [Introduction](chapters/ch1_introduction/) | n/a | roadmap illustration |
| 2 | [Background](chapters/ch2_background/) | n/a | UQ taxonomy, calibration illustrations |
| 3 | [Methodology](chapters/ch3_methodology/) | RQ1 | rank-fusion feature selection, CWE metric |
| 4 | [MLPF](chapters/ch4_mlpf/) | RQ2 | MLPF vs. ML/DNN baselines |
| 5 | [FPQR](chapters/ch5_fpqr/) | RQ3 | FPQR experiments, quantile-fraction diagnostics |
| 6 | [Conformal Prediction](chapters/ch6_conformal/) | RQ3 | architectural study, design ablation |
| 7 | [Multi-Target Forecasting](chapters/ch7_multitarget/) | RQ4 | joint P/Q/V benchmark, decision experiment |
| 10 | [Conclusion](chapters/ch10_conclusion/) | n/a | synthesis |

Each chapter folder has its own `README.md` with the exact reproduction command and the thesis
table/figure each output maps to. RQ3 covers both Chapter 5 (FPQR) and Chapter 6 (MLPGAM +
conformal calibration), per the thesis's own framing.

## Quickstart

```bash
uv sync --extra nn --dev   # omit --extra nn for Chapters 3-4 only (no torch needed)
```

<details>
<summary>Developing against an unreleased <code>twiga</code> change</summary>

```bash
uv add --editable ../twiga-forecast
```

</details>

## Data

See [`data/README.md`](data/README.md): MLVS-PT is unpublished (available from the author on
request); SPS-UK is public with a working fetch + assembly pipeline; Chapter 7's power-flow
records resolve via `EXTERNAL_DATA_DIR`.

## Tooling

Ruff, `ty`, pre-commit (Bandit, detect-secrets, Commitizen), and CI mirror `twiga-forecast`'s own
conventions.

```bash
uv run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
uv run ruff check . && uv run ruff format --check .
uv run ty check twiga_thesis chapters
uv run pytest
```

<details>
<summary>Notes from the initial reorganization</summary>

- All `chapters/` internal imports were converted from `sys.path.insert`-style and bare
  sibling-module imports to absolute package imports (`uv run ty check twiga_thesis chapters` is
  fully clean). This also caught a real bug: two output-artifact directories had been given
  `__init__.py` files, shadowing the real same-named `.py` modules.
- SPS-UK's raw-to-parquet pipeline (`twiga_thesis.datasets`) is verified end-to-end against the
  real Zenodo files. One assumption, averaging six weather stations into single Ghi/Temperature
  columns, isn't specified in the thesis text and couldn't be checked against the original
  parquet; see `data/README.md`.

</details>

---

<div align="center">

Licensed under [Apache 2.0](LICENSE) · Built on [`twiga`](https://github.com/sambaiga/twiga-forecast)

</div>
