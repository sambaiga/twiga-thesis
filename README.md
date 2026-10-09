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

| # | Chapter | RQ | |
|---|---|---|---|
| 1 | [Introduction](chapters/ch1_introduction/) | Setting the stage | <img src="assets/roadmap.png" height="150"> |
| 2 | [Background](chapters/ch2_background/) | Setting the stage | <img src="assets/ch2.png" height="150"> |
| 3 | [Methodology](chapters/ch3_methodology/) | RQ1 | <img src="assets/ch3.png" height="150"> |
| 4 | [MLPF](chapters/ch4_mlpf/) | RQ2 | MLPF vs. ML/DNN baselines |
| 5 | [FPQR](chapters/ch5_fpqr/) | RQ3 | <img src="assets/ch5.png" height="150"> |
| 6 | [Conformal Prediction](chapters/ch6_conformal/) | RQ3 | <img src="assets/ch6.png" height="150"> |
| 7 | [Multi-Target Forecasting](chapters/ch7_multitarget/) | RQ4 | <img src="assets/ch7.png" height="150"> |
| 8 | [Conclusion](chapters/ch10_conclusion/) | Synthesis | |

Each chapter folder has its own `README.md` with the exact reproduction command and the thesis
table/figure each output maps to. RQ3 covers both Chapter 5 (FPQR) and Chapter 6 (MLPGAM +
conformal calibration), per the thesis's own framing. Images are each chapter's own generated
output, not stock illustrations.

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

---

<div align="center">

Licensed under [Apache 2.0](LICENSE) · Built on [`twiga`](https://github.com/sambaiga/twiga-forecast)

</div>
