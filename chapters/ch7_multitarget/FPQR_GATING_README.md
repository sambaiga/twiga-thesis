# Chapter 6, Experiment 2: Multi-Target FPQR Evaluation

Chapter 6's three named experiments (`subsec:ch6-protocol`), each its own script in this folder:

| Experiment | Script | Question |
|---|---|---|
| 1. Probabilistic Forecasting Benchmark | `multitarget_experiment.py` | FPQR vs QR vs BNN vs Laplace, on MLVS-PT-3PHASE and DSS-EU |
| 2. Multi-Target FPQR Evaluation | `fpqr_gating_ablation.py` (this README) | Which FPQR design extensions help, on the same two datasets |
| 3. Operational Decision-Making Assessment | `operational_decision.py` | Does modelling dependence improve hosting-capacity/thermal exceedance-probability estimates, on DSS-UK and DSS-US |

`copula_variants.py` and `penetration_sweep.py` are supporting scripts, not named experiments in their own right -- see their own module docstrings.

## Objective

Chapter 6 extends FPQR from Chapter 5 to the multi-target setting: forecasting active power, reactive power, and voltage jointly at a low-voltage substation, and coupling the three learned marginals into a coherent predictive distribution. This is evaluated across three experiments; this script, `fpqr_gating_ablation.py`, implements the second: the Multi-Target FPQR Evaluation (`chap6:exp-one`'s companion section in `Chapter-6.tex`).

The objective of this experiment is to determine, for each of the proposed extensions to the original FPQR formulation, whether it improves forecasting performance and dependence representation relative to the unmodified formulation. Five extensions are evaluated, each isolated as its own condition group, 13 conditions in total: the quantile-proposal mechanism, the value-network architecture, the quantile grid (shared versus per-target), target-level output gating (the multi-target counterpart of Chapter 7's feature gating), and the probabilistic head (uncoupled FPQR margins versus a joint parametric alternative). No result from this script is written into Chapter 6's mechanism-level prose until it has been run and read; the chapter's text stays as it is until there is evidence to revise it against.

## Hypotheses tested

1. **Proposal mechanism.** The single-target ablation's finding, that a softmax proposal outperforms alternatives, transfers to the multi-target, per-target-grid setting. Reference condition: `proposal_softmax`.
2. **Value network.** An architecturally monotonic value network improves calibration over the standard value network, including a calibrated variant whose single-target failure mode (whether it reflects a tuning artefact or an architectural limitation) this experiment is positioned to resolve. Reference condition: `qvn_standard`.
3. **Quantile grid.** A per-target quantile grid (`per_target_tau`), the direct multi-target extension of Chapter 5's FPQR, improves on a single grid shared across targets. Reference condition: `grid_shared`.
4. **Target-level gating.** Target-level output gating (`use_target_gate`, introduced this session) improves forecasting performance beyond, or in combination with, Chapter 7's existing feature-group gating (`use_gate`). Four conditions cover gating disabled, each mechanism alone, and both together. Reference condition: `gate_none`.
5. **Probabilistic head.** This comparison is a check on the framework's design choice, not a test of a hypothesis this chapter argues for: multi-target FPQR (uncoupled margins) is compared against a joint parametric (multivariate-Normal) head purely as a reference point. Chapter 6's prose retains the FPQR-plus-copula architecture regardless of the outcome, per the explicit scope decision recorded when this experiment was designed. Reference condition: `head_fpqr`.

## Prerequisites

- Run from the repository root with `uv run`, or with the virtual environment active.
- MLPGAM backbone HPO parameters follow the same cache convention as the other ablation scripts (`experiment/best_params_mlpgam.json` / `_run_fpqr_gating/.../backbone_hpo/`); `--skip-hpo` reuses whatever is already cached instead of searching.
- Datasets: see **Datasets** below. Two of the three datasets this chapter uses are ready; the third is wired but needs a file.

## Datasets

All three targets in this chapter, `p_kw`, `q_kvar`, and `v_lv_pu`, are read from `examples/data/<name>.parquet` by `load_dataset()` in `twiga/utils/data_loading.py`. The three datasets do not share one origin, so each is covered separately below.

| Key | Status | Origin |
| --- | --- | --- |
| `DSS-EU:default:0.4:0.4` (default) | Ready, `examples/data/DSS-EU.parquet` already present | Simulated on the four-feeder Hybrid European MV-LV network, the non-synthetic test system of Koirala et al. (2020); built by `build_dss_eu.py` in this folder from `PhD-Thesis-Anthony`'s `PT_DSS_EV_*_PV_*.parquet` scenario files via `load_opendss_data()` |
| `DSS-US` | Ready, `examples/data/DSS-US.parquet` already present | Synthetic power-flow record, `ml-energy-disaggregation` repository |
| `MLVS-PT-3PHASE` | Ready, `examples/data/MLVS-PT-3PHASE.parquet` present | Measured record, `PhD-Thesis-Anthony` |

`DSS-UK` is now usable: `load_dataset()` has a branch for it (`DSS-UK:<feeder>:<ev>:<pv>`, feeder `"stiff"` or `"weak"`). `examples/data/DSS-UK.parquet` is a copy of `examples/data/DSS-EU_legacy_multifeeder.parquet` -- the older two-feeder ('stiff'/'weak', droop-modelled reactive power) record this session's DSS-EU rebuild superseded -- confirmed by its `feeder` column values (`stiff`/`weak`) matching `Chapter-2.tex`'s and `Chapter-6.tex`'s (`sec:ch6-datasets`) description of `DSS-UK`, not the current `DSS-EU`.

### `MLVS-PT-3PHASE`: measured record, from `PhD-Thesis-Anthony`

The same Madeira substation as `MLVS-PT`, with all three phases metered directly rather than power-flow simulated. `examples/data/MLVS-PT-3PHASE.parquet` is present in this checkout. To regenerate elsewhere, run the three-phase cell in `PhD-Thesis-Anthony/notebook/Prepare-data.ipynb`, which calls `load_three_phase_PT_substation_data()` from `PhD-Thesis-Anthony/src/utils/load_data.py` and writes the result into this repository's `examples/data/` (assuming the two repositories are checked out as siblings), or copy an already-produced file from another checkout:

```bash
cp /path/to/PhD-Thesis-Anthony/dataset/MLVS-PT-3PHASE.parquet examples/data/MLVS-PT-3PHASE.parquet
```

### `DSS-EU`: the real four-feeder Hybrid European MV-LV network

Built from `PhD-Thesis-Anthony`'s `PT_DSS_EV_*_PV_*.parquet` scenario files (`load_opendss_data()`) by `chapters/ch7_multitarget/build_dss_eu.py` in this repository, not from `ml-energy-disaggregation`. `examples/data/DSS-EU.parquet` is already present; regenerate with:

```bash
uv run python chapters/ch7_multitarget/build_dss_eu.py
```

### `DSS-US`: synthetic power-flow record, from `ml-energy-disaggregation`

An OpenDSS power-flow simulation produced in the separate `ml-energy-disaggregation` repository. Already present at `examples/data/DSS-US.parquet` in this checkout.

**Option A, copy from a local checkout.** Copy the file from wherever it was previously produced, or from another checkout of this repository.

**Option B, regenerate from the power-flow scripts.** `run_powerflow_us.py` in `ml-energy-disaggregation`, followed by `chapters/ch7_multitarget/export_us_record.py` in this one, which attaches a timestamp and the driving irradiance profile and writes `examples/data/DSS-US.parquet` directly:

```bash
uv run python chapters/ch7_multitarget/export_us_record.py
```

## Running the experiment

Groups: `proposal`, `qvn`, `grid`, `multigate`, `head` (original sweeps) plus the full FPQR hyperparameter ablations `loss`, `kappa`, `tau_embedding`, `proposal_design`, `crossing`, `quantiles`, `qvn_attention`, `qvn_umnn`, `qvn_monotone_cal`, `gradient_clip`. Each FPQR condition pins its tested values inside HPO.

```bash
# Everything on DSS-EU (default dataset), reusing cached HPO:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py

# One or more groups:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --group multigate
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --group loss kappa tau_embedding proposal_design crossing

# Both datasets:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --dataset DSS-EU:default:0.4:0.4 MLVS-PT-3PHASE

# Smoke test:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --quick

# Run HPO (shared MLPGAM parent search + per-condition FPQR head search):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --run-hpo --hpo-trials 50 --hpo-epochs 50 --hpo-objective winkler_score

# Force a fresh search, custom output root:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --run-hpo --hpo-cache-policy refresh --output-root /path/to/experiments/chapter-6-multitarget
```

After a run, print the significance report for one group (uses the latest synced run under `results/chapter-6-multitarget/fpqr_gating_ablation/`):

```bash
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --report multigate
```

This uses `twiga.core.metrics (pairwise_dm_table, mcs_ranking)`: a Diebold-Mariano/Model-Confidence-Set table on the Winkler score, the proper scoring rule the decision rests on, never on CWE.

## Deliberately not in this script

- **Copula coupling.** `twiga/distributions/nn/copula_sampling.py` (`fit_gaussian_copula`) is a post-hoc, numpy-level pathway, not wired into any `Condition`-driven model configuration, so it cannot be a group here. It is covered by `copula_variants.py` in this folder instead — a separate, hand-rolled script on the same expanding-window protocol, scored with the sample-based joint metrics (energy score, variogram score) the engine does not compute.
- **Multivariate adaptive conformal calibration.** `Chapter-6.tex` (`subsec:ch6-evaluation`) states every backbone is evaluated with this applied on top. `ConformalConfig.method` does not currently accept `"multivariate"` as a value, even though `twiga.distributions.conformal.core.Conformal` supports it internally, so no condition in this script (or `copula_variants.py`) currently applies it. Open gap, not yet closed.
- **Smoke-test verification.** A one-fold, one-epoch run of the target-level gating group on `DSS-EU:default:0.4:0.4` completed end to end this session (training, per-target point/interval/quantile evaluation, summary table, ablation figures), confirming the pipeline mechanics, not the ablation's findings. All four gating conditions produced identical metrics at that scale, the expected artefact of gates barely moving from initialisation in one epoch, not a result to draw any conclusion from.
