# Chapter 7 experiments: how to run

## Objective

Chapter 7 addresses RQ3: how lightweight deep neural network models can be developed to produce forecasts that are accurate, well calibrated, and inexpensive enough to evaluate at substation scale. `Chapter-7.tex` (`chap:7-expriment-protol`) reports two experiments; the design choices behind the proposed configuration are fixed by a separate design ablation. Results and statistical detail live in `Chapter-7.tex`, not here. This file is a run guide only.

| experiment | folder | script |
| --- | --- | --- |
| 1 — architectural evaluation (H1 backbone, H2 gating, H3 horizon-wise calibration) | `experiment_1_architectural/` | `architectural_study.py`; `mlpgam_gate_analysis.py` (single-model gate analysis) |
| 2 — probabilistic forecasting benchmark | `experiment_2_benchmark/` | `probabilistic_benchmark.py` |
| design ablation (fixes the configuration Experiment 2 uses) | `design_ablation/` | `mlpgam_design_ablation.py` and its significance/figure scripts |
| not in the chapter's protocol | `exploratory/` | calibration-source study, embedding ablation, FPQR/QR head ablations |

## How the experiments are built (aligned with Chapter 6)

Both experiments follow `chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py` and run through `twiga.experiment.ExperimentEngine`:

- **Feature extraction and selection** are not configured per experiment. `load_dataset` (through `load_and_preprocess`) selects the features for every chapter in the same way: the top weather feature (`Ghi` on both datasets), two calendar features, two autoregressive features (lag 96 and a 96-step rolling mean), and the ACF-derived input window (48 samples on both datasets, the "48-sample daily input window" of `Chapter-7.tex`). The dataset entries in `shared/protocol.py` carry only the record window, horizon (48), window stride (48, daily forecast origins) and period, as Chapter 6's do.
- **Conditions** are plain `Condition` objects built from `shared/conditions.py`: metric-type constants (`QUANTILE_METRIC_TYPES`/`PARAMETRIC_METRIC_TYPES` add CRPS wherever the engine allows; conformalised conditions use `INTERVAL_METRIC_TYPES`), the proposed configuration (`PROPOSED`, `PROPOSED_CONFORMAL`), the conformal procedures (`SCP`, `SCP_POOLED`), and the HPO building blocks.
- **HPO**: every MLPGAM-backed condition shares one backbone search per dataset (`hpo_model_cls=MLPGAMConfig`, `hpo_variant=SHARED_HPO_VARIANT`), run with the gate on and the inert target-gate fields pinned (`MLPGAM_HPO_PINS`). A head with hyperparameters of its own (FPQR) runs a head-only search on top of that backbone (`hpo_parent_model_cls`), as Chapter 6's FPQR does.
- **Specs**: `shared/protocol.py` holds `SPEC_COMMON` (the protocol and `FIXED_OVERRIDES`) and `dataset_spec(key, ...)`, which builds one `ExperimentSpec` per dataset with that dataset's own HPO window (`HPO_MONTHS`, 12 + 3 months, ending before the first test fold), like Chapter 6's `SPEC_MLVS_PT_3PHASE`. `base_config()` is the shared `PipelineConfig`. Each script runs one engine per dataset.
- **Training budget** (Chapter 6's): 100 epochs per condition, 50 HPO trials of 50 epochs, early-stopping patience 40, `num_workers=0`. The design ablation keeps its original 500-epoch / patience-100 budget so its results stay reproducible.

## Layout

```text
chapter-7-conformal/
  shared/
    protocol.py                    protocol, datasets (+ noise variants), HPO windows, SPEC_COMMON, dataset_spec, base_config
    conditions.py                  metric types, proposed config, conformal procedures, shared-backbone HPO variants and pins
    gate_faithfulness.py           gate values, gradient sensitivity, rho_faith, noise AUC; ExperimentEngine custom_scorer
    ch7_analyse_results.py         DM/MCS/Kupiec tables, conclusions markdown, figures (Experiment 2)
    ch7_architectural_analysis.py  Experiment 1 analysis: tables, DM/MCS, seed spread, faithfulness, noise test, figures
    forecast_examples.py           shared best/worst-day forecast panels
    metrics.py                     metric column sets, CWE comparison
  experiment_1_architectural/
  experiment_2_benchmark/
  design_ablation/                 conditions.py holds the ablation conditions, spec and engine
  exploratory/
```

Scripts add the chapter directory and `experiment/` to `sys.path` themselves; run them from the repository root with `uv run`. Analysis reuses `experiment/common` (`result_analysis`, `significance`, `comparison_plots`, `forecast_panels`). Every Chapter 7 entry is registered in `experiment/common/run_experiments.py` (`ch7_architectural_study`, `ch7_benchmark`, `ch7_design_ablation`).

## Backtesting protocol

Identical to Chapter 6's MLVS-PT protocol (`Chapter-6.tex`, rolling-origin paragraph) and enforced by `tests/unit/experiment/test_chapter7_protocol.py`: ten folds, four-month test window, origin advanced one month, expanding training window of at least twelve months, two-month validation set, three-month calibration set. All constants live in `shared/protocol.py`; every spec reads them.

The fold structure is identical but the calendar is not: Chapter 7's `MLVS-PT.parquet` (single-target net load) ends 2021-12-31. `load_dataset` truncates Chapter 6's `MLVS-PT-3PHASE.parquet` at the same date.

## Experiment 1: architectural study (`architectural_study.py`)

Four groups on one configuration, so the tables of all three hypotheses are comparable:

| group | hypothesis | conditions | decided on |
| --- | --- | --- | --- |
| `backbone` | H1: MLPGAM is more accurate than MLPF | `mlpf_point`, `mlpgam_point`, `mlpf_scp`, `mlpgam_scp`, each over `--seeds` training seeds (`<name>__s<seed>`) | per-fold MAE (point), Winkler score (intervals), seeds averaged per fold |
| `gating` | H2: the gate helps on its own or with its penalty | proposed MLPGAM-CRC with gate off, no penalty, tuned (proposed), `lambda_gate` 1e-4/1e-3/1e-2 | Winkler score; every fold also reports gate faithfulness |
| `gating_noise` | H2 stress test: the gate suppresses irrelevant inputs | the `gating` conditions on `<dataset>+noise` (4 standard-normal exogenous columns added) | Winkler score, change from clean data, `gate_noise_auc` |
| `calibration` | H3: one conformal quantile per horizon beats a pooled one; the learned scale and the signed score; MLPGAM interpretability | `conf_crc` (trains), `conf_scp`, `conf_scp_pooled`, `conf_crc_signed` (reuse its checkpoint) | Winkler score, coverage and width by lead time; gate-permutation agreement, branch decomposition |

Design decisions:

- **Gate on.** Every MLPGAM-backed condition sets `use_gate=True`. Runs before this change left it at the config default `False`, so they evaluated MLPGAM without its gate.
- **Fair HPO budget.** Each backbone gets `--trials-per-param` x its effective search-space size (MLPF 13 parameters, 50 trials; MLPGAM 17 parameters after pinning the inert target-gate fields, 65 trials), through `Condition.hpo_num_trials`.
- **Seeds.** Hyperparameters are tuned once per backbone and dataset; each seed retrains with them fixed (`overrides={"seed": ...}`). The seed spread is the retraining noise floor reported next to the backbone effect.
- **Faithfulness on every fold.** The gating groups use `shared.gate_faithfulness.make_gate_faithfulness_scorer()` as the `custom_scorer`: the built-in interval metrics plus per-fold `gate_rho_spearman`, `gate_rho_pearson`, `gate_noise_mean`, `gate_signal_mean`, `gate_noise_auc` and per-feature `gate__<branch>__<feature>` / `sens__<branch>__<feature>` columns.

Each dataset gets two engine runs, one for the clean groups and one for `gating_noise` on `<dataset>+noise`; the analysis reads all run directories together.

## Experiment 2: probabilistic benchmark (`probabilistic_benchmark.py`)

`mlpgam_crc` (proposed: MLPGAM-CRC with adaptive horizon-wise conformal calibration) against `mlpgam_qr`, `mlpgam_fpqr`, `mlpgam_bnn` and `mlpgam_laplace`, all on the shared gate-on MLPGAM backbone. QR/FPQR report the quantile evaluator's CRPS and BNN/Laplace the parametric evaluator's CRPS and NLL; `mlpgam_crc` reports point and interval metrics (the engine's calibrated path supports no others). The Winkler score decides every comparison.

## Output

Written to `$TWIGA_EXPERIMENTS_ROOT/chapter-7-conformal` (default `../experiments/chapter-7-conformal`, a sibling of the repository), overridable with `--output-root` on every script. Engine runs are at `<root>/<experiment>/<spec slug>/runs/<run id>/`. `ch7_analyse_results.py` and `forecast_examples.py` read the latest engine run that covers each dataset, falling back to the earlier `<experiment>/<dataset>/results/` layout. Analysis is written to `<root>/analysis`.

## Prerequisites

- Datasets `MLVS-PT` and `MLVS-UK` at `examples/data/*.parquet`.
- HPO caches are fingerprinted per experiment root. `--skip-hpo` requires a matching cache; `--hpo-cache-policy refresh` re-tunes. The trial count is not part of the fingerprint, so changing `--trials-per-param` or `--hpo-trials` needs `refresh`.

## Running

```bash
# Smoke tests (minutes)
uv run python chapters/ch6_conformal/experiment_1_architectural/architectural_study.py \
  --dataset MLVS-UK --seeds 1 --epochs 2 --hpo-epochs 2 --folds 2 --hpo-trials-floor 1 --trials-per-param 0 \
  --output-root /tmp/ch7-smoke
uv run python chapters/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py \
  --dataset MLVS-UK --epochs 2 --hpo-trials 1 --hpo-epochs 2 --folds 2 --output-root /tmp/ch7-smoke

# Experiment 1: architectural study (all groups, both datasets)
uv run python chapters/ch6_conformal/experiment_1_architectural/architectural_study.py \
  --dataset all --seeds 5 --hpo-cache-policy refresh

# Experiment 1 analysis (pass every run directory the study wrote)
uv run python chapters/ch6_conformal/shared/ch7_architectural_analysis.py \
  --study-runs <root>/architectural_study/<spec slug>/runs/<run id> [...] --copy-to-images

# Experiment 2
uv run python chapters/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py \
  --dataset all --hpo-cache-policy refresh

# Experiment 2 analysis: tables, DM/MCS, Kupiec, conclusions, figures
uv run python chapters/ch6_conformal/shared/ch7_analyse_results.py --experiment probabilistic_benchmark --dataset all

# Best/worst-day forecast panels
uv run python chapters/ch6_conformal/shared/forecast_examples.py --experiment probabilistic_benchmark

# Design ablation
uv run python chapters/ch6_conformal/design_ablation/mlpgam_design_ablation.py --group gating
uv run python chapters/ch6_conformal/design_ablation/design_ablation_significance.py --run-dir <run dir> --dataset all
```

Significance rule: Winkler score decides interval comparisons and MAE decides point comparisons, via the DM test and the Model Confidence Set. CWE, PICP and NMPI are diagnostics. `--winsorize-nrmse` on `ch7_analyse_results.py` clips NRMSE at its 97th percentile per condition, as in Chapter 6.

## What to rerun

Every Chapter 7 result produced before the Chapter 6 alignment (commit `eab8b67`) used a different configuration: gate off, window stride 1, 500 epochs, HPO window 14 + 2 months, and on SPS-UK an HPO window overlapping the test folds. Rerun the two experiments below, from the repository root, in this order. Use `--hpo-cache-policy refresh` so no cache tuned under the old configuration is reused.

| order | step | command | expected output |
| --- | --- | --- | --- |
| 0 | smoke test, Experiment 1 (minutes) | `uv run python chapters/ch6_conformal/experiment_1_architectural/architectural_study.py --dataset MLVS-UK --seeds 1 --epochs 2 --hpo-epochs 2 --folds 2 --hpo-trials-floor 1 --trials-per-param 0 --output-root /tmp/ch7-smoke` | every condition `succeeded` in `run_status.json`; `gate_*` columns in the full CSV |
| 0 | smoke test, Experiment 2 (minutes) | `uv run python chapters/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py --dataset MLVS-UK --epochs 2 --hpo-trials 1 --hpo-epochs 2 --folds 2 --output-root /tmp/ch7-smoke` | five conditions `succeeded`; `crps` for QR, FPQR, BNN, Laplace |
| 1 | Experiment 1: architectural study (H1, H2, H3) | `uv run python chapters/ch6_conformal/experiment_1_architectural/architectural_study.py --dataset all --seeds 5 --hpo-cache-policy refresh` | four run directories under `<root>/architectural_study/mlpgam_architectural_study__chapter_7/runs/`: clean groups and `+noise` group, per dataset |
| 2 | Experiment 2: probabilistic benchmark | `uv run python chapters/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py --dataset all --hpo-cache-policy refresh` | one run directory per dataset under `<root>/probabilistic_benchmark/probabilistic_forecasting_benchmark__chapter_7/runs/` |
| 3 | Experiment 1 analysis | `uv run python chapters/ch6_conformal/shared/ch7_architectural_analysis.py --study-runs <the four run directories from step 1> --copy-to-images` | tables, seed spread, faithfulness and noise tables in `<root>/analysis/architectural_study/`; figures copied to the thesis |
| 4 | Experiment 2 analysis | `uv run python chapters/ch6_conformal/shared/ch7_analyse_results.py --experiment probabilistic_benchmark --dataset all` | DM/MCS, Kupiec, conclusions in `<root>/analysis/` (reads the latest engine run per dataset) |
| 5 | forecast panels | `uv run python chapters/ch6_conformal/shared/forecast_examples.py --experiment probabilistic_benchmark` | best/worst-day panels in `<root>/analysis/figures/forecast_examples/` |

`<root>` is `$TWIGA_EXPERIMENTS_ROOT/chapter-7-conformal` (default `../experiments/chapter-7-conformal`).

Check after the full runs:

- FPQR: PICP was 0 in the 2-epoch smoke test (an untrained grid); confirm it is near the nominal 0.90 at 100 epochs.
- Gate penalty sweep: `gate_l1e-4`/`1e-3`/`1e-2` should differ from `gate_default`. The penalty ramps in over the first 5 epochs, so a 2-epoch smoke test cannot separate them.
- `run_status.json` of every run: all conditions `succeeded`.

Not rerun: the design ablation (`design_ablation/`), which fixes the proposed configuration and keeps its original budget, and the `exploratory/` scripts, which are outside the chapter's protocol. `mlpgam_gate_analysis.py` is superseded by the per-fold faithfulness columns of step 1.

### Partial rerun of the benchmark heads (gate fix and FPQR pins)

The first aligned benchmark run (`probabilistic_benchmark_v2`) trained QR, FPQR, BNN and Laplace without the feature gate: the probabilistic MLPGAM wrappers dropped the gate fields (fixed in `twiga.models.nn.mlpgam_model.mlpgam_gate_kwargs`). Its FPQR also searched the proposal family and QVN freely and failed in the first five MLVS-PT folds; FPQR now pins `proposal_type="softmax"`, `qvn_type="standard"`, `proposal_eps=1e-4` and `gradient_clip_val=1.0` (`shared/conditions.FPQR_HPO_PINS`). Rerun only the four affected heads into the same folder, so the HPO caches are found and `mlpgam_crc` is kept:

```bash
uv run python chapters/ch6_conformal/experiment_2_benchmark/probabilistic_benchmark.py \
  --dataset all --experiment-name probabilistic_benchmark_v2 \
  --conditions mlpgam_qr mlpgam_fpqr mlpgam_bnn mlpgam_laplace
```

Use the default `--hpo-cache-policy reuse`, not `refresh`. The shared MLPGAM backbone cache is still valid (that search already ran with the gate on) and is reused; the FPQR head cache no longer matches its fingerprint, because its pinned fields changed, so it is re-tuned automatically. The log reports `Reusing backbone params` for the shared backbone and a new search for `mlpgamfpqr_fpqr_head`. `ch7_analyse_results.py` takes each condition from the latest run that contains it, so the rerun heads are analysed together with the earlier `mlpgam_crc`.

### Calibration on one trained model, signed score, and interpretability (H3, H2)

The `calibration` group now trains the proposed MLPGAM-CRC model once per fold and calibrates it four ways, so the conditions differ only in the calibration step and run-to-run training variation no longer enters the comparison (separate runs of the same configuration differed by up to 5.3% in WS on MLVS-PT):

| condition | score | quantile | model |
| --- | --- | --- | --- |
| `conf_crc` | absolute residual / learned sigma (proposed) | one per horizon | trained here |
| `conf_scp` | absolute residual | one per horizon | reuses `conf_crc`'s checkpoint |
| `conf_scp_pooled` | absolute residual | one, pooled over horizons | reuses `conf_crc`'s checkpoint |
| `conf_crc_signed` | signed residual / learned sigma | split alpha/2 and 1 - alpha/2 per horizon (asymmetric) | reuses `conf_crc`'s checkpoint |

`conf_crc` also runs `shared/mlpgam_interpretability.py` as its scorer. Per fold it records the gate value, gradient sensitivity and permutation importance of every input feature and their Spearman agreement (`gate_rho_perm`, `gate_rho_spearman`). Per timestamp it records the exact additive decomposition of the forecast into the history-branch, covariate-branch and offset terms (`contrib_past`, `contrib_future`, `contrib_offset` in `predictions.parquet`). `decomp_max_rel_error` checks that the three terms sum to the forecast, and is 0 to machine precision.

Rerun the group into the existing study folder so the cached shared MLPGAM backbone is reused:

```bash
uv run python chapters/ch6_conformal/experiment_1_architectural/architectural_study.py --group calibration --dataset all
```

What it runs: one training run per fold of `conf_crc` (10 folds x 2 datasets = 20 trainings of at most 100 epochs), then three calibration-only passes per fold that load its checkpoints. No HPO is run, and the other groups are not run. One new run directory per dataset is written under `architectural_study/mlpgam_architectural_study__chapter_7/runs/`.

Before running:

1. `git pull` so the group contains the four conditions above.
2. Use the output folder that holds the existing study (default `architectural_study`; otherwise pass `--output-root`). The log should report `Reusing matching backbone params  model=mlpgam_shared_mlpgam` and, for the three reusing conditions, `Loaded model from checkpoint: .../conf_crc/checkpoints/...`.
3. The checkpoints of `conf_crc` are kept on disk because other conditions reuse them, which takes about 20 small files per dataset.

Analysis:

```bash
# H3: pass the new run directories after the four existing ones (load_study keeps the last run per condition)
uv run python chapters/ch6_conformal/shared/ch7_architectural_analysis.py \
  --study-runs <4 existing run dirs> <new MLVS-PT run dir> <new MLVS-UK run dir> --copy-to-images

# Interpretability: reads conf_crc from the latest run that contains it
uv run python chapters/ch6_conformal/shared/ch7_interpretability_analysis.py --copy-to-images

# Fixed vs adaptive forecast example (picks up the new conf_scp and conf_scp_pooled)
uv run python chapters/ch6_conformal/experiment_1_architectural/calibration_forecast_example.py --copy-to-images
```

`ch7_interpretability_analysis.py` writes, under `<root>/analysis/architectural_study/interpretability/`:

- `interpretability_features_<ds>.csv`: gate value, permutation importance and sensitivity per feature;
- `interpretability_summary.csv`: gate-permutation agreement and the covariate-branch share;
- three figures per dataset, copied to the thesis as `fig_ch7_arch_interp_*`:
  - `interp_importance_<ds>`: gate value against permutation importance;
  - `interp_branch_profile_<ds>`: mean branch terms by lead time;
  - `interp_decomposition_<ds>`: one day's forecast split into its terms.

The H3 numbers and the interpretability results in `Chapter-7.tex` must then be updated from these tables.

### Computational cost

`shared/ch7_model_cost.py` reports trainable parameters (each model rebuilt from its cached tuned hyperparameters) and per-fold test-set inference time from the saved results, for the lightweight claim of RQ3 (`tab:ch7-cost`):

```bash
uv run python chapters/ch6_conformal/shared/ch7_model_cost.py --study-name architectural_study --benchmark-name probabilistic_benchmark_v4
```

## Known issues

- Results from the earlier `architectural_evaluation.py` (removed) and the pre-alignment `probabilistic_benchmark.py` used `PipelineConfig`'s default window stride of 1, so `predictions.parquet` repeats every timestamp 2,304 times (about 133 million rows per condition) and each fold has one metric row per 30-minute origin. `forecast_examples.py` de-duplicates them. Runs with the aligned scripts use stride 48 (one origin per day).
- Before `train_start` was taken from each dataset entry, benchmark-style runs tuned on PipelineConfig's default window (2019-03 + 16 months), which on SPS-UK overlaps the test folds (2019-06 to 2020-07). Pre-alignment SPS-UK results are affected; MLVS-PT is not.
- `design_ablation/training_schedule_mechanism_figure.py` and `mlpgam_forecast_examples.py` default to run directories under the old in-repo `results/` tree; pass `--run-dir` for a current run.
