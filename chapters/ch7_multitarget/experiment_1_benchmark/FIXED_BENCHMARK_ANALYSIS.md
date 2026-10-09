# Chapter 6, Experiment 1 at fixed parameters: analysis (2026-10-06)

Runs analysed (commit `3f3b1a`, under `experiments/chapter-6-multitarget/multitarget_fixed_benchmark/runs/`):

| Dataset | Run | Protocol | Folds | Conditions |
|---|---|---|---|---|
| MLVS-PT-3PHASE | `20261006_111801…_3f3b1a` | 12-month expanding train, 2 val, 3 test, 3 calib | 10 | 6 heads + 30 conformal |
| DSS-EU (0.4/0.4) | `20261006_105509…_3f3b1a` | 4-month fit + 1 val (`train_size_mode="fixed"`), 1 test, rolling | 5 | 6 heads |
| DSS-UK (weak, 0.4/0.4) | `20261006_110203…_3f3b1a` | as DSS-EU | 6 | 6 heads |
| DSS-US (rank 0, 0.4/0.4) | `20261006_110959…_3f3b1a` | as DSS-EU | 6 | 6 heads |

Every head shares one fixed MLPF backbone (latent/hidden 64, 2 layers, dropout 0.1, AdamW 1e-3/1e-6), 60 epochs with early-stop patience 100, a 48-step lookback, and a nominal 90% interval: anchored FPQR (`conf_level=0.1`, `proposal_anchor_outer=True`), QR on its default 0.1..0.9 grid plus 0.05/0.95, BNN, and Laplace (now scored with Laplace intervals and CRPS, fix `59d4fb4`). Two further FPQR heads (learned outer levels; Chapter 5's MLPQF configuration) have native intervals that are not at 90%, so only their CRPS and NRMSE are comparable.

Regenerate everything with:

```bash
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_report.py \
    --experiment-root ../experiments/chapter-6-multitarget/multitarget_fixed_benchmark
```

Outputs land in `analysis/<dataset>/` (tables, `conclusion.md`, DM/Kupiec CSVs, figures) and `analysis/joint_metrics_by_dataset.pdf`.

## Method

Fold-then-mean aggregation; DM (Benjamini-Hochberg) and MCS (alpha 0.05) on fold-level losses through `twiga.core.metrics.result_analysis`, as Chapters 5 and 7 test them; `metric_table` ranks (PICP by distance from 0.90); exact Kupiec POF test from the saved hit sequences; stratified CWE two-sided (coverage gate within one standard error of nominal). Decisions rest on each section's deciding metric, never on the blended rank. With 5-6 folds on the DSS records the fold-level tests have little power; the per-day tests (`fixed_benchmark_analysis.py`) agree in direction.

## Results

### Average rank (folds x targets x NRMSE/CRPS/WS/CWE; lower is better)

| Dataset | FPQR (anchored) | QR | BNN | Laplace |
|---|---|---|---|---|
| MLVS-PT | **2.46** | 2.48 | 2.57 | 2.49 |
| DSS-EU | 2.98 | 1.90 | 3.48 | **1.63** |
| DSS-UK | 2.39 | 2.90 | 2.89 | **1.82** |
| DSS-US | 2.69 | 2.07 | 3.25 | **1.99** |

### MLVS-PT per target (fold means)

| Head | NRMSE P/Q/V | CRPS P/Q/V | WS P/Q/V | CWE-strat P/Q/V |
|---|---|---|---|---|
| FPQR (anchored) | 0.265/0.218/0.413 | 4.66/1.19/1.10 | 37.1/10.1/10.5 | 0.83/0.85/0.71 |
| QR | 0.262/0.222/0.387 | 4.41/1.23/0.96 | 37.1/10.7/7.4 | 0.76/0.72/0.77 |
| BNN | 0.255/0.226/0.401 | 4.46/1.30/1.15 | 36.8/9.3/11.4 | 0.79/0.85/0.61 |
| Laplace | 0.251/0.242/0.353 | 4.40/1.54/0.96 | 37.1/17.2/8.4 | 0.79/0.42/0.72 |

Fold-level significance on MLVS-PT: all four heads tie on P (CRPS, WS, NRMSE MCS); on Q, FPQR has the best CRPS and NRMSE (MCS: FPQR, BNN, QR) and BNN the best WS; on V, Laplace and QR are significantly better than FPQR on every metric.

### Joint metrics

| | MLVS-PT | DSS-EU | DSS-UK | DSS-US |
|---|---|---|---|---|
| Joint coverage: FPQR / QR / BNN / Laplace | 0.50 / 0.81 / 0.52 / 0.29 | 0.49 / 0.92 / 0.68 / 0.81 | 0.60 / 0.91 / 0.63 / 0.67 | 0.38 / 0.74 / 0.39 / 0.49 |
| Energy score, best head | Laplace (MCS: all four) | Laplace (sig.) | Laplace (sig.) | Laplace (sig. vs FPQR) |
| Thermal-event Brier skill, range | 0.15 to 0.34 | 0.02 to 0.09 | -0.22 to 0.00 | about -0.02, all heads |

- QR's joint coverage comes from over-covering each target (marginal 0.91-0.998), not from calibration.
- On DSS-UK and DSS-US no head forecasts transformer overload better than the base rate.
- Kupiec rejects nominal coverage for every head and target except FPQR on DSS-UK P; at about 40,000 test points the test rejects small deviations, so this reads as "not exactly 90%".

### Conformal calibration (MLVS-PT)

| Head | Joint coverage native / max-norm / CQR-marginal | WS P after CQR-marginal |
|---|---|---|
| FPQR (anchored) | 0.500 / 0.897 / 0.665 | 36.0 |
| QR | 0.808 / 0.917 / 0.712 | 35.2 |
| BNN | 0.519 / 0.896 / -- | -- |
| Laplace | 0.285 / 0.855 / -- | -- |

Max-norm reaches 0.86-0.92 joint coverage for every head at a WS cost (P: about 37 to 45-51). CQR-marginal gives the sharpest intervals but, calibrating each target separately, only 0.67-0.71 joint coverage. `qr_quantile_unscaled_marginal` is the only conformal condition in the WS MCS on all three targets.

### Dependence

| Residual correlation P-V | MLVS-PT | DSS-EU | DSS-UK | DSS-US |
|---|---|---|---|---|
| Observed | -0.37 | -0.99 | -0.48 | -0.62 |
| FPQR errors | +0.19 | -0.96 | -0.22 | -0.57 |

On the simulated DSS records voltage follows nearly deterministically from P and Q through the power flow, so every head's errors inherit the observed dependence without modelling it. On the measured MLVS-PT record the residual correlation is weak and has the wrong sign for P-V. The chapter's "no head reproduces the dependence" holds for measured data only.

### Coverage over time (MLVS-PT)

FPQR's and BNN's joint coverage rises from winter to summer (FPQR 0.18 to 0.70, BNN 0.21 to 0.86), driven by V; QR stays at 0.74-0.87; Laplace stays near 0.3 because of Q.

## Comparison with earlier multi-target runs on MLVS-PT-3PHASE

CRPS, the only level-free score across versions (earlier runs: QR at 80%, Laplace at an effective 80.7% with Normal-form CRPS, anchored FPQR at 95%, 4-month test windows):

| CRPS P/Q/V | v1 | v2 | v3 | v6 | this run |
|---|---|---|---|---|---|
| FPQR | 4.43/1.40/0.99 | 3.50/1.10/0.70 (PICP 0.60-0.67) | 4.41/1.20/1.00 | 4.47/1.30/1.00 | 4.67/1.19/1.10 |
| QR | 4.48/1.25/0.86 | 4.16/1.25/0.98 | -- | 4.07/1.31/0.91 | 4.41/1.23/0.96 |
| BNN | 4.75/2.02/1.25 | 5.40/1.99/1.29 | -- | 4.68/1.33/1.11 | 4.45/1.29/1.15 |

- v6's tuned configurations had collapsed (anchored FPQR PICP 0.52-0.70, grid variants up to WS 610, Laplace PICP 0.30-0.54); the fixed-parameter run is far more stable.
- Against v3, the best tuned FPQR, this run is level on Q and 6-10% behind on P and V.
- QR and BNN CRPS are stable across versions (about +/-10%): the ranking moves with FPQR's configuration.

## Consequences for Chapter-6.tex (not yet applied)

1. The current FPQR numbers are not from one benchmark run: `benchmark_dependence_check.py` substitutes the v3 grid-group FPQR (anchored at `conf_level=0.05`, a 95% interval) and `decision_summary_figure.py` hard-codes joint coverage from v3 and v6. The 0.782 "FPQR joint coverage" was largely the wider interval.
2. The per-fold "joint coverage" heatmap plotted the mean of the three targets' coverage, not joint coverage; the new figure computes it properly.
3. Experiment 1's conclusion becomes: anchored FPQR is competitive, not superior, at fixed parameters; strongest on reactive power, held back by under-coverage; a Laplace head is the strongest fixed-parameter baseline on the simulated records; joint calibration, not the head, delivers reliable joint coverage.
4. Protocol text to reconcile: MLVS-PT test windows are 3 months here (the text says four); hyperparameters are fixed, not tuned.

## Proposed next step: retune FPQR's head only, backbone fixed

Goal: test whether FPQR's under-coverage is a configuration problem, without changing anything else that this comparison holds equal.

**What is searched.** Only the FPQR head's own fields (the part of `MLPFFPQRConfig.search_space` that `MLPFConfig` does not have): `n_quantiles`, `entropy_weight`, `num_cosines` with `rope_type`, `qvn_type` (including `monotone_cal`, whose calibration term targets exactly this under-coverage), `proposal_eps`, `loss_fn`, `gradient_clip_val`, plus `lambda_cal` and `crossing_penalty`.

**What is pinned** (through `Condition.hpo_fixed_params`, which removes a field from the Optuna search and re-applies it at build time):

- the whole backbone: the 13 `MLPFConfig` search fields at `FIXED_BACKBONE`'s values (latent/hidden 64, 2 layers, dropout 0.1, width 2.0, alpha 0.1, no embeddings, SiLU, norm on, no residual, addition-comb);
- the training budget: AdamW, lr 1e-3, weight decay 1e-6, multi-step schedule, batch 64 (training-level fields, so `_TRAINING_LEVEL_HPO_FIELDS` freezes them in the sampler);
- the interval definition: `conf_level=0.1`, `proposal_anchor_outer=True`, `proposal_type="softmax"`, `per_target_tau=False`.

**Objective.** Winkler score at the 90% interval on the HPO validation window (proper, and penalises the under-coverage directly), with early stopping left on point accuracy (`50bb271`: monitoring Winkler per epoch picks each trial's sharpest epoch and collapses coverage). Report two-sided stratified CWE and PICP for the chosen trial as diagnostics.

**HPO window, without leakage.** The HPO split runs from `train_start` for `hpo_train_months + hpo_val_months` and must end before the first CV test fold:

- MLVS-PT: 14 + 6 months from 2019-03 ends 2020-11, before the first test window (2020-12). Safe.
- DSS records: the first test month is late June (EU) or June (UK/US). 3 + 1 months from January ends in April-May. Short, so either tune on MLVS-PT only and transfer the head settings to the DSS runs, or accept a 4-month HPO window.

**Budget (implemented as the `head_tuned` group).** FPQR 100 trials, QR and BNN 50 each: TPE spends its first 20 trials sampling at random, and FPQR searches 6 head settings every trial plus 2-5 conditional ones against QR's 2-3 and BNN's 3. Tuned on CRPS, with 50 epochs per trial, then retrained under the benchmark's own 60-epoch, patience-100 budget.

**Why 50 epochs per trial.** In the fixed runs (162 folds, 60-epoch ceiling) the best-validation epoch has median 19 and exceeds 20 in 49% of folds; 40 covers about 83%. Per head (median best epoch, MLVS-PT / DSS-EU / DSS-UK / DSS-US): FPQR 14/12/22/28, QR 15/18/10/16, BNN 28/33/50/40. At 20 epochs per trial BNN, whose best epoch exceeds 20 in nearly every DSS fold, would be tuned on under-trained models, and the search would favour settings that converge fast rather than settings that end best.

The learning-rate schedule changes how to read those best epochs. `multi_step` drops the rate tenfold at 50% and 90% of each run's own `max_epochs`, so a trial runs the whole schedule compressed into its length, not the first N epochs of the 60-epoch final run (drops at 30 and 54). The fixed runs' late best epochs partly reflect their drop at epoch 30. The real cost of a short trial is less time at each rate: 50 epochs (drops at 25 and 45) keeps 83% of the final run's time per phase, 40 keeps 67%, and 60 would match exactly.

**Fairness.** Tuning only FPQR gives it a search the baselines did not get. Either tune each baseline's own head fields under the same budget (QR: `loss_fn`, `kappa`, `crossing_penalty`; BNN: `kl_weight`, `n_samples`; Laplace has none), or state the asymmetry and treat the tuned FPQR as an upper bound.

**Engine change needed.** `ExperimentSpec.hpo_enabled` is all-or-nothing: switching it on would tune every head. Add a per-condition switch (for example `Condition.hpo_enabled: bool | None = None`, inheriting the spec), so that a new `fpqr_anchor_tuned` condition is tuned while `fpqr_anchor`, QR, BNN and Laplace stay fixed and directly comparable. The tuned head then reports beside the fixed one in the same run, on the same folds.

**Search strategy (Optuna 5).** Every tuned head stops early once 20 completed trials in a row fail to beat its best CRPS by 0.1% (at least 20 completions first), with the trial budget as the ceiling. This replaces `optuna.terminator`, which Optuna deprecated in 4.9 and removes in 6.0. FPQR keeps multivariate TPE because its search is wide and partly conditional. QR and BNN search a few mostly continuous settings and use the Gaussian-process sampler. Each search saves PED-ANOVA parameter importances as `backbone_hpo/<dataset>/<model>_head_tuned_importance.json`, and the report tabulates them as `hpo_importance.csv`. With `--hpo-windows K`, the HPO validation span is cut into K consecutive chunks with an expanding origin, and each trial is scored as the mean CRPS over the chunks. A Wilcoxon signed-rank pruner across windows then replaces per-epoch Hyperband. This is off by default for two reasons. First, each trial trains K times. Second, with a 0.1 threshold the test cannot prune before four windows (the smallest one-sided p is 0.25, 0.125 and 0.0625 at 2, 3 and 4 windows), so it saves compute only from K = 5. On the short DSS protocol, five windows leave validation chunks of about six days.
