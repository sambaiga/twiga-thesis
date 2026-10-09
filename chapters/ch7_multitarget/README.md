# Chapter 6 experiments: how to run

## Objective

Chapter 6 extends probabilistic forecasting from a single target to the
coupled active power, reactive power, and voltage triple at an LV secondary
substation, and asks whether modelling that coupling changes an operational
decision (hosting capacity, thermal exceedance) rather than only a generic
score. **Revised 2026-09-24: the copula-based dependence model is dropped.**
The chapter now presents multi-target FPQR plus multi-target conformal
calibration only (residual-based max-norm/Mahalanobis, and CQR's own
marginal/pooled variants) — `copula_variants.py` and `operational_decision.py`
below are retired for this chapter's own argument (their copula-family and
Brier-score conclusions are superseded, not extended); they are kept in this
directory only because other, still-copula-based work may reuse them. The
full argument for why the constraint framing matters, not only the score, is
in `POSITIONING.md`; the data-generation protocol is in `EXPERIMENT_PLAN.md`.
Results, findings, and statistical detail belong in `Chapter-6.tex` and
`DESIGN_DECISIONS.md`, not here — this file is a run guide only.

The chapter's current protocol maps onto four scripts:

| experiment | script | scope | rerun? |
|---|---|---|---|
| 1a — probabilistic forecasting benchmark | `multitarget_experiment.py` | `MLVS-PT-3PHASE` only | yes |
| 1b — PV/EV penetration sweep | `plot_ev_pv_penetration.py` | `DSS-EU` only | **no** — rebuilt from the existing `results_ev_pv_rate.xlsx`, not a live script |
| 2 — multi-target FPQR + conformal evaluation | `backbone_conformal_study.py` | `MLVS-PT-3PHASE`, `DSS-UK`, `DSS-US` (+ `DSS-EU` `none`-only) | yes |
| 3 — operational decision-making assessment | `backbone_conformal_decision.py` | `DSS-UK`, `DSS-US` | yes |

Experiment 1 was previously run across both `MLVS-PT-3PHASE` and `DSS-EU` in
one pass. It is now split: 1a reruns the FPQR/QR/BNN/Laplace benchmark on
`MLVS-PT-3PHASE` only (`--dataset MLVS-PT-3PHASE`, not `--dataset all` —
`DSS-EU` is 1b's dataset, not 1a's, and is not rerun at all); 1b's PV/EV
penetration results are rebuilt directly from the existing
`results_ev_pv_rate.xlsx` (10,920 rows, predates the current
`twiga.experiment` pipeline) via the new `plot_ev_pv_penetration.py`, per the
explicit decision not to rerun `DSS-EU`.

All engine-based scripts (`multitarget_experiment.py`,
`fpqr_gating_ablation.py`, `backbone_conformal_study.py`) import
condition/backbone/CV wiring from `twiga.experiment.ExperimentEngine`, the
same machinery Chapter 7's `mlpgam_design_ablation.py` uses.
`copula_variants.py`, `operational_decision.py`, and
`backbone_conformal_decision.py` share their backtest/metrics machinery with
`forecast_examples.py` instead (`build_backtest_folds`, `fit_and_predict`,
`hosting_capacity_rows`).

**Start here.** If you are picking this chapter up cold: run the scripts in
the numbered order under "What to rerun, and in what order" below, starting
at step 1 (Experiment 1a, `multitarget_experiment.py`). Do not jump ahead --
each step's result is a precondition later steps build on (Experiment 2's
`per_target_tau`/CV-protocol fixes are a precondition for trusting
Experiment 3, which imports Experiment 2's `BACKBONES` directly).
`--quick`/small-epoch smoke tests exist for every script and should be run
first, per script, before committing real server time.

## Current pipeline (2026-10-07): what the chapter's numbers come from

This section supersedes the table above and every older run guide below for
Experiment 1 (benchmark) and Experiment 2/3 (operational decision). The
chapter reports all four heads (FPQR, QR, BNN, Laplace) at their **fixed**
settings, on MLVS-PT-3PHASE, DSS-EU, DSS-UK and DSS-US, with max-norm and all
six CQR variants (scaled/unscaled x marginal/Bonferroni/pooled) as the
conformal set. Steps 2-4 (head tuning, the tuned FPQR's calibration, the
combined tuned set) were run as a check and are **not** what the chapter
reports: tuning improved FPQR's interval calibration but its forecasts were
less meaningful on inspection, so the chapter keeps the fixed head. Skip them
for a chapter rerun. Every command runs from the repository root.
`<experiments>` is `$TWIGA_EXPERIMENTS_ROOT`, by default `../experiments`, so
outputs land under `<experiments>/chapter-6-multitarget/`.

```bash
git pull origin thesis/browell-revision
```

### Experiment 1: benchmark

| step | script | output (under `<experiments>/chapter-6-multitarget/`) |
|---|---|---|
| 1 fixed heads + conformal | `experiment_1_benchmark/fixed_benchmark.py` | `fixed_benchmark/multitarget_fixed_benchmark/runs/` |
| 2 head tuning | `fixed_benchmark.py --group head_tuned` | same root, `backbone_hpo/` holds the tuned settings |
| 3 tuned FPQR calibration (MLVS-PT) | `fixed_benchmark.py --group head_tuned_conformal` | same root |
| 4 combined result set | `experiment_1_benchmark/combine_tuned_benchmark.py` | `multitarget_benchmark_final/` |
| 5 tables, figures, thesis images | `fixed_benchmark_report.py`, `fixed_benchmark_chapter_figures.py`, `fixed_benchmark_forecast_examples.py` | `fixed_benchmark/multitarget_fixed_benchmark/analysis/` |

1. **Fixed heads and their conformal variants** (all four datasets; the
   conformal group runs on MLVS-PT-3PHASE only, the only record with a
   calibration window):

   ```bash
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --quick
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py
   ```

2. **Head-only tuning** of FPQR, QR and BNN (backbone pinned, scale-free CRPS
   objective, 6 validation windows with Wilcoxon pruning). Only the tuned FPQR
   is reported; tuned QR and BNN barely changed and stay as a check:

   ```bash
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --group head_tuned --hpo-windows 6
   ```

3. **Calibrate the tuned FPQR on MLVS-PT-3PHASE.** A `head_tuned` run keeps no
   checkpoints, so this group refits the tuned FPQR from the cached settings
   in `backbone_hpo/` (no new search) and runs max-norm plus the six CQR
   variants on the refit's weights. Run it on the same machine as step 2:
   the refit is not bit-identical across hardware (a Mac refit scored native
   joint coverage 0.527 against the GPU run's 0.629 with identical settings).
   Pass `--tuned-hpo-root` if the tuned settings live elsewhere:

   ```bash
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --dataset MLVS-PT-3PHASE --group head_tuned_conformal
   ```

4. **Combine** the tuned FPQR with the fixed QR, BNN and Laplace. Copies
   result parquet files only; the source runs are not modified, and the
   fixed FPQR is kept under `benchmark_fixed/`/`conformal_fixed/`:

   ```bash
   E=<experiments>/chapter-6-multitarget
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/combine_tuned_benchmark.py \
     --fixed $E/fixed_benchmark/multitarget_fixed_benchmark \
     --tuned $E/fixed_benchmark/multitarget_fixed_benchmark \
     --refit $E/fixed_benchmark/multitarget_fixed_benchmark \
     --out $E/multitarget_benchmark_final
   ```

   `--fixed`, `--tuned` and `--refit` may be different roots (e.g. a copied
   `multitarget_fixed_benchmark_v1` holding the head-tuned runs).

5. **Report, chapter figures, forecast examples, thesis images**, from the
   fixed run. `--copy-to-images` writes the PDFs into
   `Papers/PhD thesis -Final-Revision/Images/Chapter-6/`. The forecast
   examples draw every head and conformal variant on FPQR's best day per
   record, one figure per (record, condition, target):

   ```bash
   F=$E/fixed_benchmark/multitarget_fixed_benchmark
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_report.py --experiment-root $F
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_chapter_figures.py --experiment-root $F --copy-to-images
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_forecast_examples.py --experiment-root $F --copy-to-images
   ```

   The same three scripts read the combined tuned set
   (`--experiment-root $E/multitarget_benchmark_final`) if it is ever needed.

   To compare tuned and fixed heads in one test family instead (all seven
   heads), read the tuned runs from their own root:

   ```bash
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_report.py \
     --experiment-root $E/fixed_benchmark/multitarget_fixed_benchmark \
     --tuned-root <root holding the head_tuned runs> --include-tuned --output <analysis folder>
   ```

### Operational decision: DSS-UK and DSS-US penetration sweep

`experiment_3_decision/fixed_decision_sweep.py` trains the four fixed heads on each
EV x PV scenario of DSS-UK (weak and stiff feeders) and DSS-US rank 8, and scores
the probability of each operational event (single-step and sustained overload at
1.0x, 1.25x and 1.5x the pre-DER headroom, customer-voltage breach and their union
on DSS-UK) through every scenario form a head supports: independent and
Schaake-coupled draws for every head, and for the BNN also its own paths with
per-cell or day-coherent noise (`FORECAST_TYPE_PLAN.md`). Conformal conditions are
off (`--with-conformal` restores them). Each head keeps its checkpoints and saves
its raw forecasts, so `forecast_type_study.py` runs on any scenario offline.

DSS-US uses rank 8, not rank 0: SMART-DS assigns DER to a fresh random set of
customers at each penetration level, so most single transformers (rank 0 among
them) do not see their DER grow from one level to the next. The sweep checks every
network's response to EV and PV at start and warns otherwise.

```bash
# Smoke test, one scenario:
uv run python chapters/ch7_multitarget/experiment_3_decision/fixed_decision_sweep.py --quick --stage all --records DSS-US:8 --ev 0.4 --pv 0

# Stage 1: EV-only and PV-only levels (21 scenarios), then stage 2: the other EV x PV points (27):
uv run python chapters/ch7_multitarget/experiment_3_decision/fixed_decision_sweep.py --stage axes
uv run python chapters/ch7_multitarget/experiment_3_decision/fixed_decision_sweep.py --stage grid

# Report by forecast type (event rates, head ranking, scenario forms, paired tests, figures):
uv run python chapters/ch7_multitarget/experiment_3_decision/decision_forecast_type_report.py \
  --experiment-root $E/fixed_decision_sweep/fixed_benchmark/multitarget_fixed_decision_sweep
```

## Next rerun (2026-10-02): MLVS-PT-3PHASE benchmark+conformal, then DSS-UK/DSS-US decision, `cwe_stratified` objective

Two runs, in this order (the second's `fpqr_anchor_outer`/`qr` conditions
reuse the first's own per-fold checkpoints and calibration-window layout via
`reuse_checkpoint_from`/`reserve_calib_window`, so this sequencing matters,
same as "What to rerun, and in what order" above). Delete stale HPO caches
first if they predate this session's `_FPQR_STABILITY_PINS` additions
(`proposal_type="softmax"`, `gradient_clip_val=1.0`), the `loss_fn` search
space restriction (`"winkler"` dropped), the `per_target_tau` un-pin for
`fpqr_anchor_outer` specifically, or the `max_abs`/`mahalanobis_*` conformal
trim — any of these change a condition's own HPO fingerprint or its final
built config, so a cache from before they landed would otherwise be
silently reused unchanged.

1. **`multitarget_experiment.py`, `benchmark` + `conformal` groups,
   `MLVS-PT-3PHASE` only** (`grid` is a supporting ablation, not part of
   this rerun):

   ```bash
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py \
     --group benchmark conformal --dataset MLVS-PT-3PHASE \
     --run-hpo --hpo-objective cwe_stratified --hpo-trials 100 --hpo-epochs 50

   # Sync (step 0 under "What to rerun, and in what order"), then analyse:
   uv run python chapters/ch7_multitarget/shared/ch6_analyse_results.py \
     --experiment multitarget_experiment --group benchmark conformal --dataset MLVS-PT-3PHASE
   ```

2. **`multitarget_decision.py`, both decisions** — the new MLPF-only, four-head
   (`fpqr_anchor_outer`/`qr`/`bnn`/`laplace`) companion to
   `backbone_conformal_decision.py`'s own backbone comparison; see its own
   section below for what it covers and why it is a separate file:

   ```bash
   uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision.py --quick

   uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision.py \
     --decisions DSS-UK DSS-US --epochs 150 \
     --run-hpo --hpo-objective cwe_stratified --hpo-trials 100 --hpo-epochs 50

   # Sync, then analyse (same ch6_analyse_results.py entry point as backbone_conformal_decision.py,
   # reads this script's own "multitarget_decision" output tree):
   uv run python chapters/ch7_multitarget/shared/ch6_analyse_results.py \
     --experiment multitarget_decision --dataset dss-uk dss-us
   ```

   `--hpo-objective` only applies to `fpqr_anchor_outer`/`qr` here (the two
   quantile-producing heads) — `bnn`/`laplace` stay `winkler_score`,
   matching `multitarget_experiment.py`'s own restriction
   (`cond.model_cls in (MLPFFPQRConfig, MLPFQRConfig)`).

## Prerequisites

- Run from the repository root with `uv run`, or from inside `experiment/`
  with the virtual environment active.
- A tuned MLPF/MLPGAM backbone at `chapters/ch7_multitarget/backbone_params.json`.
  `tune_backbone.py` writes it; every script below falls back to a hardcoded
  default if it is missing, so re-tuning is optional.
- Every script's local smoke-test command (below) finishes in minutes and
  should be run before the corresponding full server run — this replaces
  discovering a broken condition after committing server time.

## Datasets

| key | source | role |
|---|---|---|
| `DSS-EU:default:0.4:0.4` | OpenDSS power flow, EU LV feeder (`uk-mvlv`) | Experiment 1/2 reference; voltage-binding half of Experiment 3 (`DSS-UK`) |
| `DSS-UK:weak:0.4:0.4` | OpenDSS power flow, weak EU feeder | Experiment 3, hosting-capacity (voltage) decision |
| `DSS-US:0:0.4:0.4` | OpenDSS power flow, SMART-DS | Experiment 3, thermal-exceedance decision |
| `MLVS-PT-3PHASE` | measured, Madeira LV substation | Experiment 1/2, the measured (not simulated) P/Q/V record |

`MLVS-PT-3PHASE.parquet` is not produced from raw per-year CSVs available
only on the project server (`PhD-Thesis-Anthony/notebook/Prepare-data.ipynb`)
or copied into `examples/data/`; scripts that default to it fail with a
plain `FileNotFoundError` until it exists locally. `EXPERIMENT_PLAN.md`
documents how the OpenDSS scenarios themselves are generated.

## Recent fixes (2026-09-27/28) — rerun everything below, don't trust older cached results

**Directory reorg (2026-09-27)**: every script moved out of this flat
directory into a subfolder by experiment number —
`experiment_1_benchmark/`, `experiment_2_backbone_conformal/`,
`experiment_3_decision/`, `copula/` (retired scripts), `shared/` (modules
several experiments import: `forecast_examples.py`, `tune_backbone.py`,
`ch6_analyse_results.py`, `ch6_forecast_examples.py`,
`split_scheme_figure.py`). All commands below already reflect the new
paths. Large output/data directories (`results/`, `figures/`, `data/`,
`forecast_examples/`, `bakeoff/`, `study/`, `operational_decision/`,
`joint_vs_independent/`) stayed at this top level.

**HPO cache fingerprinting**: `run_backbone_hpo` now validates a cached
`*_best_params.json` against a SHA-256 fingerprint (data content, model
config, search space, protocol settings) before reusing it — a stale or
pre-fingerprint cache is silently re-tuned rather than trusted, and any
code change to the search space/objective invalidates every existing
cache for that model. `--hpo-trials` is deliberately **not** part of the
fingerprint (2026-09-28 fix) — changing it between runs no longer forces
a full re-tune.

**FPQR/QR search-space pinning, fixed 2026-09-28**: `proposal_type`/
`qvn_type` are now pinned *inside* the Optuna search itself (before any
trial samples), not just applied post-hoc to `best_params` — the old
post-hoc-only approach let a resumed/historical study's "best trial" carry
an unpinned value (e.g. `proposal_type="kumaraswamy"` leaking through
despite the intended `"softmax"` pin) whenever `--hpo-trials` was small
enough to just resume an old study rather than search fresh. **If you ever
resumed an old FPQR/QR study with a small `--hpo-trials` before this date,
delete that cache and re-tune from an empty study** — a resumed cache may
not actually reflect the pin.

**FPQR/QR HPO objective, added 2026-09-27/28**: quantile-producing
backbones (FPQR/QR) now tune against **Winkler score** by default, not
point accuracy (MAE/RMSE) — point accuracy alone can't tell a
well-calibrated quantile grid from a degenerate/collapsed one at equal
median accuracy. `get_quantile_metrics` also now computes `picp`/`nmpi`/
`ace`/`cwe`/`cwe_stratified` (all from the outermost predicted quantile
pair, no calibration needed) — pass any of them via the new
`--hpo-objective` flag (`multitarget_experiment.py`: FPQR/QR conditions
only; `backbone_conformal_study.py`: every condition, all FPQR-family).
`cwe`/`cwe_stratified` are diagnostics, not proper scoring rules (Chapter
3's own framing) — prefer `winkler_score` (the default) as the actual
optimisation target; use the others to inspect, not to optimise against,
unless you have a specific reason.

**BNN stability, fixed 2026-09-28** (`twiga/distributions/nn/bayesian.py`,
`twiga/models/nn/mlpfbnn_model.py`): three independent bugs, all silent
(no crash, just bad numbers) before this date —

1. The KL term was an unnormalised sum over every decoder weight, with no
   dataset-size correction, so it could swamp the NLL term entirely
   (`val_loss ≈ val_kl`, `val_nll` negligible by comparison) depending on
   the HPO-sampled `kl_weight`/`hidden_dim`. Fixed: KL is now scaled by
   `batch_size / dataset_size` (the standard Blundell et al. 2015
   minibatch-KL weighting), set from `BaseNeuralForecast.fit()` once the
   real training-set size is known.
2. The learned homoscedastic noise (`log_noise_var`) had no floor, so an
   overfitting trial could collapse it toward zero and make NLL explode
   catastrophically on held-out data (observed: `val_nll` in the 1e9-1e11
   range). Fixed: clamped to `[-8.0, 4.0]` (noise std ≈ 0.018-7.4).
3. `MLPFBNNConfig` never overrode `gradient_clip_val` (unlike
   `BaseFPQRConfig`/`BaseQRConfig`, which both clip "for stability") — an
   unstable trial's raw decoder output could diverge unchecked. Fixed:
   `gradient_clip_val=1.0`, matching FPQR/QR's existing convention.

**Delete any cached `mlpfbnn`/`mlpffpqr`/`mlpfqr` HPO params from before
2026-09-28 and re-tune** — none of the three BNN fixes above, nor the
FPQR/QR pinning fix, are retroactive; an old cache reflects the buggy
behaviour, not the current code.

**Interval-scoring alpha mismatch, fixed 2026-09-28**
(`twiga/forecaster/abstract.py`): every model's interval bounds used to be
scored (Winkler/PICP/NMPI/CWE) at a hardcoded `alpha=0.01` regardless of
the alpha the bounds were actually built at — Laplace/BNN's native
z-score interval defaults to `alpha=0.10` (90%), and FPQR/QR's own outer
quantile pair has whatever nominal span its fixed/learned grid implies
(e.g. QR's default `[0.1,...,0.9]` grid is an 80% interval,
`alpha=0.2`). Every model's interval metrics from before this date were
scored against the wrong nominal coverage — **re-run any full backtest you
still care about**, not just HPO.

## Recent fixes (2026-09-29 to 2026-10-01) — rerun FPQR/QR conditions again, delete old caches

**FPQR search space substantially widened, 2026-09-29**
(`MLPFFPQRConfig.search_space`): `n_quantiles` went from a 4-value
categorical (`7, 9, 15, 21`) to a real integer range `(10, 100)`, and
`num_cosines` from `(16, 32, 64)` to `(8, 128)` — the earlier ranges were
tuned on a *single-target* ablation and do not necessarily transfer to the
multi-target P/Q/V setting. `qvn_type` (all 7 architectures, including
`monotone`/`monotone_cal`), `per_target_tau`, and `gradient_clip_val` are
now genuinely searched rather than fixed at class defaults. **Any cached
`mlpffpqr`/`mlpfqr` HPO params from before this date reflect the old,
narrower space — delete and re-tune.**

**FPQR's own fixed-hparams pin removed, restored to a real joint search,
2026-09-28/29** (`multitarget_experiment.py`): an earlier attempt to pin
FPQR's hyperparameters to values known-good on a *different* dataset
(the PhD-thesis DSS reference implementation) was tried and **reverted** —
on `MLVS-PT-3PHASE` it produced a large `n_quantiles`/`gradient_clip_val`
mismatch that drastically over-covered (PICP 0.98-0.99 against a 0.90
nominal target on every one of P/Q/V). The `fpqr`/`fpqr_shared_grid`/
`fpqr_per_target_grid` conditions each now run their own full joint
backbone+FPQR-head Optuna search (`hpo_model_cls=MLPFFPQRConfig`, its own
dedicated `hpo_variant`), not a search shared with `qr`/`bnn`/`laplace`'s
backbone-only one and not a value pinned from an external reference —
`per_target_tau` is pinned per-condition via `hpo_fixed_params` (not
`overrides`), since the `grid` group's whole purpose is comparing it in a
controlled way.

**`cwe_stratified`'s "no stratum was scorable" warning, fixed 2026-09-29**
(`SPEC_MLVS_PT_3PHASE.hpo_val_months`): raised from the class default of 2
to 6 — the default was not reliably producing the ~50-sample-per-bucket
minimum `cwe_stratified` needs during HPO trials on this dataset. Diagnostic
only (never the HPO objective), so this doesn't change what any trial
optimises for, only whether the diagnostic is actually computed during HPO.

**Mahalanobis/copula calibration on FPQR's real per-row quantile grid, fixed
2026-09-30** (`twiga/core/stats/target_dependence.py`,
`twiga/distributions/conformal/multivariate.py`): `compute_pit` crashed
(`ValueError: quantile_levels must be strictly increasing`) whenever a
`per_target_tau=True` FPQR model's real `quantile_levels` (shape
`(B, N_q, H, D)`) reached it — the strict-monotonicity check was reading the
trailing *target* axis (P/Q/V), which has no reason to be monotonic, not the
quantile-index axis. `compute_pit`/`MultivariateConformal._build_pit` now
treat an adaptive per-row grid the same way quantile *values* already are
(forced monotone via `np.maximum.accumulate`, not hard-rejected); a
caller-supplied fixed `(Q,)` grid still validates strictly, since that one
really is a caller contract. This was blocking every PIT-based
`dependence_method` (`spearman`/`kendall`/`gaussian_copula`) for
`score_type="mahalanobis"` on any `per_target_tau=True` backbone — i.e.
every FPQR backbone in this chapter. **Delete any run that hit this crash
and re-run** — it was failing partway through a fold, not silently producing
wrong numbers.

**`CalibratedMonotonicQVN` boundary-tensor wiring, fixed 2026-09-30**
(`twiga/distributions/nn/fpqr/{distribution,qvn}.py`): the `qvn_type=
"monotone_cal"` condition's caller code (`distribution.py`) still built and
passed the old *midpoint* tensor under the old parameter name, while the
class's own API had moved to the true *boundary* tensor; `qvn_type=
"monotone"` had the same gap once `MonotonicQVN`'s `delta_tau` became a
required (no longer optional) argument. Both fixed, plus a real off-by-one
in `qvn.py`'s own boundary/combined alignment (`taus[:, 2:, :]`, not
`taus[:, 1:, :]`) found only once real shapes were exercised. Affects any
`qvn_type` search that samples `monotone`/`monotone_cal` — which the widened
search space above now does by default.

**QR added as a fourth head, 2026-09-30/10-01**
(`backbone_conformal_study.py`, `backbone_conformal_decision.py`): both
scripts' `--backbones`/condition grid now include `qr` alongside
`mlpf`/`mlpgam`/`nhits` — a fixed-grid baseline under the same conformal
comparison. QR reuses a dedicated, backbone-only `MLPFConfig` HPO search
(`QR_HPO_VARIANT = "shared_mlp_backbone"`) rather than its own full joint
FPQR-style study, since `MLPFQRConfig` has no FPQR-specific hyperparameters
to tune. `backbone_conformal_decision.py` also gained the `--output-root`/
`TWIGA_EXPERIMENTS_ROOT` override and the `--skip-hpo`/`--run-hpo` two-flag
pair, matching the other two scripts (it previously had neither).

**Per-target metric breakdown, added 2026-09-29**
(`ch6_analyse_results.py`): `analyse_multitarget_group` now also writes
`{dataset}_metrics_by_target.csv` alongside the existing pooled table —
diagnostic only (significance/Kupiec/the conclusion stay pooled, per this
project's own convention that DM needs a per-observation loss differential,
not a per-target split), but it is what caught the voltage-specific
amplitude-compression pattern common to all four benchmark conditions on
`MLVS-PT-3PHASE` (every method's point forecast under-represents voltage's
true daily swing; none of P/Q/V's other targets show it).

**`hpo_fixed_params`/`hpo_parent_model_cls` pins silently stopped holding
after trial 1, fixed 2026-10-01** (65 files across `twiga/models/nn/`):
every NN model wrapper's `update(trial)` reconstructed `self.model_config`
via `SomeConfig(**params)`, which silently discarded the Optuna-stripped
`search_space` and reverted it to the class's full, unstripped default on
every trial after the first — so a `Condition.hpo_fixed_params` pin (or a
`hpo_parent_model_cls`-derived backbone pin) only ever held for trial 0.
Confirmed directly: `fpqr_anchor_outer`'s own saved `best_params.json`
showed `proposal_anchor_outer: false` despite being pinned `true`. Fixed at
the single shared point: `self.model_config =
self.model_config.model_copy(update=params)`, which preserves every field
not in `params` (including the stripped search space) instead of discarding
it. **Any cached HPO params for `fpqr_shared_grid`, `fpqr_per_target_grid`,
or `fpqr_anchor_outer` from before this date reflect the broken pin — delete
and re-tune, they were not testing what their own name claims.**

**All four FPQR conditions now share `qr`/`bnn`/`laplace`'s backbone, 2026-10-01**
(`multitarget_experiment.py`): `fpqr`, `fpqr_shared_grid`,
`fpqr_per_target_grid`, and `fpqr_anchor_outer` all set
`hpo_parent_model_cls=MLPFConfig, hpo_parent_variant=SHARED_HPO_VARIANT`,
reversing the 2026-09-28 "own joint backbone+head search" decision above —
that joint search's 50-trial budget was being spent across ~15 backbone
dimensions none of the four conditions needed their own copy of, on top of
~15 FPQR-head dimensions, and the backbone-capacity asymmetry against
`qr`/`bnn`/`laplace` confounded the head-to-head benchmark comparison. Only
reliable now that the pin-loss bug above is fixed — this mechanism
(`hpo_parent_model_cls`) existed before but silently didn't work past trial
1 either.

**`fpqr_anchor_outer` condition, added 2026-09-30/10-01** (`grid` group):
pins the proposal's two outermost quantile levels to `conf_level/2` and
`1 - conf_level/2` (`BaseFPQRConfig.proposal_anchor_outer`,
`twiga/distributions/nn/fpqr/proposal.py`) instead of leaving them fully
adaptive — the guarantee a fixed-grid model like QR gets for free. Only
implemented for `proposal_type="softmax"`; silently a no-op under
`"horizon"`/`"kumaraswamy"` rather than raising, so it is searched
unconditionally in `MLPFFPQRConfig.search_space` and guarded at the
model-construction call site instead of being excluded from the search
space itself (`Condition.hpo_fixed_params` cannot reach a conditionally-
suggested field reliably — see the pin-loss bug above).

**`fpqr_lv_differenced` condition, added 2026-10-01** (`grid` group):
forecasts voltage (LV) as its own first difference rather than its raw
level, reconstructed back to level scale at evaluation
(`DataPipelineConfig.target_differencing`,
`twiga.core.data.pipeline.DataPipeline`,
`twiga.forecaster.base.BaseForecaster._undifference_predictions` — see
`Chapter-6.tex`'s own voltage-difficulty discussion). LV's own
`SignalCharacteriser` verdict on `MLVS-PT-3PHASE` is fractional integration
(`d=1`): its monthly mean drifts by ~3.5V against a ~1.8V total standard
deviation, comparable to the whole signal's own variance — every benchmark
condition's raw-level LV forecast shows this as a shared,
architecture-independent bias. `Condition.data_overrides` (new field,
parallel to `overrides` but reaching `DataPipelineConfig` instead of the
model config) is required for this, since `target_differencing` lives on a
separate config object built fresh per condition; `overrides` cannot reach
it.

**FPQR search-space ranges tightened, 2026-10-01**
(`MLPFFPQRConfig.search_space`): `entropy_weight`'s upper bound lowered from
`10.0` to `3.0` — confirmed against the first real post-pin-fix run that
`entropy_weight=9.22` was this run's single worst-performing FPQR condition
(Winkler score roughly double every sibling condition, all of which scored
`entropy_weight < 1.5`), and, checked directly against its own TensorBoard
`val_mae` curves, trained into a genuine overfitting regime (train MAE
improving, val MAE worsening) faster and more erratically than every other
condition. A large `entropy_weight` keeps the proposal's quantile grid
artificially spread against the pinball loss's own pull toward committing to
sharp, accurate quantile locations; near weight initialisation the proposal
is already close to uniform (high entropy for free), so the term costs
little early on and increasingly fights genuine specialisation as training
proceeds.

**`early_stop_patience` lowered 100 → 40, 2026-10-01** (`_SPEC_COMMON`,
`multitarget_experiment.py`): the 2026-09-18 change to 100 (below) correctly
fixed a `reduce_on_plateau` scheduler-patience race, but checked directly
against TensorBoard logs it was far more conservative than needed — 9 of
FPQR's 10 CV folds, in both this run and the one before it, reach their true
best validation epoch between epoch 4 and 72 of a 150-epoch ceiling, then
train 80-130+ further epochs into visible overfitting before patience=100
ever stops them. 40 still clears `reduce_on_plateau`'s own worst case
(`int(prob_patience_max × max_epochs) = int(0.2 × 150) = 30`) with real
margin, while cutting most of that wasted tail on every fold and every HPO
trial. `ModelCheckpoint` itself was never affected by this — it already
saves the true best-validation epoch regardless of patience — so this is a
compute-budget fix, not a correctness fix. `dropout` was checked too and
ruled out: the regressed trial's own `dropout=0.30` was already near its
`0.35` range ceiling, so under-regularisation is not a contributing factor.

**Multi-target feature/lag selection now characterises every target, not
just the first, 2026-10-01** (`twiga/utils/data_loading.py:load_dataset`):
previously ran `SignalCharacteriser` on `target_series[0]` only (`"LP"` for
`MLVS-PT-3PHASE`) and applied that single target's lag/calendar/exogenous
selections to every target — reactive power and voltage are not guaranteed
to share active power's seasonal/autoregressive structure. Now characterises
each target independently and unions the results (deduplicated) across
targets; `lookback_window_size` takes the max of every target's own
recommendation. Single-target datasets are unaffected (same one
characterisation call as before). **Changes the feature/lag set every
condition in this chapter trains on** (not an FPQR-specific change) — this
landed in the same commit range as the fixes above, so any run from before
it predates this feature set regardless of which conditions it covers.

**Checkpoint reuse (`reuse_checkpoint_from`/`reserve_calib_window`) added,
conformal group expanded then trimmed to 16 conditions, 2026-10-01/02**
(`twiga/experiment/{condition,engine}.py`, `multitarget_experiment.py`,
`ch6_analyse_results.py`): see "Checkpoint reuse" under
`multitarget_experiment.py` above for the mechanism and its preconditions.
Initially expanded to the full 36-condition sweep (every
`backbone_conformal_study.py` variant), then trimmed back to 16 once a real
run (`run_id=20261002_07253650b7d21a6dc24d5fb8748d1fdb480710_f25312`) showed
`max_abs` and all four `mahalanobis_*` variants losing to `max_norm` on every
axis for all three non-FPQR heads — see `multitarget_experiment.py`'s own
`_MULTIVARIATE_CONFORMAL_VARIANTS` comment for the numbers.
Also fixed `on_load_checkpoint()` (`twiga/forecaster/base.py`), which never
actually reached a domain='nn' model's own checkpoint-loading logic at
all — its manifest/`*.pkl`-glob file-finding only matches the ml/baseline
checkpoint format, which nn models (Lightning `*.ckpt` files instead) never
produce, so it silently returned early on every nn-domain call before this
fix, with no error. `ch6_analyse_results.py`'s `MULTITARGET_REFERENCE`
updated from `"laplace_conformal"` (no longer a condition name) to
`"laplace_max_norm"`.

## What to rerun, and in what order

As of 2026-09-28, Experiments 1a, 2, and 3 are all outstanding reruns, each
followed by its own sync-and-analyse pass (step 0). Do them in this
order — Experiment 2's `per_target_tau=True` fix and CV-protocol correction
are prerequisites for trusting Experiment 3's own backbone/conformal grid,
since `backbone_conformal_decision.py` imports its `BACKBONES` directly from
`backbone_conformal_study.py`. Delete stale HPO caches per "Recent fixes"
above before any of these, or they will silently reuse pre-fix numbers.

0. **Sync a completed run into the local, git-ignored analysis tree before
   analysing it.** `ch6_analyse_results.py`/`ch6_forecast_examples.py`
   (below) read from
   `chapters/ch7_multitarget/results/chapter-6-multitarget/<experiment>/<spec-slug>/runs/<run_id>/`
   — a **separate, local-only copy** of a run, not the external
   `../experiments/chapter-6-multitarget/...` tree every script above
   writes to directly. Nothing copies a run there automatically; a fresh
   run's results stay invisible to the analysis scripts (silently falling
   back to whatever run was synced last, however old) until you copy it
   yourself:

   ```bash
   EXT=../experiments/chapter-6-multitarget/<experiment>/<spec-slug>/runs/<run_id>
   LOCAL=chapters/ch7_multitarget/results/chapter-6-multitarget/<experiment>/<spec-slug>/runs/<run_id>
   mkdir -p "$LOCAL/results"
   cp "$EXT"/meta.json "$EXT"/run_status.json "$LOCAL/"
   cp "$EXT"/results/*.csv "$LOCAL/results/"
   # per condition, metrics/predictions parquet only -- skip checkpoints/logs/figures:
   for cond_dir in "$EXT"/conditions/*/*/*/; do
     rel=${cond_dir#"$EXT"/conditions/}
     mkdir -p "$LOCAL/conditions/$rel"
     cp "$cond_dir"*.parquet "$LOCAL/conditions/$rel" 2>/dev/null
   done
   ```

1. **`multitarget_experiment.py --dataset MLVS-PT-3PHASE`, Experiment 1a.**
   `DSS-EU` is 1b, not rerun (see Objective) — do not pass `--dataset all`
   here. `--hpo-objective` defaults to `winkler_score` for the `fpqr`/`qr`
   conditions (see "Recent fixes" above); pass `cwe_stratified` to try that
   diagnostic as the objective instead. `fpqr`/`fpqr_shared_grid`/
   `fpqr_per_target_grid` each now run their own full joint search over the
   widened space (see "Recent fixes (2026-09-29 to 2026-10-01)" above) —
   `--hpo-trials 100 --hpo-epochs 50` is the recommended budget for that
   wider space, not the script's own (lower) defaults. `--run-hpo` alone
   only searches when no matching cache exists (`--hpo-cache-policy reuse`,
   the default) — it does **not** force a fresh search if a cache already
   matches the current fingerprint. To force one regardless (e.g. after a
   code change that doesn't happen to alter the fingerprint), add
   `--hpo-cache-policy refresh` (only valid together with `--run-hpo`).

   ```bash
   uv run python chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py --dataset MLVS-PT-3PHASE --run-hpo --hpo-trials 100 --hpo-epochs 50

   # Sync (step 0 above), then analyse:
   uv run python chapters/ch7_multitarget/shared/ch6_analyse_results.py --experiment multitarget_experiment --group benchmark --dataset MLVS-PT-3PHASE

   # Good/bad-day forecast panels, per target (repeat --target for LP/LQ/LV):
   uv run python chapters/ch7_multitarget/shared/ch6_forecast_examples.py --experiment multitarget_experiment --group benchmark --dataset MLVS-PT-3PHASE --target LP --day-metric winkler --latex
   ```
2. **`backbone_conformal_study.py --dataset all`, Experiment 2.** 4 heads
   (MLPF, MLPGAM, NHiTS FPQR, plus QR's fixed-grid baseline, added
   2026-09-30/10-01) x 13 conformal conditions (`none`, `max_norm`,
   `max_abs`, four `mahalanobis_*` correlation estimators, six CQR
   `quantile_*` variants). Verify with `--quick` first: two library bugs
   were fixed 2026-09-24 that previously made `max_norm`/`mahalanobis` crash
   on every run (`_sigma_from_quantiles`'s shape assumption, now
   `twiga.distributions.ml.utils.get_sigma_from_quantile_levels`; `corr`
   never reaching `MultivariateConformal.calibrate()`), a third PIT-shape bug
   fixed 2026-09-30 blocked every `mahalanobis_{spearman,kendall,
   gaussian_copula}` variant specifically (see "Recent fixes" above), and
   `per_target_tau=True` was enabled for all three FPQR backbones after a
   pooled-CQR/max-norm blow-up was traced to a shared quantile grid letting
   one target's predicted spread collapse — re-check the `*_quantile_joint`
   and `*_q_kvar`-style sample plots after a `--quick` run before committing
   full server time, per `backbone_conformal_study.py`'s own module
   docstring. `--backbones` restricts to a subset (e.g. `--backbones qr
   mlpf`); QR reuses a dedicated backbone-only HPO search
   (`QR_HPO_VARIANT`), not its own full joint one.

   `--hpo-objective` (default `winkler_score`) applies to every condition
   here — all FPQR-family or QR, all quantile-producing.

   ```bash
   uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --quick
   uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --dataset all --run-hpo --hpo-trials 100 --hpo-epochs 50

   # Sync (step 0 above) each dataset's run, then analyse and get the DM/MCS report:
   uv run python chapters/ch7_multitarget/shared/ch6_analyse_results.py --experiment backbone_conformal_study --group backbone --dataset mlvs-pt-3phase dss-uk dss-us dss-eu
   uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --report backbone
   ```
3. **`backbone_conformal_decision.py`, Experiment 3.** `DSS-UK`/`DSS-US`
   only (`MLVS-PT-3PHASE` has no decision threshold). Same 4 heads as
   Experiment 2 (`--backbones` now includes `qr`, added 2026-09-30/10-01,
   sharing the same dedicated backbone-only HPO search there). Since the
   `ExperimentEngine` migration, the CV window (~18-week derived train, 4
   non-overlapping folds, weeks-based) is fixed internally in `build_spec()`
   rather than set via CLI flags — `--train-size`/`--split-freq`/`--expanding`/
   `--train-stride`/`--test-stride`/`--num-workers` no longer exist on this
   script (an old command using them will fail with "unrecognized
   arguments"). `--test-size`/`--step` still exist and already default to 8,
   matching the corrected protocol (fewer, larger test windows — more
   exceedance events per fold — deliberately different from Experiment 2's
   own protocol; see `Chapter-6.tex`'s experimental-protocol discussion for
   why a single-year record cannot reach winter test coverage either way).
   `--skip-hpo`/`--run-hpo` (default: skip) and `--output-root`/
   `TWIGA_EXPERIMENTS_ROOT` now match the other two scripts exactly (added
   2026-10-01 — this script previously had only a bare `--run-hpo` flag and
   no output-root override at all).

   ```bash
   uv run python chapters/ch7_multitarget/experiment_3_decision/backbone_conformal_decision.py --quick
   uv run python chapters/ch7_multitarget/experiment_3_decision/backbone_conformal_decision.py --decisions DSS-UK DSS-US --epochs 150 --run-hpo --hpo-trials 100 --hpo-epochs 50

   # Sync (step 0 above) each decision's run, then analyse:
   uv run python chapters/ch7_multitarget/shared/ch6_analyse_results.py --experiment backbone_conformal_decision --dataset dss-uk dss-us
   ```

`fpqr_gating_ablation.py`'s mechanism ablation, `copula_variants.py`, and
`operational_decision.py` are not part of this list — the first already has
a clean run `DESIGN_DECISIONS.md` cites; the latter two are retired for this
chapter (see Objective). `penetration_sweep.py` is superseded for Experiment
1b's own purpose by `plot_ev_pv_penetration.py` (no rerun, reads the existing
xlsx directly).

Steps 2-6 can run in parallel across machines if you have the server time to
spare (each uses its own output directory and its own `--run-hpo` tuning
pass); the ordering above is for running them one at a time.

## `multitarget_experiment.py`: Experiment 1a, head benchmark (MLVS-PT-3PHASE only)

**Revised 2026-09-24**: run with `--dataset MLVS-PT-3PHASE`, not `--dataset all`
— `DSS-EU` is Experiment 1b (the PV/EV penetration sweep), which is not
rerun; its results are rebuilt from the existing `results_ev_pv_rate.xlsx`
via `plot_ev_pv_penetration.py` (below), per the explicit decision not to
rerun `DSS-EU`. The `benchmark`/`grid`/`conformal` groups described below
still apply to `MLVS-PT-3PHASE` unchanged.

**Objective.** Compare FPQR against QR, BNN, Laplace, and (on
MLVS-PT-3PHASE) conformal-calibrated variants of all four, plus three
supporting-mechanism checks, under one CV protocol. Three groups:
`benchmark` (head comparison), `grid` (supporting mechanism checks: shared-
vs-per-target quantile grid, `fpqr_anchor_outer` — pins the two outermost
quantile levels to the nominal interval boundary instead of leaving them
fully adaptive, the guarantee a fixed-grid model like QR gets for free — and
`fpqr_lv_differenced`, forecasting voltage as its own first difference
rather than its raw level, added 2026-10-01; see "Recent fixes" below),
`conformal` (MLVS-PT-3PHASE only, since DSS-EU's `cv_calib_size=0` leaves
nothing to calibrate against).

**`conformal` group, 16 conditions** (expanded to 36 on 2026-10-01, trimmed
back on 2026-10-02 once a real run showed why): one `multivariate` variant,
`max_norm`, applied to all four `benchmark`-group heads (`fpqr`/`qr`/`bnn`/
`laplace`), plus the six `quantile_{scaled,unscaled}_{marginal,pooled,
bonferroni}` CQR variants applied only to the two quantile heads (`fpqr`/
`qr`), since `method="quantile"` requires a native quantile output BNN/
Laplace never produce. `max_abs` and the four `mahalanobis_*`
dependence-method variants (`backbone_conformal_study.py`'s own
`CONFORMAL_VARIANTS` still has all of these, for its own separate backbone
comparison) were dropped here after
`run_id=20261002_07253650b7d21a6dc24d5fb8748d1fdb480710_f25312` showed both
losing to `max_norm` on every axis (Winkler, PICP/Kupiec exceedance, CWE)
for all three non-FPQR heads, with the four `mahalanobis_*` dependence
estimators barely distinguishable from each other on top of that (Kendall
and Gaussian-copula identical to 4 decimal places) — see
`_MULTIVARIATE_CONFORMAL_VARIANTS`'s own comment in
`multitarget_experiment.py` for the full numbers. This empirically confirms
`MultivariateConformal`'s own documented caveat that a *fixed* correlation
doesn't reliably beat `max_norm`'s own implicit handling of it, for this
dataset. Condition names are `{head}_{variant}`, e.g. `fpqr_max_norm`,
`qr_quantile_scaled_pooled`, `laplace_max_norm`. None of these 16 conditions
retrain a model: each reuses its `benchmark`-group namesake's already-trained
per-fold weights via `Condition.reuse_checkpoint_from`
(`twiga/experiment/condition.py`/`engine.py`) and only calibrates — see
"Checkpoint reuse" below for the mechanism and its preconditions. To get
this, `fpqr`/`qr`/`bnn`/`laplace` (`benchmark` group) now set
`reserve_calib_window=True`, so they carve out the same calibration window
their `conformal`-group siblings do, even though they never calibrate
themselves (keeps their own native-interval numbers an apples-to-apples
comparison against QR/BNN/Laplace, unaffected by calibration).

```bash
# Local smoke test (5 epochs, DSS-EU only, cached backbone params):
uv run python chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py --quick

# Full server run, Experiment 1a (MLVS-PT-3PHASE only, real backbone HPO):
uv run python chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py --dataset MLVS-PT-3PHASE --run-hpo

# benchmark + conformal together, same command (benchmark MUST run first in
# the same invocation so conformal's reuse_checkpoint_from has something to
# load -- --group lists groups in run order, and checkpoints are not
# persisted across separate runs):
uv run python chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py --group benchmark conformal --dataset MLVS-PT-3PHASE --epochs 100 --skip-hpo
```

### Checkpoint reuse (`reuse_checkpoint_from`)

Added 2026-10-01 so a calibration-only condition (same model, same tuned
hyperparameters, only `conformal_config` differs) never needs to retrain —
it loads its source condition's already-trained per-fold weights instead of
calling `fit()`. Mechanism lives in `twiga/experiment/engine.py`
(`_resolve_checkpoint_reuse`, `_load_fold_checkpoint`) and
`twiga/experiment/condition.py` (`Condition.reuse_checkpoint_from`,
`Condition.reserve_calib_window`), not specific to this script — any
`ExperimentSpec` can use it.

**Preconditions, validated at run time (raises a clear `ValueError` if
violated):**

- The source condition must run, in the **same** `ExperimentEngine.run()`
  call, in a group ordered **before** the dependent condition's own group
  (`--group benchmark conformal`, not `--group conformal benchmark` or
  `--group conformal` alone — checkpoints are never persisted across
  separate runs/run_ids, only within one).
- Both conditions must share the identical CV layout (`cv_train_size`/
  `cv_test_size`/`cv_val_size`/`cv_calib_size`/`cv_stride`/`cv_window`) *and*
  the same effective calibration-window reservation — `conformal_config`
  being set or `reserve_calib_window=True` shifts every fold's train/test
  boundary even when `cv_calib_size` itself matches, so the two must agree
  on that too, not just on the raw field value.
- Both must share the same `model_cls` (a checkpoint from one architecture
  cannot load into another).
- **Not validated** (surfaces as a `RuntimeError` from `load_state_dict`
  instead, not a clean error up front): `data_overrides`/anything else that
  changes the model's input dimensionality.

A condition that is itself a `reuse_checkpoint_from` **source** for another
condition automatically keeps its per-fold checkpoints on disk afterward
(instead of the usual `cleanup_training_artifacts` deletion) — no separate
flag needed, computed from the full static condition list every run.

Output under `../experiments/chapter-6-multitarget/multitarget_experiment/`.
**Not synced automatically** — `ch6_analyse_results.py` reads a separate,
local, git-ignored copy under
`chapters/ch7_multitarget/results/chapter-6-multitarget/multitarget_experiment/`,
which only updates when you copy a run there yourself (step 0 above).
**Renamed 2026-09-19**: `ExperimentSpec.name` (the middle path segment,
derived by slugifying it) was shortened from the original
`"Chapter 6, Experiment 1: multi-target probabilistic forecasting benchmark"`
to `"Multitarget Forecasting Benchmark"` — the chapter/experiment numbering
was redundant with this file's own location and this script's own name. Runs
from before that date are still under the old, long slug; only new runs use
the short one.

## `plot_ev_pv_penetration.py`: Experiment 1b, PV/EV penetration sweep (not rerun)

**Objective.** Present the impact of increasing PV and EV penetration on
`DSS-EU` forecasting performance. **Not rerun** — `DSS-EU` is not part of
this session's revised scope, so this script rebuilds the figures directly
from the existing `results_ev_pv_rate.xlsx` (13 (ev-rate, pv-rate)
combinations x 3 targets x 5 folds x 56 daily rows, predating the current
`twiga.experiment` pipeline: two one-axis-at-a-time sweeps sharing a (0, 0)
baseline, plus four coupled diagonal points not plotted here) rather than
through `ExperimentEngine` or `penetration_sweep.py`. Reports NRMSE, NMPI,
PICP, and WS per target (CWE dropped — no per-observation loss differential
to test, per this chapter's own evaluation-criteria convention below).

Requires `openpyxl`, which is not a project dependency (this is the only
script that needs it):

```bash
uv run --with openpyxl python chapters/ch7_multitarget/experiment_1_benchmark/plot_ev_pv_penetration.py
```

Writes 12 panel PDFs (3 targets x 4 metrics) to
`chapters/ch7_multitarget/figures/` and to `Images/Chapter-6/`, where
present.

## `fpqr_gating_ablation.py`: FPQR mechanism + gating ablation (supporting, not a numbered experiment)

**Revised 2026-09-24**: no longer "Experiment 2" — that number now belongs to
`backbone_conformal_study.py` (below). This script's own conclusion (a simple
FPQR configuration: shared grid, softmax proposal, standard quantile-value
network) is a precondition `backbone_conformal_study.py` builds on, not a
numbered experiment in its own right.

**Objective.** Isolate the contribution of each FPQR mechanism component and
the target-level gate, on the multi-target MLPGAM+FPQR backbone. Five
groups, one reference condition each:

| group | reference | tests |
|---|---|---|
| `proposal` | `proposal_softmax` | quantile-fraction allocation mechanism (`BaseFPQRConfig.proposal_type`) |
| `qvn` | `qvn_standard` | quantile-value-network architecture, including the monotonic value network |
| `grid` | `grid_shared` | shared vs. per-target quantile grid |
| `multigate` | `gate_none` | target-level output gate, alone and combined with the existing feature gate |
| `head` | `head_fpqr` | FPQR vs. a joint parametric (multivariate-Normal) head — a design-sanity check, not a hypothesis the chapter argues for |

See `FPQR_GATING_README.md` for each group's pre-registered hypothesis and
`DESIGN_DECISIONS.md` for what the completed run actually found per group,
including where a condition's current DM/MCS verdict conflicts with what
`Chapter-6.tex`'s prose currently states.

```bash
# Local smoke test (one group, DSS-EU, cached backbone params):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --group proposal qvn --quick

# Full server run (every group, both datasets, real backbone HPO):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --dataset all

# proposal only, once its new proposal_horizon condition needs its first
# real run — see DESIGN_DECISIONS.md; qvn/grid/multigate/head already have
# a clean 2026-09-18 run and do not need re-running:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --group proposal --dataset all

# Significance report for one group, once it has results (auto-discovers the
# latest run under results/ that you've manually synced there -- step 0
# above; see DESIGN_DECISIONS.md):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --report grid
```

Output under `../experiments/chapter-6-multitarget/fpqr_gating_ablation/`.
**Not synced automatically** — a local, git-ignored copy under
`chapters/ch7_multitarget/results/chapter-6-multitarget/fpqr_gating_ablation/`
only updates when you copy a run there yourself (step 0 above).
**Renamed 2026-09-19**: `ExperimentSpec.name` shortened from
`"Chapter 6 multi-target FPQR + gating ablation"` to `"FPQR Gating Ablation"`,
same reason as `multitarget_experiment.py` above. `--report`'s run discovery
globs for any slug under this path, so it is unaffected either way.

## `split_scheme_figure.py`: CV split-scheme figure

Renders both datasets' actual fold layout from `fpqr_gating_ablation.py`'s
real `ExperimentSpec` against the real parquet files — the fastest way to
check a CV protocol change produced the fold count intended before spending
compute on a full run.

```bash
uv run python chapters/ch7_multitarget/shared/split_scheme_figure.py
```

No arguments. Writes both datasets' figures to
`chapters/ch7_multitarget/figures/`.

## `copula_variants.py`: copula-based dependence modelling (retired for this chapter, 2026-09-24)

**Retired**: `Chapter-6.tex` drops copula-based dependence modelling entirely
(see Objective above); this script's copula-family conclusion is no longer
part of the chapter's argument. Kept for reference/other work, not part of
the current run list.

**Objective (historical).** Test whether coupling the FPQR margins through a copula
(Gaussian, Clayton, or a three-target D-vine), on top of the shared or
per-target quantile grid, improves joint/operational-tier scores over the
uncoupled FPQR baseline, and whether the copula family matters for the
specific one-sided joint-extreme event Chapter 6's hosting-capacity criterion
scores. See `COPULA_METHODS.md` for the seven hypotheses this tests and
which metric each one is decided on (the hosting-capacity Brier/reliability
tier for the family-choice question, not Winkler score — `DESIGN_DECISIONS.md`
explains why the two can disagree). Eight conditions: shared/per-target grid
x {none, Gaussian, Clayton, D-vine}, run across `DSS-EU`, `DSS-UK`, `DSS-US`
(not `MLVS-PT-3PHASE` — this script's weekly-split backtest protocol is
verified against the OpenDSS records only).

```bash
# Local smoke test (DSS-EU only, few epochs, few sampled paths):
uv run python chapters/ch7_multitarget/copula/copula_variants.py --epochs 5 --paths 50

# Full server run (every scenario — the family-choice verdict should not
# rest on one scenario given the EU-voltage/US-thermal split). --run-hpo
# tunes the shared MLPF backbone first (tune_backbone.py's random search,
# reused in-process, once, not per scenario) rather than reading a possibly
# missing/stale backbone_params.json or silently falling back to a fixed
# default architecture -- omit it to reuse whatever is already cached:
uv run python chapters/ch7_multitarget/copula/copula_variants.py --dataset all --epochs 100 --run-hpo
```

Output under `../experiments/chapter-6-multitarget/copula_variants/<scenario-slug>/runs/<run_id>/results/`
(**moved 2026-09-20** from `chapters/ch7_multitarget/copula_variants/`,
to the same outside-the-repo location every other Chapter 6 script uses — see
`multitarget_experiment.py`'s section below):
`copula_summary.csv` (all three metric tiers), `copula_significance.json`
(the pre-registered Brier-score pair plus the Model Confidence Set), and
`copula_conformal_summary.csv` (max-norm vs. mahalanobis conformal scoring).

## `operational_decision.py`: hosting-capacity / thermal decision (retired for this chapter, 2026-09-24)

**Retired**: superseded by `backbone_conformal_decision.py` (below), now
Experiment 3 — it covers the same `DSS-UK`/`DSS-US` decision criteria with
the current (non-copula) backbone x conformal grid and, unlike this script,
actually applies conformal calibration to the decision region. Kept for
reference/other work, not part of the current run list.

**Objective (historical).** Score every Experiment 1/2 condition against a real
operator decision rather than a generic metric: `DSS-UK`'s hosting-capacity
union event (thermal or voltage, EN 50160) and `DSS-US`'s thermal-only
exceedance (ANSI C84.1). Ten conditions: the two parametric baselines
(joint/independent Gaussian) plus the eight FPQR conditions
`copula_variants.py` covers.

```bash
# Local smoke test (DSS-UK only, few epochs):
uv run python chapters/ch7_multitarget/experiment_3_decision/operational_decision.py --decisions DSS-UK --epochs 5

# Full server run (both decisions, all ten conditions). --run-hpo tunes the
# shared MLPF backbone first, once for both decisions -- see
# copula_variants.py's --run-hpo note above:
uv run python chapters/ch7_multitarget/experiment_3_decision/operational_decision.py --epochs 100 --run-hpo
```

Output under `../experiments/chapter-6-multitarget/operational_decision/runs/<run_id>/results/`
(**moved 2026-09-20**, same reason as `copula_variants.py` above):
`<decision>_{per_fold,summary,significance,value_curve}.csv` and
`operational_decision_config.json`. `plot_operational_decision.py` (below)
reads the latest run there automatically.

```bash
# Decision-value figures, once both decisions have real results:
uv run python chapters/ch7_multitarget/experiment_3_decision/plot_operational_decision.py
```

## `backbone_conformal_study.py`: Experiment 2, multi-target FPQR + conformal evaluation

**Objective.** Two questions left open once `fpqr_gating_ablation.py`
(mechanism) is taken as given: does a
different backbone architecture (MLPF, MLPGAM, NHiTS) carrying that same
simple FPQR configuration change the marginal picture, and does conformal
calibration — joint/multivariate (`max_norm`/`mahalanobis`,
`MultivariateConformal`) or CQR (`quantile`/`quantile_joint`,
`ConformalQuantileRegressor`) — change the calibrated interval picture for
any of them? `quantile`/`quantile_joint` are the same CQR score
(`score_type="scaled"`), differing only in `target_coverage` (added
2026-09-21): `quantile` calibrates each target independently (marginal
coverage, no cross-target coupling); `quantile_joint` pools the score across
targets before calibrating (`target_coverage=True`), the same construction
`max_norm` uses, giving CQR its own joint/simultaneous-coverage variant to
compare against the two `MultivariateConformal` methods on equal footing.
One group, `backbone`: 4 heads (MLPF/MLPGAM/NHiTS FPQR plus QR's fixed-grid
baseline, added 2026-09-30/10-01 — see "Recent fixes" above) x 13 conformal
conditions (`none`, `max_norm`, `max_abs`, four `mahalanobis_*` correlation
estimators, six CQR `quantile_*` variants), on all four datasets (`DSS-EU`'s
`cv_calib_size=0` skips the conformal-bearing conditions there, same reason
`multitarget_experiment.py`'s own `conformal` group is DSS-EU-exempt). QR
reuses a dedicated backbone-only `MLPFConfig` HPO search
(`QR_HPO_VARIANT = "shared_mlp_backbone"`) instead of its own full joint
study, since it has no FPQR-specific hyperparameters to tune (this script has
no per-head CLI restriction flag — every condition in `CONDITIONS` always
runs; use `--variants` to restrict the conformal axis instead, or edit
`HEADS` directly for a one-off head-only run). Runs through `ExperimentEngine`, like
`multitarget_experiment.py`/`fpqr_gating_ablation.py` — `max_norm`/
`mahalanobis` reaching a quantile-only (FPQR) backbone at all, and DSS-UK/
DSS-US reaching the engine at all, both needed a library-level fix (see
`twiga.distributions.ml.utils.get_sigma_from_quantile_levels` and
`ExperimentSpec.cv_split_freq`, both added 2026-09-21) — this script's own
module docstring has the full detail.

**Three further fixes, 2026-09-24**: (1) `get_sigma_from_quantile_levels`
only handled `quantile_levels` shapes with no batch axis; a real FPQR
forecast reports `(B, N_q, T)`, which crashed with `IndexError` — this is
why `max_norm`/`mahalanobis` had never produced a single row in any run
before this date. (2) `corr` was never passed to
`MultivariateConformal.calibrate()` for `method="multivariate"`, so
`mahalanobis` raised `ValueError` the instant it was calibrated; fixed by
computing an empirical per-horizon correlation from the calibration set's
own normalised residuals (no copula fit required). (3) All three backbones'
`per_target_tau` overrides were flipped to `True`: a shared quantile grid
across P/Q/V let one target's predicted spread collapse under the pooled
pinball loss, which then poisoned every pooled/simultaneous conformal
condition (`quantile_joint`, and `max_norm`/`mahalanobis` once (1)/(2) let
them run) via that target's near-zero denominator dominating the shared
threshold — confirmed both visually (the `*_quantile_joint`/`*_q_kvar`
sample plots) and quantitatively (NMPI/Winkler inflating 5-60x for the
non-degenerate targets under `quantile_joint`, across all three backbones).
Re-verify with `--quick` and the sample plots before trusting any full run's
`max_norm`/`mahalanobis`/`quantile_joint` numbers.

**HPO flag unified, 2026-09-24**: this script previously ran HPO by default
(`--skip-hpo` to disable), the opposite of `multitarget_experiment.py`'s/
`backbone_conformal_decision.py`'s `--run-hpo`-to-enable convention — a real
footgun (`--run-hpo` on this script used to be an unrecognised-argument
error). All three now share the same interface: `--skip-hpo` (default) /
`--run-hpo`, `--hpo-trials` (default 50), `--hpo-epochs` (default 100, epochs
per Optuna trial, distinct from `--epochs`, the final conditions' budget).

```bash
# Local smoke test (5 epochs, 1 trial, MLVS-PT-3PHASE only):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --quick

# Full server run (every dataset, real backbone HPO, all 4 heads):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --dataset all --run-hpo --hpo-trials 100 --hpo-epochs 50

# Significance report for one dataset's run (Winkler score DM/MCS; reads the
# manually-synced local copy -- step 0 above):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --report backbone
```

Output under `../experiments/chapter-6-multitarget/backbone_conformal_study/`.
**Not synced automatically** — a local, git-ignored copy under
`chapters/ch7_multitarget/results/chapter-6-multitarget/backbone_conformal_study/`
only updates when you copy a run there yourself (step 0 above).

**Known local-sandbox hang (2026-09-21), not a bug in this script**: dataset
loading (`twiga.utils.data_loading`, via `twiga.core.data.selection.compute_rf_importance`)
runs a `RandomForestRegressor(n_jobs=-1)` feature-selection step, which
spawns a `joblib`/`loky` worker pool. Under one sandboxed local shell this
pool hung indefinitely (near-zero CPU for 25+ minutes, `loky-*` semaphores
visible in `lsof`) rather than erroring — this affects any script that calls
`load_dataset()`, not only this one, and is expected to run normally on an
unsandboxed machine/server.

## `backbone_conformal_decision.py`: Experiment 3, operational decision-making assessment

**Objective.** `operational_decision.py`'s own module docstring names this
gap directly: every condition there reports its native, uncalibrated
predictive distribution, so the hosting-capacity/thermal Brier score has
never been measured against a *calibrated* region. This script closes that
gap for the same backbone x conformal grid `backbone_conformal_study.py`
evaluates on the marginal metrics, restricted to `DSS-UK`/`DSS-US` — the two
datasets with a verified decision threshold (`MLVS-PT-3PHASE` has none, so it
stays marginal-only, per `backbone_conformal_study.py`'s own scope).
`none`/`max_norm`/`mahalanobis`/`quantile`/`quantile_joint` each turn into a
differently shaped scoreable ensemble (`sample_calibrated_ensemble` for the
two joint methods, `sample_cqr_ensemble` — with `target_coverage=True` for
`quantile_joint` — for both CQR variants; see the module docstring for why
each needs its own mechanism) fed into `forecast_examples.hosting_capacity_rows`
unchanged. A secondary diagnostic, ported from the author's own prior
conformal work (`twiga.distributions.conformal.score.calibrate_score_threshold`/
`exceeds_threshold`), reports a validated-decision flag alongside the primary
Brier-score table. Hand-rolled, not `ExperimentEngine`-based — turning a
calibrated region into a scoreable ensemble is a per-fold sampling step the
engine's `Condition` loop has no hook for, the same reason `copula_variants.py`
stays hand-rolled.

**HPO, via the engine's own Phase 1, not a separate step**: earlier versions
of this paragraph described a `tune_backbone.py`/`--hpo-scenario`-based
tuning step; that mechanism is gone (an `--hpo-scenario` flag no longer
exists on this script — an old command using it fails with "unrecognized
arguments"). Since the `ExperimentEngine` migration, HPO runs the same way
as `multitarget_experiment.py`/`backbone_conformal_study.py`: `--skip-hpo`
(default, reuse a cached `*_best_params.json` if present) / `--run-hpo`
(tune for real), `--hpo-trials` (default 50), `--hpo-epochs` (default 100,
per Optuna trial, distinct from `--epochs`). Each head gets its own cache
key, per `_build_conditions`'s own per-head HPO wiring (see "Recent fixes"
above for QR's dedicated one) — no shared-backbone-file/"last scenario
tuned wins" ambiguity the old mechanism had.

```bash
# Local smoke test (5 epochs, 20 paths, one decision, one head):
uv run python chapters/ch7_multitarget/experiment_3_decision/backbone_conformal_decision.py --quick

# Full server run (both decisions, every head x conformal cell, real HPO).
# --test-size/--step default to 8 (the corrected Experiment 3 protocol: 4
# non-overlapping folds, fewer/larger test windows than Experiment 2's own
# protocol, deliberately, for more hosting-capacity/thermal exceedance events
# per fold) -- there is no --train-size/--split-freq to pass any more, the CV
# window is fixed internally in build_spec():
uv run python chapters/ch7_multitarget/experiment_3_decision/backbone_conformal_decision.py --decisions DSS-UK DSS-US --epochs 150 --run-hpo --hpo-trials 100 --hpo-epochs 50

# Restrict to specific heads (e.g. just QR, reusing its own cached backbone-only search):
uv run python chapters/ch7_multitarget/experiment_3_decision/backbone_conformal_decision.py --decisions DSS-UK DSS-US --backbones qr --run-hpo
```

Output under `../experiments/chapter-6-multitarget/backbone_conformal_decision/runs/<run_id>/results/`:
`<decision>_{per_fold,value_curve,summary,significance,conformal,calibrated_flag}.csv`
and `backbone_conformal_decision_config.json`.

## `multitarget_decision.py`: Experiment 3 companion, probabilistic head x decision (new, 2026-10-02)

**Objective.** `backbone_conformal_decision.py` above varies *backbone
architecture* (MLPF/MLPGAM/NHiTS), holding the head fixed at FPQR (plus a
separate QR entry). This script holds the *backbone* fixed at MLPF instead
and varies the *head* — `fpqr_anchor_outer`, `qr`, `bnn`, `laplace`, the same
four `multitarget_experiment.py`'s own `benchmark`/`conformal` groups
compare — asking whether the chapter's head-choice conclusion from
Experiment 1/2 (marginal metrics) also holds for the hosting-capacity/
thermal-exceedance decision. A new file, not an extension of
`backbone_conformal_decision.py`: the two scripts answer different questions
that happen to share machinery (ensemble sampling, interval metrics, DM
significance, the hosting-capacity row builder — all imported directly from
`backbone_conformal_decision.py`, not duplicated), so `backbone_conformal_decision.py`
itself is unmodified by this script's existence (only the shared conformal-set
trim below touched it, independently).

**Conformal set — the chapter's agreed methodology, not
`backbone_conformal_study.py`'s own 13-variant grid.** `max_norm` for every
head, plus the full six-variant CQR grid (`quantile_{scaled,unscaled}_
{marginal,pooled,bonferroni}`) for the two quantile-producing heads only
(`fpqr_anchor_outer`/`qr`) — `bnn`/`laplace` have no quantile grid to
calibrate a CQR score around, so they only ever get `none`/`max_norm`,
gated by each head's own `is_quantile_head` flag exactly as
`multitarget_experiment.py`'s `_build_conformal_conditions` does. `max_abs`
and the four `mahalanobis_*` correlation estimators are not offered at all
(dropped from `backbone_conformal_study.py`'s own `CONFORMAL_VARIANTS` too,
2026-10-02, matching `Chapter-6.tex`'s `sec:ch6-mahalanobis`: a fixed,
calibration-set-estimated correlation gave a 17-37% worse Winkler score than
max-norm across every head, so it is not carried forward as a reported
conformal variant anywhere in this chapter). 20 conditions per dataset: 8
each for `fpqr_anchor_outer`/`qr`, 2 each for `bnn`/`laplace`.

**Checkpoint reuse, not retraining per conformal variant.** Each head's own
`{head}_none` condition is the `reserve_calib_window=True` trained-weights
source; every other conformal variant of that head sets
`reuse_checkpoint_from="{head}_none"` — one fit per head per fold, not one
per head-x-conformal-method cell (the same fix was applied to
`backbone_conformal_decision.py`'s own `_build_conditions` at the same time,
2026-10-02 — neither script reused checkpoints across its own conformal
variants before this).

**BNN/Laplace have no quantile grid**, so `build_condition_scorer` here adds
two sampling branches `backbone_conformal_decision.py`'s own scorer never
needed: `"none"` draws directly from the head's native predictive
distribution (`ForecastResult.scale` reparameterised as Normal noise for
BNN, Laplace noise for `laplace` — both location-scale families, so
`loc + noise * scale` is an exact draw either way) rather than by
inverse-transform sampling on a quantile grid that doesn't exist; the
diagnostic `interval_lower`/`interval_upper` reported alongside use the
correct per-distribution quantile multiplier for the project's standard
`alpha=0.1` (90%) nominal interval (`norm.ppf(0.95)≈1.645` for BNN,
`-ln(alpha)≈2.303` for Laplace, exact from the Laplace inverse CDF) — not a
raw `±1` scale band, which would under-cover relative to every other
condition's own reported interval. `"max_norm"` for these heads uses
`result.scale` directly as the conformal `sigma` (BNN/Laplace already have a
native scale parameter, no `get_sigma_from_quantile_levels` derivation
needed, unlike FPQR/QR).

`--hpo-objective` only applies to `fpqr_anchor_outer`/`qr` (the two
quantile-producing heads) — `bnn`/`laplace` stay `winkler_score`.

**Three fixes, 2026-10-03/04 — delete any run from before these dates and
rerun; `qr` and two of `fpqr_anchor_outer`'s six CQR variants did not produce
a single row before them:**

1. **`qr` crashed on every condition, both datasets** (`IndexError: too many
   indices for array: array is 1-dimensional, but 2 were indexed`, in
   `backbone_conformal_decision.sample_quantile_ensemble`): QR's own
   `RawPrediction.quantile_levels` is a flat `(L,)` grid shared across every
   target, with no target axis at all (`twiga/distributions/nn/quantile.py`'s
   `self.taus`), unlike FPQR's always-at-least-2D convention —
   `sample_quantile_ensemble` assumed a target axis always exists. Fixed by
   broadcasting a genuinely 1-D `quantile_levels` to `(L, D)` before the
   existing per-target loop, same as `get_sigma_from_quantile_levels` already
   did correctly.
2. **`fpqr_anchor_outer_quantile_{scaled,unscaled}_bonferroni` crashed on
   both datasets** (`ArrowInvalid: Could not convert array(nan)...`, a
   parquet-write failure downstream of
   `twiga.core.metrics.interval.get_interval_metrics`): a Bonferroni-corrected
   CQR threshold legitimately diverges to `q_hat=inf` under this script's
   small per-fold calibration set, which makes `cwe_score` raise
   `ValueError` (correctly) — but the `except` branch's own NaN fallback
   skipped the same scalar-reduction (`.item()`) the success path applies,
   so the NaN landed in the result as an actual `0`-d `ndarray` cell, not a
   float, which pyarrow then refused to write. Fixed in
   `get_interval_metrics` itself (shared by every script that calls it, not
   `multitarget_decision.py`-specific).
3. **`--quick` crashed immediately for any `--heads` choice other than the
   default** (`ValueError: reference_conditions['decision']='laplace_max_norm'
   does not identify a condition in that group`): `--quick` filters
   `spec.conditions` down to one `{head}_none` condition but left
   `spec.reference_conditions` pointed at `"laplace_max_norm"`
   unconditionally. Fixed by pointing it at whichever single condition
   `--quick` actually kept.

**Training-level config aligned with `multitarget_experiment.py`, 2026-10-03**:
`fpqr_anchor_outer`/`qr`/`bnn`/`laplace` here all pin
`_REFERENCE_TRAINING_PINS` (optimizer/scheduler/batch-size/gradient-clip,
imported directly from `multitarget_experiment.py`, not duplicated) the same
way that script's own benchmark-group conditions do — see that script's own
section above for the values and history. `fpqr_anchor_outer`'s own
`per_target_tau`/`proposal_type`/`proposal_anchor_outer` pins are unaffected
by this and unchanged from this script's own original design.

```bash
# Local smoke test (every head):
uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision.py --quick

# Local smoke test, one head only (this exact combination crashed before fix 3 above):
uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision.py --quick --heads qr

# Full server run (both decisions, every head x conformal cell, real HPO):
uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision.py --decisions DSS-UK DSS-US --epochs 150 --run-hpo --hpo-objective cwe_stratified --hpo-trials 100 --hpo-epochs 50

# Restrict to specific heads (e.g. just BNN/Laplace, the two cheaper point-accuracy-objective ones):
uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision.py --decisions DSS-UK DSS-US --heads bnn laplace --run-hpo
```

Output under `../experiments/chapter-6-multitarget/multitarget_decision/runs/<run_id>/results/`:
`<decision>_{per_fold,value_curve,summary,significance}.csv` and
`multitarget_decision_config.json` — same shape as
`backbone_conformal_decision.py`'s own output, read by the same
`ch6_analyse_results.py` entry point (`--experiment multitarget_decision`).

## `multitarget_decision_single_split.py`: Experiment 3, one fixed event-balanced split, two stages (new, 2026-10-05)

**Objective.** `multitarget_decision.py`'s own four-fold expanding-window
protocol was confirmed to produce a systematic, monotone PICP collapse
across folds for every head and every condition alike (fold 4 PICP
0.02-0.28 against fold 1's 0.20-0.51, on both `DSS-UK` and `DSS-US`) — the
signature of distribution shift between each fold's own trailing
calibration window and its test window, not a per-head or per-method
failure. This script sidesteps that mechanism with a single, deliberately
chosen split per dataset (`cv_folds=1`, no fold marches forward in calendar
time), with `val`/`calib`/`test` sizes chosen by scanning real per-week
event counts so every portion's own event rate stays close to the dataset's
overall ~2% rate (confirmed directly against the record, not assumed from
calendar proportions): `DSS-UK` val=4w/calib=12w/test=16w, `DSS-US`
val=4w/calib=8w/test=12w.

**Two stages, one spec, two condition groups**, sharing the identical
`val`/`test` windows and differing in exactly one respect:

- `native_only` group: one `{head}_native` condition per head,
  `reserve_calib_window=False` (this script's own default) and
  `conformal_config=None` — `ExperimentEngine`'s own `use_calib` gate then
  forces `calib_size=0` for these regardless of the spec's own
  `cv_calib_size`, so they train on the *combined* train+calib span.
  Answers "how good is this head's own native predictive distribution,
  before paying any data for calibration".
- `decision` group: the same native (`{head}_none`,
  `reserve_calib_window=True`, calibration window carved from train) plus
  the full conformal grid, as `multitarget_decision.py`'s own
  `_build_conditions`. Answers "does calibration add value, at the cost of
  the training data it consumes".

**Significance testing is not computed here** — a single split gives one
value per condition, and `family_significance` needs several folds for a
fold-level Diebold-Mariano test. A defensible test under a single split
needs per-timestamp paired losses within the one test window instead (more
observations than four folds would give, in principle, but a different code
path) — left as a follow-up.

```bash
# Local smoke test, one head only:
uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision_single_split.py --decisions DSS-UK --heads qr --quick

# Full server run (both decisions, every head x conformal cell, real HPO):
uv run python chapters/ch7_multitarget/experiment_3_decision/multitarget_decision_single_split.py --decisions DSS-UK DSS-US --epochs 150 --run-hpo --hpo-objective cwe_stratified --hpo-trials 100 --hpo-epochs 50
```

Output under `../experiments/chapter-6-multitarget/multitarget_decision_single_split/runs/<run_id>/results/`:
`<decision>_{per_fold,value_curve,summary}.csv` and
`multitarget_decision_single_split_config.json` — same per_fold/value_curve
shape as `multitarget_decision.py`'s own output (no `_significance.csv`,
for the reason above), with an extra `group` column (`native_only` or
`decision`) distinguishing the two stages.

## `forecast_examples.py`: predictive distributions and the full metrics table

**Objective.** Produce the predictive-distribution figures and the complete
marginal/joint/operational metrics table (energy score, variogram score,
hosting-capacity Brier score) across every FPQR/copula/head variant, plus the
Winkler-score DM/MCS significance table `Chapter-6.tex`'s evaluation-metrics
section specifies for deciding a comparison (never CWE — diagnostic only).

**Note (2026-09-19)**: the panel-drawing primitives this script's predictive-
distribution figures use (`draw`, `_format_axes`, `place_legend_below`,
`_fmt2` — hollow-marker observed series, dashed predictive median, translucent
quantile fan) were extracted, unchanged, into `experiment/common/forecast_panels.py`
so other chapters reuse this exact style instead of a second copy. This file
now imports them from there rather than defining them locally; behaviour is
identical. `chapter-7-conformal/mlpgam_forecast_examples.py` is the first
outside consumer.

```bash
# Local smoke test (few paths, one backbone):
uv run python chapters/ch7_multitarget/shared/forecast_examples.py --backbone mlpf --paths 50 --variants all

# Full server run (both backbones; --paths 500+ needed for the joint and
# operational tiers to be evaluated at all, not just fitted):
uv run python chapters/ch7_multitarget/shared/forecast_examples.py --backbone mlpf --paths 500 --variants all
uv run python chapters/ch7_multitarget/shared/forecast_examples.py --backbone mlpgam --paths 500 --variants all

# Redraw from cached predictions instead of retraining (styling changes only):
uv run python chapters/ch7_multitarget/shared/forecast_examples.py --replot --variants all
```

Output under `chapters/ch7_multitarget/forecast_examples/`:
per-panel PDFs, `metrics_all_tiers.csv`, `winkler_dm_table.csv`,
`conformal_calibration.csv`, `pointwise_losses.csv`.

## `head_conformal_check.py`: exploratory — every backbone x head x conformal variant

**Not a chapter experiment.** `multitarget_experiment.py` (Experiment 1a,
MLPF only, FPQR/QR/BNN/Laplace) and `backbone_conformal_study.py`
(Experiment 2, MLPF/MLPGAM/NHiTS, FPQR only) stay exactly as they are and
remain the two scripts whose numbers go in the chapter. This script exists
purely to sanity-check a proposed extension — every backbone (MLPF, MLPGAM,
NHiTS) paired with every head (FPQR, QR, BNN, Laplace), under every
conformal variant (`none`/`max_norm`/`mahalanobis`/`quantile`/
`quantile_joint`), on any of the four datasets — before deciding whether any
of it is worth folding into either formal script. Run it, look at the
numbers and panels, decide; nothing it produces is claimed as a chapter
result on its own.

Reuses `backbone_conformal_study.py`'s own verified `ExperimentSpec` objects
directly (via `dataclasses.replace`, swapping only `conditions`) rather than
re-deriving CV numbers by hand — an earlier version of this script got
DSS-EU's fold count wrong doing that and crashed on a training window too
small to fit a single lookback+horizon; cloning the already-correct spec
avoids that class of mistake.

**Panels reuse `forecast_examples.py`'s own drawing utilities directly** —
`experiment/common/forecast_panels.py`'s `draw`/`place_legend_below`, the
same hollow-marker observed series / dashed median / translucent interval
convention used throughout the thesis — so this script's output reads in
the same visual style, not a different ad hoc one. Unlike
`forecast_examples.py` (a full quantile fan from several nested levels),
this draws a single interval band per condition from its own saved
`lower`/`forecast`/`upper` columns, exactly the "single prediction
interval" case `forecast_panels.draw`'s own docstring describes. One
best-day and one worst-day panel is saved per condition x target, picked by
a simple RMSE-normalised-by-mean-|Actual| proxy (adequate for choosing which
day to look at; the CSV summary is the actual metrics table).

```bash
# Quickest possible check (1 dataset, 1 backbone, 1 head, 5 epochs):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/head_conformal_check.py --quick

# One dataset, chosen backbones/heads, real epochs:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/head_conformal_check.py \
    --dataset MLVS-PT-3PHASE --backbones nhits mlpgam --epochs 30

# Every dataset, every backbone (default heads = all four), full grid:
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/head_conformal_check.py --dataset all --epochs 30

# Skip the panel plots (metrics CSV only, faster):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/head_conformal_check.py --dataset all --no-panels
```

Output under `../experiments/chapter-6-multitarget/exploratory/head_conformal_check/head_x_conformal_check/runs/<run_id>/`
-- the same external base the formal chapter scripts share (each gets its
own subfolder there; `exploratory` is this script's), so nothing this
produces (checkpoints, predictions, plots, figures) ever accumulates inside
the repo:
`results/head_conformal_check_{full,summary,representative}.csv`,
`plots/<dataset>/panels/<condition>_<target>_{best,worst}.pdf`, plus the
engine's own default 7-day sample plots under `plots/<dataset>/`. Also
writes `../experiments/chapter-6-multitarget/exploratory/results_head_conformal_check.csv`
(the across-dataset summary this script's own `main()` concatenates).

## `penetration_sweep.py`: does dependence-modelling advantage widen with penetration

**Objective.** Test whether the joint-vs-independent gap widens as EV/PV
penetration (and so P/Q/V coupling strength) increases — the mechanism axis
`EXPERIMENT_PLAN.md` §3 argues for. Uses `split_freq="weeks"` (not
`Chapter-6.tex`'s literal months-based protocol, which leaves too few folds
on these ~11-12 month synthetic records for the trend regression).

```bash
# Local smoke test (fewer epochs, default record):
uv run python chapters/ch7_multitarget/copula/penetration_sweep.py --epochs 10

# Full server run (all four penetration levels, 8-10 folds each). --run-hpo
# tunes the shared MLPF backbone first, once for the whole sweep -- see
# copula_variants.py's --run-hpo note above:
uv run python chapters/ch7_multitarget/copula/penetration_sweep.py --epochs 60 --run-hpo
```

Output under `../experiments/chapter-6-multitarget/sweep/<record-slug>/runs/<run_id>/results/`
(**moved 2026-09-20**, same reason as `copula_variants.py` above):
`sweep_per_fold.csv` and `sweep_trend.json` (the coupling-vs-advantage
regression).

## Supporting diagnostics

Back a specific claim in `Chapter-6.tex`'s prose rather than implementing one
of the three named experiments. Each takes no required arguments.

- **`dependence_cost.py`** — cost of assuming target independence, measured
  directly from the power-flow records (`Chapter-6.tex` §6.3).
- **`per_target_tau_diagnostic.py`** — whether FPQR's learned quantile grid
  tracks the `|Q''(tau)|^(-1/3)` theory-optimal shape per target; see
  `EXPERIMENT_PLAN.md` §10.4-10.8 for the full result and why it answers a
  different question than `fpqr_gating_ablation.py`'s `grid` group.
- **`joint_vs_independent.py`** — the dependence-cloud figure and energy
  score comparing joint and independent heads.
- **`residual_dependence.py`** — how much cross-target dependence survives
  after conditioning on the shared backbone representation.
- **`multitarget_bakeoff.py`** — a quick, single-scenario scoping run across
  head variants, to decide which comparisons are worth a full backtest.
- **`characterise_records.py`** — measures each record's own lag/AR-order
  structure rather than assuming the 96-step/48-step window throughout.

`build_dss_eu.py`, `export_us_record.py`, and `check_chapter.py` are one-off
data-preparation/validation utilities, not experiments in their own right.

## Evaluation criteria

Higher is better for CWE, stratified CWE, and PICP; lower is better for
CRPS, the Winkler score, and the Brier score. **Comparisons between
conditions are decided on the Winkler score** (marginal/design questions) or
the **hosting-capacity Brier score** (the copula-family and operational
questions — see `COPULA_METHODS.md`), via Diebold-Mariano and the Model
Confidence Set, **never on CWE** — it has no per-observation loss
differential to test, and stays in every table as a diagnostic alongside
PICP and nMPI.

## Shared modules

- `twiga.core.metrics (pairwise_dm_table, mcs_ranking)` — `pairwise_dm_table`,
  `mcs_ranking`, reused unchanged by every script above that reports a
  significance table.
- `twiga.experiment.ExperimentEngine`/`Condition`/`ExperimentSpec` —
  condition/backbone/CV wiring for `multitarget_experiment.py`,
  `fpqr_gating_ablation.py`, and `backbone_conformal_study.py`.
- `forecast_examples.py`'s `build_backtest_folds`/`metrics_table`/
  `hosting_capacity_rows` — reused by `copula_variants.py`,
  `operational_decision.py`, `penetration_sweep.py`, and
  `backbone_conformal_decision.py`.
- `twiga.distributions.conformal` — `MultivariateConformal` (joint,
  `max-norm`/`mahalanobis`), `ConformalQuantileRegressor` (CQR; per-target by
  default, or joint/simultaneous via `target_coverage=True`, added 2026-09-21 —
  also reachable through `ConformalConfig(method="quantile", target_coverage=True)`
  for engine-driven conditions, see `backbone_conformal_study.py`'s
  `quantile_joint`), and `score.calibrate_score_threshold`/`exceeds_threshold`
  (a method-agnostic validated-threshold diagnostic, ported from the
  author's own prior conformal work), all reused by
  `backbone_conformal_study.py`/`backbone_conformal_decision.py`.
- `experiment/common/forecast_panels.py` — `draw`/`_format_axes`/
  `place_legend_below`/`_fmt2`, this chapter's own predictive-panel style,
  extracted here and consumed by `forecast_examples.py` itself plus
  `chapter-7-conformal/mlpgam_forecast_examples.py`. Not chapter-specific.
