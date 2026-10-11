# Chapter 7 experiments: how to run

## Objective

Chapter 7 extends probabilistic forecasting from a single target to the
coupled active power, reactive power, and voltage triple at an LV secondary
substation, and asks whether modelling that coupling changes an operational
decision (hosting capacity, thermal exceedance) rather than only a generic
score. The chapter presents multi-target FPQR plus multi-target conformal
calibration (residual-based max-norm, and CQR's own marginal/pooled
variants). Copula-based dependence modelling is not part of the chapter's
argument; `copula/` keeps those scripts for other, still-copula-based work to
reuse, outside the current run list. The full argument for why the
constraint framing matters, not only the score, is in `POSITIONING.md`; the
data-generation protocol is in `EXPERIMENT_PLAN.md`. Results, findings, and
statistical detail belong in `Chapter-7.tex` and `DESIGN_DECISIONS.md`, not
here; this file is a run guide only.

The chapter's protocol maps onto four scripts:

| experiment | script | scope |
|---|---|---|
| 1a, probabilistic forecasting benchmark, fixed heads | `experiment_1_benchmark/fixed_benchmark.py` | `MLVS-PT-3PHASE`, `DSS-EU`, `DSS-UK`, `DSS-US` |
| 1b, PV/EV penetration sweep | `experiment_1_benchmark/plot_ev_pv_penetration.py` | `DSS-EU` only, rebuilt from `results_ev_pv_rate.xlsx` |
| 2, backbone architecture + conformal evaluation | `experiment_2_backbone_conformal/backbone_conformal_study.py` | `MLVS-PT-3PHASE`, `DSS-UK`, `DSS-US` (+ `DSS-EU`, `none` only) |
| 3, operational decision-making assessment | `experiment_3_decision/fixed_decision_sweep.py` | `DSS-UK`, `DSS-US` |

All engine-based scripts (`fixed_benchmark.py`, `multitarget_experiment.py`,
`fpqr_gating_ablation.py`, `backbone_conformal_study.py`) import
condition/backbone/CV wiring from `twiga.experiment.ExperimentEngine`, the
same machinery Chapter 6's `mlpgam_design_ablation.py` uses. `forecast_examples.py`
provides the shared backtest/metrics machinery (`build_backtest_folds`,
`fit_and_predict`, `hosting_capacity_rows`) that the hand-rolled scripts in
`copula/` and the retired `experiment_3_decision/operational_decision.py`
reuse instead.

`--quick`/small-epoch smoke tests exist for every script and should be run
first, per script, before committing real server time.

## Experiment 1: benchmark (`fixed_benchmark.py`)

Compares FPQR against QR, BNN and Laplace at fixed hyperparameters, so the
head ranking does not depend on a cached HPO result (`ExperimentSpec.hpo_enabled=False`):
every head sits on one fixed MLPF backbone and FPQR keeps a hand-set
configuration rather than a tuned one.

**`benchmark` group** (every dataset):

| condition | description |
|---|---|
| `fpqr` | FPQR, shared quantile grid, learned (floating) outer levels |
| `fpqr_anchor` | the same, with the outer levels pinned to `conf_level/2` and `1 - conf_level/2` (`proposal_anchor_outer=True`); without the pin, `conf_level` is metadata only for FPQR, and its native interval is whatever span the outermost learned pair settles on, not 90% |
| `fpqr_ch5` | Chapter 5's MLPQF benchmark head exactly as trained there, including its own MLPF backbone (N=90, TauRoPE, standard QVN, `entropy_weight=1`, learned outer levels); unlike the other heads it does not share `FIXED_BACKBONE`, since without that backbone it would not be Chapter 5's model |
| `qr` | fixed-grid quantile regression on its default `0.1..0.9` grid plus the `alpha/2` and `1 - alpha/2` outer levels (11 levels total), so its native interval is 90% |
| `bnn` | Bayesian neural network |
| `laplace` | parametric Laplace |

**`conformal` group** (`MLVS-PT-3PHASE` only, the only record with a
calibration window): max-norm multivariate conformal on every head, plus the
six CQR variants (scaled/unscaled x marginal/pooled/bonferroni) on the three
quantile heads. Each reuses its head's trained weights. Mahalanobis is not
offered: a fixed, calibration-set-estimated correlation loses to max-norm on
every axis for every head (`sec:ch7-mahalanobis`), so it is not carried
forward as a reported conformal variant anywhere in this chapter.

**`head_tuned` / `head_tuned_conformal` groups**: tune FPQR, QR and BNN
head-only (backbone pinned, scale-free CRPS objective, 6 validation windows
with Wilcoxon pruning), then calibrate the tuned FPQR on `MLVS-PT-3PHASE`.
Only the tuned FPQR is reported in the chapter; tuned QR and BNN are run as a
check and stay fixed. A `head_tuned` run keeps no checkpoints, so
`head_tuned_conformal` refits the tuned FPQR from the cached settings in
`backbone_hpo/` (no new search) and runs max-norm plus the six CQR variants
on the refit's weights. Run this group on the same machine as `head_tuned`:
the refit is not bit-identical across hardware. Pass `--tuned-hpo-root` if
the tuned settings live elsewhere.

**Protocol**: `DSS-EU`, `DSS-UK` (weak feeder) and `DSS-US` (rank-8
transformer), all at 0.4 EV / 0.4 PV, about eleven months of power-flow data
each, so every fold fits on 4 months and validates on the next 1, then tests
on 1, rolling, no calibration window and hence no conformal group. The
training window is held fixed (`cv_train_size_mode="fixed"`) and `cv_folds`
only caps the count, giving 5 folds on `DSS-EU` and 6 on `DSS-UK`/`DSS-US`
(their records hold one more month). `MLVS-PT-3PHASE` uses the 12-month
expanding train, 2-month validation, 3-month test, 3-month calibration,
10-fold protocol, with the `benchmark` and `conformal` groups.

```bash
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --quick
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py

# Head-only tuning (only the tuned FPQR is reported; QR/BNN stay as a check):
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --group head_tuned --hpo-windows 6

# Calibrate the tuned FPQR on MLVS-PT-3PHASE:
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py --dataset MLVS-PT-3PHASE --group head_tuned_conformal
```

**Combining the tuned FPQR with the fixed heads.** `combine_tuned_benchmark.py`
copies result parquet files only; the source runs are not modified:

```bash
E=<experiments>/ch7_multitarget
uv run python chapters/ch7_multitarget/experiment_1_benchmark/combine_tuned_benchmark.py \
  --fixed $E/fixed_benchmark/multitarget_fixed_benchmark \
  --tuned $E/fixed_benchmark/multitarget_fixed_benchmark \
  --refit $E/fixed_benchmark/multitarget_fixed_benchmark \
  --out $E/multitarget_benchmark_final
```

`--fixed`, `--tuned` and `--refit` may be different roots (e.g. a copied
`multitarget_fixed_benchmark_v1` holding the head-tuned runs).

**Report, chapter figures, forecast examples.** `--copy-to-images` writes the
PDFs into `Papers/PhD thesis -Final-Revision/Images/Chapter-7/`. The forecast
examples draw every head and conformal variant on FPQR's best day per record,
one figure per (record, condition, target):

```bash
F=$E/fixed_benchmark/multitarget_fixed_benchmark
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_report.py --experiment-root $F
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_chapter_figures.py --experiment-root $F --copy-to-images
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_forecast_examples.py --experiment-root $F --copy-to-images
```

The same three scripts read the combined tuned set
(`--experiment-root $E/multitarget_benchmark_final`) when that comparison is
needed. To compare tuned and fixed heads in one test family instead (all
seven heads), read the tuned runs from their own root:

```bash
uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_report.py \
  --experiment-root $E/fixed_benchmark/multitarget_fixed_benchmark \
  --tuned-root <root holding the head_tuned runs> --include-tuned --output <analysis folder>
```

### `multitarget_experiment.py`: supporting mechanism checks (`grid` group)

`fixed_benchmark.py` covers the chapter's own head comparison; `multitarget_experiment.py`'s
`grid` group covers three supporting mechanism checks on the same protocol:

- **`fpqr_shared_grid` / `fpqr_per_target_grid`**: shared vs. per-target
  quantile grid across P/Q/V.
- **`fpqr_anchor_outer`**: pins the proposal's two outermost quantile levels
  to `conf_level/2` and `1 - conf_level/2` (`BaseFPQRConfig.proposal_anchor_outer`),
  the guarantee a fixed-grid model like QR gets for free. Implemented only for
  `proposal_type="softmax"`.
- **`fpqr_lv_differenced`**: forecasts voltage (LV) as its own first
  difference rather than its raw level, reconstructed back to level scale at
  evaluation (`DataPipelineConfig.target_differencing`,
  `twiga.forecaster.base.BaseForecaster._undifference_predictions`). LV's own
  `SignalCharacteriser` verdict on `MLVS-PT-3PHASE` is fractional integration
  (`d=1`): its monthly mean drifts by an amount comparable to the whole
  signal's own standard deviation, which every raw-level LV forecast shows as
  a shared, architecture-independent bias.

Each head in the `benchmark`/`conformal` groups of this script shares one
backbone search (`hpo_parent_model_cls=MLPFConfig, hpo_parent_variant=SHARED_HPO_VARIANT`);
FPQR's `grid` conditions each run their own full joint backbone+FPQR-head
Optuna search, since `per_target_tau` is the quantity the `grid` group
compares and so cannot itself be tuned away.

```bash
uv run python chapters/ch7_multitarget/experiment_1_benchmark/multitarget_experiment.py \
  --group grid --dataset MLVS-PT-3PHASE --run-hpo --hpo-trials 100 --hpo-epochs 50
```

**Checkpoint reuse (`reuse_checkpoint_from`).** A calibration-only condition
(same model, same tuned hyperparameters, only `conformal_config` differs)
loads its source condition's already-trained per-fold weights instead of
calling `fit()` again. The mechanism lives in `twiga/experiment/engine.py`
(`_resolve_checkpoint_reuse`, `_load_fold_checkpoint`) and
`twiga/experiment/condition.py` (`Condition.reuse_checkpoint_from`,
`Condition.reserve_calib_window`), and is not specific to this script.
Preconditions, validated at run time: the source condition must run, in the
same `ExperimentEngine.run()` call, in a group ordered before the dependent
condition's own group; both conditions must share the identical CV layout and
the same effective calibration-window reservation; both must share the same
`model_cls`. Not validated (surfaces as a `RuntimeError` from
`load_state_dict` instead): `data_overrides` or anything else that changes
the model's input dimensionality.

## Experiment 1b: PV/EV penetration sweep (`plot_ev_pv_penetration.py`)

Presents the impact of increasing PV and EV penetration on `DSS-EU` forecasting
performance. This script rebuilds the figures directly from
`results_ev_pv_rate.xlsx` (13 (ev-rate, pv-rate) combinations x 3 targets x 5
folds x 56 daily rows: two one-axis-at-a-time sweeps sharing a `(0, 0)`
baseline, plus four coupled diagonal points not plotted here) rather than
through `ExperimentEngine`. Reports NRMSE, NMPI, PICP, and WS per target; CWE
is dropped, since it has no per-observation loss differential to test.

Requires `openpyxl`, which is not a project dependency otherwise:

```bash
uv run --with openpyxl python chapters/ch7_multitarget/experiment_1_benchmark/plot_ev_pv_penetration.py
```

Writes 12 panel PDFs (3 targets x 4 metrics) to
`chapters/ch7_multitarget/figures/` and to `Images/Chapter-7/`, where present.

## FPQR mechanism + gating ablation (`fpqr_gating_ablation.py`)

A precondition Experiment 2 builds on, not a numbered experiment in its own
right: isolates the contribution of each FPQR mechanism component and the
target-level gate, on the multi-target MLPGAM+FPQR backbone. Five groups, one
reference condition each:

| group | reference | tests |
|---|---|---|
| `proposal` | `proposal_softmax` | quantile-fraction allocation mechanism (`BaseFPQRConfig.proposal_type`) |
| `qvn` | `qvn_standard` | quantile-value-network architecture, including the monotonic value network |
| `grid` | `grid_shared` | shared vs. per-target quantile grid |
| `multigate` | `gate_none` | target-level output gate, alone and combined with the existing feature gate |
| `head` | `head_fpqr` | FPQR vs. a joint parametric (multivariate-Normal) head, a design-sanity check rather than a hypothesis the chapter argues for |

See `FPQR_GATING_README.md` for each group's hypothesis and `DESIGN_DECISIONS.md`
for what each group found.

```bash
# Local smoke test (one group, DSS-EU, cached backbone params):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --group proposal qvn --quick

# Full server run (every group, both datasets, real backbone HPO):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --dataset all

# Significance report for one group, once it has results (discovers the latest synced run):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/fpqr_gating_ablation.py --report grid
```

Output under `../experiments/ch7_multitarget/fpqr_gating_ablation/`. Not
synced automatically; see "Output and syncing a run" below.

## Experiment 2: backbone architecture + conformal evaluation (`backbone_conformal_study.py`)

Two questions, taking `fpqr_gating_ablation.py`'s mechanism choice as given:
does a different backbone architecture (MLPF, MLPGAM, NHiTS) carrying that
same FPQR configuration change the marginal picture, and does conformal
calibration, joint/multivariate (`max_norm`, `MultivariateConformal`) or CQR
(`quantile`/`quantile_joint`, `ConformalQuantileRegressor`), change the
calibrated interval picture for any of them? `quantile`/`quantile_joint` are
the same CQR score (`score_type="scaled"`), differing only in
`target_coverage`: `quantile` calibrates each target independently (marginal
coverage); `quantile_joint` pools the score across targets before
calibrating (`target_coverage=True`), the construction `max_norm` uses,
giving CQR its own joint/simultaneous-coverage variant to compare against the
two `MultivariateConformal` methods on equal footing.

One group, `backbone`: 4 heads (MLPF/MLPGAM/NHiTS FPQR plus QR's fixed-grid
baseline) x conformal conditions (`none`, `max_norm`, the six CQR
`quantile_*` variants), on all four datasets (`DSS-EU`'s `cv_calib_size=0`
skips the conformal-bearing conditions there). QR reuses a dedicated
backbone-only `MLPFConfig` HPO search (`QR_HPO_VARIANT = "shared_mlp_backbone"`)
instead of its own full joint study, since it has no FPQR-specific
hyperparameters to tune. This script has no per-head CLI restriction flag;
every condition in `CONDITIONS` always runs. Use `--variants` to restrict the
conformal axis, or edit `HEADS` directly for a one-off head-only run. Every
backbone's `per_target_tau` is enabled: a shared quantile grid across P/Q/V
lets one target's predicted spread collapse under the pooled pinball loss,
which then poisons every pooled/simultaneous conformal condition via that
target's near-zero denominator dominating the shared threshold.

`--hpo-objective` (default `winkler_score`) applies to every condition here,
all FPQR-family or QR, all quantile-producing. `--skip-hpo` (default, reuse a
cached `*_best_params.json`) / `--run-hpo` (tune for real), `--hpo-trials`
(default 50), `--hpo-epochs` (default 100, per Optuna trial, distinct from
`--epochs`, the final conditions' budget).

```bash
# Local smoke test (5 epochs, 1 trial, MLVS-PT-3PHASE only):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --quick

# Full server run (every dataset, real backbone HPO, all 4 heads):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --dataset all --run-hpo --hpo-trials 100 --hpo-epochs 50

# Significance report for one dataset's run (Winkler score DM/MCS; reads the synced local copy):
uv run python chapters/ch7_multitarget/experiment_2_backbone_conformal/backbone_conformal_study.py --report backbone
```

Output under `../experiments/ch7_multitarget/backbone_conformal_study/`. Not
synced automatically; see "Output and syncing a run" below.

**Dataset loading note.** Loading a dataset (`twiga.utils.data_loading`, via
`twiga.core.data.selection.compute_rf_importance`) runs a
`RandomForestRegressor(n_jobs=-1)` feature-selection step, which spawns a
`joblib`/`loky` worker pool. This step, and so any script that calls
`load_dataset()`, needs an unsandboxed shell to run normally; a sandboxed
local shell can leave the worker pool hanging indefinitely instead of
erroring.

## Experiment 3: operational decision (`fixed_decision_sweep.py`)

Trains the four fixed heads (`fpqr`, `qr`, `bnn`, `laplace`) on each EV x PV
scenario of `DSS-UK` (weak and stiff feeders) and `DSS-US` rank 8, and scores
the probability of each operational event (single-step and sustained overload
at 1.0x, 1.25x and 1.5x the pre-DER headroom, customer-voltage breach and
their union on `DSS-UK`) through every scenario form a head supports:
independent and Schaake-coupled draws for every head, and for the BNN also
its own paths with per-cell or day-coherent noise (`FORECAST_TYPE_PLAN.md`).
Conformal conditions are off by default; `--with-conformal` restores them.
Each head keeps its checkpoints and saves its raw forecasts, so
`forecast_type_study.py` runs on any scenario offline.

`DSS-US` uses rank 8, not rank 0: SMART-DS assigns DER to a fresh random set
of customers at each penetration level, so most single transformers (rank 0
among them) do not see their DER grow from one level to the next. The sweep
checks every network's response to EV and PV at start and warns otherwise.

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

`experiment_3_decision/backbone_conformal_decision.py`, `multitarget_decision.py`,
`multitarget_decision_single_split.py`, and `operational_decision.py` are
earlier operational-decision scripts kept in this directory for reference;
`fixed_decision_sweep.py` above is the current one.

## `forecast_examples.py`: predictive distributions and the full metrics table

Produces the predictive-distribution figures and the complete marginal/joint/
operational metrics table (energy score, variogram score, hosting-capacity
Brier score) across every FPQR/head variant, plus the Winkler-score DM/MCS
significance table for deciding a comparison (never CWE, which is diagnostic
only). Its panel-drawing primitives (`draw`, `_format_axes`,
`place_legend_below`, `_fmt2`: hollow-marker observed series, dashed
predictive median, translucent quantile fan) live in
`twiga_thesis.plotting.forecast_panels`, shared with Chapter 6's
`chapters/ch6_conformal/design_ablation/mlpgam_forecast_examples.py`.

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

Output under `chapters/ch7_multitarget/forecast_examples/`: per-panel PDFs,
`metrics_all_tiers.csv`, `winkler_dm_table.csv`, `conformal_calibration.csv`,
`pointwise_losses.csv`.

## `head_conformal_check.py`: exploratory, every backbone x head x conformal variant

Not a chapter experiment. `fixed_benchmark.py` (Experiment 1, MLPF only,
FPQR/QR/BNN/Laplace) and `backbone_conformal_study.py` (Experiment 2,
MLPF/MLPGAM/NHiTS, FPQR only) remain the two scripts whose numbers go in the
chapter. This script exists to sanity-check a proposed extension, every
backbone paired with every head under every conformal variant, on any of the
four datasets, before deciding whether any of it is worth folding into either
formal script.

It reuses `backbone_conformal_study.py`'s own `ExperimentSpec` objects
directly (via `dataclasses.replace`, swapping only `conditions`) rather than
re-deriving CV numbers by hand, and its panels reuse `forecast_examples.py`'s
own drawing utilities (`twiga_thesis.plotting.forecast_panels`) so its output
reads in the thesis's standard visual style. One best-day and one worst-day
panel is saved per condition x target, picked by a simple
RMSE-normalised-by-mean-|Actual| proxy.

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

Output under `../experiments/ch7_multitarget/exploratory/head_conformal_check/head_x_conformal_check/runs/<run_id>/`,
the same external base the formal chapter scripts share:
`results/head_conformal_check_{full,summary,representative}.csv`,
`plots/<dataset>/panels/<condition>_<target>_{best,worst}.pdf`, plus the
engine's own default 7-day sample plots under `plots/<dataset>/`.

## `split_scheme_figure.py`: CV split-scheme figure

Renders both datasets' actual fold layout from `fpqr_gating_ablation.py`'s
real `ExperimentSpec` against the real parquet files, the fastest way to
check a CV protocol change produced the fold count intended before spending
compute on a full run.

```bash
uv run python chapters/ch7_multitarget/shared/split_scheme_figure.py
```

No arguments. Writes both datasets' figures to `chapters/ch7_multitarget/figures/`.

## Supporting diagnostics

Back a specific claim in `Chapter-7.tex`'s prose rather than implementing one
of the three named experiments. Each takes no required arguments.

- **`copula/dependence_cost.py`**: cost of assuming target independence,
  measured directly from the power-flow records (`Chapter-7.tex` §7.3).
- **`experiment_2_backbone_conformal/per_target_tau_diagnostic.py`**:
  whether FPQR's learned quantile grid tracks the `|Q''(tau)|^(-1/3)`
  theory-optimal shape per target; see `EXPERIMENT_PLAN.md` §10.4-10.8 for the
  full result and why it answers a different question than
  `fpqr_gating_ablation.py`'s `grid` group.
- **`copula/joint_vs_independent.py`**: the dependence-cloud figure and
  energy score comparing joint and independent heads.
- **`copula/residual_dependence.py`**: how much cross-target dependence
  survives after conditioning on the shared backbone representation.
- **`experiment_2_backbone_conformal/multitarget_bakeoff.py`**: a quick,
  single-scenario scoping run across head variants, to decide which
  comparisons are worth a full backtest.
- **`experiment_1_benchmark/characterise_records.py`**: measures each
  record's own lag/AR-order structure rather than assuming the 96-step/48-step
  window throughout.
- **`copula/copula_variants.py`** and **`copula/penetration_sweep.py`**: the
  copula-based dependence modelling the chapter no longer argues for, kept in
  `copula/` for other work to reuse.

`experiment_1_benchmark/build_dss_eu.py`, `experiment_1_benchmark/export_us_record.py`,
and `check_chapter.py` are one-off data-preparation/validation utilities, not
experiments in their own right.

## Datasets

| key | source | role |
|---|---|---|
| `DSS-EU:default:0.4:0.4` | OpenDSS power flow, EU LV feeder (`uk-mvlv`) | Experiment 1/2 reference; voltage-binding half of Experiment 3 (`DSS-UK`) |
| `DSS-UK:weak:0.4:0.4` | OpenDSS power flow, weak EU feeder | Experiment 3, hosting-capacity (voltage) decision |
| `DSS-US:0:0.4:0.4` | OpenDSS power flow, SMART-DS | Experiment 3, thermal-exceedance decision |
| `MLVS-PT-3PHASE` | measured, Madeira LV substation | Experiment 1/2, the measured (not simulated) P/Q/V record |

`MLVS-PT-3PHASE.parquet` is not produced from raw per-year CSVs available only
on the project server (`PhD-Thesis-Anthony/notebook/Prepare-data.ipynb`) or
copied into `examples/data/`; scripts that default to it fail with a plain
`FileNotFoundError` until it exists locally. `EXPERIMENT_PLAN.md` documents
how the OpenDSS scenarios themselves are generated.

## Prerequisites

- Run from the repository root with `uv run`.
- A tuned MLPF/MLPGAM backbone at `chapters/ch7_multitarget/backbone_params.json`.
  `tune_backbone.py` writes it; every script below falls back to a hardcoded
  default if it is missing, so re-tuning is optional.
- Every script's `--quick` smoke test finishes in minutes and should be run
  before the corresponding full server run.

## Output and syncing a run

Every script above writes to `$TWIGA_EXPERIMENTS_ROOT/ch7_multitarget/<experiment>/<spec-slug>/runs/<run_id>/`
(default `../experiments/ch7_multitarget/`, a sibling of the repository), not
into the tracked tree. `shared/ch6_analyse_results.py` and
`shared/ch6_forecast_examples.py` read from a separate, local, git-ignored
copy under `chapters/ch7_multitarget/results/ch7_multitarget/<experiment>/<spec-slug>/runs/<run_id>/`
instead. A fresh run's results stay invisible to the analysis scripts
(falling back to whichever run was synced last) until that copy is made:

```bash
EXT=../experiments/ch7_multitarget/<experiment>/<spec-slug>/runs/<run_id>
LOCAL=chapters/ch7_multitarget/results/ch7_multitarget/<experiment>/<spec-slug>/runs/<run_id>
mkdir -p "$LOCAL/results"
cp "$EXT"/meta.json "$EXT"/run_status.json "$LOCAL/"
cp "$EXT"/results/*.csv "$LOCAL/results/"
# per condition, metrics/predictions parquet only, skip checkpoints/logs/figures:
for cond_dir in "$EXT"/conditions/*/*/*/; do
  rel=${cond_dir#"$EXT"/conditions/}
  mkdir -p "$LOCAL/conditions/$rel"
  cp "$cond_dir"*.parquet "$LOCAL/conditions/$rel" 2>/dev/null
done
```

```bash
uv run python chapters/ch7_multitarget/shared/ch6_analyse_results.py --experiment multitarget_experiment --group benchmark --dataset MLVS-PT-3PHASE

# Good/bad-day forecast panels, per target (repeat --target for LP/LQ/LV):
uv run python chapters/ch7_multitarget/shared/ch6_forecast_examples.py --experiment multitarget_experiment --group benchmark --dataset MLVS-PT-3PHASE --target LP --day-metric winkler --latex
```

The `ch6_` prefix on these module filenames and on `ExperimentSpec` slugs
carries over from an earlier folder-numbering scheme; it is part of the file
and path names, not a chapter reference.

## Evaluation criteria

Higher is better for CWE, stratified CWE, and PICP; lower is better for CRPS,
the Winkler score, and the Brier score. Comparisons between conditions are
decided on the Winkler score (marginal/design questions) or the
hosting-capacity Brier score (the operational questions), via Diebold-Mariano
and the Model Confidence Set, never on CWE: it has no per-observation loss
differential to test, and stays in every table as a diagnostic alongside PICP
and NMPI.

## Shared modules

- `twiga.core.metrics` (`pairwise_dm_table`, `mcs_ranking`), reused by every
  script above that reports a significance table.
- `twiga.experiment.ExperimentEngine`/`Condition`/`ExperimentSpec`, the
  condition/backbone/CV wiring for `fixed_benchmark.py`, `multitarget_experiment.py`,
  `fpqr_gating_ablation.py`, and `backbone_conformal_study.py`.
- `forecast_examples.py`'s `build_backtest_folds`/`metrics_table`/
  `hosting_capacity_rows`, reused by the scripts in `copula/` and by
  `experiment_3_decision/operational_decision.py`.
- `twiga.distributions.conformal`: `MultivariateConformal` (joint, max-norm),
  `ConformalQuantileRegressor` (CQR; per-target by default, or
  joint/simultaneous via `target_coverage=True`, also reachable through
  `ConformalConfig(method="quantile", target_coverage=True)` for
  engine-driven conditions, see `backbone_conformal_study.py`'s
  `quantile_joint`), and `score.calibrate_score_threshold`/`exceeds_threshold`
  (a method-agnostic validated-threshold diagnostic), all reused by
  `backbone_conformal_study.py`.
- `twiga_thesis.plotting.forecast_panels` (`draw`/`_format_axes`/
  `place_legend_below`/`_fmt2`), this chapter's predictive-panel style,
  consumed by `forecast_examples.py` and by Chapter 6's
  `mlpgam_forecast_examples.py`. Not chapter-specific.
