# Chapter 5: FPQR experiments

Regenerates every number in Chapter 5 under the evaluation protocol of Chapter 3,
so that the two chapters agree and the probabilistic conclusions rest on proper
scores. Naming follows the Chapter 3 convention: `s5<section><n>_*.csv`, keyed to
the section of `Chapter-5-new.tex` that consumes the result.

## Use the twiga ExperimentEngine, not hand-rolled loops

`shared/fpqr_experiment.py` defines these experiments as an `ExperimentSpec` over
`Condition` objects and runs them through `twiga.experiment.ExperimentEngine`,
the same machinery Chapter 6 and Chapter 7's scripts use. That gives:

* **Ten-fold expanding-window backtesting.** The engine defaults are
  `cv_train_size=12`, `cv_test_size=4`, `cv_val_size=2`, `cv_folds=10`, exactly
  the protocol Chapter 3 specifies. `exploratory/learned_tau.py` and
  `exploratory/tau_matrix.py` use a single train/test split instead, so their
  numbers carry no fold-level variability; treat their output as a diagnostic,
  not as a result for the thesis.
* **`controlled_fields`.** The engine holds the listed fields equal across
  conditions, so an apparent effect cannot come from a differing backbone.
* **`reference_conditions`.** Each group is compared against its own baseline,
  which is what the paired significance tests need.
* **CLI, resumption and tracking.** `--group`, `--dataset`, `--skip-hpo` and
  MLflow come free, so a partial rerun costs one flag.

`Condition.model_cls` carries the fixed-grid comparison, since `mlpfqr` is a
different config class from `mlpffpqr`.

**What the engine does not do** is capture intermediate tensors. The learned
grid is read with a forward hook on the proposal network, so the tau
diagnostics need a small `ExperimentEngine` subclass overriding `_run_condition`
to attach that hook after each fold's fit. `exploratory/learned_tau.py` is the
source for the tau figures, with the single-split caveat above.

## Protocol: fixed across every experiment

| | |
|---|---|
| Datasets | `examples/data/MLVS-PT.parquet` (primary), `MLVS-UK.parquet` (generalisation) |
| Backtesting | 10-fold expanding window: 12-month initial, 4-month window, 2-month step |
| Nominal level | `1 − α = 0.90` throughout, stated in every table |
| Quantile grid | `n_quantiles = 9` (twiga default) unless the experiment sweeps it |
| Seeds | 5 per configuration, fixed and recorded per fold |

The backtesting scheme is deliberately identical to Chapter 3's, so that Chapter
5 and Chapter 7 report comparable numbers for nominally identical models.

## Metrics: which number carries which claim

**Primary, carry the conclusions.** `crps_quantile`, `winkler_score`
(`reduction="mean"`, since the median reduction is improper), and calibration as
a hypothesis test rather than an eyeball: `christoffersen_test` for conditional
coverage and `pit_uniformity_test` on the PITs from `reliability_scores`.
`interval_coverage` at 50/80/90/95/98%, not at one level.

**Secondary, for ranking only.** `reduced_cwe` with `strata` set to the lead
time. Stratification is not optional here: Chapter 3 establishes that the
marginal form is blind to conditional calibration, and a 48-step-ahead forecast
is exactly the heteroscedastic case where that matters.

**Point accuracy.** NRMSE only. MAPE and SMAPE are dropped from the headline
tables, since net load crosses zero by construction.

## Significance, and one thing it cannot cover

`diebold_mariano_from_losses` and `giacomini_white_from_losses` (both reused via
`twiga_thesis.analysis`) take a pre-computed pointwise loss differential, so
they apply to CRPS, the interval score and pinball without modification. Use
**GW** where the comparison is nested (FPQR against fixed-grid QR is the same
architecture with the grid held fixed, which is precisely the nested case), and
**DM** otherwise. Apply `benjamini_hochberg` across the M×M grid, and report
`model_confidence_set` as a column on the benchmark table.

**The CWE cannot enter any of these tests.** DM and GW require a per-observation
loss differential `d_t`; the CWE exists only as a sample aggregate, so no `d_t`
can be formed. This is independent corroboration of the Chapter 3 argument:
a quantity that cannot be tested is not a scoring rule.

## Experiments

### `s561`: benchmark replication (Experiment 3, `sec:mlpfq-exp-setup`)

**Objective.** Replicate Chapter 5's Experiment 3 under Chapter 3's evaluation
protocol, comparing MLPQF against QR, parametric Laplace, and BNN heads with a
full metric surface rather than the chapter's original NRMSE/CWE-only report.

**Hypothesis.** MLPQF's adaptive, learned quantile grid matches or exceeds
fixed-grid QR, parametric Laplace, and BNN on proper scores (CRPS, Winkler
score), not only on NRMSE/CWE.

Implemented as the `benchmark` group in `shared/fpqr_experiment.py`
(`--group benchmark`): MLPQF (`mlpffpqr`) against QR, parametric Laplace, and
BNN, all on the MLPF backbone, full metric surface (point + interval +
quantile/parametric, not only NRMSE and CWE). The MLPQF condition sets
`num_cosines=0, rope_type="linear"` explicitly: the chapter states MLPQF embeds
learned fractions via RoPE rather than a cosine basis ("this is what
distinguishes MLPQF from FPSe2Q"), with a formula that rotates by the slot's
integer rank, which is `rope_type="linear"` in twiga's terms, not twiga's
default `"tau_pos"`. Everything else uses twiga's current defaults (plain
pinball, standard QVN, relu crossing penalty) rather than the chapter's
original Huber-pinball + squared-margin objective; `s566` below is the
ablation that settles whether that choice matters.

A `benchmark_nhits` group repeats the same four-way comparison on the N-HiTS
backbone, as a robustness check that the head ranking found on MLPF holds
across architectures. Together `benchmark` and `benchmark_nhits` are what
Chapter 5's `tab:backbone-generality` (Experiment 4, backbone generality)
reports.

**Plotting the `benchmark`/`benchmark_nhits` results.**
`experiment_4_backbone_generality/plot_backbone_generality.py` reads
`results/ch5_fpqr_full.csv`, aggregates to one row per fold (the fold-then-mean
convention `design_ablation_significance.py` also uses), and writes one bar
chart per metric (NRMSE, CRPS, Winkler score, CWE stratified) to `figures/`,
faceted by dataset and coloured by backbone (MLPF vs N-HiTS), one bar per head
(MLPQF/QR/BNN/Laplace):

```bash
uv run python chapters/ch5_fpqr/experiment_4_backbone_generality/plot_backbone_generality.py
```

Requires `arviz<1.0` and `seaborn` (both `dev` dependencies), since newer
`arviz` dropped the `arviz-doc` matplotlib style name this script's
`az.style.use(["science", "arviz-doc"])` call needs.

**FPSe2Q-MLP is deliberately not a `benchmark` condition.** It is a separate,
pre-twiga architecture (`PhD-Thesis-Anthony/src/fpq/FPQ.py`), not
self-contained, that twiga's own FPQR already supersedes architecturally.
Rather than porting it, the legacy code runs as-is, unmodified, on the same
data split, and its output is quoted as an external reference row in the
benchmark table, with a caveat that its data pipeline and features differ from
twiga's.

**Softmax proposal shape.** `Chapter-5.tex` describes Chapter 5's softmax
proposal as "a single quantile grid... applied uniformly across all forecast
horizons" (implying a shared `BxNx1` grid). Checked directly against
`PhD-Thesis-Anthony/src/net/quantile_block.py::MLPFPQNetwork` (the actual MLPQF
implementation): it instantiates `NQuantileProposal(..., M=hparams['horizon'])`,
and `NQuantileProposal.forward()` reshapes to `(B,N,M)` and softmaxes over `N`
independently per horizon column, genuinely `BxNxH`, not `BxNx1` (confirmed
numerically: horizon slices differ, not merely broadcast). The `benchmark`
group's `mlpqf` condition therefore uses twiga's existing default proposal
behaviour (`proposal_shared_horizon=False`, the mode that matches the real
code); `twiga.distributions.nn.fpqr.proposal.QuantileProposal` also exposes a
separate, opt-in `shared_across_horizon` flag for the shared-grid mode.

### `horizon_sharing`: does the proposal-sharing choice matter?

**Objective.** Measure whether the two proposal-sharing modes (`BxNx1` vs
`BxNxH`) have any measurable effect on performance.

**Hypothesis.** The two proposal-sharing modes produce indistinguishable
point/interval/quantile metrics.

`horizon_varying` (`proposal_shared_horizon=False`, the reference condition,
matching the real code's behaviour) against `horizon_shared`
(`proposal_shared_horizon=True`, the shared-grid design).

### `s562`: fixed-grid ablation (the missing control)

**Objective.** Isolate the effect of learning the quantile fractions versus
using a fixed uniform grid, holding backbone, `n_quantiles`, and training
budget fixed so the grid is the only thing that varies.

**Hypothesis.** MLPQF's learned grid outperforms the fixed uniform grid.

`mlpffpqr` against `mlpfqr`: same backbone, same `n_quantiles`, same training
budget; only the grid differs. Nested, so GW.

### `s563`: learned-τ diagnostics

**Objective.** Characterise what the learned quantile grid actually looks like
on real data: its mean shape against uniform, the spread of τ̂ across
instances, drift by lead time, and the difference between clear-sky and
overcast days.

**Hypothesis.** The learned grid allocates resolution non-uniformly, departing
further from the uniform grid at the tails and under overcast conditions,
where the conditional distribution is least Gaussian. Diagnostic rather than a
significance-tested claim (no retraining, no fold-level variability), supplying
the real-data counterpart to the illustrative `fig_fpqr_*` panels.

No retraining: `forecast()` returns `quantile_levels` as a `(B, N, H)` tensor.

### `s564`: number of quantile levels

**Objective.** Determine how many quantile levels N are needed, sweeping the
geometrically spaced grid `N ∈ {9, 14, 23, 36, 57, 90}` (ratio
`(90/9)**(1/5) ≈ 1.585`, spanning the chapter's illustrative N=9 to the N=90
used for its Experiment 3 numbers).

**Hypothesis.** Performance improves with more levels up to a point of
diminishing returns. Watch the tail levels in particular: N bounds the
resolution of the reconstructed CDF.

### `s565`: quantile crossing rate

**Objective.** Quantify the empirical quantile-crossing (inversion) rate with
and without the crossing penalty `L_nc`, and at inference under
`qvn_type='standard'`.

**Hypothesis.** The crossing penalty substantially reduces but does not
eliminate crossing: it is a soft constraint with zero gradient once ordering
holds, so a non-zero residual rate is expected.

### `s566`: loss-term ablation

**Objective.** Test whether each term of the FPQR objective (entropy
regularisation, Wasserstein proposal loss, crossing penalty) and the loss
function choice (pinball vs. Huber-pinball) individually matter, rather than
assuming the full objective is jointly necessary.

**Hypothesis.** Removing any term degrades calibration or accuracy relative to
the full objective; plain pinball outperforms Huber-pinball. twiga's own
documentation notes that `huber-pinball` degrades interval calibration; this
ablation checks that directly against the chapter's method.

### `s567`: real-time robustness

**Objective.** Re-evaluate Chapter 5's Experiment 2 (forecast rather than
observed GHI as an exogenous input) on proper scores (CRPS, Winkler score)
rather than NRMSE/CWE alone.

**Hypothesis.** Forecast accuracy and calibration degrade measurably when using
forecast GHI instead of observed GHI, and CRPS/WS make that degradation
visible more clearly than NRMSE/CWE do.

## Monotone quantile-value-network architectures

Three `qvn_type` conditions, in addition to the four standard mechanisms,
each enforcing output monotonicity a different way:

- **`qvn_lattice`** (`LatticeQVN`, Cannon, 2018, MCQRNN). Guarantees
  monotonicity via non-negative weight reparameterisation on the path from the
  quantile level to the output, not a cumulative sum over slots, so every slot
  is evaluated independently and in parallel rather than sequentially.
  Cheapest of the three to train (a single forward pass, no extra
  evaluations). The guarantee requires the context fed into its unconstrained
  path to carry no information about the quantile level itself: FPQR's usual
  fused `combined` embedding already has tau fused in via the multiplicative
  gate upstream and breaks the guarantee, so `LatticeQVN` consumes the
  backbone's raw `context_features` directly instead, the same tau-independent
  vector `AttentionQVN` uses.
- **`qvn_spline`** (`SplineQVN`, harmonic-mean derivative smoothing over
  adaptive knots, following Fritsch & Carlson, 1980's monotone piecewise-cubic
  construction). Existing spline quantile functions (Gasthaus et al., AISTATS
  2019, used in GluonTS/DeepAR) interpolate between fixed, regularly-spaced
  knots; this combines the same monotone-derivative-smoothing idea with FPQR's
  own adaptively proposed knot positions. Same per-slot sequential-integration
  cost profile as `MonotonicQVN`.
- **`qvn_umnn`** (`UMNNQVN`, Wehenkel & Louppe, NeurIPS 2019). Constrains only
  the derivative to be positive (a single softplus), then integrates it via
  Gauss-Legendre quadrature (default 8 points per segment, `umnn_quad_points`
  in `MLPFFPQRConfig`) rather than constraining the map itself, at the highest
  compute cost of the three (quadrature evaluations per segment). The
  quadrature context is the segment's lower-endpoint embedding, held fixed
  across quadrature nodes, a discrete stand-in for UMNN's usual
  continuously-varying context, since FPQR's tau embedding only supplies
  per-slot, not per-tau, context vectors.

All three are verified end to end (`tests/unit/distribution/test_qvn.py`):
correct output shape, strictly monotone output (`LatticeQVN` checked over 20
random `(context, tau)` draws, since its guarantee is per-input), and
gradients reaching the input.

## Run order

`s563` needs `s561`'s checkpoints. `s566` and `s564` are independent and can
run in parallel. `s562` is the highest-value single experiment: it is the
control the chapter's central claim requires, and it is one configuration.

## How to run

### Prerequisites

- Run from the repository root with `uv run`, or from inside `chapters/ch5_fpqr/`
  with the virtual environment active. `shared/fpqr_experiment.py` inserts the
  repository root onto `sys.path` itself, so no package installation is
  required.
- Datasets: `MLVS-PT` and `MLVS-UK`, both expected at `examples/data/*.parquet`.
  These are the same files Chapter 7 uses; see
  [`chapters/ch6_conformal/README.md`](../ch6_conformal/README.md) for how to
  obtain them if a checkout does not already have them.

### Bootstrap: seed the backbone HPO cache first

`shared/fpqr_experiment.py` runs on `twiga.experiment.engine.ExperimentEngine`,
the same shared engine Chapter 6 and Chapter 7's scripts use. Its Phase 1 tunes
the backbone (`MLPFConfig`, `SPEC.backbone_cls`) once per dataset via Optuna and
caches the result under the engine's own `backbone_hpo/<dataset>/` directory;
Phase 2 (every ablation group/condition) then reuses that cache unless
`--skip-hpo` forces it to skip HPO and fall back to config defaults. A
`--skip-hpo` run against a dataset whose cache was never populated does not
error, it silently trains on un-tuned defaults: if a condition's results look
unstable (e.g. FPQR on MLPF), confirm the cache was actually seeded for that
dataset before suspecting the ablation itself.

Run once per dataset, **without** `--skip-hpo`, before any other group.
`--dataset all` runs both datasets in one process, since the engine loops over
`dataset_keys` and caches Phase 1 HPO separately for each:

```bash
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --group grid --dataset all
```

`grid` is used here only because it is the smallest group (one condition); any
group works, since Phase 1 HPO tunes the shared backbone before Phase 2 runs
the group's own conditions, and only needs to happen once per dataset. Every
subsequent group/condition on the same dataset can then safely add
`--skip-hpo` to reuse the cached backbone rather than re-tuning it.

### `shared/fpqr_experiment.py`: every experiment except `s563`

`shared/fpqr_experiment.py` implements the `grid`, `n_quantiles`, `loss_terms`,
`proposal`, `tau_embed`, `num_cosines`, `rope_type`, `crossing`, `sort_output`,
`qvn_type`, `loss_fn`, `lambda_cal`, `prop_nq`, `stop_gradient`, `prop_loss`,
`horizon_sharing`, `best_config`, `benchmark`, and `benchmark_nhits` groups
described above (`s561`/`s562`/`s565`/`s566` and the `horizon_sharing` check
all live here; see the module's own docstring for the full group-to-hypothesis
mapping).

```bash
# List every group and flag:
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --help

# Everything, both datasets:
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py

# One group only, for example the fixed-grid control (s562):
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --group grid

# The benchmark replication (s561), one dataset:
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --group benchmark --dataset MLVS-PT

# Several groups at once, e.g. both benchmark variants:
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --group benchmark benchmark_nhits

# Reuse cached backbone HPO parameters instead of re-searching:
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --group grid --skip-hpo

# Quick smoke test with fewer epochs and folds:
uv run python chapters/ch5_fpqr/shared/fpqr_experiment.py --group grid --epochs 2 --folds 2
```

`--group`/`--dataset` both accept `all` (the default) or a space-separated
list. Output is written under `../experiments/chapter-5-fpqr/` (a sibling of
the repository root, kept out of the tracked tree since these runs regenerate
large per-fold CSVs and checkpoints), one CSV per group per dataset, following
the `s5<section><n>_*.csv` naming convention.

### `exploratory/learned_tau.py` and `exploratory/tau_matrix.py`: `s563`, single-split diagnostics

These are not on the `ExperimentEngine` (see "What the engine does not do"
above) and run as plain scripts:

```bash
uv run python chapters/ch5_fpqr/exploratory/learned_tau.py
uv run python chapters/ch5_fpqr/exploratory/tau_matrix.py
```

Both need `s561`'s checkpoints already trained, since they read a fitted MLPQF
model rather than retraining one.

### `exploratory/worked_example.py`: pipeline-figure intermediates

Also not on the `ExperimentEngine`. Captures every intermediate signal of one
MLPQF forecast (input window, GHI, learned quantile fractions and values,
realised net load) for the pipeline figure, on the day chosen as the
median-CRPS day among the most variable third by within-day irradiance:
representative rather than flattering.

```bash
uv run python chapters/ch5_fpqr/exploratory/worked_example.py
```

Needs `s561`'s checkpoints, same as `exploratory/learned_tau.py`/
`exploratory/tau_matrix.py` above. Writes `s563_worked_example.npz`, under
`exploratory/`.

### `exploratory/qvn_pit.py`: PIT and reliability per `qvn_type` mechanism

Bypasses the engine's standard reduced-prediction path to capture the full
learned quantile array (`quantile_hats`/`tau_hats`) a PIT value needs,
generalised across all eight `qvn_type` mechanisms via a forward hook, since
`FPQRDistribution.quantile_proposal`/`.quantile_output` are set under every
`qvn_type` branch.

```bash
# All eight qvn_type conditions, MLVS-PT (default):
uv run python chapters/ch5_fpqr/exploratory/qvn_pit.py

# One dataset, a subset of conditions:
uv run python chapters/ch5_fpqr/exploratory/qvn_pit.py --dataset MLVS-UK --conditions qvn_relu qvn_softplus
```

Illustrative, not a faithful reproduction of the formal backtest: one
freshly-trained model per condition on `exploratory/learned_tau.py`'s own
`SPLITS` (a single train/test window), not the 10-fold rolling-origin protocol
`shared/fpqr_experiment.py`'s `qvn_type` group runs under. Read the PIT and
reliability shapes as diagnostic of each mechanism's calibration behaviour,
not as a substitute for `shared/analyse_results.py`'s per-fold Winkler-score
comparison. Output (`.npy` PIT arrays, `.npz` forecast captures for reuse, PIT
histogram + reliability diagram PNGs, `pit_summary.csv`) lands under
`chapters/ch5_fpqr/qvn_pit/<dataset>/`.

### `qvn_forecast_examples.py`: good/bad-day forecast panels by condition

The companion good/bad-day view: what a forecast from each mechanism actually
looks like, not just whether it is calibrated on average. Needs no retraining
and no new metric computation: `predictions.parquet` already carries
`forecast`/`lower`/`upper`/`Actual`, and `metrics.parquet` already has per-day
NRMSE/CRPS/Winkler, both written by `shared/fpqr_experiment.py`'s own formal
10-fold backtest, so this script only reads them. One day is picked once,
pooled over every condition with a completed backtest (mean per-day NRMSE), so
every panel shows the same day and the comparison is fair. Reuses
`twiga.core.plot.plot_forecast` directly against `predictions.parquet`'s own
column names.

```bash
uv run python chapters/ch5_fpqr/qvn_forecast_examples.py
uv run python chapters/ch5_fpqr/qvn_forecast_examples.py --dataset MLVS-UK
```

Generalises to any group via `--group`, for example `--group proposal` for the
FPN/proposal-network ablation (`prop_softmax`/`prop_horizon`/
`prop_kumaraswamy`); a group is only reachable this way if its run retained
per-timestamp predictions (`predictions.parquet`). Output
(`good_<condition>.png`, `bad_<condition>.png`, `overview_{good,bad}_day.png`,
`forecast_examples_summary.csv`) lands under
`chapters/ch5_fpqr/qvn_forecast_examples/<dataset>/`.

### `shared/qvn_type_comparison.py`: one table across `qvn_type`'s three result files

Joins `shared/analyse_results.py`'s three separate `qvn_type` CSVs per dataset
(`_metrics`, `_significance`, `_kupiec`) into one table: NRMSE/CRPS/Winkler/
CWE-stratified/PICP/NMPI/calibration-error, a Borda-count ranking, a DM
significant win/loss count per condition, and the Kupiec calibration flag, for
both datasets side by side, plus four quantile-specific metrics
`shared/analyse_results.py`'s own `METRIC_COLS` does not carry (pinball loss,
weighted quantile loss, sharpness, tail pinball loss), read from each
condition's own `metrics.parquet` (`metric_type == "quantile"` rows).

```bash
uv run python chapters/ch5_fpqr/shared/qvn_type_comparison.py
uv run python chapters/ch5_fpqr/shared/qvn_type_comparison.py --group proposal
```

Output: `results/analysis/{group}_comparison_both_datasets.csv`. Read the
Borda ranking alongside the DM significance table, not instead of it: a
blended rank can be misled by one or two metrics even when the pairwise
significance table shows a different condition winning most comparisons.

## Results

The current, complete set of runs is under
`results/chapter_5_fpqr/runs/*/results/ch5_fpqr_full.csv` (one directory per
run, picked by timestamp), not a single flat `results/ch5_fpqr_full.csv` file.
`shared/analyse_results.py` reads the latest run that covers each group
automatically; read `ch5_fpqr_full.csv` directly rather than
`ch5_fpqr_summary.csv` from any individual run, since the summary pools
`MLVS-PT`/`MLVS-UK` into one number per condition rather than reporting the two
datasets separately.

### `shared/analyse_results.py`: metrics, significance, and figures across every completed group

Consolidates every run under `results/chapter_5_fpqr/runs/` (picking the latest
run that covers each requested group), and reports, per
`(dataset, group, condition)`: NRMSE, CRPS, Winkler score, CWE stratified,
PICP, and NMPI (mean ± sd across folds), plus a per-fold average rank across
all six. Shared fold aggregation, metric ranking, DM/MCS, coverage tests, and
comparison figures are implemented in `twiga_thesis.analysis` and
`twiga_thesis.plotting`. Figures include a multi-metric forest overview,
per-metric condition bars, an adjusted-p significance matrix, cross-dataset
metric comparisons, a benchmark-only computation-time comparison, and a
best-vs-worst-day comparison from `ch5_fpqr_representative.csv`.

```bash
# Everything that has complete data (prints a coverage table first):
uv run python chapters/ch5_fpqr/shared/analyse_results.py --group all

# One group:
uv run python chapters/ch5_fpqr/shared/analyse_results.py --group benchmark benchmark_nhits
```

Output under `chapters/ch5_fpqr/results/analysis/`.

## FPQR-MLP native comparison: not yet implemented

Figures 5.11/5.12 (`fig:mlpfq-benchmark`/`fig:experiment_four_complexity`)
currently compare MLPQF against FPQR-MLP by rerunning both models under the
original, pre-twiga implementation
(`chapters/ch5_fpqr/results/results-fpqr-benchmark`, `MLPFPQ`/`DNNFPQ`), per
`Chapter-5.tex`'s own caveat that this comparison "should be interpreted within
their respective evaluation settings rather than compared directly". A native
`twiga` rerun of both models, `mlpqf` against a new `fpqr_mlp`-style condition
using the `FPSe2Q`-MLP architecture under the same 10-fold expanding-window
protocol as `benchmark`, would put Figures 5.11/5.12 on the same basis as
`tab:mlpqf-probabilistic-heads` and `tab:backbone-generality` and drop the
legacy-codebase dependency entirely. This needs a new condition definition in
`shared/fpqr_experiment.py` for the FPQR-MLP predecessor architecture; the
current `benchmark` group only has `mlpqf`/`qr`/`bnn`/`laplace`.
