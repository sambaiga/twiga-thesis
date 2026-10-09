# Chapter 5: FPQR experiments

Regenerates every number in Chapter 5 under the evaluation protocol of Chapter 3,
so that the two chapters agree and the probabilistic conclusions rest on proper
scores. Naming follows the Chapter 3 convention: `s5<section><n>_*.csv`, keyed to
the section of `Chapter-5-new.tex` that consumes the result.

## Pending experiments — commands to run

Everything still genuinely outstanding as of 2026-09-17 (see **Next steps**
below for the full write-up behind each):

```bash
# qvn_spline/qvn_umnn -- the only two conditions across all 19 axes that have
# never run at all (item 4). Reruns the other five qvn_type conditions too,
# redundant but harmless, for one clean seven-condition comparison:
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group qvn_type --dataset all --skip-hpo

# benchmark -- corrects the pre-2026-09-17 nrmse scale_method bug (item 1) and
# produces predictions.parquet, which benchmark_nhits already has but this
# group's original 2026-09-13 run predates the feature (item 10):
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group benchmark --dataset all --skip-hpo
```

**Item 11** (a native-`twiga` rerun of the \ac{MLPQF} vs. FPQR-MLP comparison,
replacing the legacy `PhD-Thesis-Anthony` codebase Figures 5.11/5.12 currently
depend on) has **no runnable command yet** -- it needs a new FPQR-MLP-style
condition added to `fpqr_experiment.py` first (`benchmark`'s current
conditions are `mlpqf`/`qr`/`bnn`/`laplace` only, no `FPSe2Q`-\ac{MLP}
predecessor). See item 11 below before starting this one.

## Use the twiga ExperimentEngine, not hand-rolled loops

`fpqr_experiment.py` defines these experiments as an `ExperimentSpec` over
`Condition` objects and runs them through `twiga.experiment.ExperimentEngine`,
the same machinery as `chapters/ch6_conformal/crc_experiment.py`. That buys four things the
hand-rolled scripts in this folder do not have:

* **Ten-fold expanding-window backtesting.** The engine defaults are
  `cv_train_size=12`, `cv_test_size=4`, `cv_val_size=2`, `cv_folds=10`, which is
  exactly the protocol Chapter 3 specifies. `learned_tau.py` and
  `tau_matrix.py` use a single train/test split, so their numbers carry no
  fold-level variability. Chapter 3's own argument is that single splits
  mislead, and the plan records a case where a single-split ablation failed to
  reproduce on a ten-fold rerun. Treat their output as a diagnostic, not as a
  result for the thesis.
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
to attach that hook after each fold's fit. Until that exists, `learned_tau.py`
remains the source for the tau figures, with the single-split caveat stated.

## Why re-run rather than re-score

Chapter 5 currently reports NRMSE and CWE, computed under the superseded CWE and
with `winkler_score` reducing by the median, which was improper. Post-hoc
re-scoring of saved forecasts cannot fix three of the gaps (the fixed-grid
control does not exist, the loss-term ablation does not exist, and the τ tensors
were not retained), so the configurations are retrained.

## Protocol: fixed across every experiment

| | |
|---|---|
| Datasets | `examples/data/MLVS-PT.parquet` (primary), `MLVS-UK.parquet` (generalisation) |
| Backtesting | 10-fold expanding window: 12-month initial, 4-month window, 2-month step |
| Nominal level | `1 − α = 0.90` throughout, **stated in every table** (A23) |
| Quantile grid | `n_quantiles = 9` (twiga default) unless the experiment sweeps it |
| Seeds | 5 per configuration, fixed and recorded per fold |

The backtesting scheme is *deliberately identical* to Chapter 3's. Chapter 5 and
Chapter 7 currently report different numbers for nominally identical models
(A22); sharing one protocol is what stops that recurring.

## Metrics: which number carries which claim

**Primary, carry the conclusions.** `crps_quantile`, `winkler_score`
(**`reduction="mean"`**, since the median reduction is improper), and calibration as a
hypothesis test rather than an eyeball: `christoffersen_test` for conditional
coverage and `pit_uniformity_test` on the PITs from `reliability_scores`.
`interval_coverage` at 50/80/90/95/98 %, not at one level (A20).

**Secondary, for ranking only.** `reduced_cwe` with `strata` set to the lead
time. Stratification is not optional here: Chapter 3 establishes that the
marginal form is blind to conditional calibration, and a 48-step-ahead forecast
is exactly the heteroscedastic case where that matters.

**Point accuracy.** NRMSE only. MAPE and SMAPE are dropped from the headline
tables, since net-load crosses zero by construction (A12).

## Significance, and one thing it cannot cover

`diebold_mariano_from_losses` and `giacomini_white_from_losses` take a
pre-computed pointwise loss differential, so they apply to CRPS, the interval
score and pinball without modification. Use **GW** where the comparison is
nested (FPQR against fixed-grid QR is the same architecture with the grid held
fixed, which is precisely the nested case), and **DM** otherwise. Apply
`benjamini_hochberg` across the M×M grid, and report `model_confidence_set` as a
column on the benchmark table.

**The CWE cannot enter any of these tests.** DM and GW require a per-observation
loss differential `d_t`; the CWE exists only as a sample aggregate, so no `d_t`
can be formed. This is worth one sentence in the chapter, because it is
independent corroboration of the Chapter 3 argument: a quantity that cannot be
tested is not a scoring rule.

## Experiments

### `s561`: benchmark replication (Experiment 3, `sec:mlpfq-exp-setup`)

**Objective.** To replicate Chapter 5's Experiment 3 under Chapter 3's
evaluation protocol, comparing MLPQF against QR, parametric Laplace, and BNN
heads with a full metric surface rather than the chapter's original
NRMSE/CWE-only report.

**Hypothesis.** MLPQF's adaptive, learned quantile grid matches or exceeds
fixed-grid QR, parametric Laplace, and BNN on proper scores (CRPS, Winkler
score), not only on NRMSE/CWE.

Implemented as the `benchmark` group in `fpqr_experiment.py`
(`--group benchmark`): MLPQF (`mlpffpqr`) against QR, parametric Laplace, and
BNN, all on the MLPF backbone, full metric surface (point + interval +
quantile/parametric, not only NRMSE and CWE as the chapter originally reports,
A19). The MLPQF condition sets `num_cosines=0, rope_type="linear"` explicitly:
the chapter states MLPQF embeds learned fractions via RoPE rather than a
cosine basis ("this is what distinguishes MLPQF from FPSe2Q"), with a formula
that rotates by the slot's integer rank: `rope_type="linear"` in twiga's
terms, not twiga's own later-discovered `"tau_pos"` default. Everything else
uses twiga's current defaults (plain pinball, standard QVN, relu crossing
penalty) rather than the chapter's original Huber-pinball + squared-margin
objective; see `s566` below, which is exactly the ablation that settles
whether that choice matters.

A `benchmark_nhits` group repeats the same four-way comparison on the N-HiTS
backbone, as a robustness check that the head ranking found on MLPF holds
across architectures. Not part of the chapter's own named experiment, but a
supporting check, the same way `grid`/`proposal` are. Together `benchmark` and
`benchmark_nhits` are also what Chapter 5's `tab:backbone-generality`
(Experiment 4, backbone generality) reports.

**Plotting the `benchmark`/`benchmark_nhits` results.** `plot_backbone_generality.py`
reads `results/ch5_fpqr_full.csv`, aggregates to one row per fold (the same
fold-then-mean convention `design_ablation_significance.py` uses), and writes
one bar chart per metric (NRMSE, CRPS, Winkler score, CWE stratified) to
`figures/`, faceted by dataset and coloured by backbone (MLPF vs N-HiTS), one
bar per head (MLPQF/QR/BNN/Laplace):

```bash
uv run python chapters/ch5_fpqr/plot_backbone_generality.py
```

Requires `arviz<1.0` and `seaborn` (both added as `dev` dependencies; newer
`arviz` dropped the `arviz-doc` matplotlib style name this script's
`az.style.use(["science", "arviz-doc"])` call needs).

**FPSe2Q-MLP is deliberately not a `benchmark` condition.** It is a separate,
pre-twiga architecture (`PhD-Thesis-Anthony/src/fpq/FPQ.py`, ~1,940 lines
across two source packages, not self-contained) that twiga's own FPQR already
supersedes architecturally. Rather than porting it, the legacy code is run
as-is, unmodified, on the same data split, and its output is quoted as an
external reference row in the benchmark table, with a caveat noted that its
data pipeline and features differ from twiga's.

**Softmax proposal shape, validated against the actual reference code, not
just the chapter's prose.** The chapter's own Chapter 6 describes Chapter 5's
softmax proposal as "a single quantile grid... applied uniformly across all
forecast horizons" (implying a shared `BxNx1` grid). Checked directly against
`PhD-Thesis-Anthony/src/net/quantile_block.py::MLPFPQNetwork` (the actual
MLPQF implementation): it instantiates `NQuantileProposal(..., M=hparams['horizon'])`,
and `NQuantileProposal.forward()` reshapes to `(B,N,M)` and softmaxes over `N`
*independently per horizon column*: genuinely `BxNxH`, not `BxNx1` (confirmed
numerically: horizon slices differ, not merely broadcast). The chapter's prose
description is inaccurate relative to its own reference implementation; the
`benchmark` group's `mlpqf` condition therefore uses twiga's existing default
proposal behaviour (`proposal_shared_horizon=False`, the only mode that
matches the real code), not the shared-grid mode this discovery motivated
adding to `twiga.distributions.nn.fpqr.proposal.QuantileProposal` as a
separate, opt-in `shared_across_horizon` flag.

### `horizon_sharing`: does the discrepancy actually matter?

**Objective.** To measure whether the discrepancy between Chapter 6's stated
softmax proposal shape (`BxNx1`) and the reference implementation's actual
behaviour (`BxNxH`) has any measurable effect on performance, rather than
leaving it as a documentation note.

**Hypothesis.** The two proposal-sharing modes produce indistinguishable
point/interval/quantile metrics: the discrepancy is a prose inaccuracy in the
thesis, not a substantive design choice.

The `shared_across_horizon` flag above is not just documented, it is measured:
`horizon_varying` (`proposal_shared_horizon=False`, the reference condition,
matching the actual code's behaviour) against `horizon_shared`
(`proposal_shared_horizon=True`, the chapter's stated `BxNx1` design). This is
the one place in the file the two are run and scored against each other,
rather than the discrepancy being left as a documentation note.

### `s562`: fixed-grid ablation *(the missing control)*

**Objective.** To isolate the effect of learning the quantile fractions versus
using a fixed uniform grid, holding backbone, `n_quantiles`, and training
budget fixed so the grid is the only thing that varies.

**Hypothesis.** MLPQF's learned grid outperforms the fixed uniform grid.
Chapter 5 claims learning the fractions helps but never runs the one
experiment that isolates it (A18).

`mlpffpqr` against `mlpfqr`: same backbone, same `n_quantiles`, same training
budget; only the grid differs. Nested, so GW.

### `s563`: learned-τ diagnostics

**Objective.** To characterise what the learned quantile grid actually looks
like on real data: its mean shape against uniform, the spread of τ̂ across
instances, drift by lead time, and the difference between clear-sky and
overcast days.

**Hypothesis.** The learned grid allocates resolution non-uniformly, departing
further from the uniform grid at the tails and under overcast conditions,
where the conditional distribution is least Gaussian. Diagnostic rather than
a significance-tested claim (no retraining, no fold-level variability), but it
supplies the real-data counterpart to the illustrative `fig_fpqr_*` panels
(A16).

No retraining: `forecast()` returns `quantile_levels` as a `(B, N, H)` tensor.

### `s564`: number of quantile levels

**Objective.** To determine how many quantile levels N are needed, sweeping
the geometrically spaced grid `N ∈ {9, 14, 23, 36, 57, 90}` (ratio
`(90/9)**(1/5) ≈ 1.585`, spanning the chapter's illustrative N=9 to the N=90
actually used for its real Experiment 3 numbers).

**Hypothesis.** Performance improves with more levels up to a point of
diminishing returns, justifying N=90 over the smaller illustrative value with
evidence rather than assertion. Watch the tail levels in particular: N bounds
the resolution of the reconstructed CDF.

### `s565`: quantile crossing rate

**Objective.** To quantify the empirical quantile-crossing (inversion) rate
with and without the crossing penalty `L_nc`, and at inference under
`qvn_type='standard'`.

**Hypothesis.** The crossing penalty substantially reduces but does not
eliminate crossing: it is a soft constraint with zero gradient once ordering
holds, so a non-zero residual rate is expected. Reporting it converts a defect
an examiner would find into a stated limitation.

### `s566`: loss-term ablation

**Objective.** To test whether each term of the FPQR objective (entropy
regularisation, Wasserstein proposal loss, crossing penalty) and the loss
function choice (pinball vs. Huber-pinball) individually matter, rather than
assuming the full objective is jointly necessary.

**Hypothesis.** Removing any term degrades calibration or accuracy relative to
the full objective; plain pinball outperforms Huber-pinball. The latter is not
optional to test: twiga warns that `huber-pinball` degrades interval
calibration, and the chapter presents it as the method, so either the warning
or the chapter is wrong, and this ablation settles it.

### `s567`: real-time robustness

**Objective.** To re-evaluate Chapter 5's Experiment 2 (forecast rather than
observed GHI as an exogenous input) on proper scores (CRPS, Winkler score)
rather than NRMSE/CWE alone.

**Hypothesis.** Forecast accuracy and calibration degrade measurably when
using forecast GHI instead of observed GHI, and CRPS/WS make that degradation
visible more clearly than NRMSE/CWE do. This finding, that accuracy is
sensitive to irradiance forecast error, is one of the chapter's better
contributions and is currently supported only by NRMSE and CWE.

## Findings, and four bugs found while auditing them

`ch5_fpqr_full.csv` (`chapters/ch5_fpqr/results/`) has a completed run
across all 19 groups, both datasets. Reading it directly, per dataset rather
than pooled (percentages are Winkler-score change vs. each group's own
reference condition; positive is better):

**Three groups were silently no-op, confirmed and fixed.** `prop_loss`,
`stop_gradient`, and `horizon_sharing` each override a field
(`proposal_loss_fn`/`sqda_weight`, `proposal_stop_gradient`,
`proposal_shared_horizon`) that `MLPFFPQRConfig` never declared at all, even
though `FPQRDistribution` itself fully supports all four. The override was
silently accepted and never reached the model, so most of these conditions
trained the class default regardless of what was requested: `prop_loss`'s
`combined`/`sqda`/`sqda_kl` conditions were bit-identical to each other (not
merely close, exact to full float precision, every fold, both datasets) while
only `wasserstein` (which happens to match the class default) differed;
`stop_gradient`'s `softmax_sg_off`/`softmax_sg_on` pair was bit-identical on
MLVS-PT while the `kuma_*` pair was bit-identical on MLVS-UK instead (the same
underlying bug, showing up on whichever proposal family's pair the run
happened to hash together). Fixed in `twiga/models/nn/mlpffpqr_model.py` (the
four fields now declared and forwarded) and
`twiga/models/nn/prob/mlpffpqr.py` (accepted and forwarded into
`FPQRDistribution`'s `head_kwargs`); regression tests added in
`tests/unit/models/nn/prob/test_mlpffpqr_model.py::test_proposal_overrides_reach_the_distribution_head`.
**These three groups need a re-run before their numbers mean anything**; the
completed run's rows for them reflect the bug, not the design choice.

**`best_config` was never actually assembled.** Its `Condition` definition in
`fpqr_experiment.py` was a byte-for-byte copy of `default_config`'s, with a
description literally reading "update once the ablation completes", never
done. Updated to the settings confirmed on both datasets: `n_quantiles=23`
(prop_nq: +17.5% MLVS-PT / +6.5% MLVS-UK vs. the reference's 9), `crossing_penalty=100`
(crossing: +14.7% / +6.9% vs. 10), `crossing_mode="softplus"` (qvn_type:
+4.8% / +16.8% vs. relu). `proposal_type`, `rope_type`, and larger
`n_quantiles` values disagreed by dataset and are left at the reference rather
than tuned to one feeder. `proposal_loss_fn`, `proposal_stop_gradient`, and
`proposal_shared_horizon` are left at their defaults, since the groups that
would answer those questions were the three just found silently no-op.

**Fourth bug, found auditing Chapter 7, applies here too: backbone HPO never ran a real search.** `BASE_CFG` (this file) never set `num_trials`, so it used `PipelineConfig`'s own default of 10. `create_optuna_study`'s `TPESampler` uses `n_startup_trials=patience*2=20` (`twiga/forecaster/base.py:663`, `tune()`'s own default `patience=10`, also never overridden here). With `num_trials=10 < n_startup_trials=20`, every trial in every backbone HPO run this chapter has ever completed fell inside the sampler's pure-random-startup phase, where suggested hyperparameters are drawn from the seeded `TPESampler` alone, independent of any trial's observed objective. Verified directly: every `results/chapter_5_fpqr/backbone_hpo/*/logs/*/*.log` Optuna journal for this chapter shows exactly 10 trials. MLVS-PT and MLVS-UK's tuned backbones for a given model happen not to be byte-identical (unlike the exact same bug caught on Chapter 7's MLPGAM backbone, `chapters/ch6_conformal/README.md`'s 2026-09-18 entry, where they were, to 17 significant digits) -- with more models and more datasets here, the seed-determined candidate pool evidently didn't happen to pick the same winning trial index everywhere, but the search itself was never any less broken: "best of 10 random draws," not a converged, adaptive search, for every backbone (`mlpf`, `mlpffpqr`, `mlpfqr`, `mlpfbnn`, `mlpflaplace`, and every N-HiTS variant) on both datasets. Fixed by setting `num_trials=50` on `BASE_CFG`, giving ~30 trials of genuine TPE-guided search once HPO is rerun. `SPEC.fixed_overrides = {"early_stop_patience": 20}` also added: `reduce_on_plateau`'s scheduler patience (`int(sch_prob_patience * max_epochs)`) reaches up to 12 epochs at this chapter's `epochs=60`, close enough to the old `early_stop_patience=10` to race it -- the same collision confirmed and fixed for Chapter 7's MLPGAM backbone. **This requires deleting every cached `backbone_hpo/<dataset>/<model>_best_params.json` first** -- `run_backbone_hpo` reuses whatever is cached regardless of `num_trials`. Given how many of this chapter's findings rest on these backbones (including the three bugs above, all evaluated against a backbone that was never really tuned), a full rerun is needed before treating any number in this README as final, not only the three groups already flagged for a re-run above.

**Strong, dataset-consistent findings.** The proposed method beats every
benchmark baseline by a wide margin on both datasets and both backbones (QR:
-51 to -105% Winkler score, BNN: -115 to -174%, Laplace: -171 to -534%).

**Backbone generality, stated explicitly.** `benchmark` (MLPF) and
`benchmark_nhits` (N-HiTS) run the identical four-way comparison
(`mlpqf`/`qr`/`laplace`/`bnn`) on two architecturally distinct backbones.
The head ranking (MLPQF best, by Winkler score) holds on both:

| Dataset | Backbone | vs. QR | vs. BNN | vs. Laplace |
| --- | --- | --- | --- | --- |
| MLVS-PT | MLPF | -51.2% | -53.5% | -84.2% |
| MLVS-PT | N-HiTS | -43.4% | -65.0% | -72.3% |
| MLVS-UK | MLPF | -33.9% | -53.4% | -63.1% |
| MLVS-UK | N-HiTS | +5.2% | -62.4% | -46.6% |

This is the evidence base for the chapter's backbone-independence claim
(`Chapter-5.tex` line 674): the Model Confidence Set on Winkler score
retains only MLPQF on MLVS-PT with both backbones, and retains both MLPQF
and QR (statistically tied) on MLVS-UK with both backbones — MLPQF is never
significantly beaten on the deciding metric, though on MLVS-UK/N-HiTS QR's
raw mean Winkler score is numerically slightly lower (not significant, per
[[feedback_significance_first]]; do not read the `+5.2%` row above as a
loss).

**Winkler score alone overstates the win; NRMSE/CRPS tell a more qualified
story.** `Chapter-5.tex`'s `tab:backbone-generality` now reports NRMSE, CRPS,
Winkler score, and CWE stratified together, plus a per-fold average rank
across all four (lower is better). On MLVS-PT, MLPQF has the best average
rank on MLPF (2.05) but QR edges it out on N-HiTS (1.75 vs MLPQF's 1.93,
despite MLPQF alone forming the significant confidence set there). On
MLVS-UK, MLPQF does not lead on NRMSE or CRPS against QR or BNN, and QR has
the best average rank on both backbones (2.17 vs MLPQF's 2.55 on MLPF; 1.68
vs MLPQF's 1.98 on N-HiTS) despite MLPQF's clear win on CWE stratified
there. MLPQF's advantage is concentrated in interval quality, not uniform
across every metric — the blended average rank and the Winkler-score MCS
verdict diverge on the N-HiTS backbone specifically, another instance of
[[feedback_significance_first]]'s warning not to read the rank alone; see
`Chapter-5.tex:sec:mlpqf-backbone-generality` for the full table and
discussion.

`nhits_laplace` on MLVS-PT was the last missing cell in this table — **fixed
2026-09-16**: a further targeted rerun (`--group benchmark_nhits --dataset
all --skip-hpo`, run `20260916_070152_034c8d`) completed all four
`benchmark_nhits` conditions on both datasets. Since that rerun retrained
all four conditions together, not just the missing one, the other three
N-HiTS rows shifted slightly from the numbers the table above was originally
built on; `tab:backbone-generality` and its discussion in `Chapter-5.tex`
have been rewritten from this run and the table above (**Findings**) reads
from the current `ch5_fpqr_full.csv`. `analyse_results.py` picks up this
run automatically (latest by timestamp per group).
`tau_embed_type="TimeEmb"` (the reference) beats `LinearEmb`/`ConvEmb` by 9 to
74%. The full objective (pinball + entropy + crossing terms) beats every
alternative (Huber at any kappa, Winkler-loss, dropping entropy or crossing)
by 87 to 496%, not a close call.

**Findings that disagree by dataset (real hyperparameter/data interaction,
not a bug: `proposal_type`, `rope_type`, `lambda_cal`, and `n_quantiles`
beyond 23 are all correctly forwarded to the model, confirmed directly against
`MLPFFPQRModel._init_model()`'s kwarg list).** `proposal_type="softmax"` wins
clearly on MLVS-PT but loses to both alternatives on MLVS-UK. `horizon_shared`
is 1.3% worse on MLVS-PT, 7.5% better on MLVS-UK. `rope_type` alternatives
help on MLVS-UK, hurt on MLVS-PT. None of these are baked into `best_config`
above for exactly this reason.

**`qvn_monotone`/`qvn_monotone_cal`'s catastrophic regression (-57% to -238%)
is grounded in real, cited theory** (`MonotonicQVN`, cumulative-softplus
non-crossing construction, Brando et al., AISTATS 2022, correctly implemented
and cited in `twiga/distributions/nn/fpqr/qvn.py`), **not obviously the same
class of bug as the three above.** The diagnostic pattern rules out "safely
conservative but wide": PICP is *lower* than the standard QVN's (0.85 vs.
0.98 on MLVS-UK) and NMPI is *narrower*, not wider, with strongly negative
ACE, under-covering with a tight interval, the signature of an under-fit,
harder-to-optimise head rather than an architecturally-forced wide one. That
is a documented practical risk of hard-monotonicity-constraint QVNs in this
literature family, plausible as a real finding rather than a plumbing defect,
though a longer training budget was not tested here and would be the next
check before citing the magnitude specifically. `qvn_monotone` was also never
run at all on MLVS-PT (present on MLVS-UK only, 0 of 10 folds present);
worth confirming whether that run failed outright before re-running the group.

## Three new monotone QVN architectures

Added to `twiga/distributions/nn/fpqr/qvn.py` and wired through as three new
`qvn_type` values, each answering `qvn_monotone`'s under-fitting failure
differently. Three new `qvn_type` conditions, `qvn_lattice`/`qvn_spline`/
`qvn_umnn`, are in the `qvn_type` group above alongside the four already
there.

- **`qvn_lattice`** (`LatticeQVN`, Cannon, 2018, MCQRNN). Guarantees
  monotonicity via non-negative weight reparameterisation on the path from
  the quantile level to the output, not a cumulative sum over slots, so
  every slot is evaluated independently and in parallel rather than
  sequentially. Cheapest of the three to train (a single forward pass, no
  extra evaluations). The guarantee requires the context fed into its
  unconstrained path to carry no information about the quantile level
  itself; confirmed by testing that FPQR's usual fused `combined` embedding
  breaks this (already has tau fused in via the multiplicative gate
  upstream), so `LatticeQVN` consumes the backbone's raw `context_features`
  directly instead, the same tau-independent vector `AttentionQVN` uses.
- **`qvn_spline`** (`SplineQVN`, harmonic-mean derivative smoothing over
  adaptive knots, following Fritsch & Carlson, 1980's monotone piecewise-cubic
  construction). The genuinely novel piece: existing spline quantile
  functions (Gasthaus et al., AISTATS 2019, used in GluonTS/DeepAR)
  interpolate between fixed, regularly-spaced knots; this combines the same
  monotone-derivative-smoothing idea with FPQR's own adaptively *proposed*
  knot positions, which no published spline-QF work does. Same per-slot
  sequential-integration cost profile as `MonotonicQVN`.
- **`qvn_umnn`** (`UMNNQVN`, Wehenkel & Louppe, NeurIPS 2019). Constrains only
  the derivative to be positive (a single softplus), then integrates it via
  Gauss-Legendre quadrature (default 8 points per segment, `umnn_quad_points`
  in `MLPFFPQRConfig`) rather than constraining the map itself; most directly
  targets the diagnosed under-fitting mechanism, at the highest compute cost
  of the three (quadrature evaluations per segment). The quadrature context is
  the segment's lower-endpoint embedding, held fixed across quadrature nodes,
  a discrete stand-in for UMNN's usual continuously-varying context, since
  FPQR's tau embedding only supplies per-slot, not per-tau, context vectors.

All three verified end to end (`tests/unit/distribution/test_qvn.py`,
40 tests): correct output shape, strictly monotone output (`LatticeQVN`
checked over 20 random `(context, tau)` draws, since its guarantee is
per-input rather than architecturally trivial to see by inspection), and
gradients reaching the input. Not yet run through the actual ablation; see
**Next steps** below.

## Run order

`s563` needs `s561`'s checkpoints. `s566` and `s564` are independent and can run
in parallel. `s562` is the highest-value single experiment: it is the control the
chapter's central claim requires, and it is one configuration.

## How to run

### Prerequisites

- Run from the repository root with `uv run`, or from inside `experiment/` with
  the virtual environment active. `fpqr_experiment.py` inserts `experiment/`
  onto `sys.path` itself, so no package installation is required.
- Datasets: `MLVS-PT` and `MLVS-UK`, both expected at `examples/data/*.parquet`.
  These are the same files Chapter 7 uses; see
  [`chapters/ch6_conformal/README.md`](../chapter-7-conformal/README.md#datasets)
  for how to obtain them if a checkout does not already have them.

### Bootstrap: seed the backbone HPO cache first

`fpqr_experiment.py` runs on `twiga.experiment.engine.ExperimentEngine`, the
same shared engine Chapter 7's scripts use. Its Phase 1 tunes the backbone
(`MLPFConfig`, `SPEC.backbone_cls`) once per dataset via Optuna and caches the
result under the engine's own `backbone_hpo/<dataset>/` directory; Phase 2
(every ablation group/condition) then reuses that cache unless `--skip-hpo`
forces it to skip HPO and fall back to config defaults. This mirrors
`crc_experiment.py`'s `seed_mlpgam_backbone_params`/`load_backbone_params`
pattern in Chapter 7 exactly (see that chapter's own README, "Next: run
Experiment 2, then Experiment 3"), but was not previously stated here
explicitly, which matters concretely: nearly every command in this section
passes `--skip-hpo`, and a `--skip-hpo` run against a dataset whose cache was
never populated does not error, it silently trains on un-tuned defaults. If a
condition's results look unstable (e.g. FPQR on MLPF), confirm the cache was
actually seeded for that dataset before suspecting the ablation itself.

Run once per dataset, **without** `--skip-hpo`, before any other group.
`--dataset all` runs both datasets in one process, since the engine loops
over `dataset_keys` and caches Phase 1 HPO separately for each:

```bash
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group grid --dataset all
```

`grid` is used here only because it is the smallest group (one condition);
any group works, since Phase 1 HPO tunes the shared backbone before Phase 2
runs the group's own conditions, and only needs to happen once per dataset.
Every subsequent group/condition on the same dataset can then safely add
`--skip-hpo` to reuse the cached backbone rather than re-tuning it.

### `fpqr_experiment.py`: every experiment except `s563`

`fpqr_experiment.py` implements the `grid`, `n_quantiles`, `loss_terms`,
`proposal`, `tau_embed`, `num_cosines`, `rope_type`, `crossing`, `sort_output`,
`qvn_type`, `loss_fn`, `lambda_cal`, `prop_nq`, `stop_gradient`, `prop_loss`,
`horizon_sharing`, `best_config`, `benchmark`, and `benchmark_nhits` groups
described above (`s561`/`s562`/`s565`/`s566` and the `horizon_sharing` check
all live here; see the module's own docstring for the full group-to-hypothesis
mapping).

```bash
# List every group and flag:
uv run python chapters/ch5_fpqr/fpqr_experiment.py --help

# Everything, both datasets:
uv run python chapters/ch5_fpqr/fpqr_experiment.py

# One group only, for example the fixed-grid control (s562):
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group grid

# The benchmark replication (s561), one dataset:
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group benchmark --dataset MLVS-PT

# Several groups at once, e.g. both benchmark variants:
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group benchmark benchmark_nhits

# Reuse cached backbone HPO parameters instead of re-searching:
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group grid --skip-hpo

# Quick smoke-test with fewer epochs and folds:
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group grid --epochs 2 --folds 2
```

`--group`/`--dataset` both accept `all` (the default) or a space-separated
list. Output is written under `../experiments/chapter-5-fpqr/` (a sibling of
the repository root, matching Chapter 7's convention — kept out of the
tracked tree since these runs regenerate large per-fold CSVs and checkpoints),
one CSV per
group per dataset, following the `s5<section><n>_*.csv` naming convention.

### `learned_tau.py` and `tau_matrix.py`: `s563`, single-split diagnostics

These are not on the `ExperimentEngine` (see **What the engine does not do**
above) and run as plain scripts:

```bash
uv run python chapters/ch5_fpqr/learned_tau.py
uv run python chapters/ch5_fpqr/tau_matrix.py
```

Both need `s561`'s checkpoints already trained, since they read a fitted
MLPQF model rather than retraining one.

### `worked_example.py`: pipeline-figure intermediates, already run

Also not on the `ExperimentEngine`. Captures every intermediate signal of
one MLPQF forecast (input window, GHI, learned quantile fractions and
values, realised net load) for the pipeline figure, on the day chosen as the
median-CRPS day among the most variable third by within-day irradiance --
representative rather than flattering.

```bash
uv run python chapters/ch5_fpqr/worked_example.py
```

Needs `s561`'s checkpoints, same as `learned_tau.py`/`tau_matrix.py` above.
Already run: writes `s563_worked_example.npz`, present in this directory.

## Next steps

As of 2026-09-16, `results/chapter_5_fpqr/runs/` has eight run directories
covering every group at least once. Verified programmatically this session
(every (group, condition, dataset) triple checked for 10/10 fold coverage):
of 19 groups / 84 conditions, only `qvn_type`'s `qvn_spline`/`qvn_umnn` are
still genuinely missing (item 2 below) — everything else, including the
`benchmark_nhits` gap item 5 used to flag, is complete.

1. **`nrmse` throughout every run here predates a 2026-09-17 fix.**
   `twiga.core.metrics.point.nrmse()` is called per day (matching this
   chapter's own stated methodology) and its default `scale_method` was
   `"max"` (normalises by that day's maximum target value), silently
   diverging from this chapter's own documented normalisation and from the
   legacy `PhD-Thesis-Anthony` reference implementation, both of which
   normalise by the *range* (max minus min) instead. Fixed in
   `twiga/core/metrics/point.py` (default now `"range"`).
   **Validated exactly, not estimated**, for the two groups whose reruns
   retained per-timestamp predictions (`grid`, `benchmark_nhits`): recomputing
   `nrmse` under the corrected normalisation directly from
   `conditions/<dataset>/<group>/<condition>/predictions.parquet`
   reproduces the existing `ch5_fpqr_full.csv` values exactly under the old
   `"max"` method (confirming the reconstruction is correct), and the
   corrected `"range"` values are **larger**, not smaller, typically by
   50-60% on both datasets -- a single day's range is usually narrower than
   its maximum, since most days do not reach the negative net-load
   excursions (on-site generation exceeding demand) that widen the
   whole-period range this section's own methodology describes.
   `benchmark` (the \ac{MLPF} backbone -- `tab:mlpqf-probabilistic-heads`
   and half of `tab:backbone-generality`) did not retain per-timestamp
   predictions, so it cannot be corrected this way and needs an actual
   retrain. Chapter-5.tex's tables were **not** patched with the corrected
   `grid`/`benchmark_nhits` numbers, since half of `tab:backbone-generality`
   (the `benchmark` rows) can't be corrected the same way yet, and mixing
   normalisations within one table would corrupt exactly the cross-backbone
   comparison that table exists to support -- worse than leaving it
   uniformly on the old normalisation. **No other metric is affected** --
   CRPS, Winkler score, CWE, PICP, NMPI and calibration error are all on
   the raw target scale, and no significance conclusion in this chapter is
   decided on NRMSE. Pending, to correct every table consistently:
   ```bash
   uv run python chapters/ch5_fpqr/fpqr_experiment.py --group benchmark benchmark_nhits grid --dataset all --skip-hpo
   ```
2. **A real bug in the shared `pairwise_dm_table`'s `significant` column was found and
   fixed 2026-09-17, and every group's significance table below has
   already been regenerated against the fix -- this item is closed, not
  pending.** `experiment/common/significance.py`'s
   `significant = bool(p_adjusted < alpha)` only ever tested one direction
   of the one-sided Diebold-Mariano test (a raw p-value near 1, meaning
   `condition_a` is the significantly better side, was silently reported
   as not significant). Fixed by symmetrising
   (`2 * min(p, 1 - p)`) before Benjamini-Hochberg correction. This
   flipped several conclusions already written into `Chapter-5.tex`,
   notably: MLPQF is significantly better than QR on Winkler score in
   three of the four (dataset, backbone) pairs in the backbone-generality
   comparison, not the one pair the confidence set alone had suggested.
   Chapter 7's own use of the same function has not been fully
   re-audited -- one spot-checked claim (`best_config` vs
   `best_config_v1` on MLVS-PT) is confirmed safe, but its other tables
   are not yet re-verified.

3. **`prop_loss`/`stop_gradient`/`horizon_sharing`: done.** Rerun on
   2026-09-14 (`runs/20260914_110320`), after the bug that made most of
   their conditions silently no-op was fixed
   (`twiga/models/nn/mlpffpqr_model.py`, `twiga/models/nn/prob/mlpffpqr.py`).
4. **`qvn_type`: partially done, corrected 2026-09-18.** The group now has
   eight conditions, not seven (`qvn_attention` added since this item was
   last written). The latest run (`runs/20260918_002159_e7736d`) covers
   five of them on both datasets (`qvn_relu`, `qvn_softplus`,
   `qvn_monotone`, `qvn_monotone_cal`, `qvn_lattice`) — `analyse_results.py`
   output for these five is complete
   (`results/analysis/{MLVS-PT,MLVS-UK}_qvn_type_{metrics,significance,kupiec}.csv`).
   **`qvn_attention`/`qvn_spline`/`qvn_umnn` trained (TensorBoard logs exist
   for all three, both datasets) but produced no `predictions.parquet` and
   no results row** -- this run started `2026-09-18T00:21:59`, ~1h45m
   *before* the `get_quantile_metrics` mixed-shape fix
   (`c87e92f`, `02:07:35`; see `chapters/ch7_multitarget/README.md`'s
   2026-09-18 entry for the original diagnosis, found there first on FPQR's
   Kumaraswamy proposal). These three conditions' quantile-level structure
   (attention's cross-attention taus, spline/umnn's continuous
   parameterisations) plausibly hits the exact same "no symmetric quantile
   pairs" -> mixed scalar/array crash the fix addresses; a rerun after it
   should resolve them without further changes. Pending:
   ```bash
   uv run python chapters/ch5_fpqr/fpqr_experiment.py --group qvn_type --dataset all --skip-hpo
   ```
   Re-running the five already-complete conditions alongside the three
   missing ones is redundant but harmless, and keeps all eight from one run
   for a clean comparison.

   **Significance on the five complete conditions (DM, BH-corrected, per
   `analyse_results.py`)**: on both datasets, `qvn_relu`/`qvn_softplus`
   (unconstrained, crossing-penalty QVN) each significantly beat all three
   monotonicity-constrained variants (`qvn_monotone`, `qvn_monotone_cal`,
   `qvn_lattice`) on Winkler score; `qvn_relu` vs. `qvn_softplus` itself is
   not significant on either dataset. `qvn_lattice` is the single worst
   condition on both datasets by every metric (NRMSE, CRPS, Winkler score),
   confirming this session's memory note that it underfits -- now with DM
   backing, not just the raw percentages. See the `*_significance.csv`
   files above for the full pairwise table.
5. **`best_config`: done, three times.** Reran on 2026-09-14 (twice) and
   2026-09-15 against the updated definition (`n_quantiles=23`,
   `crossing_penalty=100`, `crossing_mode="softplus"`), with genuinely
   different results each time (e.g. Winkler score 41.52 -> 39.73 on
   MLVS-PT across the three) — the **latest** run
   (`runs/20260915_054803`) is the one to read.
6. **`grid`: also refreshed** on 2026-09-15 (`runs/20260915_054854`),
   `fpqr` vs. `qr_fixed`, both datasets, complete — not something this list
   previously asked for, but a legitimate rerun.
7. **`benchmark`/`benchmark_nhits`: done.** `benchmark_nhits` reran in full
   on 2026-09-16 (`runs/20260916_070152_034c8d`, `--group benchmark_nhits
   --dataset all --skip-hpo`); `nhits_laplace` on `MLVS-PT` now has 10/10
   folds, closing the gap `Chapter-5.tex`'s `tab:backbone-generality` used
   to flag as `n/a`. `benchmark` itself is unchanged from the original
   2026-09-13 run — no gap there, nothing pending.
8. **Read the per-epoch TensorBoard scalars, not just the text log, if any
   condition's sample plot looks off.** `fpqr_experiment.py`'s own run log has
   no per-epoch loss; `logs/<condition>/.../lightning_logs/backtesting_fold_*/
   events.out.tfevents.*` does. This is what distinguished a real training
   collapse from a y-axis-compression artefact during the equivalent Chapter 7
   audit, and the same check applies here if any condition's plot shows
   unexplained spikes.
9. **`s563`/`s564`/`s565`/`s566`/`s567` need no further action** beyond what
   **Findings** above already covers; their groups showed no duplicate-row
   pattern in the completed run.
10. **Per-timestamp predictions: `benchmark_nhits` already has them,
   `benchmark` just needs a plain rerun -- no new script.** Corrected
   2026-09-17: this item previously called for a bespoke new capture
   script, which was wrong. `twiga.experiment.spec.ExperimentSpec` already
   defaults `save_predictions=True`, and `engine.py` already writes
   `conditions/<dataset>/<group>/<condition>/predictions.parquet`
   (`timestamp`, `Model`, `forecast`, `Actual`, `lower`, `upper`, `Folds`,
   ...) whenever that flag is on. `benchmark_nhits`'s 2026-09-16 rerun
   (`runs/20260916_070152_034c8d`) already has this file for all four
   conditions on both datasets; `benchmark`'s original 2026-09-13 run
   simply predates `save_predictions` being added to the engine, which is
   the only reason it's missing there. Fixed by rerunning `benchmark` with
   the current codebase, same invocation as every other group here:

   ```bash
   uv run python chapters/ch5_fpqr/fpqr_experiment.py --group benchmark --dataset all --skip-hpo
   ```

   This one command (plus `benchmark_nhits`'s file, already on disk) covers
   three of the four things the analysis plan (2026-09-16) needs:

   * the example-forecast figure (one panel per condition -- `Actual` vs.
     `forecast`/`lower`/`upper` by timestamp);
   * `christoffersen_test` (temporal independence of exceedances) and
     `stratified_coverage_test` by `season` (both derivable from the
     `Actual`-within-`[lower, upper]` hit sequence; `ch5_fpqr_full.csv`
     already has `season` to join on);
   * **any future per-condition prediction-band plot** for the `benchmark`
     and `benchmark_nhits` groups' four heads (`mlpqf`/`fpqr`, `qr`, `bnn`,
     `laplace`) -- the same file serves plotting and the calibration tests,
     not two separate asks.

   **`berkowitz_test`/`pit_uniformity_test` are the one thing this does
   not cover**, and still need a real new capture: `predictions.parquet`
   only carries the reduced point/interval form (`forecast`/`lower`/`upper`
   at the nominal PI), not the full learned quantile array
   (`quantile_hats`/`tau_hats`) a PIT value needs. Getting that means
   bypassing the standard reduced-prediction path the engine returns from
   `forecaster.backtesting`/`_run_folds`, the way `worked_example.py`
   already does for its one representative day -- extended to every test
   timestamp, for `benchmark`/`benchmark_nhits`. **Not started for
   `benchmark`/`benchmark_nhits`**, but **done for the `qvn_type` group**
   specifically (2026-09-18, `qvn_pit.py`, below) -- the same
   forward-hook capture, generalised across all eight `qvn_type`
   mechanisms rather than one condition on one day, since
   `FPQRDistribution.quantile_proposal`/`.quantile_output` (confirmed
   directly against `twiga/distributions/nn/fpqr/distribution.py`) are set
   under every `qvn_type` branch, not just the default one
   `worked_example.py` targets.

   **Real bug found and fixed while building this**:
   `twiga.core.plot.plot_pit_histogram` silently rendered zero bars (no
   error, no warning, just an empty panel below the reference line) once a
   histogram had enough bars (20, the function's own default) and passed
   `geom_bar` an explicit float `width` -- confirmed directly and
   independent of this script's data: a hand-built 3-bar synthetic example
   with `width=0.05` renders fine, the same call at 20 bars with
   `width=0.046` renders nothing. Fixed by dropping the explicit `width=`
   entirely (lets-plot's own inferred per-bar width from the data's
   x-spacing renders correctly at both bar counts); `bar_width_factor`
   removed from the function's signature since there is no way left to
   honour it. Affects every caller of `plot_pit_histogram`, not only this
   script.

   ```bash
   # All eight qvn_type conditions, MLVS-PT (default):
   uv run python chapters/ch5_fpqr/qvn_pit.py

   # One dataset, a subset of conditions:
   uv run python chapters/ch5_fpqr/qvn_pit.py --dataset MLVS-UK --conditions qvn_relu qvn_softplus
   ```

   Illustrative, not a faithful reproduction of the formal backtest: one
   freshly-trained model per condition on `learned_tau.py`'s own `SPLITS`
   (a single train/test window), not the 10-fold rolling-origin protocol
   `fpqr_experiment.py`'s `qvn_type` group actually ran under -- read the
   PIT/reliability shapes as diagnostic of each mechanism's calibration
   behaviour, not as a substitute for `analyse_results.py`'s per-fold
   Winkler-score comparison. Output (`.npy` PIT arrays, `.npz` forecast
   captures for reuse, PIT histogram + reliability diagram PNGs,
   `pit_summary.csv`) lands under `chapters/ch5_fpqr/qvn_pit/<dataset>/`.

   **`qvn_forecast_examples.py` (2026-09-18)** is the companion good/bad-day
   view: what a forecast from each `qvn_type` mechanism actually looks like,
   not just whether it is calibrated on average. Unlike `qvn_pit.py`, this
   one needs no retraining and no new metric computation -- `predictions.parquet`
   already carries exactly `forecast`/`lower`/`upper`/`Actual`, and
   `metrics.parquet` already has per-day NRMSE/CRPS/Winkler, both written by
   `fpqr_experiment.py`'s own formal 10-fold backtest, so this script only
   reads them. One day is picked once, pooled over every condition with a
   completed backtest (mean per-day NRMSE), so every panel shows the same
   day and the comparison is fair -- the same principle
   `chapter-6-multitarget/forecast_examples.py` uses for its own best/worst
   day, and `draw()`/`panel_metrics()` from that script were considered for
   reuse here too, but its per-target grid convention (`levels` shape
   `(L, D)`) did not look like it would fit this single-target case cleanly,
   so this script instead reuses `twiga.core.plot.plot_forecast` directly
   against `predictions.parquet`'s own column names. **Update (2026-09-19)**:
   it does fit after all -- `chapter-7-conformal/mlpgam_forecast_examples.py`
   now calls the same `draw()` (relocated, unchanged, to
   `experiment/common/forecast_panels.py`) for exactly this single-interval
   case, by stacking `[lower, forecast, upper]` as a degenerate 3-level fan
   against `levels=[alpha/2, 0.5, 1-alpha/2]`; `draw()` resolves the median to
   the middle level and draws one band from the outer two. Migrating this
   script to the same shared `draw()` (dropping `twiga.core.plot`'s lets-plot
   rendering in favour of the thesis's matplotlib convention) is a candidate
   follow-up, not done here.

   ```bash
   uv run python chapters/ch5_fpqr/qvn_forecast_examples.py
   uv run python chapters/ch5_fpqr/qvn_forecast_examples.py --dataset MLVS-UK
   ```

   Only 5 of the 8 conditions have a completed formal backtest on MLVS-PT
   (`qvn_spline`/`qvn_umnn`/`qvn_attention` never finished one -- see item 4
   above); the script prints which are skipped rather than silently
   dropping them. Output (`good_<condition>.png`, `bad_<condition>.png`,
   `overview_{good,bad}_day.png`, `forecast_examples_summary.csv`) lands
   under `chapters/ch5_fpqr/qvn_forecast_examples/<dataset>/`.

   **Generalised to any group via `--group`** (2026-09-18,
   `GROUP_CONDITIONS`) -- e.g. `--group proposal` for the FPN/proposal-network
   ablation (`prop_softmax`/`prop_horizon`/`prop_kumaraswamy`). Not every
   group can produce panels this way, though: `proposal` was run 2026-09-13,
   before `save_predictions` existed on the engine (the same gap already
   documented above for `benchmark`'s original run), so no
   `predictions.parquet` was ever written for any of its three conditions --
   confirmed directly, no such file exists under any run directory for that
   group -- and the script reports all three missing and exits rather than
   drawing an empty panel. Getting panels for `proposal` needs a rerun:
   `uv run python chapters/ch5_fpqr/fpqr_experiment.py --group proposal --dataset all --skip-hpo`.

   **Finding**: on MLVS-PT, `qvn_lattice` predicts an almost perfectly flat
   line on *both* the picked good day and the picked bad day -- visibly
   failing to condition on the input at all, not just tracking it noisily.
   This sharpens the "`qvn_lattice` underfits" conclusion already in this
   README (item 3, `qvn_type` findings) from an aggregate metric into a
   visible failure mode: the lattice head's forecast is close to the
   unconditional mean, and it "wins" the picked bad day's NRMSE only
   because the other conditions' more responsive forecasts happened to
   misjudge that particular day's direction more.

   **`qvn_type_comparison.py` (2026-09-18)** joins `analyse_results.py`'s
   three separate `qvn_type` CSVs per dataset (`_metrics`, `_significance`,
   `_kupiec`) into one table -- NRMSE/CRPS/Winkler/CWE-stratified/PICP/NMPI/
   calibration-error, the Borda-count ranking, a DM significant win/loss
   count per condition, and the Kupiec calibration flag, for both datasets
   side by side, plus four quantile-specific metrics `analyse_results.py`'s
   own `METRIC_COLS` does not carry (pinball loss, weighted quantile loss,
   sharpness, tail pinball loss), read from each condition's own
   `metrics.parquet` (`metric_type == "quantile"` rows) rather than
   `analyse_results.py`'s `load_csv` -- that path reads only the single
   most-recently-timestamped run covering the whole group, which silently
   drops `qvn_monotone` on MLVS-PT (it only completed in an older run than
   the one currently latest). Otherwise recomputes nothing; only reads and
   joins.

   ```bash
   uv run python chapters/ch5_fpqr/qvn_type_comparison.py
   uv run python chapters/ch5_fpqr/qvn_type_comparison.py --group proposal
   ```

   Output: `results/analysis/{group}_comparison_both_datasets.csv`.

   **Finding: the Borda ranking and the DM significance table disagree on
   MLVS-UK**, which is exactly the case this repo's "significance table
   first, not avg-rank" convention exists for. `qvn_monotone` has the
   *highest* `weighted_borda` on MLVS-UK (3.68, top of the table) despite
   worse NRMSE than `qvn_relu`/`qvn_softplus` -- its CRPS and Winkler score
   happen to be very low, which is enough to win the blended rank. But the
   DM table shows it significantly *losing* 3 of its 4 pairwise comparisons
   and winning only 1: the blended rank is not evidence that `qvn_monotone`
   is actually the best MLVS-UK condition, it is evidence that a single
   composite rank can be misled by a metric or two. `qvn_softplus`
   (4 significant wins, 0 losses on MLVS-UK; 3 wins, 0 losses on MLVS-PT) is
   the condition the significance table actually supports as best on both
   datasets, consistent with the DM-significance finding already documented
   above (relu/softplus beat every monotonicity-constrained variant). No
   condition is Kupiec-calibrated on either dataset (`calibrated == False`
   throughout both `_kupiec.csv` files) -- a separate, already-known finding
   this table did not change, just re-surfaced alongside the others.

   **`--group proposal` finding: unlike `qvn_type`, none of the three
   proposal-network mechanisms differ significantly from each other, on
   either dataset.** Every one of the 3 pairwise DM comparisons on both
   MLVS-PT and MLVS-UK has `significant == False`
   (`results/analysis/{MLVS-PT,MLVS-UK}_proposal_significance.csv`) --
   `dm_significant_wins`/`dm_significant_losses` are 0 for every condition
   in `proposal_comparison_both_datasets.csv`. The metrics themselves do
   differ (e.g. Winkler score 47.5-58.6 on MLVS-PT), but not by more than
   fold-to-fold noise can explain: whether the proposal network uses a
   softmax simplex, a Kumaraswamy tail-adaptive distribution, or conditions
   on the forecast horizon does not measurably matter for this backbone,
   in contrast to how sharply the `qvn_type` (quantile-*value*) mechanism
   choice does.

11. **New, not yet built: re-run the \ac{MLPQF} vs. FPQR-MLP comparison
    (Figures 5.11/5.12, `fig:mlpfq-benchmark`/`fig:experiment_four_complexity`)
    natively in `twiga` instead of the legacy `PhD-Thesis-Anthony` codebase.**
    That comparison currently reruns both models under the original,
    pre-`twiga` implementation (`chapters/ch5_fpqr/results/results-fpqr-benchmark`,
    `MLPFPQ`/`DNNFPQ` — see `Chapter-5.tex`'s own caveat that this and
    `tab:mlpqf-probabilistic-heads` "should be interpreted within their
    respective evaluation settings rather than compared directly"). A native
    `twiga` rerun of both models — `mlpqf` against a new `fpqr_mlp`-style
    condition using the `FPSe2Q`-\ac{MLP} architecture, same 10-fold
    expanding-window protocol as `benchmark` — would let Figures 5.11/5.12
    sit on the same basis as `tab:mlpqf-probabilistic-heads` and
    `tab:backbone-generality`, closing that caveat and dropping the
    legacy-codebase dependency (and its pandas-1.5.3-pickle/`ppscore` shim)
    entirely. Not started; needs a new condition definition in
    `fpqr_experiment.py` for the FPQR-MLP predecessor architecture, since the
    current `benchmark` group only has `mlpqf`/`qr`/`bnn`/`laplace`, not the
    `FPSe2Q`-\ac{MLP} predecessor itself.

**Note on where results now live**: the current, complete set of runs is
under `results/chapter_5_fpqr/runs/*/results/ch5_fpqr_full.csv` (one
directory per run, picked by timestamp), not the single flat
`results/ch5_fpqr_full.csv` file earlier revisions of this README pointed
to. `analyse_results.py` (below) reads the latest run that covers each
group automatically; do not read `ch5_fpqr_summary.csv` from any individual
run directly, since it pools `MLVS-PT`/`MLVS-UK` into one number per
condition rather than reporting the two datasets separately.

```bash
# The one remaining ExperimentEngine gap (qvn_spline/qvn_umnn never run --
# Chapter 6 scope, not Chapter 5; benchmark_nhits is closed as of 2026-09-16):
uv run python chapters/ch5_fpqr/fpqr_experiment.py --group qvn_type --dataset all --skip-hpo
```

### `analyse_results.py`: metrics, significance, and figures across every completed group

Consolidates every run under `results/chapter_5_fpqr/runs/` (picking the
latest run that covers each requested group, so `best_config`/`grid` come
from their newest reruns rather than the original 2026-09-13 run), and
reports, per `(dataset, group, condition)`: NRMSE, CRPS, Winkler score, CWE
stratified, PICP, and NMPI (mean +/- sd across folds), plus a per-fold
average rank across all six. Shared fold aggregation, metric ranking, DM/MCS,
coverage tests, and comparison figures are implemented in
`experiment/common/` and `twiga.experiment`. Figures include a multi-metric
forest overview, per-metric condition bars, an adjusted-p significance matrix,
cross-dataset metric comparisons, a benchmark-only computation-time comparison,
and a best-vs-worst-day comparison from `ch5_fpqr_representative.csv`.

```bash
# Everything that has complete data (skips qvn_umnn/qvn_spline
# automatically -- see the coverage table it prints first):
uv run python chapters/ch5_fpqr/analyse_results.py --group all

# One group:
uv run python chapters/ch5_fpqr/analyse_results.py --group benchmark benchmark_nhits
```

Output under `chapters/ch5_fpqr/results/analysis/`.
