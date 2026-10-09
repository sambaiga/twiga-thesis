# Chapter 6: analysis plan for the results already on disk

Scope: this is the post-hoc analysis step, not a plan for what else to run.
`EXPERIMENT_PLAN.md` covers data generation and the methodology audit;
`POSITIONING.md` sets the T1-T7 test plan and the objectives every result is
checked against; `README.md` tracks which scripts have and have not been
run. This document covers what to do with the two runs that already exist
(`multitarget_experiment.py`'s 2026-09-18 run and `fpqr_gating_ablation.py`'s
2026-09-18 run) and where each item on the T1-T7 list stands once those
results are read.

## 1. The convention this inherits

Chapters 5, 6, and 7 use common analysis owners: `twiga.core.metrics.result_analysis`
provides fold summaries, rankings, significance orchestration, and coverage
tables; `twiga.core.metrics (pairwise_dm_table, mcs_ranking)` provides DM/MCS tests;
`twiga.core.metrics.comparison_plots` provides thesis-styled forest, bar,
normalized, best/worst, and significance-matrix figures. Run discovery remains
in `twiga.experiment.run_catalog`. Each chapter retains its result adapters,
metric definitions, references, and interpretation.

The rule that matters most, per [[feedback_significance_first]]: **CWE never
decides a comparison.** It has no per-observation loss differential to feed
a Diebold-Mariano test, and is reported only as a diagnostic alongside PICP
and nMPI. A blended Borda rank is reported for orientation, but the verdict
comes from the DM/MCS table, checked directly.

Chapter 6 already commits to this in writing -- `multitarget_experiment.py`'s
own docstring names `pairwise_dm_table`/`mcs_ranking` and states the
CWE-never-decides rule -- and `fpqr_gating_ablation.py` already has a working
`significance_report()` that reuses `significance.py` verbatim. What is
missing is running it against the results that exist, and giving
`multitarget_experiment.py` the same capability.

## 2. What the two completed runs currently produce

| script | groups (run 2026-09-18) | dataset | current output | what's missing |
|---|---|---|---|---|
| `multitarget_experiment.py` | `benchmark`, `grid`, `conformal` | MLVS-PT-3PHASE only | `ch6_benchmark_{full,seasonal,summary,representative}.csv`; summary has mean+std and `Delta <metric> vs ref (%)` per condition | no `metric_table`-style Borda rank, no `dm_and_mcs`, no `kupiec_table`. There is no `--report` flag at all -- unlike `fpqr_gating_ablation.py`, this script cannot produce a significance table today. |
| `fpqr_gating_ablation.py` | `proposal`, `qvn`, `grid`, `multigate`, `head` | MLVS-PT-3PHASE only | same four-CSV shape, plus a working `significance_report(full_csv, group)` | the flag exists but has not been run against this run's output for any of the five groups -- do this first, it costs nothing. |

Both runs cover MLVS-PT-3PHASE only; DSS-EU is not yet in either. Any table
built now is a single-dataset result and should say so.

## 3. The three-step build

**Step 1 -- run what already works.** `fpqr_gating_ablation.py --report
<group>` for each of `proposal`, `qvn`, `grid`, `multigate`, `head` against
the 2026-09-18 run. Zero new training, this just reads `ch6_fpqr_gating_full.csv`.
Do this before writing any new code, since it may already answer several of
Chapter 6's own ablation questions.

**Step 2 -- give `multitarget_experiment.py` the same capability.** Add a
`significance_report(full_csv, group)` to `multitarget_experiment.py`,
mirroring `fpqr_gating_ablation.py`'s function rather than writing a new one:
same run-discovery, same `fold_level`/`dm_and_mcs`/`kupiec_table` calls from
`chapter-5-fpqr/analyse_results.py`, parameterised on `benchmark`, `grid`,
`conformal`. This closes the literal gap between what the docstring already
promises and what the script can do. Once in place, run it against the
existing `ch6_benchmark_full.csv` for all three groups -- again, no
retraining needed.

**Step 3 -- the `head` group's missing proper score.** `fpqr_gating_ablation.py`'s
`head` group compares `head_fpqr` against `head_joint_parametric`, and per
`EXPERIMENT_PLAN.md` SS10.8 neither the engine's aggregate path nor
`ch6_fpqr_gating_full.csv` carries a joint proper score: `MultivariateNormalDistribution.get_log_likelihood`
exists but its `forecast()` output does not carry `log_likelihood` through, so
the parametric evaluator falls back to marginal `normal_nll`, and the DM
table Step 2 produces for `head` will therefore only ever compare marginal
skill -- exactly the framing `POSITIONING.md` argues is the wrong one for
this comparison. `energy_score` is already implemented
(`twiga/core/metrics/parametric.py`) and already used, ad hoc, in
`joint_vs_independent.py`, `multitarget_bakeoff.py`, and
`forecast_examples.py`'s joint-tier metrics (`metrics_all_tiers.csv`), which
is also where `winkler_dm_table.csv` already runs a DM/MCS table across many
conditions including joint vs independent heads.

Two options, not mutually exclusive:

- **(a) now, no engineering.** Cite `forecast_examples.py`'s
  `metrics_all_tiers.csv`/`winkler_dm_table.csv` for the head-comparison
  claim, footnoted as coming from the standalone script's sample-based joint
  tier, not from the engine-driven `head` ablation. This is already run and
  already contains the 1.34% energy-score finding `POSITIONING.md` discusses.
- **(b) later, real fix.** Wire `energy_score` (or the joint log-likelihood,
  once `forecast()` carries it) into the engine's aggregate path
  (`twiga/experiment/analysis.py`'s `PARAMETRIC_COLS`) so the `head` group's
  own DM/MCS table can use a proper multivariate score directly, rather than
  reasoning across two different scripts' outputs. This is write-path work,
  not analysis, and should be scoped separately from this plan.

Recommend (a) for the current write-up and defer (b); it is already flagged
as open in `EXPERIMENT_PLAN.md` and re-deriving it here would duplicate that
record.

## 4. Where this leaves the T1-T7 test plan

| item | what it needs | status given what's on disk |
|---|---|---|
| T1, US thermal exceedance | `operational_decision.py` DSS-US run | **not started.** DSS-US has never been run (README, 2026-09-15). This is `POSITIONING.md`'s own "cheapest decisive test" and is the highest-priority run outstanding, ahead of anything in this document. |
| T2, EU voltage tail | `operational_decision.py` DSS-UK run + reliability/resolution against realised frequency | **largely done.** DSS-UK has a real run: `dss-uk_{per_fold,summary,significance,value_curve}.csv` plus `reliability_operational_decision.pdf`, `resolution_operational_decision.pdf`, `brier_score`/`brier_skill_score`/`event_rate_error` figures -- this is close to the exact protocol T2 asks for (predicted exceedance vs realised frequency, reliability curve). Confirm the scored event matches T2's definition (`V outside EN 50160 and P above its 90th percentile`) rather than a generic hosting-capacity decision value before citing it as T2's answer; if it is a different event, T2 still needs its own scan. |
| T3, constraint-aware quantile placement | new objective term or conditioning | **not started**, and per `POSITIONING.md`'s own sequencing should not start until T1 shows the constraint framing separates the two heads at all. |
| T4, scenario-conditional what-if | pooled model with `ev_pen`/`pv_pen` as inputs | **not started**, gated on T1 by `POSITIONING.md`'s own ordering. |
| T5, state estimation (U1a/U1b) | `experiments/lv-multitarget/state_estimation.py` with forecast-derived weights | **not started** in this results tree; U1a is a few lines once T1/T2 tables exist to draw predictive std from. |
| T6, forecast-in-the-loop power flow | feed forecast P/Q into OpenDSS, compare `V` to `V_pf` | **not started.** `joint_vs_independent.py` already produces the paired samples T6b needs. |
| T7, decision-level evaluation | translate exceedance probabilities into a replacement/hosting-capacity decision | depends on T1/T2; partially answerable once T2's DSS-UK significance table is confirmed against T2's exact event definition. |

## 5. Sequencing

1. Step 1 above (`fpqr_gating_ablation.py --report`, all five groups) --
   free, do first.
2. Step 2 above (`multitarget_experiment.py` significance report) -- a small
   code addition mirroring existing code, then run against the existing CSV.
3. Confirm whether the existing DSS-UK `operational_decision.py` run already
   answers T2 as defined, or scores a different event (per SS4 above).
4. Run `operational_decision.py --epochs 100` for DSS-US -- this is T1, the
   test `POSITIONING.md` itself ranks as the single most decisive and
   cheapest step outstanding, and nothing later in the T-list should be
   started before it reports.
5. Only after T1/T2 report: T7 (decision-level framing of both), then T6a
   (physical consistency check, cheap given `joint_vs_independent.py`
   already exists), then T3/T4/T5 in `POSITIONING.md`'s own order.
6. Step 3(b) (wiring a proper joint score into the engine path) can happen
   at any point once someone is touching `twiga/experiment/analysis.py`; it
   is not on the critical path for T1-T7.

## 6. Caveats every table here inherits

Unchanged from `EXPERIMENT_PLAN.md` SS10.9, restated because Steps 1-2
produce tables that carry them without saying so on the page:

- Fold test windows overlap (fold 0 tests June-August, fold 1 July-September),
  so the reported fold standard deviation understates uncertainty.
- Evaluation covers June-December only; winter, which matters for an
  EV/PV-penetration study, is never in a test fold.
- One fixed seed across conditions -- the reported spread is fold variation,
  not seed variation.
- Both completed runs (`multitarget_experiment.py`, `fpqr_gating_ablation.py`)
  cover MLVS-PT-3PHASE only; any table built now should say so rather than
  imply cross-dataset coverage.
