# Chapter 6: which FPQR mechanism and which copula become the thesis design

This tracks, per ablation axis, which variant the evidence actually supports
versus what `Chapter-6.tex` currently states in its mechanism-level prose
(SS3-4 for FPQR, `subsec:ch6-copula` for the dependence model). See
`ANALYSIS_PLAN.md` for the general Chapter 6 analysis convention and
`POSITIONING.md` for why the operational-metric axis matters as much as the
Winkler-score axis. Decision rule, stated once rather than per row: **adopt
the reference (simplest) condition unless an alternative wins on the axis's
own decisive metric at the DM/MCS significance level; where the decisive
metric is operational (Brier/reliability) rather than Winkler, the Winkler
table is a secondary diagnostic only.**

Evidence below marked `[RUN 20260918_091304_9d1579]` comes from
`fpqr_gating_ablation.py --report <group>` against the existing MLVS-PT-3PHASE
run, re-executed 2026-09-19 after fixing that command (see **Fixes made to
get here** below) — this is real, already-computed evidence, not a
projection. Rows marked `PENDING` need a run that has not happened yet.

## FPQR mechanism axes

### Proposal (`proposal_softmax` reference)

| condition | status |
|---|---|
| `proposal_kumaraswamy` | **[RUN]** DM vs. softmax: not significant (winner=softmax). MCS retains only `proposal_softmax`. |
| `proposal_horizon` (new, added this session) | **PENDING** — condition did not exist in the 2026-09-18 run; needs `--group proposal --dataset all` re-run before it has any evidence at all. |

**`Chapter-6.tex` says**: the horizon-conditioned proposal is adopted (SS3.1,
`eq:horizon-context`; also named in the Experiment 2 description, line ~484).
**Tension, already documented before any new run**: `BaseFPQRConfig`'s own
`proposal_type` field docstring records the single-target Chapter 5 ablation's
finding as "softmax wins; horizon worst" — i.e. this is not merely untested,
it is the specific option Chapter 5's own ablation found *worst*. The text's
own rationale (SS3.1: "extending to multiple targets introduces additional
variability... a shared discretisation may be suboptimal when uncertainty
evolves with lead time") is a legitimate reason the multi-target case could
differ, but it is a claim, not yet a result. **Verdict: cannot be settled
until `proposal_horizon` is run** — this is the single most consequential
pending run in this table, since it is the one axis where the text's claim
and the existing (single-target) evidence point in opposite directions.

### QVN (`qvn_standard` reference)

| condition | status |
|---|---|
| `qvn_monotone` | **[RUN]** DM vs. standard: not significant (winner=standard). **Not retained by the MCS** (excluded from `{qvn_monotone_cal, qvn_standard}`). |
| `qvn_monotone_cal` | **[RUN]** DM vs. standard: not significant (winner=standard). Retained by the MCS. |

**Which condition matches the text.** `Chapter-6.tex` SS"Monotonic Value
Network" describes `Q(tau_n) = Q(tau_1) + integral of a strictly positive
derivative` with no calibration loss or scale anchor mentioned — that
construction is `qvn_monotone`, not `qvn_monotone_cal` (the calibrated
variant, with an added CALCE loss and temperature/scale anchor). This
matters because the two conditions do not behave alike here: the MCS retains
`qvn_monotone_cal` but excludes plain `qvn_monotone`.

**Flag, pre-existing in the codebase, not introduced here.**
`BaseFPQRConfig`'s own `qvn_type` docstring records "Monotone: best Pinball"
from the single-target ablation; `fpqr_formulation.md` SS12 records the same
condition as "confirmed problematic... should be avoided." This run's own
MCS result (excludes plain `monotone`) sides with `fpqr_formulation.md`'s
reading, on Winkler score, in the multi-target setting.

**Verdict: as currently written, `Chapter-6.tex`'s adopted construction
(plain `monotone`) is the one excluded by the MCS.** Either the text should
describe the calibrated variant instead (which the MCS does retain, alongside
the plain standard baseline), or the adopted architecture should be
`qvn_standard`/`qvn_monotone_cal`, not plain `qvn_monotone` as currently
written. Neither pairwise comparison reaches significance on its own, so this
reads as "no evidence to prefer monotonicity over the standard crossing-penalty
network at all," which is itself a fact the prose does not currently state.

### Grid (`grid_shared` reference)

| condition | status |
|---|---|
| `grid_per_target` | **[RUN]** DM vs. shared: not significant (winner=`grid_per_target` on the raw pairwise mean, but not significant). **MCS retains only `grid_shared`.** |

**Two different questions, kept separate per this session's earlier finding**
(`ANALYSIS_PLAN.md` SS4 / `EXPERIMENT_PLAN.md` SS10.4-10.8): this DM/MCS
table answers "does per-target grid improve forecast accuracy" (no, on this
evidence). `EXPERIMENT_PLAN.md`'s own diagnostic answers a different
question, "does the per-target grid learn something closer to the
`|Q''|^(-1/3)` theory-optimal shape than a shared grid does" (at 31 slots,
yes in 6/6 comparisons, direction-consistent but none individually
significant after correction). **Both are true at once and neither
contradicts the other**: a mechanism can track a theoretical shape better
without that translating into a significantly lower Winkler score. Report
both; do not let one stand in for the other.

**`Chapter-6.tex`**: no explicit "per-target adopted" commitment was found in
the earlier search, so no prose conflict here. **Verdict: adopt
`grid_shared` as the default**, on the MCS result, and cite the theory-tracking
diagnostic separately as a mechanism-fidelity finding, not a performance
claim.

### Multigate (`gate_none` reference)

| condition | status |
|---|---|
| `gate_feature_only` | **[RUN]** no significant pairwise differences among any of the four conditions. |
| `gate_target_only` | **[RUN]** same. **MCS retains only `gate_target_only`**, excluding `gate_none`, `gate_feature_only`, and `gate_both`. |
| `gate_both` | **[RUN]** same. |

**Verdict: `gate_target_only` is the single condition the MCS retains**,
despite no individual pairwise comparison reaching significance — this is
exactly what an MCS is for (a joint elimination test with more power than
six separate pairwise tests at a corrected threshold). `Chapter-6.tex`
currently wraps its target-gating description in `\cld{}` (draft-claim
markup, lines ~431/444) rather than stating it plainly; **this result is
consistent with un-flagging target-gating specifically** (not combined
gating, and not feature-only gating alone) once someone reviews the prose
against this table.

### Head (`head_fpqr` reference, explicitly not a hypothesis test)

| condition | status |
|---|---|
| `head_joint_parametric` | **[RUN]** DM vs. FPQR: not significant on Winkler (winner=`head_fpqr` on the raw comparison). **MCS retains only `head_joint_parametric`**, excluding `head_fpqr`. |

Per `FPQR_GATING_README.md`, this group is a design-sanity check, not a
hypothesis Chapter 6 argues for: the FPQR+copula architecture is retained
regardless of this outcome, "so the FPQR+copula design is judged against a
real alternative rather than assumed to win" (module docstring). **On
Winkler score alone, the joint parametric head is what the MCS prefers.**
This is the same tension `POSITIONING.md` already discusses for the
energy-score result: a global/marginal proper score can favour the simpler
joint-parametric head while the FPQR+copula design's actual argument is about
resolution at the constraint tail and per-target grid adaptivity, which
Winkler alone does not test. No prose change needed (the text never claims
FPQR wins this comparison), but this result should be stated explicitly
rather than left implicit, since a reader who finds `ch6_fpqr_gating_full.csv`
independently will find the MCS excluding FPQR here.

## Dependence axis

### Copula family (Gaussian / Clayton / D-vine)

**Status: PENDING.** `copula_variants.py` (Experiment 2b) has never been run
at all — no output directory exists. The only evidence currently on disk is
`forecast_examples.py`'s `winkler_dm_table.csv`, and it is silent on this
question by construction: `COPULA_METHODS.md`'s own $H_3$ hypothesis (why
Clayton should win) is stated for the hosting-capacity Brier/reliability
tier, a one-sided joint-extreme event, not for Winkler score, and the
existing table's six Gaussian-vs-Clayton-vs-vine pairwise rows are all
non-significant (p=1.0) on Winkler regardless. That null result cannot be
read as evidence against Clayton or the D-vine, because it never tested the
metric their own hypothesis names as decisive.

**`Chapter-6.tex` already states, unhedged**, that "a Gaussian copula is
adopted... because it provides a tractable representation of dependence...
computationally attractive" (line ~241), and explicitly defers the
tail-dependence trade-off to a discussion section that could not be located
with content. **This is the one commitment in the current text least
supported by evidence today** — not because it is wrong, but because the
ablation built to justify it specifically (`copula_variants.py`'s
`significance(..., metric="Brier score")`, pre-registered for exactly this
question) has not been run.

**Action needed before this row can be filled**: run `copula_variants.py`
(local smoke test first, then the full server run across all three scenarios
per `ANALYSIS_PLAN.md`/this plan's Step 4), and read `significance()`'s
Brier-score output as the deciding table, `family_significance()`'s Winkler
output as a secondary cross-check only.

### Conformal score type (`max-norm` reference, `mahalanobis` alternative)

**Status: PENDING**, blocked on the same run as above, and additionally on a
data-quality issue found while reviewing evidence for this table:
`forecast_examples.py`'s `conformal_calibration.csv` currently has a
degenerate calibration split (`n_calib=2`, `mean_width=inf`) for every copula
row, which makes it unusable as evidence for this axis regardless of which
copula wins. `copula_variants.py`'s own `copula_conformal_summary.csv` needs
checking for the same defect once that script has been run.

## Fixes made to get here

`fpqr_gating_ablation.py`'s `--report`/`significance_report()` could not
previously produce any of the `[RUN]` evidence above — two independent bugs,
neither introduced this session, both now fixed:

1. **Path bug.** `--full-csv`'s default (`HERE.parent.parent.parent /
   "experiments" / "chapter-6-multitarget" / "fpqr_gating_ablation" /
   "results" / "ch6_fpqr_gating_full.csv"`) omitted the
   `<experiment_slug>/runs/<run_id>` path segments `ExperimentEngine.run()`
   actually writes under, so it could never have resolved to a real file.
   Replaced with `discover_runs()`/`latest_run_for_group()`, mirroring
   `chapter-5-fpqr/analyse_results.py`'s pattern exactly: reads the *latest*
   synced run under `results/chapter-6-multitarget/fpqr_gating_ablation/`
   that covers the requested group, so a future re-run of one group alone
   (e.g. just `proposal`/`qvn`, per this table's pending rows) is picked up
   automatically without touching this script again.
2. **Invalid `version` argument.** The DM call passed `version="two-sided"`,
   which `diebold_mariano_from_losses` has never accepted (only
   `"univariate"`/`"multivariate"`) — this call could not have succeeded
   regardless of the path bug. Fixed to omit the argument, taking
   `pairwise_dm_table`'s own default (`"multivariate"`), matching
   `chapter-5-fpqr/analyse_results.py`'s `dm_and_mcs()` call exactly.

Both fixes are in `fpqr_gating_ablation.py`; no results were regenerated,
only read correctly for the first time.

## Prose changes to consider once the pending rows are filled

- **Copula (line ~241)**: contingent on `copula_variants.py`'s Brier-score
  result. If Gaussian is not favoured on that metric, replace "a Gaussian
  copula is adopted... because it provides a tractable representation" with
  whichever family the Brier-score MCS retains, or keep Gaussian and state
  plainly that the family-choice ablation found no significant separation.
- **QVN (SS"Monotonic Value Network")**: the construction as written matches
  `qvn_monotone`, which this run's MCS excludes. Either redescribe the
  construction to match `qvn_monotone_cal` (which the MCS retains) or note
  explicitly that the ablation did not find monotonicity itself to help on
  Winkler score.
- **Proposal (SS3.1, line ~484)**: do not describe the horizon-conditioned
  proposal as adopted until `proposal_horizon` has an actual DM/MCS result —
  currently this is a rationale, not a finding, and the only prior evidence
  (Chapter 5, single-target) found the same mechanism worst.
- **Multigate (lines ~431, ~444, currently `\cld{}`)**: this run's MCS
  supports un-flagging target-only gating specifically, not combined or
  feature-only gating.
- **Head**: no prose change needed; the text does not claim FPQR wins this
  comparison, but the MCS's exclusion of `head_fpqr` should be stated in the
  ablation's own results discussion rather than left for a reader of the raw
  CSV to discover.
