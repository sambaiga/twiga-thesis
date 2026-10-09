# Chapter 6 plan: what each forecast type gives for multi-target P/Q/V

Agreed 2026-10-07. Replaces multi-target conformal as a chapter contribution.

## Why

Under the fixed MLPF backbone the four heads have statistically indistinguishable point
accuracy; they differ in how they represent uncertainty. The benchmark scored their intervals
but never used what each head actually outputs. Multi-target conformal was dropped as a
contribution: the offline study (`experiment_1_benchmark/conformal_study.py`) showed that
calibration restores coverage but, where the model has little skill (MLVS-PT voltage), the
calibrated interval is 1.4-1.55x wider than the target's own 90% range, so stratified CWE falls.

## RQ4 (proposed)

How do quantile, parametric and Bayesian forecasts of P, Q and V differ in what they deliver
for single-target, combined-target and operational uses, under a common backbone?

## Four messages

1. Accuracy is shared; reliability depends on the type (QR over-covers, FPQR/BNN under-cover V,
   Laplace under-covers Q on MLVS-PT). Evidence: existing benchmark tables.
2. Parametric is sharp and efficient where uncertainty is regular; quantile is safer on measured
   data. Laplace's MLVS-PT Q failure is a too-small learned scale, not error shape (checked:
   MLVS Q errors are near-symmetric, skew 0.05; DSS-EU Q is skewed, 2.45, and Laplace wins there).
3. Whether joint modelling matters depends on the record: errors weakly dependent on MLVS-PT,
   strongly on DSS-EU/DSS-US (distance correlation figure).
4. Each type supplies dependence differently (quantile/Laplace need a copula, BNN scenarios carry
   it); this should show in combined-target scores and decisions.

## What each output can do

| Type | Output | Direct use | Needs |
|---|---|---|---|
| FPQR / QR | quantile grid per target and step | single-target exceedance probability | copula for joint/day quantities |
| Laplace | closed-form distribution | exact probabilities, cheap samples | copula for joint |
| BNN | scenarios (one weight draw per day, all targets) | joint and day quantities directly | nothing; finite ensemble |

## Analyses

- **A. Single target**: exceedance probability per target (Brier, reliability, resolution);
  FPQR learned-level placement per target.
- **B. Combined targets**: CRPS of apparent power S = sqrt(P^2 + Q^2); energy score (P, Q, V) and
  variogram score (targets and lead times); each head as independent samples, Schaake-coupled
  samples, and BNN native scenarios.
- **C. Decisions**: `experiment_3_decision/fixed_decision_sweep.py` without conformal conditions.

## Chapter 6 structure (after results are reviewed)

- Methods: framework (existing); "what each forecast type gives" (replaces the multi-target
  conformal sections); evaluation extended with Brier, CRPS(S), energy, variogram, REV.
- Experiment 1: 1.1 accuracy and reliability (existing); 1.2 single target (A); 1.3 combined
  targets (B); 1.4 dependence (existing correlation figures, moved); 1.5 calibration note
  (one paragraph and a small table).
- Experiment 2: decision sweep (C). Penetration subsection unchanged.

## Order of work

1. Save BNN scenarios in the raw export; MLVS-PT raw export from the saved checkpoints;
   analyses A and B offline. Review.
2. DSS-EU, DSS-UK, DSS-US benchmark heads with raw export (server or overnight). A and B. Review.
3. Decision sweep without conformal (server).
4. Chapter 6 restructure, every change in `\cld{}`.

## Step 1 results: MLVS-PT (2026-10-07)

Fixed heads, weights reloaded from the fixed run (`raw_export --reuse-run`), 10 folds, Wilcoxon
signed-rank over folds.

- **A. Single target.** Laplace has the lowest Brier score for all three tail events (skill 0.27,
  0.25, 0.29), significantly better than QR and BNN on P (p < 0.03); differences on Q and V are
  not significant. Laplace's closed-form distribution gives well-calibrated tail probabilities
  even though its central 90% interval under-covers Q. BNN's scenario-share probabilities are the
  least reliable.
- **B. Combined targets.**
  - Variogram score (dependence across lead times and targets): the Schaake coupling improves
    every head in 9-10 of 10 folds (p <= 0.004). BNN's native scenarios beat its independent
    draws (p = 0.027) but lose to Schaake-coupled BNN draws (p = 0.004).
  - Energy score (cross-target only, per step): Schaake is slightly worse (significant for FPQR
    and Laplace), consistent with the weak cross-target dependence on MLVS-PT.
  - CRPS of apparent power: Schaake brings no gain; BNN's native scenarios are its best form
    (8 of 10 folds, p = 0.014).
- Reading: on the measured record the dependence that matters is across lead times, and an
  empirical copula of recent days supplies it better than the BNN's weight draws. Message 4 must
  be tested on the strongly dependent DSS records (step 2).

## Step 2 command (server)

    uv run python chapters/ch7_multitarget/experiment_1_benchmark/fixed_benchmark.py \
        --dataset DSS-EU:default:0.4:0.4 DSS-UK:weak:0.4:0.4 DSS-US:8:0.4:0.4 --group raw_export

Then, per record (`dss-eu_default_0_4_0_4`, `dss-uk_weak_0_4_0_4`, `dss-us_8_0_4_0_4`):

    uv run python chapters/ch7_multitarget/experiment_1_benchmark/forecast_type_study.py \
        --run <experiments>/chapter-6-multitarget/fixed_benchmark/multitarget_fixed_benchmark \
        --dataset <record>

## The copula: what it changes and why it matters

Independent draws and Schaake-coupled draws use the same values in every cell (target x half-hour);
they differ only in how those values are joined into paths.

- Independent draws join the per-cell values at random: no dependence across half-hours or targets.
- The Schaake shuffle reorders the same values to follow the ranks of recent observed days (an
  empirical copula), so each path inherits how a real day moved across hours (persistence: a high
  evening stays high) and across targets (high P with high Q).

The fans (each head's own output) are identical with and without the copula. The copula matters for
anything that depends on more than one cell: sustained events (an overload lasting two hours), joint
events (low V while P is high), and whole-day quantities (daily energy, evening peak), whose spread
independent draws make too narrow because ups and downs cancel.

Evidence so far (MLVS-PT): the variogram score improves by 4-15% for every head in 9-10 of 10
folds; per-step scores do not change. Cross-target gains are expected on DSS-EU and DSS-US, where the
errors are strongly dependent through the power flow. Thesis line: quantile and Laplace heads describe
each target at each half-hour well but say nothing about how they move together; the copula adds that
from recent observed days at no training cost, and the BNN gets it from its own weight draws.

Figures (``forecast_type_study.py``): ``--outputs-day`` draws each head's own output (fan or weight
draws); ``--copula-day`` draws independent against Schaake-coupled paths over the same fan.

## Decision sweep: first analysis and the changes it led to (2026-10-07)

Axes stage, 21 scenarios (DSS-UK weak and stiff, DSS-US rank 0), fixed heads, 6 folds, conformal
conditions ignored. Paired Wilcoxon tests over scenario-folds where the event occurs on 2-98% of
test steps or days.

- Single-step overload: every head has positive Brier skill (0.28-0.38 at 1.0x); Laplace is
  significantly best (p < 0.01 against all three), BNN lowest -- as in analysis A on MLVS-PT.
- Sustained overload (two hours): independent draws under-forecast (mean probability 0.26-0.32
  against a realised 0.41-0.45); Schaake coupling lifts them to 0.39-0.44 and lowers the Brier score
  (1.25x: better in 70% of scenario-folds, median -5.3%, p = 2e-8; 1.0x: -2.3%, p = 0.002). No
  effect on single-step events. The copula's value shows in operational events.
- BNN native paths under-forecast sustained overload even more (0.24 against 0.45) and lose to the
  coupled marginal heads (p <= 0.03): per-cell noise breaks the weight draws' persistence.
- Day-event skill against the base rate stays near zero (resolution about 0.02).

Data anomaly found and fixed: SMART-DS assigns DER to a fresh random set of customers at each
penetration level, so 14 of 30 DSS-US transformers do not export more as PV rises and on 12 the
EV 0.2 record equals EV 0. Rank 0 is one of them (it exports most at PV 0.2, least at PV 0.6), so
it is not a penetration sweep. The sweep now uses rank 8, the most loaded transformer whose mean
net load and 98th-percentile apparent power respond steadily to EV and PV;
`check_penetration_response` warns at the start of every run. DSS-UK (substation aggregate) passes.

Changes to `fixed_decision_sweep.py`:
- conformal conditions removed (`--with-conformal` restores them);
- every head scored in every scenario form it supports (`decision_scoring.scenario_forms`):
  independent and Schaake for all heads, the BNN also as its normal marginal and as native paths
  with per-cell or day-coherent noise, so its distribution and its dependence are separated;
- a 1.5x overload tier, so one tier stays informative at high EV;
- checkpoints kept and raw forecasts saved per condition (`keep_checkpoints`,
  `save_raw_predictions`), as Experiment 1's `raw_export`;
- report: `decision_forecast_type_report.py` (event rates and scorable share, single-step head
  ranking, forms on sustained and combined events, paired tests, two figures).

The 21 axes scenarios must be rerun with these changes (rank 8, new forms and tier).

## Step 2 results: DSS records (2026-10-07, `multitarget_fixed_benchmark_v2`)

Server run of `raw_export` (DSS-EU 5 folds, DSS-UK weak 6, DSS-US rank 0 6; validation window as
reference). Paired Wilcoxon tests pooled over records x heads x folds (81 pairs with MLVS-PT).

- **Copula across the day (variogram score):** Schaake beats independent draws in 94% of pairs
  (p = 5e-14) and on every record (78-100%, p <= 0.003). QR gains most (-6 to -15%).
- **Copula across targets (energy score):** depends on the record, as message 3 predicts --
  DSS-EU, where V follows P and Q through the power flow, 100% of pairs better (p = 6e-5; QR -6.2%,
  Laplace -4.3%); DSS-US 83% (p = 1e-4); DSS-UK no effect; MLVS-PT slightly worse (13%).
- **Apparent power CRPS:** no effect of coupling (48%, p = 0.8).
- **BNN own scenarios against its Schaake-coupled draws:** no better on apparent power or the
  energy score, worse on the variogram score (native better in 26%, p = 4e-4). Message 4 is not
  supported in its current form: the BNN's weight draws carry less useful dependence than an
  empirical copula of recent days. The day-noise variant (decision sweep) tests whether the noise
  model is the cause.
- **Single-target probabilities (A):** no head wins everywhere. Laplace is best or near-best on
  DSS-EU (all three targets 0.31) and MLVS-PT; FPQR is best on DSS-UK V (0.49) and Q (0.43); BNN is
  weakest on every record. On DSS-US rank 0 every head has negative skill (thresholds from the
  validation month; rank 0's DER allocation anomaly).

DSS-US moved to rank 8 in the benchmark too (`fixed_benchmark.DSS_US`), so both experiments use
the same transformer; its benchmark, raw export and the chapter's DSS-US tables need a rerun.

## Benchmark rerun (v2) and the Chapter 6 update (2026-10-07)

`multitarget_fixed_benchmark_v2` (server raw_export run, same fixed settings) reproduces the
original fixed benchmark: QR is identical (deterministic), the other heads move by a few percent,
and the significance conclusions (MCS sets, DM directions) hold. Run-to-run variation does change
some ranks and coverages (MLVS-PT: best rank QR 2.22 in v2 against FPQR 2.46 originally; FPQR
DSS-UK joint coverage 0.60 -> 0.45), so the chapter must rest on DM/MCS, not on ranks.

Source of the chapter's numbers: v2 for MLVS-PT, DSS-EU and DSS-UK (the same weights the
forecast-type analyses use), DSS-US from the pending rank-8 run; the original run is cited as a
reproducibility check.

Chapter structure (every change in `\cld{}`):
1. RQ4 reframed: what quantile, parametric and Bayesian forecasts give for single-target,
   combined-target and operational uses under one backbone.
2. Methods: keep the framework; replace the multi-target conformal sections with "what each
   forecast type gives" (output table, quantile CDF with exponential tails, Laplace closed form, BNN
   weight draws, the Schaake shuffle); evaluation adds Brier and its decomposition, CRPS of apparent
   power, energy score, variogram score, REV.
3. Experiment 1: 1.1 accuracy and reliability (benchmark tables and DM figures, v2); 1.2 forecast
   outputs (fan and weight-draw figures); 1.3 single-target probabilities (A); 1.4 dependence (distance
   correlation, then copula figures and the energy/variogram results, B); 1.5 calibration note.
4. Experiment 2: decision sweep by forecast type (rank-8 rerun): Laplace on single-step events,
   copula on sustained events, BNN native vs day noise vs coupled.
5. Conclusion and RQ4 answer rewritten around the four messages; message 4 stated as tested and
   not supported unless the day-noise variant changes it.

## Significance: DM on per-window losses replaces Wilcoxon (2026-10-07)

All significance now uses the thesis's DM test with Benjamini-Hochberg adjustment and the MCS
(`dm_and_mcs`), applied to per-window losses (one score per forecast day): 153-901 paired days per
record instead of 5-10 folds. `forecast_type_study.py` writes `dm_tests.csv`; the decision sweep
saves a daily Brier loss per form and event, and `decision_forecast_type_report.py` tests each
scenario and adjusts across scenarios. The earlier Wilcoxon results above are superseded by:

- Variogram score: Schaake significantly better for all heads on all records, except BNN on DSS-US.
- Energy score: Schaake significantly better on DSS-EU and DSS-US for all heads; significantly worse
  on MLVS-PT for all heads and on DSS-UK for three of four (better for QR).
- CRPS of apparent power: Schaake mostly better on DSS-EU and DSS-US, worse on MLVS-PT.
- BNN native vs Schaake: mixed across records and scores.
- A (Brier MCS per target): Laplace in nearly every set; FPQR in most; BNN only DSS-UK P.

## DSS-US rank 8 results (2026-10-08, run 20261007_1918156c..., in `multitarget_fixed_benchmark_v2/runs`)

Chapter result set: `benchmark_chapter` = original fixed runs (MLVS-PT, DSS-EU, DSS-UK) + rank 8.
Chapter figures, forecast examples (DSS-US day now 5 Oct 2019, fold 4) and the DSS-US rows of the
benchmark and joint tables regenerated from it.

- Benchmark: point accuracy higher than rank 0 (NRMSE 0.16-0.29). FPQR has the lowest P NRMSE and
  is in the CRPS MCS for all three targets; QR alone in the P WS MCS, Laplace alone in the Q WS MCS;
  Laplace best average rank (2.00); BNN last (3.50).
- Joint: QR 0.871, Laplace 0.667, FPQR 0.528, BNN 0.476. EnS: FPQR lowest (1.596), not
  significantly different from QR and Laplace; all three significantly better than BNN. Brier skill
  positive for every head (0.01-0.10), unlike rank 0.
- A: Brier MCS P {FPQR, Laplace, QR}, Q {Laplace}, V {Laplace, QR}; BNN negative skill.
- B (DM): Schaake significantly better on EnS and VS for every head; BNN native significantly worse
  than BNN Schaake on all three combined-target scores.

Chapter 6 updated (all in \cld): DSS-US description (why rank 8), DSS-US table and joint rows,
DSS ranking paragraph, forecast-example paragraph and captions, joint-coverage/EnS/Brier sentences,
fold coverage, dependence; introduction, framework and study descriptions reframed to the
forecast-type focus. Still to restructure after the decision rerun: the results' conformal
subsection and Experiment 2.

Run location: moved on 2026-10-08 from `multitarget_decision_single_split 2` into
`multitarget_fixed_benchmark_v2/runs`, with its analyses (`analysis/dss-us_8_0_4_0_4`,
`analysis/forecast_type_study/dss-us_8_0_4_0_4`).

## Conformal removed from Chapter 6 (2026-10-08)

The methods subsection, the calibrated-results subsection (table, figure, conformal example
figures) and the old native vs max-norm Experiment 2 are gone. In their place (all in \cld):
- Results 1.x "Single-Target Event Probabilities" (tab:ch6-single-target: BSS per record/head/target,
  bold = Brier MCS) and "Joint Scenarios and the Empirical Copula" (fig:ch6-copula from
  copula_fpqr_anchor_p_*.pdf; tab:ch6-copula: % change Schaake vs independent with DM stars).
- RQ4 paragraph rewritten around what each output provides.
- Experiment 2 now states the decision-sweep design only; a `% TODO(ch6)` marks where the
  results go after the server rerun.
- The protocol's calibration window is now described as supplying the reference rates and
  the copula template. References to the earlier conformal chapter (prior work) are kept.
Backup before this change: scratchpad `Chapter-6.before-noconformal.tex`.

## Decision sweep results (2026-10-08, `multitarget_fixed_decision_sweep`, 48 scenarios)

Report: `decision_forecast_type_report.py` (benchmark-analysis format: `analysis/decision/ch6_decision*.tex|csv`,
`analysis/chapter_figures/fig_ch6_decision_*`). Significance: fold-level DM + MCS (`dm_and_mcs`) with each scored
(scenario, fold) as a paired observation, per network (heads) or pooled (forms). The runs predate the per-day Brier
export (12992da), so per-day DM needs a re-score from the kept checkpoints. Incomplete: DSS-UK weak EV0/PV0
(QR, Laplace only), EV0/PV0.2 (none); EV0.2/PV0 complete but unstatused.

1. Event rates rise with EV on both UK feeders (sustained 1.0x overload: ~0.01 at no DER to 1.00 from EV 0.4);
   DSS-US rank 8 is driven by PV 0.6 (0.69-0.80 at every EV level). Customer-voltage breaches only on weak DSS-UK.
2. Single-step events: all heads skilful (BSS 0.3-0.65 at 1.0x). Laplace best on DSS-UK (sole MCS member on stiff
   1.0x, better than FPQR); FPQR best on DSS-US (QR and Laplace significantly worse at 1.25x/1.5x). BNN lowest.
3. Sustained events: Schaake significantly better than independent draws for every head at 1.0x and 1.25x (Brier
   0.176 -> 0.162 FPQR, 0.180 -> 0.160 QR); independent draws under-forecast (mean prob ~0.37-0.45 vs realised 0.53),
   Schaake removes most of the gap (0.49-0.52).
4. BNN: native per-step-noise paths barely beat independent draws; day-coherent noise and Schaake are both in the MCS
   and clearly better, so the BNN's dependence comes from its noise model, not its weight draws.
5. Value: step-event REV 0.3-0.75 over C/L 0.05-0.2. Sustained-event BSS and REV are <= 0 on DSS-UK for every head
   (climatology of the training window is hard to beat once the event is near-certain), positive on DSS-US for the
   quantile and Laplace heads (BSS 0.10-0.27); BNN strongly negative everywhere.

Chapter 6 Experiment 2 text to align: it says DM on per-day Brier within each scenario, adjusted across scenarios;
the analysis now uses paired scenario-fold losses (state this, or re-score for per-day DM).
