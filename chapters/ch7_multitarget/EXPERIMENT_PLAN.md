# LV multi-target study: data generation plan

Generates the P, Q, V record at two LV secondary substations under a grid of EV
and PV penetrations, for the multi-target forecasting chapter. The forecasting
question is whether modelling the covariance across P, Q and V buys anything an
independent model cannot; this stage produces the data that question is asked of.

## 1. Feeder selection

Selection was made on **measured P–V coupling**, not on customer count. The
chapter's claim concerns the coupling between active power, reactive power and
voltage, so the pair must differ in exactly that, and must differ in it for
electrical rather than size reasons.

`measure_sensitivity.py` loads each of the 75 LV networks, solves at the nominal
1 kW per customer, doubles every load, re-solves, and records the shift in the
worst-bus voltage per MW at the substation. The measure spans a 3.9× range
(0.128 to 0.493 pu/MW) across the estate. Note only 25 of the 75 networks are
distinct topologies; the rest are repeats.

| | Feeder A — `network_21_1_1135` | Feeder B — `network_17_1_1149` |
|---|---|---|
| Role | stiff, weak coupling | weak, strong coupling |
| dV/dP | **0.128** pu/MW (minimum of 75) | **0.446** pu/MW (97th percentile) |
| Customers | 157 | 188 |
| Topology | 5 outgoing ways, meshed | **1** long radial |
| Transformer | 315 kVA | 500 kVA |
| V<sub>min</sub> at nominal | 0.998 pu | 0.942 pu |
| Voltage drop at 2× load | 2.3 % | 14.3 % |

The customer counts are deliberately close (157 vs 188) so the contrast is
electrical and not a diversity artefact. Feeder B was preferred over the single
network with marginally higher coupling (`network_23_1_1103`, 0.487 pu/MW), which
has only 111 customers: its substation aggregate would be small and spiky, which
would confound coupling strength with loss of diversity.

**Validated on a 2-day run.** At 60 % EV, Feeder A stays within EN 50160 at every
interval (V 0.998–1.046 pu) while Feeder B breaches it in 8.3 % of intervals
(V 0.905–1.049 pu). The pair therefore separates on the axis of interest before
any forecasting is done.

## 2. Profiles

One year, 2019, 30-minute resolution, 17,520 intervals.

| Component | Source | Model |
|---|---|---|
| Household demand | Low Carbon London, 5,560 real half-hourly UK household series | 400-series pool; one real series per customer, scaled to that customer's annual consumption. Weekday-aligned to the 2019 calendar. |
| PV | Measured UK irradiance and air temperature, SPS-UK, the same weather record used elsewhere in the thesis | NREL PVWatts v5: cell temperature from NOCT, γ = −0.0047 /°C, 0.86 system derate, 0.96 inverter, clipping at unity DC/AC |
| EV | Norwegian charging sessions (plug-in time, session energy) | Uncontrolled 7.4 kW domestic charge (32 A, 230 V) at 90 % efficiency until session energy is met; 2–5 sessions per week per driver |

Real per-customer series are used rather than a scaled aggregate because the
quantity of interest is the *diversified* substation sum. A single aggregate
shape replicated across customers understates coincidence at the peak and
overstates it off-peak, which is precisely the structure the forecaster must see.

Sanity checks against UK norms: PV yield **1062 kWh/kWp/yr** (capacity factor
0.121), household aggregate peak/mean **1.88**, EV **2660 kWh/yr** per vehicle at
a diversified **1.23 kW** per EV.

## 3. Scenario grid

EV ∈ {0, 20, 40, 60} % × PV ∈ {0, 20, 40, 60} % × 2 feeders = **32 yearly runs**.

Penetration is the fraction of customers with the technology, the convention in
LV hosting-capacity work. Customer assignment and device sizing are drawn from a
fixed seed, so the same houses hold PV and EVs across every scenario and the
grid is a controlled comparison rather than 32 independent draws.

The grid is the mechanism axis, not decoration: coupling strengthens with
penetration, so the joint-versus-independent gap should widen monotonically along
it. A single penetration level would give a number; the sweep gives a reason.

## 4. Power-flow protocol

- **Engine.** OpenDSS via `OpenDSSDirect.py`, native time-series mode with
  `LoadShape` objects at `stepsize=30m`, rather than snapshots looped from
  Python.
- **Tolerances.** `maxiterations=100`, `maxcontroliter=50`; convergence recorded
  per interval and reported per scenario. The 2-day validation had zero
  divergences in all eight scenarios.
- **Recorded per interval.** Substation P (kW) and Q (kvar) from the circuit
  total power, so transformer losses are included as a DSO meter would see them;
  mean LV busbar voltage in per unit; and the network-wide minimum and maximum
  bus voltage.
- **Compliance.** EN 50160 for LV supply, 230 V +10 % / −6 %, i.e. 0.94–1.10 pu.
  Voltage extremes are tracked network-wide rather than at the substation, since
  compliance is a property of each customer's supply terminal.
- **PV modelling.** `PVSystem` elements at unity power factor with 5 % cut-in and
  cut-out, sized at 1.1× DC rating for the inverter.
- **EV modelling.** Constant-power (`model=1`) loads at 0.99 PF at the same
  terminal as the host customer.

## 5. Running it

```bash
.venv/bin/python experiments/lv-multitarget/profile_networks.py      # estate survey
.venv/bin/python experiments/lv-multitarget/measure_sensitivity.py   # coupling, drives selection
.venv/bin/python experiments/lv-multitarget/build_profiles.py        # profile library
.venv/bin/python experiments/lv-multitarget/run_powerflow.py --quick --horizon 96   # validation
.venv/bin/python experiments/lv-multitarget/run_powerflow.py                        # full year
```

Output: `data/substation_pqv_17520.parquet`, one row per interval per scenario,
with `timestamp, feeder, ev_pen, pv_pen, p_kw, q_kvar, v_pu_mean, v_pu_min,
v_pu_max, converged`.

## 6. Demand calibration

Per-customer demand is calibrated to a **target after-diversity maximum demand**
rather than left to fall out of an annual consumption figure. The diversity of a
group is a property of which real series happen to be drawn, so it is measured on
the assembled aggregate and every customer scaled by the ratio the group needs to
reach the target coincident peak. `TARGET_ADMD_KW` is the single knob;
set it to `None` to fall back on raw annual consumption.

Default is **1.3 kW**, the common DNO design figure for new domestic
connections. Published values run 1.1-1.5 kW for electrically heated stock and
nearer 0.6-0.8 kW for gas-heated homes.

**This choice needs a decision, because it changes the story.** At 1.3 kW,
Feeder B breaches EN 50160 in **17.7 %** of intervals with *no* EVs or PV at all,
rising to 41.7 % at 60 % EV. A feeder already non-compliant before any low-carbon
technology is a poor baseline: it makes the penetration sweep look like it barely
matters, when in fact the feeder was overloaded to begin with. Feeder A stays
fully compliant at 1.3 kW even at 60 % EV, so the calibration does not disturb the
stiff end of the pair.

Recommendation: **1.0 kW**, which keeps the Feeder B baseline close to compliant
and lets the breaches emerge with penetration, which is the gradient the study is
trying to show. Run the sweep at one value and report it; do not mix.

The completed 17,520-step run in `data/` predates this calibration and used the
uncalibrated 0.58 kW. Re-run once the ADMD value is settled.

## 7. What this data does and does not support

P, Q and V are generated by the power-flow equations, and forecasts built on them
will be checked for power-flow coherence against those same equations. That is
the right design, because it provides an exact feasibility oracle that measured
substation data cannot, but the claim it supports must be stated accordingly:
**independent marginals produce infeasible P/Q/V triples where the joint head
does not**, demonstrated under controlled coupling. It is not evidence about how
either model behaves on a specific real feeder.

## 8. State estimation

The substation P, Q and V record on its own does **not** give state estimation;
it is the measurement input to one. This adds `state_estimation.py`, which
converts a feeder to pandapower with `from_opendss`, solves the true power flow
as ground truth, feeds substation measurements to weighted-least-squares
estimation, and scores the estimated voltage at every customer terminal against
the truth.

**Substation-only measurement is not observable.** On Feeder A the estimator has
347 measurements against 659 required, even after crediting every load-free bus
as a known zero injection. State estimation on this network therefore requires
smart-meter pseudo-measurements, which is itself a result worth reporting and
matches the finding in `book/08-forecasting/06-lv-state-estimation.qmd`.

With full smart-meter coverage the estimator converges at every step and already
separates coherent from incoherent substation input:

| substation input | voltage RMSE (pu) | max abs error (pu) |
|---|---|---|
| true P, Q, V | 0.00448 | 0.01004 |
| **coherent** perturbation | **0.00488** | 0.01014 |
| **incoherent** perturbation | **0.01148** | 0.01805 |

A coherent triple, where P, Q and V move together as the physics requires, gives
a state estimate barely worse than perfect measurement. Three independently
perturbed marginals of the same magnitude give **2.4x the voltage error**. That
is the operational form of the chapter's claim, and it is the argument for the
joint head that marginal scores cannot make.

**Caveat.** The perturbations here are synthetic stand-ins for forecast error,
which establishes the mechanism and the measurement chain. The chapter's actual
result must substitute real joint and independent forecasts from the trained
models for the coherent and incoherent rows.

## 9. Reactive power: what was wrong, what was fixed, and what remains

**The artefact.** In the first run every load carried a fixed power factor of
0.95 and every PV inverter ran at unity, so reactive power was active power
times a constant. Measured on the generated year: `R^2` of Q on P alone was
**0.9999**, and Q/P sat at 0.3355 with a standard deviation of 0.005, which is
exactly `tan(acos(0.95))`. The three targets carried one degree of freedom. Any
multi-target result on that data would have been an artefact of the load model,
and it is the first thing a rapporteur would check.

**The fix, in two parts, both standards-based.**

1. *Varying load power factor.* Domestic aggregate power factor moves with the
   appliance mix through the day, motor-dominated overnight and more resistive
   at the evening peak. Each customer now carries an independent reactive shape
   through the LoadShape `qmult` array, with a diurnal profile, a per-customer
   offset and interval noise, bounded to 0.88-0.995.
2. *PV inverter Volt-VAr control.* EN 50549-1 and IEEE 1547-2018 require LV
   generators to provide reactive support as a function of terminal voltage. An
   `InvControl` with a Q(U) droop of +/-0.33 pu (cos phi 0.95, as EREC G99)
   now runs on every PV system. This is the mechanism that matters, because it
   makes Q a function of V and so makes the three quantities simultaneously
   coupled rather than P determining the other two.

The control loop initially failed to converge (`Max Control Iterations
Exceeded`) at `deltaQ_factor=0.3`, which would have produced meaningless
reactive power. Damped to 0.1 with `maxcontroliter=200` it now converges and is
active in 99 % of intervals, drawing between -12.5 and +7.8 kvar.

**What the fix achieves, measured on two winter weeks, Feeder B at 20 % EV:**

| configuration | PV | R^2(Q given P) | R^2(Q given P and V) |
|---|---|---|---|
| fixed PF, unity PV | 0 % | 0.985 | 0.990 |
| fixed PF, unity PV | 40 % | 0.848 | 0.972 |
| varying PF + Volt-VAr | 0 % | **0.955** | **0.980** |
| varying PF + Volt-VAr | 40 % | **0.820** | **0.968** |

**What it does not achieve, and why that is acceptable.** Q remains about 98 %
determined by P and V together. That is no longer an artefact: at a substation
aggregating 150 to 190 domestic customers, reactive power really is largely
determined by active power and voltage. The fix removes the indefensible part,
the exact algebraic identity, and leaves the physically real part.

**The consequence for the chapter's framing.** The claim should not be that the
targets have interesting independent variation, because they do not. It is that
**independent marginals produce triples that are not realisable network
states**, and strong coupling makes that failure more severe rather than less:
if Q is 98 % determined by P and V, a model that samples Q without regard to its
own P and V samples will produce infeasible combinations almost every time. This
is the strongest available setting for the feasibility argument, not the
weakest. It also makes the Cholesky diagonal floor in the joint head
load-bearing rather than incidental, since the covariance being learned is
near-singular by construction.

The quantity to report once models are trained is therefore the **conditional**
covariance of the forecast residuals, after the forecaster has conditioned on
time, weather and lags, not the marginal correlation of the targets.

## 10. Methodology audit of the forecasting setup

An audit of the modelling and evaluation setup against the thesis's own theory,
prompted by the observation that the first grid ablation gave contradictory
answers: MAE said the per-target grid was 16% worse, the Winkler score said it
was 21% better, and the fold standard deviation on MAE was 3.33 against a mean
of 7.53.

### 10.1 Defects found and fixed

Four defects, all now covered by regression tests that fail without the fix.

**Entropy diluted across targets** (`fpqr/distribution.py`). The loss took
`entropies.mean()`. With per-target grids the entropy tensor gains a target
axis, so the mean divided the anti-collapse pressure by the number of targets:
each grid received a third of the gradient a single grid receives, and
`entropy_weight` silently meant something different in the two arms of the
ablation it was meant to control. Entropy is now averaged over batch and horizon
but summed over targets, which was verified to restore the per-grid gradient to
0.031250 in both arms.

**Quantile levels not sliced per target** (`core/metrics/quantile.py`).
`evaluate_quantile_forecast` sliced the target axis of the quantiles but passed
the whole `quantile_levels` array to every target. With a learned grid per
target this scores each target against the wrong levels. In practice the failure
was quieter: `get_quantile_metrics` catches per-metric exceptions and logs a
warning, so CRPS and the pinball loss disappeared from the results table rather
than appearing wrong. A silently absent proper score in a wide table is easy to
miss, so the regression test asserts the columns are present before asserting
they are correct.

**Checkpoint save and load disagreed** (`models/nn/core/base_model.py`).
`on_save_checkpoint` omits the pipeline path when there is no pipeline, but
`on_load_checkpoint` read the key unconditionally. Any run without a pipeline
therefore raised `KeyError` when Lightning reloaded the best checkpoint, failing
after training had already succeeded.

**Incompatible-checkpoint recovery did not recover** (`models/nn/core/base.py`).
On a checkpoint that does not match the architecture, `fit` retries with
`ckpt_path=None`, but reused the trainer that had just failed. That trainer
still holds the rejected checkpoint in its connector, so the retry restored the
same weights and failed identically. The retry now builds a fresh trainer. This
fires whenever two architectures share a checkpoint directory, which is what an
ablation over head variants does.

### 10.2 Checked and found sound

**No leakage from the configured date spans.** `train_start`/`train_end` overlap
`test_start`/`test_end` in the dataset config, and `run_forecast` builds the two
masks independently without removing the overlap. This is inert here because the
engine passes the whole frame to `backtesting`, which re-partitions it with
expanding-window `TimeBasedCV`. The realised folds are strictly ordered, for
example fold 0 trains on 13 Jan to 12 May, validates on 13 May to 12 June and
tests on 13 June to 12 August.

**No checkpoint bleed between conditions.** `resume_training` defaults to
`True`, so a model resumes from whatever checkpoint sits in its directory. The
experiment engine sets it to `False` explicitly, so conditions cannot inherit
each other's weights. Direct use of the model classes outside the engine does
not get that protection, and the diagnostic now sets it and a per-run checkpoint
directory itself.

### 10.3 Evaluation aligned with Chapter 3

Every condition now reports a proper scoring rule alongside the diagnostics:
CRPS, the Winkler score, marginal CWE, stratified CWE, PICP and nMPI. Chapter 3
argues CWE is a diagnostic rather than a proper score, so a comparative claim
cannot rest on it, and PICP and nMPI are reported beside the CWE scores as the
two components they combine, so a change in CWE can be attributed rather than
merely observed.

Four things had to be fixed to make that reporting real.

**No proper score was being computed.** The grid conditions requested
`metric_types=["point", "interval"]`, which yields CWE but no proper score, and
interval metrics alone discard most of a quantile head's output. The grid
conditions now also request `"quantile"` (CRPS, pinball). The Gaussian head
conditions request `"parametric"` (CRPS, NLL), which required adding
`"parametric"` to the engine's `evaluation_methods` mapping: the entry was missing, so a
parametric head could not report a proper score at all.

**Stratified CWE was implemented but never wired up.** `conditional_cwe` existed
in `core/metrics/interval.py` and was covered by its own unit tests, but it was
not in the registry `get_interval_metrics` iterates, so it could never appear in
any results table. It is now computed as `cwe_stratified`, stratified by lead
time. Lead time is the default stratifier because it is always available at
forecast time and uncertainty grows along it, making it the regime axis a
horizon forecast is most likely to get wrong. It is computed over the whole test
sample rather than per day, because conditional calibration is a property of the
forecaster across regimes and a per-day computation would leave roughly five
samples per stratum.

Validated on a fixture where the two forecasts are constructed to have identical
marginal width, which is exactly the case marginal CWE cannot resolve:

| forecast | PICP | nMPI | Winkler | CWE stratified |
|---|---|---|---|---|
| lead-adaptive  | 0.950 | 0.834 | 6.578 | 0.803 |
| constant width | 0.914 | 0.834 | 8.705 | 0.623 |

**Failing metrics vanished instead of reporting.** Both metric registries caught
per-metric exceptions and logged a warning, dropping the key. A dropped column
is invisible in a wide results table and reads as though the metric was never
requested; this is how the missing CRPS went unnoticed, and marginal CWE does
the same whenever its reference width exceeds the scale factor. Failures now
record NaN, which keeps the schema stable across conditions and makes the gap
visible. `conditional_cwe` likewise returned `0.0` when no stratum was scorable,
which is indistinguishable from a genuine worst-possible score since higher CWE
is better; it now returns NaN.

**CWE was ranked backwards in the results table.** `MINIMIZE_METRICS` contained
`CWE`, so the table highlighted the *lowest* CWE as best, while the delta column
has always treated CWE as higher-is-better. The same pair of numbers was
therefore reported as a win by one part of the table and a loss by the other.
CWE is an F-beta harmonic mean of a coverage score and a width score, so 1.0 is
best; it and `CWE_STRATIFIED` are now maximised, and the Winkler score, which
was in neither set, is now correctly minimised.

### 10.4 The diagnostic, restated as a falsifiable test

The original diagnostic asked whether the learned grids differ across targets.
Any two randomly initialised heads answer yes, so the question carries no
information. Chapter 5's own quadrature result gives a directional prediction
instead:

    dtau* proportional to |Q''(tau)|^(-1/3)

Measured on the reference scenario, the three targets have genuinely different
optimal grids, so the theory predicts three different learned grids and says
which way each should go:

| target | skew | kurtosis | optimal tail mass | optimal centre mass |
|---|---|---|---|---|
| p_kw    | +0.12 | -0.55 | 0.243 | 0.383 |
| q_kvar  | +1.06 | +1.39 | 0.190 | 0.369 |
| v_lv_pu | -0.59 | +0.06 | 0.272 | 0.320 |

The diagnostic now scores the learned interval widths against these, per target,
across seeds, with a shared-grid control trained under identical conditions. The
test is whether the per-target grid tracks the theory grid better than one
shared grid does, which is the claim per-target grids exist to make. Equal
values are a clean null.

Two supporting instruments were added because the headline correlation is not
interpretable on its own:

* **Clamp fraction**, which gates the result. `tau_hats` is clamped to
  [0.01, 0.99] and a pinned level says nothing about where the model wants its
  levels, so a high count means the run is uninterpretable rather than negative.
* **Distance from the uniform grid**, which separates the two failure modes. A
  proposal driven to uniform has reduced to fixed-grid quantile regression and
  earns nothing, even though it is not collapsed.

The diagnostic also standardises on training rows only. It previously used
whole-series statistics, which leaked test mean and variance into the scaler and
therefore into the theory grid the learned grid is scored against.

### 10.5 The finding that blocks the ablation

At the default `entropy_weight = 1.0`, the proposal collapses. Most quantile
levels sit pinned at the clamp bound: 60.7% and 65.2% across two seeds for the
**shared** grid, against 44.7% for per-target. The shared grid is the untouched
Chapter 5 path, so this is not a regression introduced by per-target grids; it
is a property of this configuration on this data, and it is worse in the arm
that serves as the control.

The mechanism is visible in the proposal: `probs = softmax(z_out, dim=1)` and
`taus = cumsum(probs)`, so mass concentrating on a single slot drives every
boundary to one end. The observed grid was crushed against the lower bound with
nothing at the upper, which distinguishes it from the symmetric tail
concentration the `|Q''|^(-1/3)` rule legitimately predicts for a heavy-tailed
target.

**Both arms of the grid ablation are therefore degenerate, and comparing them
answers nothing.** The contradictory first result is explained: it compared two
collapsed proposals at a fold spread that swamps the effect.

This bounds Chapter 5's contribution from both sides. Too little entropy and the
grid collapses; too much and it goes uniform, which is fixed-grid quantile
regression and concedes the contribution. The learned grid is worth reporting
only in the window between, and only if it tracks the theory grid there.

### 10.6 The entropy sweep, and what it settles

Three entropy weights, two seeds, both arms, twelve runs. Collapse is tunable
and the two seeds agree closely.

| entropy weight | clamped, shared | clamped, per-target | TV from uniform, shared | TV from uniform, per-target |
|---|---|---|---|---|
| 1  | 0.668 | 0.295 | 0.730 | 0.655 |
| 5  | 0.186 | 0.049 | 0.603 | 0.464 |
| 20 | 0.000 | 0.000 | 0.364 | 0.221 |

At weight 20 nothing is pinned, and the grid is still far from uniform, so the
proposal has not simply been regularised into fixed-grid quantile regression.
That is the setting in which the question can be asked at all.

Spearman rho between the learned interval widths and the widths the
`|Q''|^(-1/3)` rule prescribes, with two-tailed p-values:

| entropy weight | target | rho shared | rho per-target | p shared | p per-target |
|---|---|---|---|---|---|
| 1  | p_kw    | -0.339 | -0.233 | 0.372 | 0.546 |
| 1  | q_kvar  | -0.441 | -0.250 | 0.235 | 0.516 |
| 1  | v_lv_pu |  0.119 |  0.250 | 0.761 | 0.516 |
| 5  | p_kw    | -0.117 |  0.033 | 0.765 | 0.932 |
| 5  | q_kvar  | -0.167 | -0.083 | 0.668 | 0.831 |
| 5  | v_lv_pu |  0.433 |  0.533 | 0.244 | 0.139 |
| 20 | p_kw    |  0.033 |  0.033 | 0.932 | 0.932 |
| 20 | q_kvar  | -0.083 | -0.083 | 0.831 | 0.831 |
| 20 | v_lv_pu |  0.533 |  0.533 | 0.139 | 0.139 |

Three things follow, and none of them supports the per-target grid.

**At the only interpretable weight, per-target and shared are identical.** At
weight 20 the two arms give the same rho for all three targets, to three
decimals. The per-target offsets did not differentiate: each target learned the
same grid. This is the strongest observation in the table, and unlike the
correlations it does not depend on the significance of rho: the two arms
produced the same grid, so there was nothing for a test to detect.

**Where per-target does appear better, the proposal is collapsed.** Its apparent
advantage at weights 1 and 5 sits at 29.5% and 4.9% of levels pinned against
66.8% and 18.6% for shared. Per-target collapses *less*, which moves the
correlation, but a comparison between two degenerate grids is not evidence about
adaptivity. The effect disappears exactly when the confound does.

**No correlation reaches significance, but at nine slots that means very
little.** With rho computed over nine slots, |rho| must exceed 0.666 for p<0.05.
The largest observed is 0.533, for voltage, and every p-value is at least 0.139.

That threshold calculation is not a power analysis, and it was initially
over-read here as a null result. The power analysis is the relevant one:

| slots | critical \|rho\| | power at rho=0.533 | power at rho=0.40 |
|---|---|---|---|
| 9  | 0.666 | 0.31 | 0.18 |
| 21 | 0.433 | 0.71 | 0.44 |
| 31 | 0.355 | 0.88 | 0.61 |
| 41 | 0.308 | 0.96 | 0.74 |

At nine slots the test has **31% power** against an effect the size of the
largest one observed. A null was therefore roughly the likely outcome whether or
not a real effect of that size exists, so the rho column at `n_quantiles=9` is
uninformative in both directions. It is not evidence that the grid fails to
track theory; it is evidence that this test, at this size, cannot tell. Reaching
80% power needs about 26 slots at rho=0.533, or 47 at rho=0.40.

Repeating seeds does not help: the two seeds produce near-identical grids, so
the rank correlations are identical and seed variation adds no information.
Slots, not seeds, are the sample size for this test.

Even the power figures above are optimistic. The widths are a cumsum of a
softmax, so neighbouring slots are correlated by construction, the effective
sample size is below the nominal one, and the p-values are anti-conservative in
consequence. Rho is therefore best read as an effect size. The claim carrying
the least statistical baggage is the paired one, whether the two arms differ at
all, since they see identical data and differ in exactly one thing.

What the table does support without reference to rho is the identity of the two
arms at weight 20, and the disappearance of the per-target advantage exactly
when the collapse confound disappears. Those two observations point the same way
as Chapter 5's own negative, which found the grid adapts to lead time but not to
the day. The target axis was where the physics gave the strongest prior that
grids should differ, since the three targets have measurably different optimal
grids. Whether the grid adapts there is re-tested at `n_quantiles=31`, which is
the first size with adequate power.

### 10.7 Re-test at 31 slots, where the test has power

`n_quantiles` is the sample size of the rho test, so the sweep was repeated at
31 slots, the first size with adequate power (0.88 against rho=0.533; critical
|rho| falls from 0.666 to 0.355).

**The entropy weight does not transfer across slot counts.** A weight of 20 was
clean at 9 slots and collapses 62% of levels at 31. The pinball term sums over
slots and grows linearly in N while the entropy it balances is bounded by
log(N), so the weight must scale roughly linearly with N. Reaching a comparable
state at 31 slots needed 200, an order of magnitude above the library default.
This is now documented on `entropy_weight` in both `FPQRDistribution` and
`MLPFFPQRConfig`, because nothing in the code said so and the failure is silent.

| entropy weight | clamped shared | clamped per-target | offset magnitude |
|---|---|---|---|
| 100 | 0.164 | 0.072 | 0.542 |
| 200 | 0.067 | 0.033 | 0.439 |

**The per-target grids do differentiate at 31 slots.** The target offsets settle
around 0.44, and the three grids are visibly distinct. This overturns the
reading taken at 9 slots, where the two arms were identical: that identity was
an artifact of too few slots to express a difference, not evidence that the
offsets stay at zero.

Rho against the theory-optimal grid at 31 slots:

| entropy weight | target | rho shared | rho per-target | p per-target |
|---|---|---|---|---|
| 100 | p_kw    |  0.074 |  0.105 | 0.575 |
| 100 | q_kvar  | -0.412 | -0.395 | 0.028 |
| 100 | v_lv_pu |  0.325 |  0.347 | 0.056 |
| 200 | p_kw    |  0.102 |  0.205 | 0.269 |
| 200 | q_kvar  | -0.394 | -0.283 | 0.122 |
| 200 | v_lv_pu |  0.350 |  0.433 | 0.015 |

**Per-target tracks theory better than shared in 6 of 6 comparisons.** That is
the paired claim, and it needs no assumption about slots being independent. The
two entropy weights are not independent of each other, so the honest unit is the
three targets: 3 of 3 is a sign test at p=0.25. The direction is exactly what
the theory predicts and it is perfectly consistent, but three targets cannot
carry the claim to significance.

**No individual correlation survives correction.** The smallest p is 0.015 for
voltage under per-target grids; Bonferroni over the six tests at a weight puts
the threshold at 0.0083. The p-values are anti-conservative in any case, since
neighbouring slot widths are correlated by construction.

**Reactive power runs against theory.** Its rho is negative in every cell,
between -0.28 and -0.41, and it is the most consistent signal in the table. The
learned grid places its levels roughly opposite to where the `|Q''|^(-1/3)` rule
says they belong for q_kvar. That deserves an explanation before any positive
claim is made about the other two targets, since a mechanism that works should
not be reliably backwards on one of three.

**Status: open, not settled.** At 9 slots the question could not be asked. At 31
it can, and the answer is suggestive in the direction the theory predicts but
short of significant. Closing it needs more targets rather than more seeds or
slots, because the paired sign test is limited by having three targets, and the
per-target rho values are limited by multiple comparisons rather than by n.

### 10.8 Remaining gap: no proper score can separate the head comparison

`normal_crps` scores each target's marginal. The joint head's claim is about the
dependence between targets, which a marginal score cannot see, so CRPS will
report the joint and independent heads as near-equivalent by construction.

`MultivariateNormalDistribution.get_log_likelihood` is the joint log score and
does penalise getting the dependence wrong, but its `forecast` returns `loc`,
`scale`, `scale_tril` and `correlation` and no `log_likelihood`, so the
parametric evaluator falls back to `normal_nll` on the marginal `loc` and
`scale`. Both metrics available to the head group are therefore marginal.

The chapter's headline comparison currently has no proper score that can
distinguish its two arms. Closing this needs the joint log score computed at
evaluation time from `scale_tril` and the observations, or an energy score over
samples drawn from the joint. Until then the head comparison rests on the
feasibility argument and the interval diagnostics, not on a proper score.

### 10.9 Caveats to state rather than fix

* **Fold test windows overlap.** Fold 0 tests June to August, fold 1 July to
  September. The folds are not independent samples, so the reported standard
  deviation understates uncertainty and a paired significance test across folds
  would violate its own assumptions.
* **Evaluation covers June to December only.** With 12 months and an expanding
  window, the early months are never in a test fold, so winter is never
  evaluated. This matters for a study whose mechanism is EV and PV penetration.
* **One fixed seed across conditions.** The engine sets a single seed, which is
  the right choice for a paired comparison but means the reported spread is
  fold variation, not seed variation.
