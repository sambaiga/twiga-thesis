# Positioning: what Chapters 5 and 6 contribute beyond another forecasting algorithm

## The problem with the current framing

Chapter 6 currently argues that a joint predictive distribution over active power,
reactive power and voltage beats independent marginals because the three are
coupled by the power-flow equations. That is true and it is testable, but as
stated it competes on the same ground as every other forecasting paper: a lower
score on a generic metric.

Two results already sit awkwardly with that framing.

* The energy score, which is the proper multivariate score, puts the independent
  head **1.34 % ahead** of the joint one (11.484 against 11.638) even though the
  joint head recovers the dependence well: it learns a P-V correlation of -0.64
  against an observed -0.72, where the independent head asserts zero.
* Chapter 5 reports an honest negative: the learned quantile grid adapts to lead
  time but not to the day being forecast.

Neither is fatal, but together they say the generic-score framing is not where
the contribution lies.

## Alignment with the thesis objective

The thesis is on probabilistic forecasting for low-voltage distribution under DER
penetration, aimed at fundamental problems that add operational value and
thereby accelerate the energy transition. Everything below is checked against
that objective, and the check is what excludes several otherwise reasonable
directions.

| thesis objective | what this plan contributes |
|---|---|
| probabilistic, not point | constraints are tail events, so the deliverable is an exceedance probability rather than a forecast value |
| low-voltage distribution | evaluation at the secondary substation and at customer terminals, under EN 50160 and ANSI C84.1, not at system level |
| under DER penetration | the EV and PV grid is the mechanism axis, and results are reported as a function of penetration rather than averaged over it |
| fundamental problem | LV networks are unobservable and their constraints are joint in P, Q and V; both are structural, not incidental to one dataset |
| operational value | outputs are hosting capacity, exceedance risk and estimated customer voltage, which are quantities an operator already acts on |
| accelerates the transition | the headline is how much more DER a feeder can carry at an accepted risk, which is a connection decision rather than a metric |

Two consequences of taking that seriously:

* **Scope excludes control design and optimisation.** The power flow is used as
  an assessment oracle, not inside a controller. Designing a dispatch or
  scheduling rule is a different contribution and would dilute this one.
* **A metric improvement that does not change an operator's decision is not a
  result here.** That is why the energy-score outcome below is reported as a
  finding about what average scores can see, rather than buried.

## The observation this plan is built on

**The binding constraint differs between European and North American LV
distribution, and both constraints are joint functionals of the forecast
triple.** The same forecast object therefore answers two different operational
questions, and neither can be answered from marginal distributions.

| | EU, `uk-mvlv` | US, SMART-DS |
|---|---|---|
| secondary substation | 500 kVA serving 188 customers | 25 kVA serving about 6 |
| binding constraint | **voltage**, EN 50160 +/-10 % | **thermal**, transformer kVA rating |
| headroom measured here | thermal rarely binds | 107 % of a 25 kVA unit at 6 customers |
| tolerance | +/-10 % | +/-5 %, ANSI C84.1 Range A |
| what an operator asks | will voltage breach at the customer terminal, and how much more PV can connect | which transformers must be replaced before EV adoption overloads them |

## Why marginals cannot answer either question

This is the structural argument, and it is what separates the work from a better
score on a benchmark.

**Thermal.** A transformer is rated in kVA, so the constraint is

    S = sqrt(P^2 + Q^2) > S_rated

`S` is not a function of the marginal of `P` or the marginal of `Q`. Two models
with identical marginals and different dependence give different exceedance
probabilities. The constraint is a joint functional by construction.

**Voltage.** A breach matters operationally when it coincides with high loading
or high export, so the quantity of interest is a joint tail event such as
`P(V < V_min and P > P_high)`, not the marginal probability of a low voltage.

**Hosting capacity.** "How much PV can this feeder take" is
`P(any constraint violated) < epsilon`, a joint probability over the whole
triple across a horizon.

An independent model asserts that every cross-target correlation is zero. On this
data the observed P-V correlation is -0.72. For a joint tail event that is not a
cosmetic difference.

## What this does to the energy-score result

It converts it from an awkward negative into a finding worth stating:

> Joint modelling does not improve average predictive skill, and does change the
> estimated probability of constraint violation, because the constraint is a
> joint tail functional to which an average score is largely blind.

The energy score is a global measure dominated by the bulk of the distribution.
Every operational constraint is a tail event. The two should not be expected to
agree, and the claim is falsifiable either way: compute the exceedance
probability under both heads and compare with the realised frequency. If they
agree, joint modelling is genuinely unnecessary here and that is a useful
negative result for the field. If they diverge, the work has shown where it
matters and why the generic score missed it.

## The link to Chapter 5: constraint-aware quantile placement

Chapter 5 learns where to place quantile levels and finds the grid adapts to lead
time but not to the day. That is unsurprising in hindsight: nothing in the
pinball objective knows what the quantiles are **for**.

Operationally the distribution does not matter uniformly. What matters is
resolution near the constraint. The 97th percentile of apparent power against a
transformer rating is worth more than ten levels in the bulk.

**Proposal: let the operational constraint determine where the learned grid goes.**
Chapter 5 asks where quantiles should be placed; Chapter 6 supplies the answer
that the objective was missing. The two datasets then become a natural
experiment, because the constraint sits in a different place in each:

* on the US network the decision-relevant region is the upper tail of `S`;
* on the European network it is the tail of `V`.

If the learned grid moves toward the thermal tail on one network and the voltage
tail on the other, that is evidence the method tracks the operational problem
rather than fitting noise. It is also a far stronger test than the
`|Q''|^(-1/3)` correlation currently used, which has no operational meaning and
which the power analysis showed the diagnostic had only 31 % power to detect at
nine slots.

## Contribution, stated plainly

Not a new forecasting architecture. The contribution is:

1. The object a distribution operator needs is a **calibrated joint predictive
   distribution over coupled quantities**, because the decisions are joint tail
   functionals rather than marginal expectations.
2. Such a forecast should be **evaluated against the constraint it will be used
   for**, not only against generic proper scores, and the two can disagree.
3. Where a model learns its own quantile levels, those levels should be placed
   **where the constraint binds**, which gives the Chapter 5 mechanism the
   objective it was missing.
4. The same forecast object serves **two regulatory regimes with different
   binding constraints**, which is a stronger generalisation claim than a lower
   score on one dataset.

The models are the vehicle. The claim is about what should be forecast, and how
it should be judged.

## Scenario-conditional forecasting: the forecast as a planning tool

Everything above treats the forecast as an operational object, answering "what
will happen tomorrow". The same model answers a planning question that is worth
more, and the data already supports it.

### The idea

The scenario grid is 2 feeders x 4 EV x 4 PV penetration levels, and `ev_pen`
and `pv_pen` are columns of the dataset rather than only a filename convention.
Train **one** model with penetration as an exogenous input rather than one model
per scenario, and the penetration input can then be varied at inference. The
model becomes a counterfactual generator:

    "what would the P, Q, V distribution at this substation look like
     if EV penetration were 60 % instead of 40 %?"

The trends the power flow produces are strong and monotone, so there is a real
signal to learn:

| | EV 0.0 -> 0.6 | PV 0.0 -> 0.6 |
|---|---|---|
| peak active power | 157 -> 308 kW | little change |
| minimum active power | unchanged | 29 -> **-183 kW**, reverse flow |
| minimum busbar voltage | 1.034 -> 1.026 pu | unchanged |

### Why this is worth more than another metric

A year of time-series power flow takes hours per scenario. A trained conditional
forecaster returns the predictive distribution in milliseconds. Hosting-capacity
assessment needs many penetration levels across many feeders, so the forecaster
becomes a **learned surrogate for repeated power-flow simulation**, and one that
carries uncertainty where the simulation gives a single deterministic answer.

### The operational output: probabilistic hosting capacity

Hosting capacity is conventionally the largest DER penetration at which no
constraint is violated, computed deterministically. With a joint probabilistic
forecast conditioned on penetration it becomes

    HC(epsilon) = max { pen : P(constraint violated | pen) <= epsilon }

a risk-based limit rather than a worst-case one. Because the constraints are the
joint functionals of the previous section, this needs exactly the joint
predictive distribution the chapter argues for. The three ideas close on each
other: joint forecasting, constraint-based evaluation, and penetration
conditioning are the same contribution seen from three sides.

Reported as a curve of exceedance probability against penetration, per feeder
and per constraint, the deliverable is the thing a DSO actually wants: how much
more EV or PV this feeder takes before the risk of a breach passes what the
operator will accept.

### What has to be verified first

The failure mode is a model that ignores the penetration input and fits the
marginal behaviour of the pooled data. That must be tested, not assumed:

* **Sensitivity.** Vary only `ev_pen` at inference, with weather and calendar
  held fixed, and confirm the forecast moves in the direction and roughly the
  magnitude the power flow gives. A model whose output is flat in the input has
  learned nothing about penetration.
* **Interpolation.** Train on penetrations {0.0, 0.2, 0.6}, test at 0.4 against
  the power-flow ground truth for 0.4, which the model has never seen.
* **Extrapolation.** Train on {0.0, 0.2, 0.4}, test at 0.6. This is the case
  planning actually needs and the one most likely to fail; a negative result
  here bounds the claim honestly rather than after someone else finds it.
* **Confounding.** Penetration is fixed within a scenario while weather varies
  across the year, so the model can reach the right answer through season rather
  than through penetration. The sensitivity test with weather held fixed is what
  separates the two.

## Further use cases, in order of what they cost to build

These reuse the same forecast object. None needs a new model.

### U1. State estimation with forecast-derived uncertainty

`experiments/lv-multitarget/state_estimation.py` already feeds forecast P, Q and
V into a pandapower weighted-least-squares estimator and scores the resulting
customer-terminal voltages against the OpenDSS truth. What it does not yet do is
take the **uncertainty** from the forecast. The measurement weights are fixed
constants:

    STD_V_SUBSTATION = 0.002    # pu
    STD_PQ_SUBSTATION = 0.02    # fraction of reading
    STD_PQ_SMARTMETER = 0.10

Assumed pseudo-measurement uncertainty is the standing weakness of LV state
estimation: the network is unobservable, the estimator leans on
pseudo-measurements, and their weights are guessed. A calibrated predictive
distribution supplies them per time step instead of by assumption.

**A limitation, checked rather than assumed.** `pp.create_measurement` accepts
only a scalar `std_dev`, and `estimate()` exposes no covariance argument:
pandapower's WLS takes a **diagonal** measurement covariance. The joint head's
off-diagonal terms therefore cannot enter the estimator directly. That splits
the use case in two, and only the second tests the joint claim:

* **U1a, heteroscedastic weights.** Replace the constants with the forecast's own
  predictive standard deviation at each step. Cheap, and it improves the
  estimator, but it is a claim about calibration rather than about dependence:
  an independent head supplies these weights just as well.
* **U1b, ensemble state estimation.** Draw coherent triples from the joint
  predictive distribution, as `joint_vs_independent.py` already does, run the
  estimator on each draw, and report the distribution of estimated customer
  voltages. The correlation now matters, because a sample from independent
  marginals can be a triple no network state realises, and the estimator's
  response to such a triple is the operational cost of ignoring the coupling.
  This routes around the diagonal-R limitation instead of fighting it.

**Decides:** whether the covariance changes an operational quantity measured in
volts at a customer terminal rather than in nats.

### U1c. Forecast-in-the-loop power flow: assessment without optimisation

The forecast drives an OpenDSS power flow, and the power flow reports the
consequence. No control law is designed and nothing is optimised: the network
model is used as an oracle, which is what a distribution operator does in
day-ahead security assessment.

**The physical consistency test.** This is the cleanest use of the joint claim
available, and it needs only a power flow.

    forecast (P, Q, V)
        -> feed the forecast P and Q into the network as load
        -> solve the power flow
        -> obtain V_pf, the voltage those injections actually produce
        -> compare against V, the voltage the model forecast

A forecast whose three targets are mutually coherent should satisfy
`V ~ V_pf`. A forecast assembled from independent marginals need not: it can be
individually well calibrated on each target and still assert a voltage that the
active and reactive power it predicts would not produce. The discrepancy
`|V - V_pf|` is a direct, physical measure of incoherence, in per unit.

This replaces the linearised oracle proposed earlier. Fitting `V = a + bP + cQ`
reached only `R2 = 0.23` on the UK data because the varying MV source drives much
of the voltage independently. Solving the actual power flow removes the need to
fit anything, and it accounts for the source, the network impedance and the
Volt-VAr response exactly rather than approximately.

**Risk screening across DER penetration.** The same loop, run as a scan rather
than a search:

* draw sampled triples from the predictive distribution for a day;
* solve the power flow for each draw;
* count the fraction of draws that breach a constraint, giving
  `P(voltage violation)`, `P(transformer overload)`, `P(export beyond a
  threshold)` for that day;
* repeat across the EV and PV penetration grid.

The output is an exceedance-probability curve against penetration, per feeder
and per constraint. Reading the tolerated penetration off that curve gives a
risk-based hosting capacity. It is a scan over the penetration levels already
simulated, not an optimisation, and no control is designed.

**Why the forecast must be joint and probabilistic.** A power flow needs a
coherent `(P, Q)` pair as input; an independent sampler supplies pairs whose
combination may correspond to no realisable operating point, and the power flow
will faithfully report the consequences of an input that does not occur.
Sampling, rather than a point forecast, is what turns the power flow into a
distribution of network states and therefore into a probability rather than a
single verdict.

**Implementation reuses what exists.** `run_powerflow.py` already assembles
loads from shapes and solves the yearly series; the change is to drive the load
from forecast samples for the assessed day instead of from the historical
profile. `joint_vs_independent.py` already produces the paired samples.

### U2. Reverse power flow and export events

At 60 % PV the data already reaches `p_min = -183 kW`: sustained export. Export
drives curtailment, protection coordination and tariff design, and it is a
directional event, so `P(export > threshold)` is a natural probabilistic target
that needs no new data. Cheapest item on this list.

### U3. Flexibility need assessment

Not a dispatch rule, which would require designing a controller. The assessment
question is how much flexibility a feeder would need and when: for each day and
penetration level, the magnitude and duration of the excursion beyond the
constraint, read off the sampled power flows. The output is a distribution of
"how many kW for how many hours", which is what a flexibility procurement or a
reinforcement business case is built from. Whether anyone dispatches it, and
how, is outside this chapter.

### U4. Headroom assessment

The operational inverse of hosting capacity, on a daily rather than a planning
horizon: given tomorrow's forecast and the constraint, how much additional EV
load could this feeder absorb, and with what confidence? Same power-flow scan as
U1c with an added load block, reported as a headroom distribution rather than a
schedule.

## Test plan

Ordered by how decisive each step is, cheapest first.

### T1. Constraint exceedance under both heads, US network

The cheapest decisive test, and the reason to prioritise the SMART-DS
generation.

* Draw paired samples from the joint and independent heads, as
  `joint_vs_independent.py` already does.
* For each sample compute `S = sqrt(P^2 + Q^2)` and the exceedance indicator
  against the transformer rating.
* Report predicted exceedance probability from each head against the realised
  frequency in the held-out period, with a reliability curve.
* **Decides:** whether the joint head's advantage is real but invisible to
  average scores. If both heads reproduce the realised frequency, the chapter's
  claim does not hold operationally and should be reported as such.

### T2. Joint voltage tail event, European network

* Same protocol with `P(V outside EN 50160 band and P above its 90th
  percentile)`.
* **Decides:** whether the same conclusion holds where voltage rather than
  thermal loading binds.

### T3. Constraint-aware quantile placement

* Add a term that weights the pinball loss toward the levels near the
  constraint, or condition the proposal on the constraint level.
* Compare the learned grid on the two networks: thermal tail on SMART-DS,
  voltage tail on `uk-mvlv`.
* **Decides:** whether the Chapter 5 mechanism becomes useful once it is given
  an operational objective. This is the novel piece and should not start until
  T1 has shown the constraint framing separates the models at all.

### T4. Scenario-conditional what-if

Only after T1, since a surrogate for a constraint probability is worth building
only if the constraint probability distinguishes anything.

* Retrain with `ev_pen` and `pv_pen` as exogenous inputs on the pooled scenario
  grid, one model rather than sixteen.
* Run the sensitivity, interpolation and extrapolation checks above.
* Produce the exceedance-probability curve against penetration for each feeder
  and each constraint, and read the risk-based hosting capacity off it.
* Compare against the deterministic hosting capacity the power flow gives, and
  report the runtime difference, which is the practical argument for the
  surrogate.
* **Decides:** whether the forecaster is usable as a planning tool, not only an
  operational one.

### T5. State estimation, U1a then U1b

* U1a first, since it is a few lines: swap the constant `std_dev` values for the
  forecast's predictive standard deviation per step and re-score.
* U1b next: sample coherent triples, run WLS per draw, report the spread of
  estimated customer voltages under each head.
* **Decides:** whether joint modelling changes an operational estimate. This is
  the sharpest available test of the chapter's claim, because it measures the
  consequence in volts rather than in a score.

### T6. Forecast-in-the-loop power flow

No optimisation, no control design: the power flow is used as an oracle.

* **T6a, consistency.** Feed forecast P and Q into OpenDSS, solve, and compare
  the resulting `V_pf` against the forecast `V`. Report the distribution of
  `|V - V_pf|` under the joint and independent heads. **Decides** whether the
  coupling matters, measured in per unit against the physics rather than against
  a score.
* **T6b, risk screening.** Solve the power flow per sampled triple, count
  constraint breaches, and plot exceedance probability against EV and PV
  penetration for each feeder. **Decides** the risk-based hosting capacity, and
  whether the two heads disagree about it.

### T7. Decision-level evaluation

* Translate exceedance probabilities into the decision they support: transformer
  replacement priority in the US case, PV hosting capacity in the European one.
* Report how often each head would lead to a different decision, which is the
  quantity that actually matters to an operator.

## Risks to state rather than discover late

* **T1 may show no difference.** Then the chapter's headline claim weakens and
  the honest result is that independent marginals suffice at this aggregation
  level. Better to find that now than after the write-up.
* **The US scenario is climate-transplanted** if SMART-DS profiles are used on a
  European network. Only scenario B2, the native network, avoids this.
* **Aggregation differs by an order of magnitude** between the two systems, about
  6 customers against 188. That is a genuine confound for a cross-system claim
  and should be reported as an axis of the study rather than hidden.
* **Conditional extrapolation may fail.** Predicting at a penetration outside
  the training range is the case planning needs and the one most likely to
  break. If it does, the honest claim is interpolation only, which is still
  useful because it removes most of the simulation burden within the grid.
* **Constraint-aware placement changes the objective**, so Chapter 5's existing
  results are not directly comparable to it. It is an extension, not a
  correction.
