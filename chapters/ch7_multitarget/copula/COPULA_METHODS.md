# Copula methods in twiga: theory, mathematics, literature, and what is novel here

Seven pieces of code implement copula-based dependence modelling across
`twiga/distributions/nn/copula.py`, `twiga/distributions/nn/vine_copula.py`,
`twiga/distributions/nn/copula_sampling.py`, and
`twiga/distributions/conformal/multivariate.py`. This document gives the theory
and mathematics behind each, grounds it in the literature it follows, and
states plainly what is novel relative to that literature and what
hypothesis each piece lets Chapter 6 test empirically. Code references are
exact (file, class/function) so this document can be checked against the
implementation directly rather than taken on faith.

## 0. The common foundation: Sklar's theorem

Every method below rests on the same decomposition. For a random vector
$Y = (Y_1, \ldots, Y_D)$ with joint CDF $F$ and marginal CDFs
$F_1, \ldots, F_D$, Sklar's theorem (Sklar 1959) guarantees a function
$C : [0,1]^D \to [0,1]$, the copula, such that

$$
F(y_1, \ldots, y_D) = C\bigl(F_1(y_1), \ldots, F_D(y_D)\bigr).
$$

$C$ is itself a valid joint CDF on uniform margins, and it is unique when
the margins are continuous. The practical consequence every method here
exploits: the margins and the dependence structure can be estimated
separately, then recombined. In twiga, the margins are whatever a quantile
head (FPQR) already learned; the copula supplies only the missing piece,
the dependence.

**Reference.** Sklar, A. (1959). Distribution functions of $n$ dimensions
and their margins (original title in French). *Publications de l'Institut
de Statistique de l'Université de Paris*, 8, 229-231.

---

## 1. Static Gaussian copula (post-hoc, fitted once)

**Code.** `fit_gaussian_copula`, `quantile_pit`, `coupled_samples` in
`twiga/distributions/nn/copula_sampling.py`.

### Theory

A quantile head gives each target's own inverse CDF, $F_d^{-1}$, at a
discrete set of learned levels. The probability integral transform (PIT),

$$
U_d = F_d(Y_d),
$$

maps each target back to a uniform margin (Rosenblatt 1952): this is
exactly what removes the marginal shape and leaves only the dependence,
the operational content of Sklar's theorem. `quantile_pit` computes this
numerically: since the quantile function is known only at the levels the
head proposed, $F_d$ is taken to be piecewise linear between them.

A Gaussian copula assumes the dependence among the $U_d$ is that of a
multivariate normal transformed through its own margins:

$$
C_R(u_1, \ldots, u_D) = \Phi_R\bigl(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_D)\bigr),
$$

where $\Phi$ is the standard normal CDF and $\Phi_R$ the CDF of a
$D$-dimensional normal with correlation matrix $R$. $R$ is estimated by
transforming the PIT values to normal scores and taking their empirical
correlation:

$$
Z_d = \Phi^{-1}(U_d), \qquad \hat R = \operatorname{corr}(Z_1, \ldots, Z_D).
$$

This is estimated on a **held-out** split (the validation rows, not the
test rows the model is scored on), so the correlation the copula asserts
never sees the period it is later used to couple.

### What is implemented

`fit_gaussian_copula(observations, quantiles, quantile_levels)` returns
$\hat R$, a single $D \times D$ matrix pooled across every batch element
and lead time (not one per horizon step). `coupled_samples(...,
family="gaussian")` draws from it: standard normals are rotated by
$\hat R$'s Cholesky factor, mapped through $\Phi$ back to correlated
uniforms, then through each target's own quantile function (inverse PIT)
to physical units. The margins are provably unchanged by this; only the
joint arrangement of draws differs (verified in
`test_margins_are_unchanged_by_coupling`).

### Literature and what is novel

This is a direct, simplified instance of Wen & Torkkola's "deep generative
quantile-copula" decoupling: train quantile margins, separately fit a
copula on PIT values (Wen & Torkkola 2019). Their framework allows the
copula itself to be an arbitrary generative model; twiga's static piece
restricts it to Gaussian, the same pragmatic choice Salinas et al.'s GPVar
makes for neural multivariate forecasting (Salinas et al. 2019): a
closed-form density and a one-line Cholesky sampler, not a claim that
dependence is actually elliptical.

**Nothing here is methodologically novel**: it is a correctly-implemented,
well-attributed instance of an established pattern. Its role in Chapter 6
is as the **baseline** the more elaborate pieces below (2-7) have to beat:
any claim that a fancier copula helps has to clear the bar this cheap,
one-line post-hoc fit already sets.

**Hypothesis it lets the chapter test.** $H_0$: coupling FPQR's margins
with a *static* Gaussian copula, fitted once and reused for every
forecast, recovers enough of the true cross-target dependence to improve
joint/operational scores (energy score, hosting-capacity Brier score) over
sampling the margins independently.

**References.**
Rosenblatt, M. (1952). Remarks on a multivariate transformation. *Annals of
Mathematical Statistics*, 23(3), 470-472.
Wen, R., & Torkkola, K. (2019). Deep generative quantile-copula models for
probabilistic forecasting. arXiv:1907.10697.
Salinas, D., Bohlke-Schneider, M., Callot, L., Medico, R., & Gasthaus, J.
(2019). High-dimensional multivariate forecasting with low-rank Gaussian
copula processes. *NeurIPS*.

---

## 2. Conditional Gaussian copula (trained jointly, per-lead-time)

**Code.** `GaussianCopula` in `twiga/distributions/nn/copula.py`.

### Theory

Section 1's copula is a single fixed matrix: wrong wherever the true
dependence is not stationary. Chapter 6's own motivating example is
exactly this failure mode: the correlation between active and reactive
power changes *sign* with the mix of photovoltaics and electric vehicles
on a feeder, so no single matrix is right across the whole operating
range.

`GaussianCopula` instead predicts a correlation matrix as a function of
the backbone latent $x$ (the same representation the quantile margins are
read off), separately for each lead time $h = 1, \ldots, H$. It is
parameterised through a Cholesky factor $L_h$ with unit-norm rows, so
$R_h = L_h L_h^\top$ has unit diagonal by construction, a valid
correlation matrix, not merely a positive semi-definite one, with no
extra normalisation step needed.

The copula log-density used for training (so the copula's parameters are
fit by maximum likelihood jointly with the margins, not a two-stage
post-hoc procedure) is the standard Gaussian copula density in terms of
the whitened scores $z = \Phi^{-1}(u)$:

$$
\log c(u; R_h) = -\tfrac12 \log|R_h| - \tfrac12 \left( z^\top R_h^{-1} z - z^\top z \right).
$$

### What is implemented

`GaussianCopula.forward(x)` returns $L_h$ of shape $(B, H, D, D)$;
`.log_prob(u, corr_tril)` implements the density above via a triangular
solve (`torch.linalg.solve_triangular`), avoiding an explicit matrix
inverse; `.sample`/`.sample_uniform` draw coupled uniforms and push them
through the margins' inverse CDF, exactly mirroring Section 1's mechanics
but with a per-instance, per-lead-time $R_h$ instead of one fixed matrix.
The off-diagonal layer is zero-initialised, so the copula starts at
independence and has to earn any dependence it adds during training
(`test_starts_at_independence`).

### Literature and what is novel

Conditioning a copula's parameters on covariates is itself an established
idea (conditional-copula literature, e.g. Patton 2006 for time-varying
conditional copulas in econometrics; Krupskii & Joe 2020 for
factor-copula conditioning). What is specific to this implementation,
relative to Wen & Torkkola (2019) and Salinas et al.'s GPVar (2019): both
of those either fit the copula post-hoc (the former) or condition a
low-rank covariance on an RNN hidden state built for panels of many
exchangeable series (the latter, GPVar). `GaussianCopula` predicts a
**full** (not low-rank) correlation matrix, because Chapter 6's problem is
the opposite shape from GPVar's: a handful of named, physically coupled
targets ($P$, $Q$, $V$), not hundreds of interchangeable ones; a full
matrix is the right complexity for three targets, where GPVar's low-rank
approximation would be solving a scaling problem twiga does not have.

**What is genuinely novel here, in the narrow but real sense of "not found
assembled this way in the cited literature":** predicting a full-rank,
per-lead-time correlation matrix from the same backbone latent an FPQR
quantile head already conditions on, trained end-to-end with the pinball
objective via the Gaussian copula's own log-likelihood term. The pieces
(conditional copulas, Gaussian copulas for neural forecasting, FPQR
margins) are each established; assembling them this way, at this scale,
for this problem shape, is the contribution.

**Hypothesis it lets the chapter test.** $H_1$: the correlation between
$P$ and $Q$ genuinely varies with the conditioning covariates (time of
day, PV output, EV load) in a way a single static matrix cannot
represent, and a model that predicts a covariate-conditioned correlation
captures more of the true joint density (higher held-out copula
log-likelihood, better joint/operational scores) than the static copula
of Section 1.

**References.**
Patton, A. J. (2006). Modelling asymmetric exchange rate dependence.
*International Economic Review*, 47(2), 527-556.
Krupskii, P., & Joe, H. (2020). Flexible copula models with dynamic
dependence and application to financial data. *Econometrics and
Statistics*, 16, 148-167.
(Wen & Torkkola 2019; Salinas et al. 2019 as in Section 1.)

---

## 3. HorizonRoPEConditioner: how the per-lead-time conditioning is computed

**Code.** `HorizonRoPEConditioner` in `twiga/distributions/nn/copula.py`,
built on `twiga.models.nn.core.embedding.RotaryEmbedding`.

### Theory

Section 2 needs $H$ different correlation matrices, one per lead time. The
naive way to get them is $H$ independent linear output blocks from one
flat layer, which is what the first version of this code did. That treats
lead time $h$ and $h+1$ as unrelated: nothing in the parameterisation says
the correlation an hour ahead should resemble the correlation an hour and
a half ahead.

Rotary Position Embedding (RoPE; Su et al. 2021) encodes a discrete
position by rotating pairs of feature dimensions by an angle proportional
to that position, rather than by adding a learned or fixed positional
vector. Two positions close together produce nearly-aligned rotations
(their dot product decays smoothly with distance); two far apart produce
nearly-orthogonal ones. Applied to lead time, this is exactly the
smoothness structure the flat linear layer lacked: the projection at $h$
and $h+1$ is close in the rotated space by construction, not by accident
of training.

Formally, for a 2-D feature pair $(a, b)$ and integer position $p$, with
base frequency $\omega$:

$$
\operatorname{rotate}(a, b; p) = \bigl(a\cos(p\omega) - b\sin(p\omega),\; a\sin(p\omega) + b\cos(p\omega)\bigr).
$$

`RotaryEmbedding` applies this pairwise across the embedding's feature
axis with a geometrically spaced set of frequencies (as in Su et al.
2021), giving each position a unique, smoothly-varying rotation.

### What is implemented

`HorizonRoPEConditioner(hidden_size, embed_dim, forecast_horizon)`
broadcasts the backbone latent $x$ to every lead time, projects it once
(`nn.Linear(hidden_size, embed_dim)`), then rotates the result by integer
positions $0, \ldots, H-1$. This is **exactly**
`twiga.distributions.nn.fpqr.tau_embed.TauRoPEEmbedding`'s
`rope_type="linear"` construction, applied to the horizon axis instead of
FPQR's quantile-slot axis: the same primitive, a different axis.
`GaussianCopula`'s `offdiag_layer` and `ClaytonCopula`'s `theta_layer`
(Section 4) both consume this conditioner's output rather than the raw
latent directly.

### Literature and what is novel

RoPE itself is Su et al. (2021)'s contribution, already in production use
in FPQR's own tau embedding inside this codebase. **The novel step here is
reusing FPQR's own rotary embedding machinery to condition a copula's
dependence parameter on lead time, instead of introducing a second,
unrelated conditioning mechanism.** This is a design-consistency
contribution rather than a modelling one: it means the copula's notion of
"nearby lead times behave similarly" is expressed with the same tool FPQR
already uses to express "nearby quantile levels behave similarly," rather
than inventing a parallel positional scheme. No published copula work
known to this document does this; the reuse is specific to having FPQR
and a lead-time-conditioned copula share one codebase.

**Hypothesis it lets the chapter test.** $H_2$: sharing statistical
strength across nearby lead times (via rotation) produces a
better-calibrated, lower-variance correlation estimate at longer
lookaheads than treating every lead time as an independent fit,
particularly relevant since later lead times have less directly
informative signal in the backbone latent and benefit most from borrowing
structure from their neighbours.

**Reference.**
Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021 / updated
2024). RoFormer: Enhanced transformer with rotary position embedding.
arXiv:2104.09864.

---

## 4. ClaytonCopula: a one-parameter Archimedean alternative

**Code.** `ClaytonCopula` in `twiga/distributions/nn/copula.py`.

### Theory

A Gaussian copula has a structural limitation with physical consequences
here: it assigns *vanishing* probability to two targets being jointly
extreme, however strong their correlation, because the multivariate
normal's tail decays too fast. Undervoltage coinciding with thermal
overload is exactly a one-sided joint-extreme event, the case Chapter 6's
hosting-capacity criterion is written about (`Chapter-6.tex`,
`eqn:ch6-hosting-capacity`), so a copula family with genuine lower-tail
dependence is a better structural match than a Gaussian one, whatever
correlation is fitted.

Archimedean copulas are built from a generator function
$\varphi : [0,1] \to [0, \infty)$, strictly decreasing with
$\varphi(1) = 0$:

$$
C(u_1, \ldots, u_D) = \varphi^{-1}\bigl(\varphi(u_1) + \cdots + \varphi(u_D)\bigr).
$$

Clayton's generator is $\varphi(t) = (t^{-\theta} - 1)/\theta$ for
$\theta > 0$ (Clayton 1978), giving

$$
C(u_1, \ldots, u_D) = \left(\sum_{d=1}^{D} u_d^{-\theta} - D + 1\right)^{-1/\theta}.
$$

As $\theta \to 0$, $C$ approaches the independence copula; as
$\theta \to \infty$, it approaches perfect (comonotone) dependence. The
$D$-dimensional density, obtained by differentiating $C$ $D$ times (Joe
1997, ch. 5), is:

$$
c(u_1, \ldots, u_D; \theta) = \left[\prod_{k=0}^{D-1} (1 + k\theta)\right]
(u_1 \cdots u_D)^{-\theta - 1}
\left(\sum_{d=1}^{D} u_d^{-\theta} - D + 1\right)^{-(1/\theta + D)}.
$$

This trades away Gaussian's free pairwise correlations for a single,
*exchangeable* parameter shared across every pair (a genuine restriction:
Clayton cannot separately represent "$P$ and $Q$ correlate strongly, $P$
and $V$ weakly," only one overall strength), in exchange for lower-tail
dependence the Gaussian family structurally cannot provide at any
correlation.

Sampling uses the Marshall-Olkin frailty representation (Marshall & Olkin
1988): Clayton's generator is the Laplace transform of a
$\mathrm{Gamma}(1/\theta, 1)$ random variable, so

$$
V \sim \mathrm{Gamma}(1/\theta, 1), \qquad X_1, \ldots, X_D \sim \mathrm{Exp}(1) \text{ i.i.d.}, \qquad
U_d = \left(1 + \frac{X_d}{V}\right)^{-1/\theta}
$$

gives $U_1, \ldots, U_D$ marginally uniform and jointly
Clayton($\theta$) (this document verified the marginal-uniformity claim
analytically for this construction, and it is confirmed empirically in
`test_clayton_samples_are_marginally_uniform`).

### What is implemented

`ClaytonCopula(num_targets, hidden_size, forecast_horizon)` uses the same
`HorizonRoPEConditioner` as `GaussianCopula` (Section 3), with a final
`nn.Linear(embed_dim, 1)` and a softplus to keep $\theta > 0$, initialised
so $\theta$ starts near zero (independence), mirroring `GaussianCopula`'s
zero-initialised off-diagonal. `.log_prob` implements the density above
exactly (verified against the closed-form bivariate density,
`test_clayton_log_prob_matches_bivariate_closed_form`, matching to
$10^{-5}$). `.sample_uniform` implements the Marshall-Olkin construction;
the empirical Kendall's $\tau$ of its draws matches the value
`.kendall_tau(theta)` predicts to within Monte Carlo error
(`test_clayton_samples_reproduce_the_requested_kendall_tau`).
$\tau = \theta / (\theta + 2)$ (`.kendall_tau`) gives a single, bounded,
interpretable dependence summary in place of `GaussianCopula.correlation`'s
matrix.

### Literature and what is novel

Archimedean copulas, Clayton's family, its density, and the Marshall-Olkin
sampler are all established (Clayton 1978; Nelsen 2006; Joe 1997; Marshall
& Olkin 1988): no claim of novelty on the copula theory itself. What is
implemented here that is not standard: a **backbone-conditioned** Clayton
copula, predicting $\theta$ per lead time from the same latent an FPQR
quantile head uses, trained by the same maximum-likelihood machinery
`GaussianCopula` uses, and sharing its `HorizonRoPEConditioner` so the two
are interchangeable at the call site (same constructor and forward
contract, different generator). Conditional Archimedean copulas exist in
the econometrics literature (Patton 2006 covers Clayton as one case), but
not, to this document's knowledge, conditioned this way on a neural
forecaster's backbone latent for multi-target quantile forecasting.

**Hypothesis it lets the chapter test.** $H_3$: for the specific one-sided
joint-extreme event Chapter 6's hosting-capacity criterion scores
(simultaneous undervoltage and overload), a Clayton-coupled ensemble
gives a better-calibrated exceedance probability (lower Brier score on
the hosting-capacity event, better Brier skill score / reliability) than
a Gaussian-coupled one, because Clayton's lower-tail dependence is the
structurally correct shape for that event and Gaussian's is not, even
though Clayton is otherwise the less flexible copula (fewer parameters,
one dependence number instead of a matrix).

**References.**
Clayton, D. G. (1978). A model for association in bivariate life tables
and its application in epidemiological studies of familial tendency in
chronic disease incidence. *Biometrika*, 65(1), 141-151.
Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman &
Hall.
Marshall, A. W., & Olkin, I. (1988). Families of multivariate
distributions. *Journal of the American Statistical Association*,
83(403), 834-841.

---

## 5. Static Clayton copula (post-hoc, fitted once)

**Code.** `fit_clayton_theta` in `twiga/distributions/nn/copula_sampling.py`;
the `family="clayton"` branch of `coupled_samples`.

### Theory

The static counterpart of Section 4, mirroring Section 1's relationship
to Section 2. Clayton's $\theta$ has no direct correlation interpretation
(unlike Gaussian's $R$, which *is* a Pearson correlation matrix on normal
scores), so it cannot be estimated the same way Section 1 estimates $R$.
Instead it is estimated by inverting the population relationship between
Clayton's $\theta$ and Kendall's rank correlation $\tau$:

$$
\tau = \frac{\theta}{\theta + 2} \quad\Longleftrightarrow\quad \theta = \frac{2\tau}{1 - \tau}.
$$

This is the classical method-of-moments estimator for Archimedean
copulas (Genest & MacKay 1986): compute the empirical Kendall's $\tau$ of
the PIT values (which is invariant to the marginal transform, so it can
be computed directly on held-out PIT pairs), then invert the closed-form
relationship above. For more than two targets, the exchangeable Clayton
model shares one $\theta$ across every pair, so the empirical $\tau$ is
averaged over all $\binom{D}{2}$ target pairs, mirroring
`fit_gaussian_copula`'s single pooled correlation matrix.

### What is implemented

`fit_clayton_theta(observations, quantiles, quantile_levels)` computes
PIT values via `quantile_pit` (identical to Section 1's), takes Kendall's
$\tau$ for every target pair via `scipy.stats.kendalltau`, averages, and
inverts. `coupled_samples(..., family="clayton", theta=...)` draws from
the fitted $\theta$ via the same Marshall-Olkin construction Section 4's
`ClaytonCopula.sample_uniform` uses (verified to give matching results by
construction, both implement the identical formula).

### Literature and what is novel

Genest & MacKay (1986)'s $\tau$-inversion estimator is standard practice
for Archimedean copulas; nothing here is a new estimator. Novelty is
limited to integration: this is the drop-in, no-training-required Clayton
counterpart to Section 1's Gaussian fit, usable identically in
`chapters/ch7_multitarget/forecast_examples.py`'s
`"fpqr shared/per-target + clayton copula"` conditions, so the static
Gaussian-vs-Clayton comparison ($H_0$ vs. a Clayton-flavoured $H_0$) can
be run without touching a trained model at all.

**Hypothesis it lets the chapter test.** $H_4$: even without conditioning
$\theta$ on covariates at all (the static, cheapest-possible fit, exactly
analogous to Section 1), Clayton's lower-tail dependence alone, not the
conditioning of Sections 2-4, accounts for most of the hosting-capacity
benefit a copula can offer, making the marginal value of conditioning
(Sections 2-4 over Sections 1/5) a separate, testable question rather
than one bundled with the choice of copula family.

**Reference.**
Genest, C., & MacKay, J. (1986). The joy of copulas: Bivariate
distributions with uniform marginals. *The American Statistician*, 40(4),
280-283.

---

## 6. Copula-informed multivariate conformal calibration

**Code.** `MultivariateConformal(score_type="mahalanobis")` in
`twiga/distributions/conformal/multivariate.py`.

### Theory

Split conformal prediction (Vovk et al. 2005; Lei et al. 2018) calibrates
a prediction region by taking the $(1-\alpha)$ empirical quantile of a
nonconformity score on held-out calibration data, giving
distribution-free, finite-sample coverage. `MultivariateConformal`'s
existing `"max-norm"` score pools scale-normalised absolute errors across
targets by taking the max:

$$
\gamma_{i,h} = \max_d \frac{|y_{i,h,d} - \mu_{h,d}|}{\sigma_{h,d}},
$$

which implicitly treats every target as equally likely to be jointly
extreme (an isotropic assumption in the normalised space), regardless of
whatever dependence structure a copula elsewhere in the pipeline has
already estimated.

`"mahalanobis"` replaces the max with the quadratic form a fitted
correlation $R_h$ induces:

$$
\gamma_{i,h} = \sqrt{z_{i,h}^\top R_h^{-1} z_{i,h}}, \qquad
z_{i,h,d} = \frac{y_{i,h,d} - \mu_{h,d}}{\sigma_{h,d}}.
$$

This is the natural conformal counterpart to Sections 1-2's Gaussian
correlation: rather than treating each target as independently extreme,
it weights the joint deviation by how surprising it is *given the
learned dependence*.

### What is implemented, and the honest correction from testing it

`get_scores(..., score_type="mahalanobis")` computes the formula above via
`np.linalg.solve`, accepting either a single shared correlation
($H \times D \times D$, e.g. from Section 1) or a per-sample one
($N \times H \times D \times D$, e.g. from Section 2's
`GaussianCopula.correlation`). `generate_intervals` reports the same
axis-aligned box as `"max-norm"`, $\hat q \cdot \sigma$.

**This module's own docstring states, and its test suite verifies, a
correction to the original design intent.** The reported box is provably
a *superset* of the ellipsoid the score is actually calibrated against (a
correlation matrix's unit diagonal means the ellipsoid's support along
any one axis equals the box's half-width exactly, but the box's corners
lie outside the ellipsoid whenever the correlation is non-identity).
Consequently `"mahalanobis"` gives **valid but conservative** coverage
(strictly above nominal, confirmed empirically to run 2-6 points high
depending on correlation strength; see
`tests/unit/conformal/test_conformal_multivariate.py::TestMultivariateConformalMahalanobis`),
and under a *stationary* correlation it does **not** produce a tighter
interval than `"max-norm"`: direct comparison at multiple correlation
strengths found `"max-norm"`'s own calibration, which already implicitly
reflects whatever correlation is present in the calibration sample
through the empirical distribution of the max, gives an equal-or-smaller
threshold every time tested.

The score type is retained because it is the only one of the three that
can consume a correlation that genuinely **varies per sample** (Section
2's conditional `GaussianCopula` output): `"max-norm"` has no mechanism
to use predicted dependence that changes with the conditioning input,
only whatever correlation happens to be present, unconditionally, in the
calibration sample. It is not, and should not be presented as, a general
efficiency upgrade over `"max-norm"` when correlation is fixed.

### Literature and what is novel

Copula-based conformal prediction for multivariate targets is an active,
recent research area: Messoudi et al. (2021) fit a copula on
nonconformity scores directly to build a joint conformal region; a 2024
ICLR paper extends copula-conformal ideas to multi-step time series.
`MultivariateConformal`'s `"mahalanobis"` score is a much simpler,
self-contained instance of the same underlying idea (use a fitted
dependence structure to shape a conformal region rather than assuming
independence), implemented as a drop-in `score_type` on an existing class
rather than a new conformal method. Its honest characterisation
(valid-but-conservative, not a free efficiency win under stationary
correlation) is itself the finding worth reporting, since the intuitive
claim, "accounting for correlation should tighten the region," turned out
false for the box-shaped output this module reports, and demonstrating
*why* (the box-vs-ellipsoid mismatch) is a small, self-contained, correct
piece of methodology in its own right.

**Hypothesis it lets the chapter test.** $H_5$ (revised after empirical
verification): `"mahalanobis"` conformal calibration only earns its
complexity over `"max-norm"` when the copula's correlation genuinely
varies per instance (Section 2/4's conditional copulas) rather than being
fixed across the calibration set: test by comparing realised interval
width at matched, empirically-verified joint coverage between
`"max-norm"` and `"mahalanobis"` fed a per-sample-varying correlation, not
a shared one.

**References.**
Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a
Random World*. Springer.
Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.
(2018). Distribution-free predictive inference for regression. *Journal
of the American Statistical Association*, 113(523), 1094-1111.
Messoudi, S., Destercke, S., & Rousseau, S. (2021). Copula-based
conformal prediction for multi-target regression. *Pattern Recognition*,
120, 108101.

---

## 7. A three-target D-vine: TACTiS's benefit without attention

**Code.** `DVineCopula3`, `h_gaussian`/`h_clayton` (and their inverses) in
`twiga/distributions/nn/vine_copula.py`; `fit_dvine3` and
`coupled_samples(..., family="vine")` in `copula_sampling.py` for the
static post-hoc counterpart.

### Theory

Sections 1-6 each commit to one global parametric family (Gaussian or
Clayton) across all three targets at once. TACTiS's actual methodological
point (Drouin et al. 2022; Ashok et al. 2024) is that this single-family
commitment is itself the limitation attention lifts, by learning a
non-parametric copula directly from data. A D-vine (pair-copula
construction; Bedford & Cooke 2001; Aas, Czado, Frigessi & Bakken 2009)
gets the same "no single global family" property through ordinary
factorisation instead: with $Q$ central, the three-target density is
written as a product of three *bivariate* copulas,

$$
c(u_P, u_Q, u_V) = c_{PQ}(u_P, u_Q) \cdot c_{QV}(u_Q, u_V) \cdot
c_{PV \mid Q}\bigl(h(u_P \mid u_Q),\, h(u_V \mid u_Q)\bigr),
$$

where $h(u \mid v)$ is a pair copula's *h-function*, the conditional
distribution function $F(U \le u \mid V = v)$. Tree 1 has the two
unconditional edges $C_{PQ}$, $C_{QV}$; tree 2 has the one conditional
edge $C_{PV \mid Q}$, built from each tree-1 edge's h-function. Each edge
is free to be a different family (Gaussian or Clayton here), so the joint
structure is never assumed elliptical, or any other single shape, end to
end: only pairs are fit to a family, and different pairs may disagree.

For the Gaussian pair copula, the h-function has the closed form

$$
h_{\text{Gaussian}}(u, v; \rho) = \Phi\!\left(\frac{\Phi^{-1}(u) - \rho\,\Phi^{-1}(v)}{\sqrt{1-\rho^2}}\right),
$$

and for the Clayton pair copula,

$$
h_{\text{Clayton}}(u, v; \theta) = v^{-\theta - 1}\bigl(u^{-\theta} + v^{-\theta} - 1\bigr)^{-(1/\theta + 1)}.
$$

### What is implemented

Each edge reuses `GaussianCopula(num_targets=2, ...)`/
`ClaytonCopula(num_targets=2, ...)` exactly as Sections 2/4 built them,
conditioned via the same `HorizonRoPEConditioner` (Section 3): no new
copula-family code, no attention. `DVineCopula3.log_prob` sums the three
edges' log-densities; `.sample_uniform` implements the standard top-down
D-vine simulation recursion (Aas et al. 2009, Algorithm 1, specialised to
three variables): draw three independent latent uniforms, invert the
$C_{PQ}$ edge to get $u_P$ given $u_Q$, invert the $C_{PV \mid Q}$ edge in
pseudo-observation space to get the tree-2 latent for $V$, then invert
$C_{QV}$ to recover $u_V$.

Verified two ways beyond the standard shape/gradient/independence checks:
an all-Gaussian D-vine's `log_prob` matches a directly-computed 3D
Gaussian copula density to $10^{-7}$, and its samples reproduce an exact
target $3 \times 3$ correlation matrix; with mixed Gaussian/Clayton
edges, the sampler reproduces each edge's requested dependence
independently, including the $P$-$V$ marginal correlation that emerges
purely from the $Q$-mediated path when no direct $P$-$V$ edge is fit at
all: the qualitative signature a correct vine, and not a full correlation
matrix, should produce.

### Literature and what is novel

Vine copulas, their h-functions, and their simulation algorithm are all
established (Bedford & Cooke 2001; Aas et al. 2009): no claim of novelty
on the vine construction itself. What this session adds: building the
vine's edges out of the *specific* conditional copula classes
(`GaussianCopula`/`ClaytonCopula`, themselves conditioned via
`HorizonRoPEConditioner`) already built for Sections 2-4, so a vine is
available at zero additional conditioning-mechanism cost. The "attention
alternative" is assembled entirely from pieces this session already
needed to build anyway, not a separate architecture bolted on afterward.

**Hypothesis it lets the chapter test.** $H_6$: a D-vine, which commits to
no single global family across all three targets, captures
joint/operational performance (energy score, hosting-capacity Brier
score) at least as well as the single-family conditional copulas of
Sections 2 and 4, without needing TACTiS's attention machinery: directly
answering whether this chapter's three-target problem needs attention's
generality at all, or whether a vine of the existing small heads already
captures what matters.

**References.** (Bedford & Cooke 2001; Aas et al. 2009; Drouin et al.
2022; Ashok et al. 2024, as in the module docstring above.)

---

## Summary: hypothesis per method

| # | Method | Code | Hypothesis |
|---|---|---|---|
| 1 | Static Gaussian copula | `copula_sampling.fit_gaussian_copula` | $H_0$: static Gaussian coupling beats independent sampling on joint/operational scores |
| 2 | Conditional GaussianCopula | `copula.GaussianCopula` | $H_1$: dependence genuinely varies with covariates; conditioning captures more of it than a static matrix |
| 3 | HorizonRoPEConditioner | `copula.HorizonRoPEConditioner` | $H_2$: sharing structure across nearby lead times (via rotation) improves the correlation estimate at longer lookaheads |
| 4 | ClaytonCopula (conditional) | `copula.ClaytonCopula` | $H_3$: Clayton's lower-tail dependence gives a better-calibrated hosting-capacity exceedance probability than Gaussian, for this one-sided event |
| 5 | Static Clayton copula | `copula_sampling.fit_clayton_theta` | $H_4$: lower-tail dependence alone (without conditioning) accounts for most of a Clayton copula's benefit, separating the copula-family question from the conditioning question |
| 6 | Mahalanobis conformal | `conformal.multivariate.MultivariateConformal` | $H_5$ (revised): only earns its cost over max-norm when the correlation varies per sample, not when it is fixed |
| 7 | D-vine copula | `vine_copula.DVineCopula3` | $H_6$: no single global family is at least as good as Sections 2/4's single-family conditional copulas on joint/operational scores, without attention |

None of these hypotheses is settled by the code existing; each needs the
corresponding experiment run (`forecast_examples.py --paths N` for 1, 2,
4, 5, 7, with $N > 0$ so joint/operational tiers are actually populated; a
dedicated conformal calibration run for 6) to produce evidence either
way. Section 6's table row is written as "revised" deliberately: it is
the one hypothesis in this document that was tested during
implementation and found to require correction from its original
framing, which is the point of writing hypotheses down before citing
results.
