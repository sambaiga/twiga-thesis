"""Soundness and sensitivity analysis for the CWE interval metric.

This module underpins the Chapter 6 revision (see ``Feedback/revision-plan.md``,
Phase 1).  It provides three things:

1. :func:`soundness_sweep` - the **ideal-forecaster test**.  Given a known target
   distribution, sweep a forecast family that contains the truth and report where
   each metric is optimised.  A sound metric is optimised *at* the truth.
2. :func:`ranking_stability` - the **hyperparameter sensitivity test**.  Measure how
   often varying a CWE hyperparameter re-orders a set of candidate models.
3. :func:`cwc_score` - the CWC baseline, which ``twiga`` does not implement, needed
   to reproduce the Chapter 6 comparison figures.

All CWE numbers come from :mod:`twiga.core.metrics.cwe`, so the analysis tracks the
library rather than the (superseded) formulae currently written in the thesis.

Note:
    The thesis defines CWC inconsistently: Chapter 2 uses
    ``NMPI + lambda * exp(-gamma * |PICP - alpha|)`` (lower is better) while
    Chapter 6 uses ``(1 - NMPI) * exp(-eta * (PICP - (1 - alpha))^2)`` (higher is
    better).  :func:`cwc_score` implements the **Chapter 6** form, which is the one
    the CWE comparison is built against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from twiga.core.metrics.cwe import cwe_score
from twiga.core.metrics.interval import interval_coverage, nmpi, winkler_score

__all__ = [
    "cwc_score",
    "gaussian_crps",
    "ranking_stability",
    "soundness_sweep",
]


# ── Baseline metric not present in twiga ──────────────────────────────────────


def cwc_score(
    nmpi_val: ArrayLike,
    picp: ArrayLike,
    alpha: float = 0.10,
    eta: float = 30.0,
) -> np.ndarray:
    r"""Coverage Width Criterion (CWC), Chapter 6 formulation.

    .. math::
        \mathrm{CWC} = (1 - \mathrm{NMPI})
                       \exp\!\bigl(-\eta (\mathrm{PICP} - (1-\alpha))^2\bigr)

    Higher is better.  Unlike CWE this penalises coverage on **both** sides of the
    nominal level, but it ignores point-forecast accuracy and carries the free
    parameter ``eta``.

    Args:
        nmpi_val: Normalised interval width(s).
        picp: Empirical coverage(s) in ``[0, 1]``.
        alpha: Significance level; nominal coverage is ``1 - alpha``.
        eta: Coverage-penalty strength (30 in the thesis, following Jensen et al.).

    Returns:
        CWC score(s), same broadcast shape as the inputs.
    """
    nmpi_arr = np.asarray(nmpi_val, dtype=float)
    picp_arr = np.asarray(picp, dtype=float)
    return (1.0 - nmpi_arr) * np.exp(-eta * (picp_arr - (1.0 - alpha)) ** 2)


def gaussian_crps(mu: ArrayLike, sigma: ArrayLike, y: ArrayLike) -> np.ndarray:
    r"""Closed-form CRPS for a Gaussian predictive distribution.

    .. math::
        \mathrm{CRPS}(\mathcal{N}(\mu,\sigma), y)
        = \sigma\Bigl[z(2\Phi(z)-1) + 2\varphi(z) - \tfrac{1}{\sqrt{\pi}}\Bigr],
        \quad z = \frac{y-\mu}{\sigma}

    Lower is better.  Used because CRPS is a strictly proper scoring rule and so
    provides the reference behaviour against which CWE and CWC are judged.

    Args:
        mu: Predictive mean(s).
        sigma: Predictive standard deviation(s), strictly positive.
        y: Observation(s).

    Returns:
        Pointwise CRPS values.
    """
    mu_a, sigma_a, y_a = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(y, float)
    z = (y_a - mu_a) / sigma_a
    return sigma_a * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


# ── 1. Ideal-forecaster (soundness) test ──────────────────────────────────────


def soundness_sweep(
    sigmas: Sequence[float] | np.ndarray | None = None,
    alpha: float = 0.10,
    n_samples: int = 200_000,
    kappa_mode: str = "nominal",
    nmpi_scale_mode: str = "range",
    seed: int = 0,
    cwe_kwargs: dict | None = None,
) -> dict[str, np.ndarray | float]:
    r"""Sweep a Gaussian forecast family through the truth and score each member.

    The target is :math:`Q = \mathcal{N}(0, 1)`.  The forecast family is
    :math:`P_\sigma = \mathcal{N}(0, \sigma)`, which contains the truth at
    :math:`\sigma = 1`.  A **sound** metric is optimised at :math:`\sigma = 1`;
    any other optimum means an inhonest forecast scores better than the truth.

    Args:
        sigmas: Forecast standard deviations to sweep.  Defaults to 0.30-2.00.
        alpha: Significance level; the interval is the central ``1 - alpha`` one.
        n_samples: Monte-Carlo draws from the target distribution.
        kappa_mode: Rule defining ``kappa`` for CWE's reference width.
            ``"nominal"`` (default) uses the calibrated interval width
            ``q_{1-alpha/2} - q_{alpha/2}``; ``"std"``, ``"mad"`` and ``"iqr"``
            are the dispersion measures used in the thesis, none of which is
            tied to ``alpha``.
        nmpi_scale_mode: Denominator ``R`` for NMPI (``"range"``, ``"max"``).
        seed: RNG seed.
        cwe_kwargs: Extra keyword arguments forwarded to ``cwe_score``.

    Returns:
        Dict with the swept ``sigmas``, the diagnostic series (``picp``, ``nmpi``),
        each metric series (``cwe``, ``cwc``, ``winkler``, ``crps``), and the
        location of each metric's optimum (``*_argopt``).  ``winkler`` and ``crps``
        are negatively oriented (lower is better); ``cwe`` and ``cwc`` positively.
    """
    if sigmas is None:
        sigmas = np.linspace(0.30, 2.00, 171)
    sigmas = np.asarray(sigmas, dtype=float)
    cwe_kwargs = cwe_kwargs or {}

    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n_samples)  # draws from the true Q = N(0, 1)
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    # Reference quantities are properties of the *target*, not of any forecast.
    if kappa_mode == "nominal":
        # The width a correctly calibrated central (1 - alpha) interval actually
        # has.  This is the only reference for which the CWE width term and the
        # CWE coverage term peak at the same forecast.
        kappa = float(np.quantile(y, 1 - alpha / 2) - np.quantile(y, alpha / 2))
    elif kappa_mode == "std":
        kappa = float(np.std(y))
    elif kappa_mode == "mad":
        kappa = float(stats.median_abs_deviation(y))
    elif kappa_mode == "iqr":
        kappa = float(stats.iqr(y))
    else:
        raise ValueError(f"unknown kappa_mode {kappa_mode!r}")

    scale_r = float(y.max() - y.min()) if nmpi_scale_mode == "range" else float(np.abs(y).max())
    ref = kappa / scale_r

    picp_s, nmpi_s, cwe_s, cwc_s, wink_s, crps_s = ([] for _ in range(6))
    for s in sigmas:
        lower, upper = -z * s, z * s
        lo = np.full_like(y, lower)
        up = np.full_like(y, upper)

        picp = float(interval_coverage(y, lo, up))
        nmpi_v = float(nmpi(lo, up, scale=scale_r))
        # Point forecast is the (correct) mean 0, so the accuracy weight lambda is
        # identical across the sweep and cannot drive the optimum.
        err = float(np.mean(np.abs(y - 0.0))) / scale_r

        picp_s.append(picp)
        nmpi_s.append(nmpi_v)
        cwe_s.append(float(np.ravel(cwe_score(nmpi_v, picp, err, ref, alpha=alpha, **cwe_kwargs))[0]))
        cwc_s.append(float(cwc_score(nmpi_v, picp, alpha=alpha)))
        wink_s.append(float(winkler_score(y, lo, up, alpha=alpha)))
        crps_s.append(float(np.mean(gaussian_crps(0.0, s, y))))

    out: dict[str, np.ndarray | float] = {
        "sigmas": sigmas,
        "picp": np.array(picp_s),
        "nmpi": np.array(nmpi_s),
        "cwe": np.array(cwe_s),
        "cwc": np.array(cwc_s),
        "winkler": np.array(wink_s),
        "crps": np.array(crps_s),
        "kappa": kappa,
        "scale_r": scale_r,
        "ref": ref,
    }
    # Positively oriented metrics are maximised; negatively oriented ones minimised.
    for name, better in (("cwe", "max"), ("cwc", "max"), ("winkler", "min"), ("crps", "min")):
        series = out[name]
        idx = int(np.argmax(series) if better == "max" else np.argmin(series))
        out[f"{name}_argopt"] = float(sigmas[idx])
    return out


# ── 2. Hyperparameter ranking-stability test ──────────────────────────────────


def ranking_stability(
    param: str,
    values: Sequence[float],
    n_trials: int = 400,
    n_models: int = 6,
    alpha: float = 0.10,
    seed: int = 0,
    ref_sampler: Callable[[np.random.Generator], float] | None = None,
) -> float:
    """Fraction of random model sets whose CWE ranking changes with a parameter.

    A parameter that only rescales the score leaves the ordering intact and is
    therefore free.  A parameter that re-orders models is a live degree of freedom
    - a place where a reported conclusion can be tuned.

    Args:
        param: CWE keyword to vary.  One of ``"k"``, ``"beta"``, ``"tol"``,
            ``"penalty_under"``, ``"penalty_over"``, or the pseudo-parameter
            ``"ref"`` which varies the reference width ``kappa / R``.
        values: Values of ``param`` to compare.
        n_trials: Number of random model populations to draw.
        n_models: Candidate models per population.
        alpha: Significance level.
        seed: RNG seed.
        ref_sampler: Optional callable drawing the reference width per trial.

    Returns:
        Fraction of trials in ``[0, 1]`` whose ranking is not invariant.
    """
    rng = np.random.default_rng(seed)
    if ref_sampler is None:

        def ref_sampler(r: np.random.Generator) -> float:
            return float(r.uniform(0.12, 0.40))

    def _score(models: list[tuple[float, float, float]], ref: float, **kw) -> np.ndarray:
        return np.array(
            [float(np.ravel(cwe_score(n, p, e, ref, alpha=alpha, **kw))[0]) for p, n, e in models],
        )

    changed = 0
    for _ in range(n_trials):
        models = [
            (float(rng.uniform(0.50, 0.99)), float(rng.uniform(0.05, 0.60)), float(rng.uniform(0.03, 0.20)))
            for _ in range(n_models)
        ]
        ref = ref_sampler(rng)
        baseline = None
        for v in values:
            if param == "ref":
                order = tuple(np.argsort(-_score(models, ref * v)))
            else:
                order = tuple(np.argsort(-_score(models, ref, **{param: v})))
            if baseline is None:
                baseline = order
            elif order != baseline:
                changed += 1
                break
    return changed / n_trials
