r"""Decision scoring by forecast type, for the Chapter 6 operational decision experiment.

Each probabilistic head exposes a different object, and each is turned into a
decision through that object rather than through one sampler for all:

=================  ==================================  ===========================================
Head               Native output                       Route to an event probability or warning
=================  ==================================  ===========================================
QR, FPQR, CQR      marginal quantiles per target/step  draws from each cell's own quantile
                                                       function with exponential tails,
                                                       reordered by a Schaake shuffle
Laplace            closed-form marginal per target     closed-form draws, reordered by the
                                                       same Schaake shuffle
BNN                weight-draw scenarios plus noise    its own paths, one weight draw per day
Max-norm           a calibrated region at one level    warning when the region reaches the
conformal                                              violation set; conformal p-value as a
                                                       plausibility, not a probability
=================  ==================================  ===========================================

The events (:class:`DecisionLimits`), each needing the joint forecast:

* ``overload_<k>x_step``: apparent power :math:`\sqrt{P^2+Q^2}` exceeds ``k``
  times the pre-DER headroom at a step (joint P and Q), for each severity tier;
* ``overload_<k>x_day``: a sustained overload, the tier exceeded for at least
  ``min_duration`` consecutive steps (joint over the horizon);
* ``voltage_step``: the lowest customer voltage leaves its statutory band at a
  step. Statutory limits apply at customer terminals, not at the busbar, so
  the customer voltage is the busbar voltage less the LV feeder drop,
  :math:`V_c = V - (aP + bQ + c)`, a linearisation of
  :math:`\Delta V \approx (RP + XQ)/V` fitted on the days before the test
  window (joint P, Q and V), with its residual resampled into every scenario.
  The observed event uses the recorded customer voltage. Only where the
  record has customer voltages;
* ``union_day``: a sustained overload or a customer-voltage breach at any step.

Probability heads are scored on the probability tier (Brier, fair Brier for
ensembles, skill against the base rate of the days before the test window,
relative economic value) and every head on the warning tier at shared
cost-loss ratios, where a conformal region at level ``alpha = 2 C/L`` is the
decision a user with ratio ``C/L`` would take on a one-sided event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.stats import laplace, norm
from twiga.core.metrics.decision import probability_scores, relative_economic_value_curve, warning_scores
from twiga.distributions.scenarios import quantile_inverse, schaake_shuffle

COST_LOSS_RATIOS = (0.05, 0.1, 0.2)


class HeadKind(StrEnum):
    """The kind of predictive object a head exposes."""

    QUANTILE = "quantile"
    LAPLACE = "laplace"
    BNN = "bnn"


@dataclass(frozen=True)
class VoltageDrop:
    r"""Busbar-to-customer voltage drop :math:`V - V_c = aP + bQ + c + e`, fitted on past days.

    The aggregate busbar P and Q cannot see where on the feeder load clusters,
    so the fit leaves a residual :math:`e` whose upper tail is where customer
    breaches sit. The residuals are kept so scenarios carry that uncertainty.

    Attributes:
        coefficients: ``(a, b, c)``.
        residuals: Residuals of the fit on the past days, in per unit.
    """

    coefficients: tuple[float, float, float]
    residuals: np.ndarray

    def fitted(self, values: np.ndarray, target_index: tuple[int, int, int]) -> np.ndarray:
        """Mean drop :math:`aP + bQ + c` for values of shape ``(..., N, H, D)``."""
        p_idx, q_idx, _ = target_index
        a, b, c = self.coefficients
        return a * values[..., p_idx] + b * values[..., q_idx] + c


@dataclass(frozen=True)
class DecisionLimits:
    """Operating limits that define the events.

    Attributes:
        rating: Apparent-power reference: the 98th percentile of the network's
            no-DER loading, i.e. its pre-DER peak headroom, so event rates move
            with penetration.
        target_index: Positions of P, Q and V in the forecaster's target order.
        severities: Overload tiers as multiples of ``rating``: 1.0 is the headroom
            exhausted (a reinforcement trigger), 1.25 an emergency-loading tier.
        min_duration: Consecutive steps above a tier that make a day's sustained
            overload (4 half-hours: two hours), as loading guides judge overload
            by duration.
        voltage_band: Statutory customer-voltage band in per unit, ``(low, high)``.
        voltage_drop: The fitted drop model; ``None`` scores thermal events only.
    """

    rating: float
    target_index: tuple[int, int, int]
    severities: tuple[float, ...] = (1.0, 1.25)
    min_duration: int = 4
    voltage_band: tuple[float, float] = (0.94, 1.10)
    voltage_drop: VoltageDrop | None = None


@dataclass
class Predictive:
    """One fold's predictive output for the test windows, in original units.

    Attributes:
        kind: The head's kind of predictive object.
        loc: Point forecast, shape ``(N, H, D)``.
        quantiles: Quantile values, ``(N, L, H, D)``, quantile heads only.
        levels: Their levels per cell, ``(N, L, H, D)``, quantile heads only.
        scale: Laplace dispersion ``b``, ``(N, H, D)``, Laplace only.
        paths: Weight-draw means, ``(S, N, H, D)``, BNN only.
        noise_scale: Learned observation noise, ``(N, H, D)``, BNN only.
    """

    kind: HeadKind
    loc: np.ndarray
    quantiles: np.ndarray | None = None
    levels: np.ndarray | None = None
    scale: np.ndarray | None = None
    paths: np.ndarray | None = None
    noise_scale: np.ndarray | None = None


# ── Events ────────────────────────────────────────────────────────────────────


def overload_name(severity: float, scope: str) -> str:
    """Event name for an overload tier: ``overload_1x_step``, ``overload_1.25x_day``."""
    return f"overload_{severity:g}x_{scope}"


def events_from_values(
    values: np.ndarray, limits: DecisionLimits, v_customer: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Event indicators for observations or scenarios.

    Args:
        values: Shape ``(..., N, H, D)``: observations, or ``P`` scenarios ahead of them.
        limits: Event definitions.
        v_customer: Lowest customer voltage, ``(..., N, H)``: the recorded value for
            observations, a drop-model draw for scenarios. When ``None`` and a drop
            model is set, the drop model's mean is used.

    Returns:
        Indicators: step events ``(..., N, H)``, day events ``(..., N)``. Voltage
        events are present only when ``limits.voltage_drop`` is set.
    """
    p_idx, q_idx, v_idx = limits.target_index
    apparent = np.hypot(values[..., p_idx], values[..., q_idx])
    events: dict[str, np.ndarray] = {}
    for severity in limits.severities:
        step = apparent > severity * limits.rating
        events[overload_name(severity, "step")] = step
        events[overload_name(severity, "day")] = _has_run(step, limits.min_duration)
    if limits.voltage_drop is not None:
        if v_customer is None:
            v_customer = values[..., v_idx] - limits.voltage_drop.fitted(values, limits.target_index)
        low, high = limits.voltage_band
        voltage = (v_customer < low) | (v_customer > high)
        sustained = events[overload_name(min(limits.severities), "day")]
        events |= {"voltage_step": voltage, "union_day": sustained | voltage.any(axis=-1)}
    return events


def fit_voltage_drop(frame: pd.DataFrame, columns: tuple[str, str, str, str]) -> VoltageDrop:
    """Least-squares fit of the busbar-to-customer drop :math:`V - V_c = aP + bQ + c`, keeping the residuals.

    Args:
        frame: Observations before the test window.
        columns: Names of P, Q, busbar V and lowest customer V.

    Returns:
        The fitted :class:`VoltageDrop`.
    """
    p, q, v, v_customer = (frame[name].to_numpy(dtype=float) for name in columns)
    design = np.column_stack([p, q, np.ones_like(p)])
    drop = v - v_customer
    coefficients, *_ = np.linalg.lstsq(design, drop, rcond=None)
    return VoltageDrop(
        coefficients=(float(coefficients[0]), float(coefficients[1]), float(coefficients[2])),
        residuals=drop - design @ coefficients,
    )


def scenario_customer_voltage(paths: np.ndarray, limits: DecisionLimits, rng: np.random.Generator) -> np.ndarray:
    """Customer voltage for each scenario: busbar V less the fitted drop and a resampled residual.

    Drawing the residual from the past days' fit carries the drop model's own
    uncertainty (the feeder-position effects busbar P and Q cannot see) into
    the event probability, rather than treating the mapping as exact.
    """
    drop = limits.voltage_drop
    residual = rng.choice(drop.residuals, size=paths.shape[:-1])
    return paths[..., limits.target_index[2]] - drop.fitted(paths, limits.target_index) - residual


def _has_run(flags: np.ndarray, length: int) -> np.ndarray:
    """Whether ``flags`` holds ``length`` consecutive true values along the last axis."""
    if length <= 1:
        return flags.any(axis=-1)
    counts = np.cumsum(flags, axis=-1, dtype=int)
    padded = np.concatenate([np.zeros((*flags.shape[:-1], 1), dtype=int), counts], axis=-1)
    window = padded[..., length:] - padded[..., :-length]
    return (window == length).any(axis=-1)


def _sustained_plausibility(step: np.ndarray, length: int) -> np.ndarray:
    """Largest plausibility over runs of ``length`` steps, each run as plausible as its least plausible step."""
    if length <= 1:
        return step.max(axis=-1)
    windows = np.lib.stride_tricks.sliding_window_view(step, length, axis=-1)
    return windows.min(axis=-1).max(axis=-1)


# ── Probability heads ─────────────────────────────────────────────────────────


def bnn_paths(pred: Predictive, n_paths: int, rng: np.random.Generator, noise: str = "cell") -> np.ndarray:
    """BNN scenarios: each path keeps one weight draw for the whole day and all targets, plus noise.

    The shared weight draw carries the model's own dependence across targets
    and steps (epistemic). The learned observation noise is added either
    independently per cell (``noise="cell"``, the model's own assumption) or
    once per path and target, held over the whole day (``noise="day"``), so the
    noise keeps the day's persistence instead of breaking it at every step.

    Raises:
        ValueError: On an unknown ``noise`` mode.
    """
    draws = pred.paths[np.arange(n_paths) % pred.paths.shape[0]]
    if noise == "cell":
        shocks = rng.standard_normal(draws.shape)
    elif noise == "day":
        shocks = np.broadcast_to(rng.standard_normal((*draws.shape[:2], 1, draws.shape[-1])), draws.shape)
    else:
        raise ValueError(f"unknown noise mode {noise!r}")
    return draws + shocks * pred.noise_scale[None]


def marginal_paths(pred: Predictive, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Independent draws from each marginal, shape ``(P, N, H, D)``.

    Quantile heads draw from their grid, Laplace from its closed form, the BNN
    from the normal with its predictive mean and standard deviation.
    """
    u = rng.uniform(size=(n_paths, *pred.loc.shape))
    if pred.kind is HeadKind.LAPLACE:
        return laplace.ppf(u, pred.loc[None], pred.scale[None])
    if pred.kind is HeadKind.BNN:
        return norm.ppf(u, pred.loc[None], pred.scale[None])
    q = np.moveaxis(pred.quantiles, 1, -1)  # (N, H, D, L)
    tau = np.moveaxis(pred.levels, 1, -1)
    return quantile_inverse(u, q, tau)


def scenario_paths(pred: Predictive, n_paths: int, template: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Joint scenarios over targets and steps, shape ``(P, N, H, D)``.

    BNN: its own paths. Quantile and Laplace heads: marginal draws reordered by
    the Schaake shuffle on ``template`` (``P`` historical days); with no template
    the draws keep independent margins (the "no dependence" ablation).
    """
    if pred.kind is HeadKind.BNN:
        return bnn_paths(pred, n_paths, rng)
    paths = marginal_paths(pred, n_paths, rng)
    return paths if template is None else schaake_shuffle(paths, template)


def scenario_forms(
    pred: Predictive, n_paths: int, template: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Every way a head's forecast becomes joint scenarios, ``(P, N, H, D)`` each.

    * ``independent`` -- draws from each marginal on its own (no dependence);
    * ``schaake`` -- the same draws reordered to the ranks of ``template`` days (an empirical copula);
    * BNN only: ``native`` -- its weight draws with per-cell noise, as the model assumes, and
      ``native_day`` -- its weight draws with day-coherent noise.

    The BNN's marginal forms need its predictive ``scale``; without it only the native forms are returned.
    """
    forms: dict[str, np.ndarray] = {}
    if pred.kind is not HeadKind.BNN or pred.scale is not None:
        independent = marginal_paths(pred, n_paths, rng)
        forms |= {"independent": independent, "schaake": schaake_shuffle(independent, template)}
    if pred.kind is HeadKind.BNN:
        forms |= {"native": bnn_paths(pred, n_paths, rng, "cell"), "native_day": bnn_paths(pred, n_paths, rng, "day")}
    return forms


def probabilities_from_paths(paths: np.ndarray, limits: DecisionLimits, rng: np.random.Generator) -> dict:
    """Probability of every event as the share of the given joint scenarios in which it occurs."""
    v_customer = scenario_customer_voltage(paths, limits, rng) if limits.voltage_drop is not None else None
    joint = events_from_values(paths, limits, v_customer)
    return {name: indicator.mean(axis=0) for name, indicator in joint.items()}


def event_probabilities(
    pred: Predictive,
    limits: DecisionLimits,
    n_paths: int,
    template: np.ndarray | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Probability of every event, as the share of joint scenarios in which it occurs."""
    paths = scenario_paths(pred, n_paths, template, rng)
    v_customer = scenario_customer_voltage(paths, limits, rng) if limits.voltage_drop is not None else None
    joint = events_from_values(paths, limits, v_customer)
    return {name: indicator.mean(axis=0) for name, indicator in joint.items()}


# ── Conformal regions ─────────────────────────────────────────────────────────


def _thermal_radius(loc: np.ndarray, sigma: np.ndarray, limit: float, target_index: tuple[int, int, int]) -> np.ndarray:
    r"""Smallest box radius at which the region reaches an apparent-power limit, shape ``(N, H)``.

    The largest apparent power in the box :math:`|\mu| + r\sigma` per target is
    reached at its outer corner, so the radius solves
    :math:`(a + r s_p)^2 + (b + r s_q)^2 = R^2` with :math:`a = |\mu_P|`, :math:`b = |\mu_Q|`.
    """
    p_idx, q_idx, _ = target_index
    a, b = np.abs(loc[..., p_idx]), np.abs(loc[..., q_idx])
    s_p, s_q = sigma[..., p_idx], sigma[..., q_idx]
    quad_a = s_p**2 + s_q**2
    quad_b = 2.0 * (a * s_p + b * s_q)
    quad_c = a**2 + b**2 - limit**2
    root = (-quad_b + np.sqrt(np.maximum(quad_b**2 - 4.0 * quad_a * quad_c, 0.0))) / (2.0 * quad_a)
    return np.where(quad_c >= 0.0, 0.0, root)


def _voltage_radius(loc: np.ndarray, sigma: np.ndarray, limits: DecisionLimits, margin: float = 0.95) -> np.ndarray:
    r"""Smallest box radius at which the customer voltage can reach either band edge, shape ``(N, H)``.

    Over the box, the customer voltage :math:`V - aP - bQ - c` moves by at most
    :math:`r(\sigma_V + |a|\sigma_P + |b|\sigma_Q)` from its value at the centre.
    The region covers the forecast targets, not the drop model, so the centre
    is shifted by the drop residual's ``margin`` quantile toward each band edge.
    """
    p_idx, q_idx, v_idx = limits.target_index
    drop = limits.voltage_drop
    a, b, _ = drop.coefficients
    low, high = limits.voltage_band
    centre = loc[..., v_idx] - drop.fitted(loc, limits.target_index)
    lowest = centre - np.quantile(drop.residuals, margin)
    highest = centre - np.quantile(drop.residuals, 1.0 - margin)
    spread = sigma[..., v_idx] + abs(a) * sigma[..., p_idx] + abs(b) * sigma[..., q_idx]
    return np.maximum(np.minimum((high - highest) / spread, (lowest - low) / spread), 0.0)


def conformal_decisions(
    loc: np.ndarray,
    sigma: np.ndarray,
    conformal,
    limits: DecisionLimits,
    cost_loss_ratios: tuple[float, ...] = COST_LOSS_RATIOS,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Warnings and conformal plausibilities of a max-norm region.

    At ratio ``C/L`` the region is taken at ``alpha = 2 C/L``: for a one-sided
    event, acting when a central region's bound crosses the limit is acting when
    the exceedance probability passes ``alpha / 2``. The region reaches an event
    exactly when the radius it needs is within the calibrated radius, i.e. when
    the event's conformal p-value exceeds ``alpha``. A sustained overload is as
    plausible as its least plausible step, maximised over the day's runs; the
    region's coverage holds per step, not over the day.

    Args:
        loc: Region centre, ``(N, H, D)``.
        sigma: Region scale per target, ``(N, H, D)``.
        conformal: A calibrated :class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`.
        limits: Event definitions.
        cost_loss_ratios: Operating points.

    Returns:
        ``(warnings, plausibility)``: warnings per ratio and event, and the
        conformal p-value of each event.
    """
    plausibility: dict[str, np.ndarray] = {}
    for severity in limits.severities:
        step = conformal.p_value(_thermal_radius(loc, sigma, severity * limits.rating, limits.target_index))
        plausibility[overload_name(severity, "step")] = step
        plausibility[overload_name(severity, "day")] = _sustained_plausibility(step, limits.min_duration)
    if limits.voltage_drop is not None:
        voltage = conformal.p_value(_voltage_radius(loc, sigma, limits))
        sustained = plausibility[overload_name(min(limits.severities), "day")]
        plausibility |= {"voltage_step": voltage, "union_day": np.maximum(sustained, voltage.max(axis=-1))}
    warnings = {ratio: {name: p > 2.0 * ratio for name, p in plausibility.items()} for ratio in cost_loss_ratios}
    return warnings, plausibility


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_probabilities(
    probabilities: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    reference_rates: dict[str, float],
    ensemble_size: int | None,
    cost_loss_ratios: tuple[float, ...] = COST_LOSS_RATIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probability-tier and warning-tier rows for one condition and fold, plus its value curve.

    Returns:
        ``(rows, curve)``: one row per (tier, event, metric) and the REV curve per event.
    """
    rows, curves = [], []
    for name in observed:
        p, o = probabilities[name], observed[name]
        for metric, value in probability_scores(p, o, reference_rates[name], ensemble_size).items():
            rows.append({"tier": "probability", "event": name, "metric": metric, "value": value})
        ratios, value = relative_economic_value_curve(p, o, reference_rate=reference_rates[name])
        curves.append(pd.DataFrame({"event": name, "cost_loss": ratios, "value": value}))
        for ratio in cost_loss_ratios:
            for metric, v in warning_scores(p > ratio, o, ratio, reference_rates[name]).items():
                rows.append({"tier": "warning", "event": name, "cost_loss": ratio, "metric": metric, "value": v})
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def score_warnings(
    warnings: dict[float, dict[str, np.ndarray]],
    plausibility: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    reference_rates: dict[str, float],
) -> pd.DataFrame:
    """Warning-tier rows for a conformal region, and its plausibility's reliability as a diagnostic."""
    rows = []
    for ratio, by_event in warnings.items():
        for name, warn in by_event.items():
            for metric, v in warning_scores(warn, observed[name], ratio, reference_rates[name]).items():
                rows.append({"tier": "warning", "event": name, "cost_loss": ratio, "metric": metric, "value": v})
    for name, p in plausibility.items():
        # Plausibility is not a calibrated probability; its Brier score is reported
        # as a diagnostic so the chapter can show how far it is from one.
        scores = probability_scores(p, observed[name], reference_rates[name])
        rows.append({"tier": "plausibility", "event": name, "metric": "brier_score", "value": scores["brier_score"]})
    return pd.DataFrame(rows)


def reference_rates(
    history: np.ndarray, limits: DecisionLimits, v_customer: np.ndarray | None = None
) -> dict[str, float]:
    """Event rates over historical days, the climatology the skill and value scores compare against.

    Args:
        history: Observed days before the test window, shape ``(K, H, D)``.
        limits: Event definitions.
        v_customer: Recorded lowest customer voltage on those days, ``(K, H)``.
    """
    events = events_from_values(history, limits, v_customer)
    return {name: float(indicator.mean()) for name, indicator in events.items()}


def history_template(history: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """``n_paths`` template days for the Schaake shuffle, the most recent first.

    Uses the latest ``n_paths`` days before the test window (closest in season,
    after Scheuerer et al., 2017), drawing with replacement when there are fewer.

    Args:
        history: Observed days before the test window, oldest first, ``(K, H, D)``.
        n_paths: Template size, equal to the number of scenarios.
        rng: Generator for the draw with replacement.

    Returns:
        Template, shape ``(n_paths, H, D)``.
    """
    if history.shape[0] >= n_paths:
        return history[-n_paths:]
    return history[rng.integers(0, history.shape[0], size=n_paths)]
