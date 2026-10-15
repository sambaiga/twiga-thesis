r"""Chapter 6, Experiment 3: does calibrated conformal help the decision, per backbone.

``backbone_conformal_study.py`` (Experiment 2) answers whether backbone
architecture or conformal calibration changes the *marginal* metric/coverage
picture. This script answers the companion decision-criterion question:
does conformal calibration change the *operational* picture, scored as a
hosting-capacity/thermal-exceedance Brier score rather than a generic
interval metric.

Engine-based, via :class:`~twiga.experiment.ExperimentEngine` and
:class:`~twiga.experiment.Condition`'s ``custom_scorer`` hook. Turning a
calibrated region into a Monte-Carlo ensemble and scoring a decision event
on it has no equivalent among the engine's four built-in ``metric_types``
evaluators (point/interval/quantile/parametric), so ``custom_scorer`` runs
that step directly: it receives the already fit and (when the condition
carries a ``conformal_config``) calibrated forecaster for one fold, and
returns the same ``(pred_df, metric_df)`` shape the built-in evaluators do.

Monte Carlo is still required with the copula dropped, not because of it.
The hosting-capacity/thermal event is a union over a nonlinear function,
:math:`S=\sqrt{P^2+Q^2}>S^{\mathrm{rated}}` or :math:`V` out of band, which
has no closed form regardless of how -- or whether -- P/Q/V are coupled: for
``max_norm``/``mahalanobis`` the calibrated object is a geometric region (a
box or an ellipsoid), not a density, so there is no formula to integrate;
for CQR (``quantile``/``quantile_joint``) the margins are sampled
independently (no coupling at all, already the no-copula case), and even
then :math:`S`'s distribution under two independent nonlinear-combined
margins has no closed form either. Sampling is how the union probability is
estimated either way.

Same head x conformal grid as ``backbone_conformal_study.py`` (MLPF, MLPGAM,
NHiTS FPQR, each carrying the FPQR mechanism ablation's simple configuration,
plus QR's fixed-grid baseline, added 2026-09-30 alongside that script's own
QR head; ``none``/``max_norm``/``quantile``/``quantile_joint`` conformal --
``max_abs``/``mahalanobis_*`` dropped 2026-10-02, matching
``backbone_conformal_study.py``'s own trim and Chapter-6.tex's
``sec:ch6-mahalanobis``: a fixed, calibration-set-estimated correlation gave
a 17-37% worse Winkler score than max-norm across every head, so it is not
carried forward as a reported conformal variant here either), restricted
here to ``DSS-UK``/``DSS-US``, the two datasets with a verified decision
threshold (``MLVS-PT-3PHASE`` has none, per ``backbone_conformal_study.py``'s
own docstring). QR reuses the same dedicated, backbone-only HPO search
(``QR_HPO_VARIANT``) there too, rather than its own full joint study.

Turning one calibrated forecaster into a scoreable ensemble, per method
--------------------------------------------------------------------------
* ``none``: sample directly from the model's own raw (uncalibrated)
  predicted quantile grid by inverse-transform sampling -- no calibration,
  today's baseline.
* ``max_norm``: :func:`sample_calibrated_ensemble` draws a Monte-Carlo
  ensemble from the calibrated
  :class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`
  region -- the axis-aligned box ``max_norm`` calibrates.
* ``quantile``/``quantile_joint`` (CQR): the already-calibrated
  :class:`~twiga.distributions.conformal.cqr.ConformalQuantileRegressor`
  resizes only the outermost predicted quantile pair per target; sampling
  is inverse-transform per target, independently -- CQR has no cross-target
  mechanism (the deliberate contrast against the two joint methods above),
  and pooling (``quantile_joint``) only changes the calibration threshold,
  not the sampling step.

The calibration window is a real calendar reservation
(``ExperimentSpec.cv_calib_size``), the same mechanism Experiment 2 uses --
not the test window halved after the fact, which this script's own
hand-rolled predecessor used before this migration.

Usage::

    uv run python experiment/ch7_multitarget/backbone_conformal_decision.py --decisions DSS-UK
    uv run python experiment/ch7_multitarget/backbone_conformal_decision.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_2_backbone_conformal"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from twiga.core.metrics import pairwise_dm_table  # noqa: E402
from twiga.core.metrics.interval import get_interval_metrics
from twiga.core.utils import configure, get_logger, slugify
from twiga.distributions.ml.utils import get_sigma_from_quantile_levels
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec, start_run
from twiga.models.nn import MLPFConfig
from twiga.utils.data_loading import load_dataset
from twiga.utils.run_forecast import PipelineConfig

from chapters.ch7_multitarget.experiment_2_backbone_conformal.backbone_conformal_study import (  # noqa: E402
    CONFORMAL_VARIANTS,
    DATASETS,
    DSS_UK,
    DSS_US,
    HEADS,
    QR_HPO_VARIANT,
)
from chapters.ch7_multitarget.experiment_3_decision.operational_decision import (  # noqa: E402
    INCLUDE_VOLTAGE,
    VOLTAGE_BANDS,
)
from chapters.ch7_multitarget.shared.forecast_examples import hosting_capacity_rows  # noqa: E402
from chapters.ch7_multitarget.shared.tune_backbone import tune as tune_backbone  # noqa: E402
from twiga_thesis.paths import thesis_data_dir

# Same nominal level throughout the project (Chapter 5/7's own convention,
# and TwigaForecaster's own `interval_alpha` default): a 90% central
# interval, alpha=0.10.
INTERVAL_ALPHA = 0.1

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
# parents[3]: experiment_3_decision -> ch7_multitarget -> experiment ->
# twiga-forecast (repo root) -> projects -- matches
# multitarget_experiment.py's/backbone_conformal_study.py's own output_dir
# convention, a sibling of the repo, not inside it. Overridable via
# --output-root or the TWIGA_EXPERIMENTS_ROOT environment variable (added
# 2026-10-01 for uniformity with those two scripts -- this one previously
# had no override at all, hardcoding the "projects"-sibling path).
DEFAULT_OUTPUT_ROOT = (
    Path(os.environ.get("TWIGA_EXPERIMENTS_ROOT", str(script_dir.parents[3] / "experiments"))) / "ch7_multitarget"
)
SCENARIOS = {"DSS-UK": DSS_UK, "DSS-US": DSS_US}
CONFORMAL_METHODS = tuple(CONFORMAL_VARIANTS)


def decision_rating(scenario: str, quantile: float) -> float:
    """Apparent-power rating at *quantile* of the whole record's own S, not per fold.

    A threshold recomputed per fold would move with the season, and the
    decision would then compare exceedance probabilities defined against
    different levels, which is not a comparison at all.
    """
    data, targets, *_ = load_dataset(scenario, data_path=str(thesis_data_dir()))
    p, q = data[targets[0]].to_numpy(), data[targets[1]].to_numpy()
    return float(np.quantile(np.hypot(p, q), quantile))


# ── Ensemble sampling: turn one calibrated forecaster into scoreable draws ────


def sample_unit_ball(
    n_paths: int, n_samples: int, n_horizons: int, n_targets: int, rng: np.random.Generator
) -> np.ndarray:
    """``n_paths`` draws, uniform within the unit ``n_targets``-ball.

    Shape ``(n_paths, n_samples, n_horizons, n_targets)``.

    Direction uniform on the unit sphere (a normalised Gaussian vector), radius
    ``p ** (1 / n_targets)`` for ``p ~ Uniform(0, 1)`` -- the standard
    construction for a uniform density over the ball's volume, not merely its
    surface.
    """
    direction = rng.normal(size=(n_paths, n_samples, n_horizons, n_targets))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-12
    radius_fraction = rng.uniform(0.0, 1.0, size=(n_paths, n_samples, n_horizons, 1))
    return direction * radius_fraction ** (1.0 / n_targets)


def sample_calibrated_ensemble(
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    q_hat: np.ndarray,
    score_type: str,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte-Carlo ensemble drawn from one :class:`MultivariateConformal` calibrated region.

    ``"max_norm"`` draws independent per-target uniform samples inside the
    axis-aligned box :class:`MultivariateConformal.generate_intervals` itself
    reports (``mu +/- q_hat[h] * sigma``). ``"mahalanobis"`` does **not**
    reuse that box -- ``generate_intervals`` reports the same conservative
    box for both score types (see :mod:`twiga.distributions.conformal.multivariate`'s
    module docstring), which would erase the point of comparing the two.
    Instead it draws properly from the calibrated ellipsoid: a point uniform
    in the unit ``D``-ball, scaled by ``q_hat[h]``, transformed by the
    Cholesky factor of ``corr[h]``, added to ``mu``.

    Args:
        mu: Predictive means, shape ``(N, H, D)``.
        sigma: Adaptive scale, shape ``(N, H, D)``.
        corr: Correlation, shape ``(H, D, D)``. Ignored for ``"max_norm"``.
        q_hat: Calibrated per-horizon radius, shape ``(H,)``.
        score_type: ``"max_norm"`` or ``"mahalanobis"``.
        n_paths: Ensemble size.
        rng: Generator.

    Returns:
        Ensemble, shape ``(n_paths, N, H, D)``.
    """
    n_samples, n_horizons, n_targets = mu.shape
    if score_type == "mahalanobis":
        unit_ball = sample_unit_ball(n_paths, n_samples, n_horizons, n_targets, rng) * q_hat[None, None, :, None]
        cholesky_factor = np.linalg.cholesky(corr)  # (H, D, D)
        correlated_offset = np.einsum("hde,pnhe->pnhd", cholesky_factor, unit_ball)
        return mu[None] + sigma[None] * correlated_offset
    radius = q_hat[None, :, None] * sigma  # (N, H, D)
    lower, upper = mu - radius, mu + radius
    draws = rng.uniform(0.0, 1.0, size=(n_paths, n_samples, n_horizons, n_targets))
    return lower[None] + draws * (upper - lower)[None]


def interpolate_inverse_cdf(
    quantile_grid: np.ndarray, levels_sorted: np.ndarray, query_probabilities: np.ndarray
) -> np.ndarray:
    """Linear-interpolated inverse CDF: value at probability ``query_probabilities`` for each ``(path, day, step)``.

    Args:
        quantile_grid: Quantile grid, shape ``(N, L, H)``, sorted along ``L``
            to match ``levels_sorted``.
        levels_sorted: Probability levels, shape ``(L,)``, ascending.
        query_probabilities: Query probabilities, shape ``(P, N, H)``.

    Returns:
        Interpolated values, shape ``(P, N, H)``.
    """
    upper_index = np.clip(np.searchsorted(levels_sorted, query_probabilities), 1, len(levels_sorted) - 1)
    lower_index = upper_index - 1
    level_lower, level_upper = levels_sorted[lower_index], levels_sorted[upper_index]
    weight = np.where(level_upper > level_lower, (query_probabilities - level_lower) / (level_upper - level_lower), 0.0)
    quantile_grid_broadcast = np.broadcast_to(
        quantile_grid[None], (query_probabilities.shape[0], *quantile_grid.shape)
    )  # (P, N, L, H)
    value_lower = np.take_along_axis(quantile_grid_broadcast, lower_index[:, :, None, :], axis=2)[:, :, 0, :]
    value_upper = np.take_along_axis(quantile_grid_broadcast, upper_index[:, :, None, :], axis=2)[:, :, 0, :]
    return value_lower * (1.0 - weight) + value_upper * weight


def average_quantile_levels(quantile_levels: np.ndarray) -> np.ndarray:
    """FPQR's own per-sample levels, averaged to a fixed ``(N_q, T)`` grid.

    With ``per_target_tau=True`` (every backbone in this script), the real
    shape is ``(B, N_q, H, T)`` -- a per-horizon, not just per-sample, grid --
    so both the batch and horizon axes are averaged away here. Without it,
    the shape is the older ``(B, N_q, T)`` and only the batch axis is
    averaged. Either way this is the same simplification
    :meth:`~twiga.forecaster.result.RawPrediction.to_dataframe` and
    ``fit_and_predict`` already use for exactly this reason: sampling needs
    one representative grid per target, not a per-instance (or per-horizon)
    one.
    """
    if quantile_levels.ndim == 4:
        return quantile_levels.mean(axis=(0, 2))
    return quantile_levels.mean(axis=0)


def sample_quantile_ensemble(
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
    *,
    lower_adj: np.ndarray | None = None,
    upper_adj: np.ndarray | None = None,
) -> np.ndarray:
    """Sample per target, independently, by inverse-transform sampling on the (optionally CQR-adjusted) grid.

    Args:
        quantiles: Physical-unit quantiles, shape ``(N, L, H, D)``.
        quantile_levels: Probability levels, shape ``(B, L, H, D)`` with
            ``per_target_tau=True`` or ``(B, L, D)`` without it (FPQR's own
            conventions, averaged over batch and, if present, horizon here)
            or already-reduced ``(L, D)``, or a flat ``(L,)`` grid shared
            across every target and the batch (QR's own fixed-grid
            convention, which carries no target axis at all).
        n_paths: Ensemble size.
        rng: Generator.
        lower_adj: CQR-adjusted lower bound, shape ``(N, H, D)``, replacing
            the outermost predicted quantile when given. ``None`` samples
            directly from the raw predicted grid (the ``"none"`` condition).
        upper_adj: CQR-adjusted upper bound, matching ``lower_adj``.

    Returns:
        Ensemble, shape ``(n_paths, N, H, D)``.
    """
    n_samples, n_levels, n_horizons, n_targets = quantiles.shape
    if quantile_levels.ndim == 1:
        # QR's own convention: one grid, shared across every target (and the
        # batch) -- broadcast to (L, D) so the per-target indexing below is
        # uniform across every quantile head, FPQR's or QR's.
        levels_grid = np.broadcast_to(quantile_levels[:, None], (n_levels, n_targets))
    else:
        levels_grid = average_quantile_levels(quantile_levels) if quantile_levels.ndim > 2 else quantile_levels
    ensemble = np.empty((n_paths, n_samples, n_horizons, n_targets))
    for target_idx in range(n_targets):
        sort_order = np.argsort(levels_grid[:, target_idx])
        levels_sorted = levels_grid[sort_order, target_idx]
        # Select the target axis (basic indexing) before applying `sort_order`
        # (fancy indexing): combining a fancy index with a separated basic
        # index in one call reorders the output axes instead of keeping
        # them in place (same trap documented in get_sigma_from_quantile_levels).
        target_quantiles = quantiles[:, :, :, target_idx][:, sort_order, :].copy()  # (N, L, H)
        if lower_adj is not None:
            target_quantiles[:, 0, :] = lower_adj[:, :, target_idx]
            target_quantiles[:, -1, :] = upper_adj[:, :, target_idx]  # type: ignore[index]
            if target_quantiles.shape[1] > 2:
                # A large q_hat can push an adjusted outer level past its inner
                # neighbour; clip so the grid stays non-decreasing along levels,
                # which interpolate_inverse_cdf's interpolation assumes.
                target_quantiles[:, 0, :] = np.minimum(target_quantiles[:, 0, :], target_quantiles[:, 1, :])
                target_quantiles[:, -1, :] = np.maximum(target_quantiles[:, -1, :], target_quantiles[:, -2, :])
        # Full (0, 1), not [levels_sorted[0], levels_sorted[-1]]: CQR widens the
        # boundary values at the *same* nominal probability levels to fix
        # undercoverage, so the tail mass beyond those levels must still be
        # sampled -- interpolate_inverse_cdf extrapolates linearly past the outer
        # grid point using its adjacent segment's slope, reaching into that
        # widened boundary rather than truncating at it.
        query_probabilities = rng.uniform(0.0, 1.0, size=(n_paths, n_samples, n_horizons))
        ensemble[..., target_idx] = interpolate_inverse_cdf(target_quantiles, levels_sorted, query_probabilities)
    return ensemble


def interval_metric_rows(
    model: str, targets: list[str], true: np.ndarray, lower: np.ndarray, upper: np.ndarray, loc: np.ndarray
) -> list[dict]:
    """Point + interval metrics (NRMSE-equivalent, PICP, NMPI, Winkler score, CWE), per target.

    ``backbone_conformal_study.py`` (Experiment 2) reports these same metrics
    via the engine's built-in ``"interval"`` evaluator; this script bypasses
    that evaluator entirely (``custom_scorer`` replaces it, since turning a
    calibrated region into a decision-scoreable ensemble has no built-in
    equivalent -- see the module docstring), so without this, Experiment 3
    reports operational (Brier-family) metrics only, with no way to check
    whether a backbone/conformal condition that wins the *decision* also has
    a sane *marginal* interval -- exactly the check that would have caught
    the NHITS/DSS-UK interval collapse from this script's own results
    directly, rather than only via Experiment 2's.

    Args:
        model: Condition name.
        targets: Target names, e.g. ``["p_kw", "q_kvar", "v_lv_pu"]``.
        true: Ground truth, shape ``(N, H, D)``.
        lower: Interval lower bound, shape ``(N, H, D)``.
        upper: Interval upper bound, shape ``(N, H, D)``.
        loc: Point forecast, shape ``(N, H, D)``.

    Returns:
        Tidy rows, tier ``"interval"``, matching :func:`_operational_rows`'s
        own ``{tier, metric, target, model, value}`` shape.
    """
    rows = []
    for target_idx, target in enumerate(targets):
        metrics = get_interval_metrics(
            true=true[:, :, target_idx],
            lower=lower[:, :, target_idx],
            upper=upper[:, :, target_idx],
            loc=loc[:, :, target_idx],
            alpha=INTERVAL_ALPHA,
        )
        row = metrics.iloc[0]
        for metric_name in ("rmse", "mae", "picp", "nmpi", "winkler_score", "cwe", "cwe_stratified"):
            if metric_name in row.index:
                rows.append(
                    {
                        "tier": "interval",
                        "metric": metric_name,
                        "target": target,
                        "model": model,
                        "value": row[metric_name],
                    }
                )
    return rows


def build_condition_scorer(
    method_name: str,
    rated: float,
    voltage_band: tuple[float, float],
    include_voltage: bool,
    n_paths: int,
    seed: int,
):
    """Build a ``custom_scorer(forecaster, test_df)`` for one backbone x conformal-method condition."""

    def custom_scorer(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)  # fixed per fold, matching the pre-migration script's own convention
        collection = forecaster.forecast(test_df)
        result = next(iter(collection))
        loc = np.asarray(result.loc)
        truth = np.asarray(result.ground_truth)
        quantiles = np.asarray(result.quantiles)
        quantile_levels = np.asarray(result.quantile_levels)

        if method_name == "none":
            ensemble = sample_quantile_ensemble(quantiles, quantile_levels, n_paths, rng)
            interval_lower, interval_upper = quantiles[:, 0, :, :], quantiles[:, -1, :, :]
        elif method_name == "max_norm":
            # mahalanobis dropped from CONFORMAL_VARIANTS (2026-10-02, see
            # module docstring), so this branch is max_norm-only now; the
            # correlation argument sample_calibrated_ensemble still accepts
            # is passed as identity, matching max_norm's own axis-aligned box
            # (score_type="max-norm" never uses the correlation argument).
            conformal = forecaster.conformal[result.model_name]
            sigma = get_sigma_from_quantile_levels(quantiles, quantile_levels)
            q_hat = np.asarray(conformal.q_hat)
            n_targets = loc.shape[-1]
            identity_corr = np.broadcast_to(np.eye(n_targets), (loc.shape[1], n_targets, n_targets))
            ensemble = sample_calibrated_ensemble(loc, sigma, identity_corr, q_hat, "max-norm", n_paths, rng)
            radius = q_hat[None, :, None] * sigma
            interval_lower, interval_upper = loc - radius, loc + radius
        else:  # "quantile" / "quantile_joint"
            conformal = forecaster.conformal[result.model_name]
            lower, upper = quantiles[:, 0, :, :], quantiles[:, -1, :, :]
            lower_adj, upper_adj = conformal.generate_intervals(lower, upper)
            interval_lower, interval_upper = np.asarray(lower_adj), np.asarray(upper_adj)
            ensemble = sample_quantile_ensemble(
                quantiles,
                quantile_levels,
                n_paths,
                rng,
                lower_adj=interval_lower,
                upper_adj=interval_upper,
            )

        days = np.arange(truth.shape[0])
        preds = {method_name: (None, None, ensemble, None)}
        curve_rows: list[dict] = []
        operational_df = hosting_capacity_rows(
            preds, truth, result.targets, rated, days, voltage_band, include_voltage, curve_sink=curve_rows
        )
        interval_df = pd.DataFrame(
            interval_metric_rows(method_name, result.targets, truth, interval_lower, interval_upper, loc)
        )
        curve_df = pd.DataFrame(curve_rows)
        if not curve_df.empty:
            curve_df["tier"] = "value_curve"
        metric_df = pd.concat([operational_df, interval_df, curve_df], ignore_index=True)
        thermal = np.hypot(ensemble[..., 0], ensemble[..., 1]) > rated
        union_hat = thermal.any(axis=-1).mean(axis=0)  # (N,), per-day predicted union probability
        pred_df = pd.DataFrame({"union_hat": union_hat})
        return pred_df, metric_df

    return custom_scorer


def family_significance(table: pd.DataFrame) -> pd.DataFrame:
    """BH-corrected pairwise Diebold-Mariano on the Brier score across every backbone x conformal cell."""
    sub = table[table.metric == "Brier score"]
    if sub.empty:
        return pd.DataFrame()
    per_fold = sub.groupby(["model", "Folds"])["value"].mean().unstack("Folds")
    losses = {name: per_fold.loc[name].to_numpy() for name in per_fold.index}
    if len(losses) < 2:
        return pd.DataFrame()
    return pairwise_dm_table(losses, h=1, alpha=0.05, version="univariate")


def _build_conditions(
    rated: float,
    voltage_band: tuple[float, float],
    include_voltage: bool,
    n_paths: int,
    seed: int,
    hpo_objective: str = "winkler_score",
):
    """3 FPQR backbones + QR x 5 conformal methods, each carrying a custom_scorer for this decision's own threshold.

    Mirrors ``backbone_conformal_study.py``'s own ``_build_conditions``
    (same name, same per-head HPO wiring) as closely as this script's
    per-decision parametrisation (rated threshold, voltage band) allows --
    that one builds its condition list once at module level since nothing
    it needs varies per dataset; this one must rebuild per decision key,
    since ``rated``/``voltage_band``/``include_voltage`` do.

    ``hpo_fixed_params`` pins ``proposal_type``/``qvn_type``/``per_target_tau``
    inside the Optuna search itself, matching
    ``backbone_conformal_study.py``'s own ``_build_conditions`` -- added
    2026-09-29, this script never had it at all (unlike that one), so every
    field here was previously searched blind by whatever free draw HPO made
    and then silently forced to this dict's own ``overrides`` afterward,
    same "tuned for an unpinned default, then relabelled" issue fixed
    upstream in ``multitarget_experiment.py``'s FPQR conditions. QR
    (2026-10-01, added alongside ``backbone_conformal_study.py``'s own QR
    head) reuses that script's dedicated backbone-only HPO search
    (``QR_HPO_VARIANT``) rather than its own full joint study, for the same
    reason: ``MLPFQRConfig`` has no FPQR-specific hyperparameters to tune.
    """
    conditions = []
    for backbone_name, (config_cls, overrides) in HEADS.items():
        hpo_fixed = {k: overrides[k] for k in ("proposal_type", "qvn_type", "per_target_tau") if k in overrides}
        if backbone_name == "qr":
            hpo_model_cls, hpo_variant = MLPFConfig, QR_HPO_VARIANT
            description_prefix = "QR (fixed-grid baseline)"
        else:
            hpo_model_cls, hpo_variant = config_cls, ""
            description_prefix = f"{backbone_name.upper()} FPQR"
        # "none" is this backbone's trained-weights source for every other
        # conformal variant below (reuse_checkpoint_from) -- one fit per
        # backbone per fold, not one per backbone-x-conformal-method cell,
        # matching multitarget_experiment.py's own benchmark/conformal split.
        # reserve_calib_window=True so "none"'s own CV layout carves out the
        # same calibration window its calibrated siblings do (required for
        # reuse_checkpoint_from -- see ExperimentEngine's own validation).
        none_name = f"{backbone_name}_none"
        for method_name, conformal_config in CONFORMAL_VARIANTS.items():
            conditions.append(
                Condition(
                    name=f"{backbone_name}_{method_name}",
                    group="decision",
                    description=f"{description_prefix}, conformal={method_name}",
                    overrides=overrides,
                    model_cls=config_cls,
                    hpo_model_cls=hpo_model_cls,
                    hpo_variant=hpo_variant,
                    hpo_fixed_params=hpo_fixed,
                    conformal_config=conformal_config,
                    reserve_calib_window=(method_name == "none"),
                    reuse_checkpoint_from=None if method_name == "none" else none_name,
                    hpo_objective_metric=hpo_objective,
                    custom_scorer=build_condition_scorer(
                        method_name, rated, voltage_band, include_voltage, n_paths, seed
                    ),
                )
            )
    return conditions


def build_spec(key: str, args) -> ExperimentSpec:
    """One ExperimentSpec per decision (DSS-UK/DSS-US), each with its own rating/voltage band."""
    scenario = SCENARIOS[key]
    include_voltage = INCLUDE_VOLTAGE[key]
    voltage_band = VOLTAGE_BANDS[key]
    rated = decision_rating(scenario, args.rated_q)
    log.info("%s rating   %.1f kVA at the %.0f%% quantile of recorded apparent power", key, rated, args.rated_q * 100)

    return ExperimentSpec(
        name=f"Backbone Conformal Decision — {key}",
        output_prefix=f"ch6_backbone_conformal_decision_{key.lower()}",
        condition_cls=HEADS["mlpgam"][0],
        backbone_cls=HEADS["mlpgam"][0],
        conditions=_build_conditions(rated, voltage_band, include_voltage, args.paths, args.seed, args.hpo_objective),
        datasets={scenario: DATASETS[scenario]},
        cv_split_freq="weeks",
        cv_train_size=26,  # nominal -- see backbone_conformal_study.py's own note, cv_folds is the real lever
        cv_test_size=args.test_size,
        # val=2/calib=8 (was 4/4, 2026-10-04) -- see multitarget_decision.py's
        # own comment for the real per-fold exceedance-event counts this is
        # based on (as low as 12-14 events/fold at calib=4 weeks, 40-91 at
        # calib=8); same fold count/stride/test window, just rebalanced from
        # a validation window that doesn't need 4 weeks for early stopping.
        cv_val_size=2,
        cv_calib_size=8,
        cv_stride=args.step,
        cv_folds=4,
        cv_window="expanding",
        # num_workers=0: the default of 8 forks DataLoader workers after the
        # MPS/Metal backend is already initialised, which macOS repeatedly
        # crashed on during verification (RuntimeError: Training failed <-
        # DataLoader worker exited unexpectedly / Please call
        # iter(combined_loader) first) -- main-process loading is more than
        # enough for these fold sizes and avoids the fork entirely.
        fixed_overrides={"early_stop_patience": 100, "num_workers": 0},
        hpo_train_months=4,
        hpo_val_months=1,
        reference_conditions={"decision": "mlpgam_none"},
        # custom_scorer returns a decision pred_df (a single `union_hat`
        # column), not an Actual/forecast frame -- the engine's built-in
        # 7-day condition plot doesn't apply here and would just fail
        # (caught, but noisy) every condition.
        save_condition_plots=False,
    )


def run_decision(key: str, args, out_dir: Path, output_root: Path) -> pd.DataFrame:
    """Run one decision (DSS-UK hosting-capacity or DSS-US thermal) end to end, every backbone x conformal cell."""
    spec = build_spec(key, args)
    engine = ExperimentEngine(spec)
    cfg = PipelineConfig(
        dataset_name=SCENARIOS[key],
        data_path=str(thesis_data_dir()),
        output_dir=str(output_root / "backbone_conformal_decision" / key.lower()),
        epochs=5 if args.quick else args.epochs,
        num_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )
    groups = ["decision"]
    conditions = spec.conditions
    if args.quick:
        conditions = [c for c in conditions if c.name == f"{args.backbones[0]}_none"]
        spec.conditions = conditions

    summary = engine.run(base_cfg=cfg, groups=groups, dataset_keys=[SCENARIOS[key]], skip_hpo=args.skip_hpo)

    # engine.run() writes each run under a timestamped subdirectory and
    # best-effort updates `<experiment_root>/latest` to point at it -- but
    # that update (ExperimentEngine._update_latest_symlink) silently no-ops
    # (OSError, caught and logged at debug level only) on a filesystem that
    # refuses an unprivileged symlink, the *default* state on Windows
    # (requires Developer Mode or an elevated process). Relying on "latest"
    # here meant this function silently returned an empty DataFrame -- no
    # per_fold/value_curve/summary/significance CSVs at all, just one
    # easy-to-miss warning log -- on exactly that platform. Finding the real
    # run directory directly (newest mtime under runs/, not the whole
    # dataset output tree, so a previous run's stale metrics.parquet is
    # never picked up) has no symlink dependency at all.
    experiment_root = Path(cfg.output_dir) / slugify(spec.name)
    runs_dir = experiment_root / "runs"
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime) if runs_dir.exists() else []
    run_root = run_dirs[-1] if run_dirs else experiment_root / "latest"
    parquet_files = list(run_root.rglob("metrics.parquet"))
    if not parquet_files:
        log.warning("%s: no metrics.parquet found under %s — run produced no results", key, run_root)
        return pd.DataFrame()
    full = pd.concat(
        [pd.read_parquet(p) for p in parquet_files],
        ignore_index=True,
    )
    full["decision"] = key

    # Split off the relative-economic-value curve rows (tier="value_curve",
    # one row per cost-loss ratio rather than per metric) before the
    # metric/target/model summary below -- grouping them in with the rest
    # would average across cost-loss ratios, which is meaningless, and their
    # own "metric" column is empty (they carry "cost_loss_ratio" instead).
    is_curve = full["tier"] == "value_curve"
    curve_full, full = full[is_curve].copy(), full[~is_curve].copy()
    if not curve_full.empty:
        curve_full.to_csv(out_dir / f"{key.lower()}_value_curve.csv", index=False)

    full.to_csv(out_dir / f"{key.lower()}_per_fold.csv", index=False)

    agg = full.groupby(["metric", "target", "model"])["value"].agg(["mean", "std", "min", "max"]).reset_index()
    agg.to_csv(out_dir / f"{key.lower()}_summary.csv", index=False)
    print(f"\n=== {key} summary (mean across {spec.cv_folds} folds) ===")
    print(agg.pivot_table(index="metric", columns="model", values="mean").round(4).to_string())

    dm = family_significance(full)
    if not dm.empty:
        dm.to_csv(out_dir / f"{key.lower()}_significance.csv", index=False)
        print(f"\n=== {key}: backbone x conformal, Brier score DM table ===")
        print(dm.to_string(index=False))

    log.info("%s: engine summary written under %s", key, summary)
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--backbones", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260906, help="fixed; folds supply the variation")
    parser.add_argument("--test-size", type=int, default=8, help="weeks; fewer, larger test windows than Experiment 2")
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rated-q", type=float, default=0.98)
    parser.add_argument(
        "--skip-hpo",
        dest="skip_hpo",
        action="store_true",
        default=True,
        help="reuse cached backbone HPO params if present (default: True, matching "
        "multitarget_experiment.py's/backbone_conformal_study.py's own --skip-hpo/--run-hpo "
        "convention -- this script previously exposed only --run-hpo, added the explicit "
        "--skip-hpo flag 2026-10-01 for CLI uniformity across all three scripts).",
    )
    parser.add_argument(
        "--run-hpo",
        dest="skip_hpo",
        action="store_false",
        help="run real backbone HPO through the engine, replacing the old --hpo-scenario=all/"
        "tune_backbone.py-based tuning -- the engine already tunes per dataset x head through "
        "its own Phase 1 HPO, same as Experiment 2.",
    )
    parser.add_argument("--hpo-trials", type=int, default=50)
    parser.add_argument("--hpo-epochs", type=int, default=100, help="epochs per HPO trial, not per final condition")
    parser.add_argument(
        "--hpo-objective",
        default="winkler_score",
        help="HPO objective for every condition (every head here is quantile-producing, unlike "
        "multitarget_experiment.py's mixed head set). 'winkler_score' (default) or any name "
        "get_quantile_metrics computes, e.g. 'cwe_stratified'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root directory for experiment artifacts and HPO caches",
    )
    parser.add_argument("--quick", action="store_true", help="short run for a smoke test")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.paths = 20
        args.decisions = args.decisions[:1]
        args.backbones = args.backbones[:1]

    configure(level="INFO")
    run_root = start_run(
        args.output_root / "backbone_conformal_decision",
        meta={"decisions": args.decisions, "backbones": args.backbones, "epochs": args.epochs, "seed": args.seed},
    )
    out_dir = run_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {key: run_decision(key, args, out_dir, args.output_root) for key in args.decisions}
    (out_dir / "backbone_conformal_decision_config.json").write_text(
        json.dumps(
            {
                "decisions": args.decisions,
                "backbones": args.backbones,
                "conformal_methods": list(CONFORMAL_METHODS),
                "scenarios": {k: SCENARIOS[k] for k in args.decisions},
                "epochs": args.epochs,
                "seed": args.seed,
                "test_size_weeks": args.test_size,
                "step_weeks": args.step,
                "paths": args.paths,
                "alpha": args.alpha,
            },
            indent=2,
        )
        + "\n"
    )
    log.info("wrote per-decision results under %s", out_dir)
    for key, table in results.items():
        log.info("  %s: %d rows", key, len(table))


if __name__ == "__main__":
    main()
