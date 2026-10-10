"""Condition building blocks for Chapter 3's meteorological covariate ablation (RQ2).

Four conditions on the same MLPF backbone, one per covariate combination, varying only which exogenous
(known-future) features reach the model. ``load_dataset`` (via ``twiga.utils.run_forecast.build_data_config``)
normally selects the weather columns automatically -- ranked against the target by ``weather_top_k`` and surfaced
to the model through ``DataPipelineConfig.known_future_features`` -- the same auto-selection mechanism
``chapters/ch3_methodology/figures/feature_evidence.py`` already documents evidence for (its own rank-fusion
consensus: per-metric rankers disagree on whether Ghi or Temperature ranks first for the reactive-power target
on DSS-EU). This experiment deliberately bypasses that auto-selection for each of its four conditions, setting
``known_future_features`` explicitly via ``Condition.data_overrides`` (the same mechanism Chapter 6/7's own
``fpqr_lv_differenced`` condition uses to reach a ``DataPipelineConfig`` field directly, via
``data_overrides={"target_differencing": ...}`` in ``multitarget_experiment.py``) -- here, overriding the feature
set *is* the experiment, not a workaround, so every condition states its own exogenous list explicitly rather than
deferring to the dataset's own selection.

Calendar features and autoregressive lags are left untouched in every condition (``calendar_features``/
``past_features`` are not overridden), so the only thing that varies across the four conditions is the
meteorological covariate set.
"""

from __future__ import annotations

from twiga.experiment import Condition
from twiga.models.nn import MLPFConfig

GHI = "Ghi"
TEMPERATURE = "Temperature"

# One shared backbone HPO search per dataset, reused by every condition (chapters/ch6_conformal/shared/
# conditions.py's own SHARED_HPO_VARIANT pattern): the ablation is about the exogenous feature set, not about
# giving each condition its own independently-tuned backbone capacity.
SHARED_HPO_VARIANT = "shared_mlpf"

METRIC_TYPES = ["point", "interval"]

CONDITIONS: list[Condition] = [
    Condition(
        name="ghi_time",
        group="covariate_influence",
        description="Global horizontal irradiance plus calendar/time features only (no temperature)",
        data_overrides={"known_future_features": [GHI]},
        model_cls=MLPFConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        metric_types=METRIC_TYPES,
    ),
    Condition(
        name="temp_time",
        group="covariate_influence",
        description="Temperature plus calendar/time features only (no irradiance)",
        data_overrides={"known_future_features": [TEMPERATURE]},
        model_cls=MLPFConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        metric_types=METRIC_TYPES,
    ),
    Condition(
        name="ghi_temp_time",
        group="covariate_influence",
        description="Both irradiance and temperature, plus calendar/time features -- the full meteorological set",
        data_overrides={"known_future_features": [GHI, TEMPERATURE]},
        model_cls=MLPFConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        metric_types=METRIC_TYPES,
    ),
    Condition(
        name="time_only",
        group="covariate_influence",
        description="Calendar/time features only -- no meteorological covariate at all (reference)",
        data_overrides={"known_future_features": []},
        model_cls=MLPFConfig,
        hpo_model_cls=MLPFConfig,
        hpo_variant=SHARED_HPO_VARIANT,
        metric_types=METRIC_TYPES,
    ),
]
