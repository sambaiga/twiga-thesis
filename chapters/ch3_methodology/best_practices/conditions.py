"""Condition building blocks for the Chapter 3 protocol comparison (single split vs expanding-window backtest).

Follows ``chapters/ch6_conformal/shared/conditions.py``'s shape: conditions are plain
:class:`~twiga.experiment.Condition` objects. Every condition here is a simple baseline -- two statistical
(seasonal-naive, naive) and three classical ML (random forest, CatBoost, linear regression) -- on their class
defaults, with HPO disabled at the spec level (``protocol.SPEC_SINGLE_SPLIT``/``SPEC_EXPANDING``'s own
``hpo_enabled=False``): the question under test is which backtesting protocol gives a trustworthy accuracy
estimate, not which hyperparameters are best for any one model.

Metric types
------------
``METRIC_TYPES`` reports the wider five-metric point set (NRMSE, MAE, MAPE, sMAPE, nBias), not
``ch6_conformal/shared/conditions.py``'s narrower ``POINT_METRIC_TYPES``: this experiment's own question is
explicitly which point metric to trust under which protocol (a scale-dependent metric such as MAE can rank
models differently from a percentage metric such as sMAPE when net load crosses zero), so every metric in that
set is reported for every condition rather than picking one in advance.
"""

from __future__ import annotations

from twiga.experiment import Condition
from twiga.models.baseline.naive_model import NAIVEConfig
from twiga.models.baseline.seasonal_naive_model import SEASONALNAIVEConfig
from twiga.models.ml.catboost_model import CATBOOSTConfig
from twiga.models.ml.lineareg_model import LINEAREGConfig
from twiga.models.ml.randomforest_model import RANDOMFORESTConfig

# Point metrics: the wider set, since this experiment is explicitly about which metric to trust under which
# protocol, not ch6_conformal's narrower POINT_METRICS (mae/rmse/smape/corr) chosen for a different purpose.
METRIC_TYPES = ["point"]
POINT_METRICS = ["nrmse", "mae", "mape", "smape", "nbias"]

CONDITIONS: list[Condition] = [
    Condition(
        name="seasonal_naive",
        group="baselines",
        description="Seasonal-naive baseline: the previous day's value at the same time of day",
        model_cls=SEASONALNAIVEConfig,
        metric_types=METRIC_TYPES,
        hpo_enabled=False,
    ),
    Condition(
        name="naive",
        group="baselines",
        description="Naive baseline: the immediately preceding observation",
        model_cls=NAIVEConfig,
        metric_types=METRIC_TYPES,
        hpo_enabled=False,
    ),
    Condition(
        name="random_forest",
        group="baselines",
        description="Random forest regressor on the autoregressive and exogenous feature set",
        model_cls=RANDOMFORESTConfig,
        metric_types=METRIC_TYPES,
        hpo_enabled=False,
    ),
    Condition(
        name="catboost",
        group="baselines",
        description="CatBoost regressor on the autoregressive and exogenous feature set",
        model_cls=CATBOOSTConfig,
        metric_types=METRIC_TYPES,
        hpo_enabled=False,
    ),
    Condition(
        name="linear_regression",
        group="baselines",
        description="Linear regression on the autoregressive and exogenous feature set",
        model_cls=LINEAREGConfig,
        metric_types=METRIC_TYPES,
        hpo_enabled=False,
    ),
]
