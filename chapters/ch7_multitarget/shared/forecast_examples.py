"""Predictive distributions against ground truth for P, Q and V, best and worst day.

The summary tables say which model scores better on average. They do not show
what a forecast looks like when it works and when it does not, and for a
multi-target problem that matters: a model can score well on active power and be
useless on voltage, and an aggregate hides it.

Each panel follows the Chapter 6 convention: observed series in orange with
markers, predictive median dashed, and the full set of predictive quantiles
drawn as a translucent fan rather than a single nominal interval, so the shape
of the distribution is visible and not just its 95% width. Panel titles carry
NRMSE, CWE and CRPS for that day and target.

Quantiles are each model's own. The Gaussian heads are evaluated analytically on
a dense level grid; FPQR is drawn on the levels it actually learned, which are
few and unevenly spaced by construction. That difference is the point of the
method and is left visible rather than smoothed away.

Panels are saved individually as PDFs, following the thesis convention of
composing sub-figures in LaTeX, plus one overview PNG per day for quick reading.

    uv run python experiment/ch7_multitarget/forecast_examples.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import laplace, norm  # noqa: E402
import torch  # noqa: E402
from twiga.core.backtester import TimeBasedCV  # noqa: E402
from twiga.core.data.processing import get_feature_sequences, get_target_sequences  # noqa: E402
from twiga.core.metrics.interval import (  # noqa: E402
    cwe_score,
    interval_coverage,
    nmpi,
    winkler_score,
)
from twiga.core.metrics.parametric import (  # noqa: E402
    brier_decomposition,
    brier_score,
    energy_score,
    relative_economic_value,
    variogram_score,
)
from twiga.core.metrics.quantile import crps_quantile  # noqa: E402
from twiga.core.stats.association import distance_correlation  # noqa: E402
from twiga.core.utils import get_logger  # noqa: E402
from twiga.distributions.conformal.multivariate import MultivariateConformal  # noqa: E402
from twiga.distributions.nn.copula_sampling import (  # noqa: E402
    coupled_samples,
    fit_clayton_theta,
    fit_dvine3,
    fit_gaussian_copula,
)
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig, MLPFFPQRModel  # noqa: E402
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig, MLPFLaplaceModel  # noqa: E402
from twiga.models.nn.mlpfmvlaplace_model import MLPFMVLaplaceConfig, MLPFMVLaplaceModel  # noqa: E402
from twiga.models.nn.mlpfmvnormal_model import MLPFMVNormalConfig, MLPFMVNormalModel  # noqa: E402
from twiga.models.nn.mlpfnormal_model import MLPFNormalConfig, MLPFNormalModel  # noqa: E402
from twiga.models.nn.mlpgamfpqr_model import MLPGAMFPQRConfig, MLPGAMFPQRModel  # noqa: E402
from twiga.models.nn.mlpgamlaplace_model import MLPGAMLaplaceConfig, MLPGAMLaplaceModel  # noqa: E402
from twiga.models.nn.mlpgammvnormal_model import MLPGAMMVNormalConfig, MLPGAMMVNormalModel  # noqa: E402
from twiga.models.nn.mlpgamnormal_model import MLPGAMNormalConfig, MLPGAMNormalModel  # noqa: E402
from twiga.utils.data_loading import load_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twiga.core.metrics import mcs_ranking, pairwise_dm_table

from twiga_thesis.paths import thesis_data_dir
from twiga_thesis.plotting.forecast_panels import draw, place_legend_below
from twiga_thesis.plotting.style import configure, pad_limits, panel  # noqa: E402
from twiga_thesis.plotting.tokens import (  # noqa: E402
    HIGHLIGHT,
    PRIMARY,
    diverging_cmap,
    matrix_cmap,
    tint,
)

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
OUT_ROOT = script_dir.parent / "forecast_examples"
# Rebound in main() to a per-backbone subdirectory so the two runs do not
# overwrite each other; defined here so module-level references stay valid.
OUT = OUT_ROOT

# Axis labels are the symbol and unit only. At a third of the text width a
# spelled-out label eats the panel; the quantity is named in the caption.
AXIS = {"p_kw": "$P$ (kW)", "q_kvar": "$Q$ (kvar)", "v_lv_pu": "$V$ (pu)"}


def _cap(label: str) -> str:
    """First letter upper-cased, the rest left alone.

    ``str.capitalize()`` also lower-cases every other character, which
    mangles an already-cased acronym like ``"MLPGAM"`` into ``"Mlpgam"``; a
    condition/model name here may be exactly that.
    """
    return label[:1].upper() + label[1:] if label else label


# The Chapter 6 convention: the observed series carries the highlight colour so
# the eye lands on it first, the prediction takes the primary.
OBSERVED = HIGHLIGHT
PREDICTED = PRIMARY
# The fan sits in a tint of the prediction colour rather than the colour itself,
# so the median line stays legible on top of it.
FAN = tint(PRIMARY, 0.55)
SPINE_COLOR = "gray"
# Correlation is signed with a meaningful zero, so it needs a diverging map, and
# the palette's teal and orange are the only pair that separates by hue.
CORR_CMAP = diverging_cmap()
# Distance correlation is bounded in [0, 1] and unsigned, so it takes the grey
# matrix map the thesis uses for its significance matrices. A diverging map
# would imply a sign the statistic does not have.
DCOR_CMAP = matrix_cmap()
# Dense grid for the analytic fan of a parametric head. Spans 1% to 99% so the
# tails show. FPQR is drawn on its own learned levels instead.
GRID_LEVELS = np.round(np.concatenate([[0.01], np.arange(0.05, 0.96, 0.05), [0.99]]), 4)

# Fixed per-edge family for the D-vine copula (see vine_copula.DVineCopula3's
# docstring): all-Gaussian, the simplest starting point. A learned per-edge
# family gate is a stated future extension, not built here.
VINE_EDGE_FAMILIES = ("gaussian", "gaussian", "gaussian")


def _theta_to_rho(family: str, theta: float) -> float:
    """Kendall's tau -> equivalent elliptical correlation, for display only.

    Gaussian's theta already is a correlation. Clayton's is not, so this
    mirrors the same tau_to_rho diagnostic conversion the Clayton branch of
    fit_and_predict uses (rho = sin(pi*tau/2), Kendall 1949's relation),
    purely so the vine's three edges combine into one displayable 3x3
    correlation matrix for correlation_panel. The model itself never uses
    this converted value.
    """
    if family == "gaussian":
        return theta
    tau = theta / (theta + 2.0)
    return float(np.sin(np.pi * tau / 2.0))


# Every parametric head reports a "scale", but not on the same footing, so
# neither the inverse CDF nor the sampler may be applied to it blindly.
#
#   normal              loc and standard deviation; the Gaussian case.
#   laplace-dispersion  the marginal Laplace head reports the Laplace b directly,
#                       since b is the distribution's own second parameter.
#   laplace-sd          the joint heads report sqrt(diag(Sigma)), a standard
#                       deviation, so that they stay interchangeable with the
#                       independent heads downstream.
#
# Each entry is (inverse-CDF distribution, scale -> dispersion, scale -> mixture
# sigma). The second converts the reported scale into the parameter the inverse
# CDF expects: a Laplace of variance s^2 has b = s / sqrt(2), so reading b
# straight off a reported standard deviation draws the fan sqrt(2) too wide. The
# third converts it into the sigma of the normal variance mixture used for
# sampling, since a Laplace of dispersion b is sqrt(W) * N(0, 2b^2) with W
# exponential: sampling at sigma = b rather than b*sqrt(2) draws an ensemble
# sqrt(2) too narrow.
PARAMETRIC_FAMILY = {
    "normal": (norm, 1.0, 1.0),
    "laplace-dispersion": (laplace, 1.0, np.sqrt(2.0)),
    "laplace-sd": (laplace, 1.0 / np.sqrt(2.0), 1.0),
}


# One week at 30-minute resolution. Every target of every record puts a
# significant autocorrelation peak here, and a 48-step lookback cannot reach it,
# so the weekly value enters as an explicit feature instead of by carrying seven
# days of context.
WEEKLY_LAG = 336


def _calendar_matrix(data, ts, calendar):
    """Calendar features, preferring the ones twiga's selection chose.

    ``load_dataset`` ranks candidate calendar signals against the target with
    :func:`select_top_features` and returns the winners. Earlier versions of this
    script discarded that and hard-coded four Fourier terms, which is a guess
    where the library had a measurement.

    Args:
        data: the loaded frame, which may already carry the named columns.
        ts: timestamps, used to rebuild anything the frame does not carry.
        calendar: column names selected by :func:`load_dataset`.

    Returns:
        Float array of shape ``(rows, features)``, always with the daily cycle
        present so that a selection returning only coarse features still leaves
        the model able to place a half-hour within the day.
    """
    columns, names = [], []
    for name in calendar:
        if name in data.columns:
            columns.append(data[name].to_numpy(dtype=np.float64))
            names.append(name)
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    for name, values in (
        ("hour_sin", np.sin(2 * np.pi * hour / 24)),
        ("hour_cos", np.cos(2 * np.pi * hour / 24)),
        ("dow_sin", np.sin(2 * np.pi * ts.dt.dayofweek.to_numpy() / 7)),
        ("dow_cos", np.cos(2 * np.pi * ts.dt.dayofweek.to_numpy() / 7)),
    ):
        if name not in names:
            columns.append(values)
            names.append(name)
    return np.column_stack(columns)


def build_windows(scenario: str, lookback: int, horizon: int):
    """Supervised windows in physical units, with the scaler fitted on train rows."""
    data, targets, calendar, exog, _lags, _lat, _lon, _lookback = load_dataset(
        scenario, data_path=str(thesis_data_dir())
    )
    y = data[targets].to_numpy(dtype=np.float64)
    x_exog = data[exog].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(data["timestamp"])
    cal = _calendar_matrix(data, ts, calendar)

    n_windows = len(data) - lookback - horizon
    fit_rows = int(0.8 * n_windows) + lookback
    mu, sd = y[:fit_rows].mean(0), y[:fit_rows].std(0)
    yz = (y - mu) / sd
    ez = (x_exog - x_exog[:fit_rows].mean(0)) / x_exog[:fit_rows].std(0)

    weekly = np.roll(yz, WEEKLY_LAG, axis=0)
    weekly[:WEEKLY_LAG] = yz[:WEEKLY_LAG]  # no earlier week exists; hold the value
    past = np.c_[yz, weekly, ez, cal]
    future = np.c_[ez, cal]
    idx = np.arange(n_windows)
    xp = np.stack([past[i : i + lookback] for i in idx])
    xf = np.stack([future[i + lookback : i + lookback + horizon] for i in idx])
    yy = np.stack([yz[i + lookback : i + lookback + horizon] for i in idx])
    stamps = np.stack([ts.to_numpy()[i + lookback : i + lookback + horizon] for i in idx])
    pad = np.zeros((n_windows, horizon, xp.shape[2] - xf.shape[2]))
    x = np.concatenate([xp, np.concatenate([pad, xf], axis=2)], axis=1).astype(np.float32)

    # Chronological 70/10/20. Validation is carved off the tail of training, not
    # taken from test: early stopping and best-checkpoint selection both read the
    # validation split, so scoring on it would leak the test period into the
    # model that is then plotted against it.
    cut = int(0.8 * n_windows)
    val_cut = int(0.875 * cut)
    return {
        "x_tr": x[:val_cut],
        "y_tr": yy[:val_cut].astype(np.float32),
        "x_val": x[val_cut:cut],
        "y_val": yy[val_cut:cut].astype(np.float32),
        "x_te": x[cut:],
        "y_te": yy[cut:].astype(np.float32),
        "stamps_te": stamps[cut:],
        "mu": mu,
        "sd": sd,
        "targets": targets,
        # Column counts within `future`/the trailing block of `past`, so
        # _cfg() can tell a model architecture there is a real future-covariate
        # block instead of declaring num_calendar_features=num_known_future_features=0
        # regardless of what the tensor actually contains (see _cfg()'s own
        # docstring for why that silently degraded every backbone -- NHITS
        # simply refuses to run rather than doing so quietly).
        "num_calendar": cal.shape[1],
        "num_exog": ez.shape[1],
    }


# Backbone size and optimiser settings, shared by every condition. Tuning each
# head separately would confound a difference between likelihoods with a
# difference in how well each was tuned, so tune_backbone.py fits these once and
# writes them here; these values are the fallback when it has not been run.
BACKBONE_FALLBACK = {
    "latent_size": 64,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-6,
}


def backbone_settings():
    """Tuned backbone settings when available, the fallback otherwise."""
    path = script_dir / "backbone_params.json"
    if not path.exists():
        log.warning("no tuned backbone at %s; using defaults %s", path, BACKBONE_FALLBACK)
        return dict(BACKBONE_FALLBACK)
    payload = json.loads(path.read_text())
    tuned = {k: payload["best"][k] for k in BACKBONE_FALLBACK}
    log.info("tuned backbone from %s: %s (validation CRPS %.5f)", path.name, tuned, payload["best_score"])
    return tuned


def _fold_windows(lo, hi, stride, *, past, future, yz, ts, lookback, horizon):
    """Windowed features/targets/timestamps for rows ``[lo, hi)``.

    Uses :func:`twiga.core.data.processing.get_feature_sequences`/
    ``get_target_sequences`` rather than a hand-rolled sliding-window loop --
    verified byte-identical to the prior implementation, which this replaces
    per this session's "don't redefine twiga utilities" review. The lookback
    is allowed to reach back before ``lo``: that is not leakage, since at
    forecast time the recent past is known, and restricting the lookback to
    the evaluation period instead would discard the first two days of every
    test month.

    Future covariates are combined with the lookback features by padding
    at the *front*, not :func:`twiga.core.data.processing.stack_features`'s
    end-padding: ``future``'s columns (``[ez, cal]``) must land in the same
    column positions as ``past``'s trailing columns (``past = [yz, weekly,
    ez, cal]``), and ``stack_features``'s generic end-padding was checked and
    found to silently misalign them instead, feeding the model future
    covariates under the wrong past-feature columns. This is the one place
    in this function that stays hand-rolled for that reason.

    Returns:
        ``(x, y, stamps)`` -- combined past+future features
        ``(n, lookback + horizon, n_past_features)``, targets
        ``(n, horizon, n_targets)``, and forecast timestamps ``(n, horizon)``.
        Empty (zero-row) arrays when the range is too short for one window.
    """
    s0 = max(lo - lookback, 0)
    if hi - s0 < lookback + horizon:
        empty_x = np.empty((0, lookback + horizon, past.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, horizon, yz.shape[1]), dtype=np.float32)
        empty_stamps = np.empty((0, horizon), dtype=ts.to_numpy().dtype)
        return empty_x, empty_y, empty_stamps

    xp = get_feature_sequences(past[s0:hi], lookback, horizon, stride=stride)
    xf = get_target_sequences(future[s0:hi], lookback, horizon, stride=stride)
    yy = get_target_sequences(yz[s0:hi], lookback, horizon, stride=stride)
    stamps = get_target_sequences(ts.to_numpy()[s0:hi].reshape(-1, 1), lookback, horizon, stride=stride)[..., 0]
    pad = np.zeros((xp.shape[0], horizon, xp.shape[2] - xf.shape[2]))
    x = np.concatenate([xp, np.concatenate([pad, xf], axis=2)], axis=1).astype(np.float32)
    return x, yy.astype(np.float32), stamps


def build_backtest_folds(
    scenario,
    lookback,
    horizon,
    train_months=4,
    test_months=1,
    step_months=1,
    train_stride=1,
    test_stride=48,
    expanding=False,
    split_freq="months",
):
    """Rolling-origin folds: a fixed training span, the next month held out.

    Replaces the single chronological split, which tested only on October to
    December and so barely exercised photovoltaic generation in a chapter about
    it. Rolling the origin puts every season in some test fold, and the spread
    across folds measures seasonal variation rather than the seed variation a
    repeated fit would give.

    The training span is a fixed width that slides, so every fold is fitted on
    the same amount of data and a difference between folds is a difference in
    season rather than in sample size. Passing ``expanding=True`` trains instead
    on all history up to each test month, which is what an operator would have,
    at the cost of confounding season with training-set size: later folds would
    then differ from earlier ones in both at once.

    Two strides, deliberately. Test windows are strided by a full day so the
    evaluation set is non-overlapping: overlapping test windows share most of
    their forecast span, which inflates the apparent sample size and understates
    the uncertainty of every score computed on them. Training windows may
    overlap, where the redundancy is harmless augmentation rather than a
    correlated evaluation.

    The scaler is fitted on each fold's training rows alone, so no fold sees the
    mean or spread of its own test period.

    Fold boundaries come from :class:`~twiga.core.backtester.TimeBasedCV`, the
    same class :class:`~twiga.experiment.ExperimentEngine` uses, rather than a
    hand-rolled month loop -- this script predates ``TimeBasedCV`` and never
    went back to adopt it, which is what caused a real bug: the original loop
    required the *entire* record to cover ``train_months + test_months`` before
    yielding even one fold, so it silently returned zero folds on a one-year
    record at Chapter 6's literal ``train_months=12``. ``TimeBasedCV`` degrades
    gracefully instead (confirmed empirically: 3 folds on DSS-EU's ~11-month
    span at the same settings, where the old loop gave 0), so this script's
    protocol no longer needs the shortened ``train_months``/``test_months``
    values that bug forced on ``copula_variants.py``, ``penetration_sweep.py``
    and ``operational_decision.py`` -- restoring the chapter's literal figures
    is a caller-side follow-up, not part of this refactor.

    Args:
        scenario: dataset key passed to :func:`load_dataset`.
        lookback: input length in steps.
        horizon: forecast length in steps.
        train_months: training-span width per fold (and the width of the
            first fold when ``expanding`` is true), in ``split_freq`` units
            despite the name -- kept as ``train_months`` for backward
            compatibility with existing callers, all of which pass calendar
            months.
        test_months: held-out span per fold, in ``split_freq`` units.
        step_months: the origin's advance between folds, in ``split_freq``
            units.
        train_stride: step between training window starts.
        test_stride: step between test window starts; 48 is one day at 30 min.
        expanding: grow the training span fold by fold rather than sliding a
            fixed width; confounds season with sample size when compared.
        split_freq: the calendar unit ``train_months``/``test_months``/
            ``step_months`` are counted in -- one of ``TimeBasedCV``'s
            ``["days", "weeks", "months", "years"]`` (``"minutes"``/``"hours"``
            are valid there too but meaningless at this function's coarse
            fold granularity). Default ``"months"`` matches every existing
            caller. Use ``"weeks"`` to reach a fold count integer-month
            stepping cannot hit on an ~11-12 month record (`penetration_sweep.py`'s
            own default is a fixed 4/1-month train/test width; on such a
            short record, going below a 1-month step needs a finer unit, not
            a smaller ``step_months`` value, since ``step_months`` must stay
            a positive integer in whichever unit is chosen).

    Returns:
        List of dicts in the same shape :func:`build_windows` returns, each with
        an added ``fold`` label naming the held-out period.
    """
    data, targets, calendar, exog, _lags, _lat, _lon, _lookback = load_dataset(
        scenario, data_path=str(thesis_data_dir())
    )
    y = data[targets].to_numpy(dtype=np.float64)
    x_exog = data[exog].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(data["timestamp"])
    cal = _calendar_matrix(data, ts, calendar)
    months = ts.dt.to_period("M")

    cv = TimeBasedCV(
        split_freq=split_freq,
        train_size=train_months,
        test_size=test_months,
        stride=step_months,
        window="expanding" if expanding else "rolling",
        num_splits=36,
        date_column="timestamp",
    )
    folds = []
    for bundle in cv.split(data):
        train_rows = np.asarray(bundle.scheme["train_idx"])
        test_rows = np.asarray(bundle.scheme["test_idx"])
        if not train_rows.size or not test_rows.size:
            continue
        train_period = sorted(months.iloc[train_rows].unique())
        test_period = sorted(months.iloc[test_rows].unique())

        # Scaling on this fold's training rows only.
        mu, sd = y[train_rows].mean(0), y[train_rows].std(0)
        yz = (y - mu) / sd
        ez = (x_exog - x_exog[train_rows].mean(0)) / x_exog[train_rows].std(0)
        weekly = np.roll(yz, WEEKLY_LAG, axis=0)
        weekly[:WEEKLY_LAG] = yz[:WEEKLY_LAG]  # no earlier week exists; hold the value
        past = np.c_[yz, weekly, ez, cal]
        future = np.c_[ez, cal]

        train_lo, train_hi = max(train_rows[0], WEEKLY_LAG), train_rows[-1] + 1
        test_lo, test_hi = max(test_rows[0], WEEKLY_LAG), test_rows[-1] + 1
        x_tr_full, y_tr_full, _ = _fold_windows(
            train_lo, train_hi, train_stride, past=past, future=future, yz=yz, ts=ts, lookback=lookback, horizon=horizon
        )
        x_te, y_te, stamps_te = _fold_windows(
            test_lo, test_hi, test_stride, past=past, future=future, yz=yz, ts=ts, lookback=lookback, horizon=horizon
        )
        if x_tr_full.shape[0] < 8 or not x_te.shape[0]:
            continue

        # Validation is the tail of the training span, so early stopping never
        # reads the held-out month. Cutting the already-windowed array by
        # position is equivalent to cutting the start-index list beforehand,
        # since both produce windows in the same stride order.
        cut = int(0.875 * x_tr_full.shape[0])
        x_tr, y_tr = x_tr_full[:cut], y_tr_full[:cut]
        x_val, y_val = x_tr_full[cut:], y_tr_full[cut:]
        folds.append(
            {
                "x_tr": x_tr,
                "y_tr": y_tr,
                "x_val": x_val,
                "y_val": y_val,
                "x_te": x_te,
                "y_te": y_te,
                "stamps_te": stamps_te,
                "mu": mu,
                "sd": sd,
                "targets": targets,
                "fold": str(test_period[0]) if len(test_period) == 1 else f"{test_period[0]}..{test_period[-1]}",
                "train_span": f"{train_period[0]}..{train_period[-1]}",
                # See build_windows()'s own num_calendar/num_exog for why these
                # are needed -- x_exog's column count is stable across folds
                # (unlike ez, which is re-scaled per fold), cal likewise.
                "num_calendar": cal.shape[1],
                "num_exog": x_exog.shape[1],
            }
        )
    return folds


def _cfg(cls, d, *, epochs, seed, horizon, lookback, backbone=None, **extra):
    """Build a model config matching build_windows()'/build_backtest_folds()'s own tensor layout.

    ``x``'s trailing feature axis is ``[target, weekly-lag, exog, calendar]``
    for the lookback rows, and ``[0-padded target+weekly-lag, exog, calendar]``
    for the horizon rows (see build_windows()'s own comments). That column
    order matches BaseArchitecture.set_num_feature_covariate_column()'s
    ``[target, raw_past, known_future, calendar]`` split exactly, so
    ``num_past_features`` here is the weekly-lag block's own width (the same
    as the target's, since it is the target rolled by a week), not the total
    non-target width -- passing the total, as an earlier version of this
    function did, told every architecture there were zero future covariates
    regardless of what the tensor actually contained: MLPF/MLPGAM degrade
    silently (their own future-covariate encoder branch is simply skipped),
    NHITS refuses to run at all (``ValueError: Exogenous features must be
    provided in NHITS``), since its architecture requires at least one.
    """
    settings = backbone or BACKBONE_FALLBACK
    num_target = d["y_tr"].shape[2]
    num_calendar = d["num_calendar"]
    num_exog = d["num_exog"]
    total = num_target + num_target + num_exog + num_calendar
    if total != d["x_tr"].shape[2]:
        raise ValueError(
            f"Feature-count mismatch: target={num_target} + weekly-lag={num_target} + exog={num_exog} + "
            f"calendar={num_calendar} = {total}, but x_tr's own feature axis is {d['x_tr'].shape[2]}. "
            "build_windows()/build_backtest_folds()'s tensor layout and _cfg()'s feature counts have "
            "drifted apart -- fix whichever changed rather than silently feeding a model the wrong split."
        )
    return cls(
        num_target_feature=num_target,
        num_past_features=num_target,
        num_calendar_features=num_calendar,
        num_known_future_features=num_exog,
        num_forecast_period_features=0,
        forecast_horizon=horizon,
        lookback_window_size=lookback,
        latent_size=int(settings["latent_size"]),
        hidden_dim=int(settings["hidden_dim"]),
        num_layers=int(settings["num_layers"]),
        dropout=float(settings["dropout"]),
        optimizer_params={"lr": float(settings["lr"]), "weight_decay": float(settings["weight_decay"])},
        max_epochs=epochs,
        batch_size=64,
        seed=seed,
        **extra,
    )


def sample_parametric(out, family, n_samples, rng):
    """Draw an ensemble from a parametric head, jointly when it emits a factor.

    Both families are normal variance mixtures, so one construction serves each
    and only the mixing weight differs: the Gaussian has weight one, the Laplace
    an exponential draw. When the head emits ``scale_tril`` the draw is joint, so
    the sampled trajectories carry the head's cross-target dependence and a
    coherent triple can be read off any one sample. That dependence is invisible
    in a marginal fan, which is the reason for sampling at all.

    A joint Laplace shares one mixing weight across targets within a sample,
    since the weight scales the whole vector; independent margins each draw their
    own.

    Args:
        out: forecast dict from a parametric head, carrying ``loc`` and
            ``scale``, and ``scale_tril`` when the head is joint.
        family: key into :data:`PARAMETRIC_FAMILY`.
        n_samples: ensemble size.
        rng: seeded generator.

    Returns:
        Array of shape ``(n_samples, days, steps, targets)``, standardised.
    """
    loc = np.asarray(out["loc"])
    shape = (n_samples, *loc.shape)
    z = rng.standard_normal(shape)
    joint = "scale_tril" in out
    if family.startswith("laplace"):
        weight = rng.exponential(size=(*shape[:-1], 1) if joint else shape)
        z = np.sqrt(weight) * z
    if joint:
        return loc[None] + np.einsum("dhij,sdhj->sdhi", np.asarray(out["scale_tril"]), z)
    _, _, to_sigma = PARAMETRIC_FAMILY[family]
    return loc[None] + z * (np.asarray(out["scale"])[None] * to_sigma)


def fit_and_predict(
    name, model_cls, cfg, d, ckpt_root, family="normal", n_paths=0, rng=None, copula_family=None, path_days=None
):
    """Train one condition and return its predictive quantiles in physical units.

    Args:
        name: label for the condition, used for the checkpoint directory.
        model_cls: forecaster class to instantiate.
        cfg: model configuration for that class.
        d: supervised windows from :func:`build_windows`.
        ckpt_root: directory under which this condition's checkpoints are kept.
        family: key into :data:`PARAMETRIC_FAMILY`, naming the marginal family
            and how the head's reported scale maps onto its dispersion. Ignored
            by quantile heads, which carry their own levels.
        n_paths: sampled trajectories to return alongside the quantiles, 0 for
            none.
        rng: generator for those samples.
        copula_family: for a quantile head, ``None`` (no coupling), ``"gaussian"``,
            ``"clayton"``, or ``"vine"`` -- whether and how to draw the
            trajectories through a copula fitted on the validation split. The
            margins are untouched either way, so this changes the paths and
            not the fan, which is precisely what the copula of Section 6.5
            contributes. ``"clayton"`` trades Gaussian's free pairwise
            correlations for a single exchangeable dependence parameter with
            lower-tail dependence (see
            :class:`~twiga.distributions.nn.copula.ClaytonCopula`). ``"vine"``
            (three targets only) factors the joint dependence into three
            independently-fitted pairwise copulas instead of one shared
            family across all three targets at once -- the non-parametric-
            in-spirit property TACTiS demonstrates, without attention (see
            :class:`~twiga.distributions.nn.vine_copula.DVineCopula3` and
            ``experiment/ch7_multitarget/COPULA_METHODS.md``).
        path_days: indices into ``d["x_te"]``/``d["y_te"]`` to restrict the
            forecast (and therefore quantiles, correlation, and sampled paths)
            to, or ``None`` for the whole test set. Sampling ``n_paths``
            trajectories is quadratic in nothing but linear in the number of
            rows forecast, so on a test set with thousands of windows (as
            ``forecast_examples.py``'s own ``main`` has), sampling every row
            when only a handful of days are ever plotted or scored allocates
            an array orders of magnitude larger than anything downstream
            reads -- this is exactly the mechanism that exhausted swap on one
            run (500 paths x ~2900 test windows x 48 steps x 3 targets is
            ~1.7 GB of float64 *per condition*, before even accounting for
            every condition's array coexisting in memory at once). Passing
            the small set of days actually needed bounds this regardless of
            the full test set's size.

    Returns:
        ``(quantiles, levels, paths, correlation)`` where quantiles is ``(days,
        n_levels, steps, targets)`` -- ``days`` is ``len(path_days)`` when
        given, the full test set otherwise -- levels is ``(n_levels,)`` for a
        fixed grid or ``(n_levels, targets)`` when the model learns one grid
        per target, paths is ``(n_paths, days, steps, targets)`` or an empty
        array, and correlation is the cross-target correlation the condition
        asserts, identity where the condition asserts none. For
        ``copula_family="clayton"`` this is a diagnostic conversion, not the
        fitted parameter itself: Kendall's tau implied by the fitted Clayton
        theta is mapped to an equivalent elliptical correlation via rho =
        sin(pi * tau / 2) (Greek-letter relation, Kendall 1949), purely so
        the panel that plots this matrix stays comparable across conditions
        -- the model itself never uses this converted value.
    """
    model = model_cls(model_config=cfg)
    model.data_pipeline = None
    model.resume_training = False
    model.checkpoints_path = ckpt_root / name.replace(" ", "_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["x_tr"], d["y_tr"], d["x_val"], d["y_val"])
    model.model.eval()
    x_te = d["x_te"] if path_days is None else d["x_te"][path_days]
    with torch.no_grad():
        out = model.model.forecast(torch.from_numpy(x_te))

    mu, sd = d["mu"], d["sd"]
    if "quantiles" in out:
        q = np.asarray(out["quantiles"])  # (B, N, H, D) standardised
        lv = np.asarray(out["quantile_levels"])
        # Levels may be (B,N,H) shared or (B,N,H,D) per target; average the axes
        # that are not slot or target, since the panel shows one representative
        # grid rather than a per-instance one.
        n_t = q.shape[-1]
        levels = lv.mean(axis=(0, 2)) if lv.ndim == 4 else np.tile(lv.mean(axis=(0, 2))[:, None], (1, n_t))
        pinned = float(((lv <= 0.0101) | (lv >= 0.9899)).mean())
        flag = "  <-- COLLAPSED, raise entropy_weight" if pinned > 0.10 else ""
        log.info("    %s: %.1f%% of quantile levels pinned at the clamp%s", name, pinned * 100, flag)

        # A quantile head states margins only, so a joint law has to come from
        # somewhere before trajectories can be drawn. Without the copula the
        # margins are sampled independently, which is the baseline the coupled
        # version has to beat; with it the dependence is fitted on validation
        # rows so it never sees the period the paths are drawn against.
        corr = np.eye(n_t)
        clayton_theta = None
        vine_theta = None
        if copula_family == "gaussian":
            with torch.no_grad():
                val = model.model.forecast(torch.from_numpy(d["x_val"]))
            corr = np.asarray(
                fit_gaussian_copula(d["y_val"], np.asarray(val["quantiles"]), np.asarray(val["quantile_levels"]))
            )
        elif copula_family == "clayton":
            with torch.no_grad():
                val = model.model.forecast(torch.from_numpy(d["x_val"]))
            clayton_theta = fit_clayton_theta(
                d["y_val"], np.asarray(val["quantiles"]), np.asarray(val["quantile_levels"])
            )
            tau = clayton_theta / (clayton_theta + 2.0)
            rho = np.sin(np.pi * tau / 2.0)  # diagnostic display conversion only, see docstring
            corr = np.eye(n_t) * (1.0 - rho) + rho
        elif copula_family == "vine":
            if n_t != 3:
                raise ValueError(f"copula_family='vine' requires exactly 3 targets, got {n_t}")
            with torch.no_grad():
                val = model.model.forecast(torch.from_numpy(d["x_val"]))
            vine_theta = fit_dvine3(
                d["y_val"],
                np.asarray(val["quantiles"]),
                np.asarray(val["quantile_levels"]),
                edge_families=VINE_EDGE_FAMILIES,
            )
            rho_pq = _theta_to_rho(VINE_EDGE_FAMILIES[0], vine_theta[0])
            rho_qv = _theta_to_rho(VINE_EDGE_FAMILIES[1], vine_theta[1])
            rho_pv_q = _theta_to_rho(VINE_EDGE_FAMILIES[2], vine_theta[2])
            # Implied marginal P-V correlation, for display only: the direct
            # (P,Q,V)-order path plus the Q-mediated one, inverting the
            # partial-correlation relationship the vine's tree-2 edge uses.
            rho_pv = rho_pq * rho_qv + rho_pv_q * np.sqrt(max(0.0, (1 - rho_pq**2) * (1 - rho_qv**2)))
            corr = np.array([[1.0, rho_pq, rho_pv], [rho_pq, 1.0, rho_qv], [rho_pv, rho_qv, 1.0]])
        paths = np.empty((0,))
        if n_paths:
            seed = int((rng or np.random.default_rng(0)).integers(2**31))
            if copula_family == "clayton":
                paths = coupled_samples(q, lv, n_paths, family="clayton", theta=clayton_theta, random_state=seed)
            elif copula_family == "vine":
                paths = coupled_samples(
                    q,
                    lv,
                    n_paths,
                    family="vine",
                    theta=vine_theta,
                    edge_families=VINE_EDGE_FAMILIES,
                    random_state=seed,
                )
            else:
                paths = coupled_samples(
                    q, lv, n_paths, corr if copula_family == "gaussian" else None, random_state=seed
                )
            paths = paths * sd + mu
        return q * sd + mu, levels, paths, corr

    loc = np.asarray(out["loc"])
    scale = np.asarray(out["scale"])
    dist, to_dispersion, _ = PARAMETRIC_FAMILY[family]
    z = dist.ppf(GRID_LEVELS)[None, :, None, None]
    q = loc[:, None] + z * (scale[:, None] * to_dispersion)
    paths = np.empty((0,))
    if n_paths:
        paths = sample_parametric(out, family, n_paths, rng or np.random.default_rng(0)) * sd + mu
    # A joint head publishes the correlation it learned; an independent head
    # asserts none, which is the identity rather than a missing value.
    corr = (
        np.asarray(out["correlation"]).reshape(-1, loc.shape[-1], loc.shape[-1]).mean(axis=0)
        if "correlation" in out
        else np.eye(loc.shape[-1])
    )
    return q * sd + mu, np.tile(GRID_LEVELS[:, None], (1, loc.shape[-1])), paths, corr


def _interp_at(qd: np.ndarray, levels_sorted: np.ndarray, p: float) -> np.ndarray:
    """Linear interpolation of ``qd`` (N, L, H), sorted along L, at probability ``p``."""
    idx = np.clip(np.searchsorted(levels_sorted, p), 1, len(levels_sorted) - 1)
    lo, hi = idx - 1, idx
    weight = (p - levels_sorted[lo]) / (levels_sorted[hi] - levels_sorted[lo])
    return qd[:, lo, :] * (1.0 - weight) + qd[:, hi, :] * weight


def quantile_loc_scale(q: np.ndarray, lv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A location/scale pair from a quantile forecast, for conformal calibration.

    Conformal calibration (:class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`)
    needs a point forecast and an adaptive scale, not a set of quantile levels.
    The median supplies the point forecast; the interquartile range supplies
    the scale, via the same normal-approximation convention
    :func:`twiga.distributions.ml.utils.get_sigma_prediction` already uses for
    quantile ML models: ``sigma = (q75 - q25) / (2 * 0.6745)``.

    ``lv`` is the single representative grid :func:`fit_and_predict` already
    returns (levels averaged over batch and horizon), not the per-instance
    levels FPQR actually learns, so this is the same display-level
    simplification the correlation panel and the pinned-quantile diagnostic
    already accept, not a new approximation introduced here.

    Args:
        q: Quantile forecasts in physical units, shape ``(N, L, H, D)``.
        lv: Probability levels, shape ``(L,)`` shared across targets, or
            ``(L, D)`` one grid per target.

    Returns:
        ``(mu, sigma)``, each shape ``(N, H, D)``.
    """
    n, _l, h, d = q.shape
    mu = np.empty((n, h, d))
    sigma = np.empty((n, h, d))
    for t in range(d):
        levels_t = lv[:, t] if lv.ndim == 2 else lv
        order = np.argsort(levels_t)
        levels_sorted = levels_t[order]
        # Two-step indexing: combining the fancy `order` index with a trailing
        # scalar `t` in one call triggers numpy's advanced-indexing axis
        # reordering (the L axis would end up first, not the batch axis).
        qt = q[:, order, :, :][..., t]  # (N, L, H)
        med = _interp_at(qt, levels_sorted, 0.5)
        lo = _interp_at(qt, levels_sorted, 0.25)
        hi = _interp_at(qt, levels_sorted, 0.75)
        mu[..., t] = med
        sigma[..., t] = (hi - lo) / (2 * 0.6745)
    return mu, np.maximum(sigma, 1e-6)


def conformal_rows(preds, truth, days, alpha=0.1):
    """Multivariate adaptive conformal calibration on top of every condition.

    ``Chapter-6.tex`` (``subsec:ch6-evaluation``) requires every backbone
    evaluated under the multivariate adaptive conformal calibration of
    ``sec:ch6-conformal``. Each condition's own forecast supplies a
    location/scale pair via :func:`quantile_loc_scale` (this covers the
    parametric heads too: :func:`fit_and_predict` returns a quantile array
    for every condition, dense analytical levels for the parametric heads
    and the learned grid for FPQR), and its correlation -- fitted copula for
    the coupled FPQR variants, the head's own learned covariance for
    ``joint``/``joint laplace``, identity for every uncoupled condition --
    supplies :class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`'s
    ``"mahalanobis"`` score. ``"max-norm"`` is calibrated alongside it as the
    baseline pooling rule, so the two can be read off the same table; see
    ``COPULA_METHODS.md`` section 6 for why ``"mahalanobis"`` is expected to
    be valid but conservative, not a free efficiency win, except where the
    correlation genuinely varies by condition.

    The evaluated days supply no separate calibration split, so they are
    themselves halved chronologically: the first half calibrates ``q_hat``
    and the second is scored, which is a within-fold split rather than a
    leak, since no scored day was used for calibration.

    Args:
        preds: fitted conditions, keyed by name, as returned by
            :func:`fit_and_predict`.
        truth: observations in physical units, ``(days, steps, targets)``.
        days: indices of the evaluation days.
        alpha: miscoverage level; ``1 - alpha`` is the nominal joint coverage.

    Returns:
        Tidy frame with one row per condition per ``score_type``: nominal and
        empirical joint coverage (all targets simultaneously within bounds,
        averaged over the scored days and horizon) and the mean interval
        width. Empty if there are too few days to split into a calibration
        and an evaluation half.
    """
    n_days = len(days)
    n_calib = n_days // 2
    if n_calib < 1 or n_days - n_calib < 1:
        return pd.DataFrame()

    y = truth[days]
    rows = []
    for name, (q, lv, _paths, corr) in preds.items():
        # quantile_loc_scale runs over every row q has, which is only the same
        # set as `days` when a caller's q already covers exactly the days being
        # scored (true of copula_variants.py's per-fold q); forecast_examples.py's
        # main() hands q covering the full path_days superset instead, so `days`
        # (already local positions into that superset) must select here too.
        mu, sigma = quantile_loc_scale(q, lv)
        mu, sigma = mu[days], sigma[days]
        h, d = mu.shape[1], mu.shape[2]
        corr_h = np.broadcast_to(corr, (h, d, d))
        mu_c, mu_e = mu[:n_calib], mu[n_calib:]
        sigma_c, sigma_e = sigma[:n_calib], sigma[n_calib:]
        y_c, y_e = y[:n_calib], y[n_calib:]

        for score_type in ("max-norm", "mahalanobis"):
            conf = MultivariateConformal(score_type=score_type, alpha=alpha)
            if score_type == "mahalanobis":
                conf.calibrate(mu_c, y_c, sigma_c, corr_h)
            else:
                conf.calibrate(mu_c, y_c, sigma_c)
            lower, upper = conf.generate_intervals(mu_e, sigma_e)
            covered = np.all((y_e >= lower) & (y_e <= upper), axis=-1)
            rows.append(
                {
                    "model": name,
                    "score_type": score_type,
                    "alpha": alpha,
                    "nominal_coverage": 1.0 - alpha,
                    "n_calib": n_calib,
                    "n_eval": n_days - n_calib,
                    "coverage": float(covered.mean()),
                    "mean_width": float((upper - lower).mean()),
                }
            )
    return pd.DataFrame(rows)


def median_of(q, levels, t_i):
    """Slice nearest the 0.5 level for one target."""
    col = levels[:, t_i] if levels.ndim == 2 else levels
    return q[:, int(np.abs(col - 0.5).argmin()), :, t_i]


def panel_metrics(y, q, levels, t_i, scale_ref):
    """NRMSE, CWE and CRPS for one day and target, from that model's own levels."""
    col = levels[:, t_i] if levels.ndim == 2 else levels
    order = np.argsort(col)
    lv, qq = col[order], q[order]
    med = qq[int(np.abs(lv - 0.5).argmin())]

    nrmse = float(np.sqrt(np.mean((y - med) ** 2)) / scale_ref)
    crps = float(np.mean(crps_quantile(y[None, :], qq.T[None, :, :].transpose(0, 2, 1), quantile_levels=lv)))

    lo, hi = qq[0], qq[-1]
    alpha = float(max(lv[0] * 2, 1e-3))
    nm = float(nmpi(lo, hi, scale=scale_ref))
    picp = float(interval_coverage(y, lo, hi))
    ref = float(np.quantile(y, 1 - alpha / 2) - np.quantile(y, alpha / 2)) / scale_ref
    try:
        cwe = float(np.ravel(cwe_score(min(nm, 1.0), picp, None, min(max(ref, 1e-6), 1.0), alpha=alpha))[0])
    except Exception as exc:  # noqa: BLE001
        log.warning("      CWE unavailable (nmpi=%.3f, ref=%.3f): %s", nm, ref, exc)
        cwe = float("nan")
    return nrmse, cwe, crps


def quantile_at(q, levels, level, t_i):
    """Slice of ``q`` nearest a probability level, for one target."""
    col = levels[:, t_i] if levels.ndim == 2 else levels
    return q[:, int(np.abs(col - level).argmin()), :, t_i]


def _operational_rows(model, target, truth, ensemble, threshold, predicted, realised, flip=False, curve_sink=None):
    """Rows for one operational event: rate, error, Brier and its decomposition.

    Args:
        model: condition name.
        target: label for the event, ``S`` or a voltage side.
        truth: observed series the event is defined on.
        ensemble: sampled series, first axis the ensemble.
        threshold: event threshold.
        predicted: predicted event rate, pooled.
        realised: realised event rate.
        flip: the event is an exceedance encoded as an indicator, so the
            ``<= threshold`` convention of the underlying metrics already
            matches and no complement is needed.
        curve_sink: if given, the full relative-economic-value curve (one row
            per cost-loss ratio) is appended to this list rather than only
            reducing it to its peak, matching how Richardson (2000) plots
            value against cost-loss ratio rather than reporting one number --
            the peak alone cannot show which users the forecast serves.

    Returns:
        List of tidy metric rows.
    """
    decomposition = brier_decomposition(truth, ensemble, threshold=threshold)
    ratios, value = relative_economic_value(truth, ensemble, threshold=threshold)
    finite = value[np.isfinite(value)]
    if curve_sink is not None:
        curve_sink.extend(
            {"model": model, "target": target, "cost_loss_ratio": float(r), "value": float(v)}
            for r, v in zip(ratios, value, strict=True)
            if np.isfinite(v)
        )
    out = [
        ("Pr(event) predicted", predicted),
        ("Pr(event) realised", realised),
        ("Event rate error (%)", 100 * (predicted - realised) / realised if realised else float("nan")),
        ("Brier score", float(np.mean(brier_score(truth, ensemble, threshold=threshold)))),
        ("Brier skill score", decomposition["brier_skill_score"]),
        ("Reliability", decomposition["reliability"]),
        ("Resolution", decomposition["resolution"]),
        # The peak of the value curve is the best a user of any cost-loss ratio
        # could obtain; reporting it beside the Brier score separates a forecast
        # that scores well from one that would change a decision.
        ("Peak relative economic value", float(finite.max()) if finite.size else float("nan")),
    ]
    return [{"tier": "operational", "metric": m, "target": target, "model": model, "value": v} for m, v in out]


def pointwise_losses(preds, truth, targets, rated, days, voltage_band=(0.9, 1.1), alpha=0.1):
    """Per-forecast losses, for the tests that need a series rather than a mean.

    Diebold-Mariano and the model confidence set are asymptotic in the length of
    the loss series. A fold mean gives one value per fold, and at eight folds the
    normal approximation is poor: simulated under a true null, the test rejects
    at roughly $0.077$ where it should reject at $0.05$. One value per forecast
    day gives around 250, which is the range where both hold their size.

    These are proper scores, not point-forecast errors. The test needs only a
    stationary loss differential, so any pointwise score serves, and the scores
    used here are the ones the comparison rests on: the Winkler score (the one
    Chapter-6.tex actually specifies for significance -- see
    ``twiga.core.metrics``'s documentation for
    why CWE cannot furnish a loss series at all), the continuous ranked
    probability score for the marginal tier, the energy score for the joint
    tier, and the Brier contribution for the operational event.

    Args:
        preds: fitted conditions, keyed by name.
        truth: observations in physical units, ``(days, steps, targets)``.
        targets: target names.
        rated: apparent-power threshold for the operational event.
        days: indices of the evaluation days.
        voltage_band: statutory limits in per unit.
        alpha: nominal miss rate for the Winkler score, matching
            :func:`metrics_table`'s own default.

    Returns:
        Tidy DataFrame with day, model, metric and value, one row per forecast.
    """
    rows = []
    observed = truth[days]
    for name, (q, lv, paths, _) in preds.items():
        for j, day in enumerate(days):
            crps = float(
                np.mean(
                    [
                        crps_quantile(
                            observed[j, :, i],
                            q[day][np.argsort(lv[:, i] if lv.ndim == 2 else lv), :, i],
                            quantile_levels=np.sort(lv[:, i] if lv.ndim == 2 else lv),
                        )
                        for i in range(len(targets))
                    ]
                )
            )
            rows.append({"day": int(day), "model": name, "metric": "CRPS", "value": crps})

            # Winkler score, averaged across targets, matching metrics_table's
            # own per-target computation but per day rather than pooled over
            # every day at once -- this is the series pairwise_dm_table/
            # mcs_ranking need.
            ws = float(
                np.mean(
                    [
                        winkler_score(
                            observed[j : j + 1, :, i],
                            quantile_at(q, lv, alpha / 2, i)[day : day + 1],
                            quantile_at(q, lv, 1 - alpha / 2, i)[day : day + 1],
                            alpha=alpha,
                        )
                        for i in range(len(targets))
                    ]
                )
            )
            rows.append({"day": int(day), "model": name, "metric": "Winkler", "value": ws})

            if not np.asarray(paths).size:
                continue
            ens = np.asarray(paths)[:, day]
            rows.append(
                {
                    "day": int(day),
                    "model": name,
                    "metric": "Energy score",
                    "value": float(np.mean([energy_score(observed[j, h], ens[:, h]) for h in range(ens.shape[1])])),
                }
            )
            # Brier contribution for the thermal event, averaged over the day.
            s_obs = np.hypot(observed[j, :, 0], observed[j, :, 1])
            s_hat = np.hypot(ens[..., 0], ens[..., 1])
            prob = (s_hat > rated).mean(axis=0)
            rows.append(
                {
                    "day": int(day),
                    "model": name,
                    "metric": "Brier score",
                    "value": float(np.mean((prob - (s_obs > rated).astype(float)) ** 2)),
                }
            )
            # And for the voltage event, where the band is actually crossed.
            v_obs, v_hat = observed[j, :, 2], ens[..., 2]
            low, high = voltage_band
            event = (v_obs < low) | (v_obs > high)
            prob_v = ((v_hat < low) | (v_hat > high)).mean(axis=0)
            if event.any() or prob_v.any():
                rows.append(
                    {
                        "day": int(day),
                        "model": name,
                        "metric": "Brier score (voltage)",
                        "value": float(np.mean((prob_v - event.astype(float)) ** 2)),
                    }
                )
    return pd.DataFrame(rows)


def metrics_table(preds, truth, targets, rated, days, alpha=0.1, voltage_band=(0.9, 1.1), scale_ref=None):
    """Every metric the evaluation methodology specifies, in its three tiers.

    The tiers answer different questions and are not interchangeable, which is
    why they are reported together rather than reduced to one headline number.

    Marginal, per target: NRMSE for the point forecast, CRPS and the Winkler
    score for the distribution. These carry the comparative conclusions. CWE
    appears as a labelled diagnostic only, never as a ranking, since the
    propriety claim made for it was withdrawn.

    Joint, across targets: the energy score and the variogram score. Both are
    proper. The second is present because the first has limited power against a
    misspecified dependence structure, so reporting the energy score alone risks
    concluding that dependence does not matter from a statistic that cannot see
    it. Where the two disagree, the disagreement is the finding.

    Operational: the probability that apparent power exceeds the rating, against
    the frequency actually realised, plus the Brier score for that event. This
    is the tier a decision rests on, and the one a factorised likelihood cannot
    produce without assuming a coupling.

    Args:
        preds: fitted conditions, keyed by name.
        truth: observations in physical units, ``(days, steps, targets)``.
        targets: target names.
        rated: apparent-power rating the exceedance event is defined against.
        days: indices of the evaluation days.
        alpha: nominal miss rate for the Winkler score and the interval.
        voltage_band: statutory limits in per unit; EN 50160 is (0.9, 1.1) and
            ANSI C84.1 Range A is (0.95, 1.05).
        scale_ref: NRMSE/NMPI normalisation range per target, or ``None`` to
            compute it from ``truth`` (its full first axis, not just ``days``,
            so a subsampled ``truth`` still needs the true full-test range
            passed in explicitly here, or NRMSE/NMPI stop being "constant
            across days and panels" as documented at the call site).

    Returns:
        Tidy DataFrame with tier, metric, target, model and value.
    """
    rows = []
    observed = truth[days]
    s_obs = np.hypot(observed[..., 0], observed[..., 1])
    realised = float((s_obs > rated).mean())
    if scale_ref is None:
        scale_ref = np.ptp(truth.reshape(-1, truth.shape[2]), axis=0)

    for name, (q, lv, paths, _) in preds.items():
        for i, target in enumerate(targets):
            y = observed[..., i]
            med = quantile_at(q, lv, 0.5, i)[days]
            lo = quantile_at(q, lv, alpha / 2, i)[days]
            hi = quantile_at(q, lv, 1 - alpha / 2, i)[days]
            col = lv[:, i] if lv.ndim == 2 else lv
            order = np.argsort(col)
            crps = float(np.mean(crps_quantile(y, q[days][:, order, :, i], quantile_levels=col[order])))
            ws = float(winkler_score(y, lo, hi, alpha=alpha))
            picp = float(interval_coverage(y, lo, hi))
            for metric, value in (
                ("NRMSE", float(np.sqrt(np.mean((y - med) ** 2)) / scale_ref[i])),
                ("CRPS", crps),
                ("Winkler", ws),
                ("PICP (diagnostic)", picp),
                ("NMPI (diagnostic)", float(nmpi(lo, hi, scale=scale_ref[i]))),
            ):
                rows.append({"tier": "marginal", "metric": metric, "target": target, "model": name, "value": value})

        if not np.asarray(paths).size:
            continue
        ens = np.asarray(paths)[:, days]
        es = float(
            np.mean(
                [energy_score(observed[d, h], ens[:, d, h]) for d in range(len(days)) for h in range(observed.shape[1])]
            )
        )
        vs = float(
            np.mean(
                [
                    variogram_score(observed[d, h], ens[:, d, h])
                    for d in range(len(days))
                    for h in range(observed.shape[1])
                ]
            )
        )
        for metric, value in (("Energy score", es), ("Variogram score", vs)):
            rows.append({"tier": "joint", "metric": metric, "target": "P,Q,V", "model": name, "value": value})

        # Thermal event: apparent power beyond the screening threshold.
        s_hat = np.hypot(ens[..., 0], ens[..., 1])
        predicted = float((s_hat > rated).mean())
        rows += _operational_rows(name, "S", s_obs, s_hat, rated, predicted, realised)

        # Voltage event: an excursion beyond the statutory band. This is the
        # binding constraint on a weak feeder, where thermal loading never is,
        # so the tier would otherwise not measure the case study's own event.
        v_obs, v_hat = observed[..., 2], ens[..., 2]
        for edge, side in ((voltage_band[0], "low"), (voltage_band[1], "high")):
            if side == "low":
                obs_event, pred_event = v_obs < edge, v_hat < edge
            else:
                obs_event, pred_event = v_obs > edge, v_hat > edge
            if not obs_event.any():
                continue
            rows += _operational_rows(
                name,
                f"V {side}",
                obs_event.astype(float),
                pred_event.astype(float),
                0.5,
                float(pred_event.mean()),
                float(obs_event.mean()),
                flip=True,
            )

    return pd.DataFrame(rows)


def hosting_capacity_rows(
    preds, truth, targets, rated, days, voltage_band=(0.9, 1.1), include_voltage=True, curve_sink=None
):
    r"""The hosting-capacity union event, `eqn:ch6-hosting-capacity` in Chapter-6.tex.

    The operational tier of :func:`metrics_table` scores the thermal and
    voltage events separately, and marginally at each timestep. Chapter 6's
    own hosting-capacity criterion is neither: it is the probability that
    *either* event occurs *anywhere* across the forecast horizon,

    .. math::

        \Pr\!\left(\bigcup_{t=1}^{H} \{S_t > S^{\mathrm{rated}}\}
        \cup \{V_t \notin [\underline{V}, \overline{V}]\}\right).

    Evaluating the union requires the joint density -- a sample that gets P
    right and Q wrong at the same timestep understates the true thermal risk,
    and a sample that violates voltage at one timestep and clears the rating
    at another still counts as a single breach for this criterion, which
    marginal per-timestep scoring cannot represent.

    Args:
        preds: fitted conditions, keyed by name (as :func:`metrics_table`).
        truth: observations in physical units, ``(days, steps, targets)``.
        targets: target names; index 2 (``v_lv_pu``) is read for the voltage
            term when ``include_voltage`` is set.
        rated: apparent-power rating the thermal event is defined against.
        days: indices of the evaluation days.
        voltage_band: statutory limits in per unit; EN 50160 is (0.9, 1.1),
            ANSI C84.1 Range A is (0.95, 1.05).
        include_voltage: include the voltage term in the union. ``False``
            reduces this to the thermal-only exceedance event, the criterion
            that binds on DSS-US, where voltage never does (Chapter-6.tex,
            subsec:ch6-protocol).
        curve_sink: if given, forwarded to :func:`_operational_rows` so the
            full relative-economic-value curve is collected rather than only
            its peak.

    Returns:
        Tidy DataFrame with tier "operational", one row per model per metric,
        following :func:`_operational_rows`'s convention: the event is
        encoded as a 0/1 indicator ahead of the call, matching the existing
        voltage-event branch above.
    """
    observed = truth[days]
    s_obs = np.hypot(observed[..., 0], observed[..., 1])
    thermal_obs = (s_obs > rated).any(axis=-1)  # one flag per forecast day, not per timestep
    if include_voltage:
        v_obs = observed[..., 2]
        voltage_obs = ((v_obs < voltage_band[0]) | (v_obs > voltage_band[1])).any(axis=-1)
        union_obs = thermal_obs | voltage_obs
    else:
        union_obs = thermal_obs
    realised = float(union_obs.mean())

    label = "hosting-capacity" if include_voltage else "thermal"
    rows = []
    for name, (_q, _lv, paths, _) in preds.items():
        if not np.asarray(paths).size:
            continue
        ens = np.asarray(paths)[:, days]  # (n_paths, days, horizon, targets)
        s_hat = np.hypot(ens[..., 0], ens[..., 1])
        thermal_hat = (s_hat > rated).any(axis=-1)  # (n_paths, days)
        if include_voltage:
            v_hat = ens[..., 2]
            voltage_hat = ((v_hat < voltage_band[0]) | (v_hat > voltage_band[1])).any(axis=-1)
            union_hat = thermal_hat | voltage_hat
        else:
            union_hat = thermal_hat
        predicted = float(union_hat.mean())
        rows += _operational_rows(
            name,
            label,
            union_obs.astype(float),
            union_hat.astype(float),
            0.5,
            predicted,
            realised,
            flip=True,
            curve_sink=curve_sink,
        )

    return pd.DataFrame(rows)


def correlation_panel(preds, observed, residual, targets):
    """Cross-target correlation: two references, then what each condition asserts.

    Both references are needed, and reading the wrong one inverts the conclusion.
    The raw panel is the empirical correlation of the test series. It is not what
    a likelihood has to represent: every head sits on a backbone that already
    couples the targets through its conditional mean, so what is left for the
    likelihood is the correlation of the residuals. Judged against the raw panel
    the joint heads look far too weak; judged against the residual panel they are
    close. ``residual_dependence.py`` measures the same quantity on a longer run.

    The residual reference subtracts the pointwise mean of every condition's
    median forecast, which estimates the shared conditional mean without
    privileging one head's own fit.

    An independent head and an uncoupled quantile head both assert the identity,
    which is the point of putting them beside the joint heads.
    """
    labels = [AXIS[t_].split()[0].strip("$") for t_ in targets]
    panels = [("observed (raw)", observed), ("observed (residual)", residual)]
    panels += [(n, c) for n, (_, _, _, c) in preds.items()]
    cols = min(4, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.0 * rows), layout="constrained", squeeze=False)
    image = None
    for k, ax in enumerate(axes.ravel()):
        if k >= len(panels):
            ax.axis("off")
            continue
        name, m = panels[k]
        m = np.asarray(m)
        image = ax.imshow(m, vmin=-1, vmax=1, cmap=CORR_CMAP)
        ax.set_xticks(range(len(targets)), labels)
        ax.set_yticks(range(len(targets)), labels)
        for i in range(len(targets)):
            for j in range(len(targets)):
                # White on the saturated ends, dark on the pale centre, so every
                # cell stays legible whatever its value.
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(m[i, j]) > 0.55 else "0.15",
                )
        ax.text(0.02, 0.98, _cap(name), transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=list(axes.ravel()), shrink=0.8, label="Correlation", ticks=[-1, -0.5, 0, 0.5, 1])
    fig.savefig(OUT / "correlation_matrices.pdf", bbox_inches="tight")
    fig.savefig(OUT / "correlation_matrices.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def dependence_matrix(values, max_samples=2000):
    """Pairwise distance correlation of a ``(rows, targets)`` array.

    Pearson answers a narrower question than the one the chapter asks. It
    measures linear association, so it reports near zero for a relationship that
    is real but neither linear nor monotone, and Section 6.3 shows that is the
    prevailing case: across scenarios where the linear correlation of active and
    reactive power passes through zero, the distance correlation does not.
    Distance correlation is zero if and only if the two variables are
    independent and presumes no functional form, so it is the measure the
    argument actually rests on. The price is that it carries no sign, which is
    why the signed Pearson matrices are kept alongside rather than replaced.
    """
    n_targets = values.shape[-1]
    out = np.eye(n_targets)
    for i in range(n_targets):
        for j in range(i + 1, n_targets):
            out[i, j] = out[j, i] = distance_correlation(values[:, i], values[:, j], max_samples=max_samples)
    return out


def dependence_panel(preds, observed, residual, targets, centred=False, stem="dependence_matrices"):
    """Distance correlation: two references, then what each condition generates.

    The model panels are measured on each condition's own sampled ensemble, not
    read off a parameter, so every panel on this figure is the same statistic
    computed the same way. That is what makes the comparison fair: a joint
    Gaussian head states a correlation, a copula states a correlation over
    uniforms, and a factorised head states nothing at all, but all three can be
    sampled and all three samples can be measured.

    Two versions are drawn, and the difference between them is the chapter's
    argument rather than a detail. Pooled over lead times, an ensemble carries
    dependence from two sources: the likelihood, and the conditional mean, which
    moves all three targets together through the shared backbone. The pooled
    figure measures their sum, which is what a downstream power-flow study
    actually receives, and on it a factorised likelihood is far from
    independent. Centring each draw on its own predictive mean removes the
    second source and leaves what the likelihood alone contributes, which is the
    quantity that separates the conditions.

    A condition that carries no dependence lands near zero but not exactly at
    zero, since distance correlation is non-negative and its finite-sample
    estimate is biased upward.

    Args:
        preds: fitted conditions, keyed by name.
        observed: reference matrix for the raw series.
        residual: reference matrix for the backbone residuals.
        targets: target names, for the axis labels.
        centred: subtract each draw's own predictive mean before measuring, so
            the statistic reflects the likelihood rather than the shared mean.
        stem: output filename stem.
    """
    labels = [AXIS[t_].split()[0].strip("$") for t_ in targets]
    panels = [("observed (raw)", observed), ("observed (residual)", residual)]
    for name, value in preds.items():
        paths = np.asarray(value[2])
        if not paths.size:
            continue
        sample = paths - paths.mean(axis=0, keepdims=True) if centred else paths
        panels.append((name, dependence_matrix(sample.reshape(-1, sample.shape[-1]))))
    cols = min(4, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.0 * rows), layout="constrained", squeeze=False)
    image = None
    for k, ax in enumerate(axes.ravel()):
        if k >= len(panels):
            ax.axis("off")
            continue
        name, m = panels[k]
        image = ax.imshow(m, vmin=0, vmax=1, cmap=DCOR_CMAP)
        ax.set_xticks(range(len(targets)), labels)
        ax.set_yticks(range(len(targets)), labels)
        for i in range(len(targets)):
            for j in range(len(targets)):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if m[i, j] > 0.55 else "0.15",
                )
        ax.text(0.02, 0.98, _cap(name), transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=list(axes.ravel()), shrink=0.8, label="Distance Correlation")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def cloud_panel(preds, truth, targets, day, step):
    """Pairwise sample clouds for every condition at one lead time.

    Rows are the three target pairs and columns the conditions, so a column is
    one model's joint law and a row is one constraint's plane. The marginal
    histograms are drawn on each panel's top strip: they are what a per-target
    figure would show, and they coincide across columns even where the clouds do
    not, which is the whole argument for looking at the joint distribution.
    """
    drawable = [(n, v[2]) for n, v in preds.items() if np.asarray(v[2]).size]
    if not drawable:
        log.warning("no sampled paths; skipping sample clouds (pass --paths)")
        return
    pairs = [(0, 1), (0, 2), (1, 2)]
    n_col = len(drawable)
    fig = plt.figure(figsize=(3.4 * n_col, 3.5 * len(pairs)), layout="constrained")
    gs = fig.add_gridspec(2 * len(pairs), n_col, height_ratios=[1, 4] * len(pairs))
    for r, (a, b) in enumerate(pairs):
        for c, (name, paths) in enumerate(drawable):
            ax = fig.add_subplot(gs[2 * r + 1, c])
            axt = fig.add_subplot(gs[2 * r, c], sharex=ax)
            xs, ys = paths[:, day, step, a], paths[:, day, step, b]
            ax.scatter(xs, ys, s=4, c=PREDICTED, alpha=0.20, lw=0)
            ax.scatter(
                [truth[day, step, a]],
                [truth[day, step, b]],
                marker="*",
                s=150,
                c=OBSERVED,
                ec="k",
                lw=0.4,
                zorder=5,
            )
            ax.set_xlabel(AXIS[targets[a]])
            if c == 0:
                ax.set_ylabel(AXIS[targets[b]])
            axt.hist(xs, bins=30, color=PREDICTED, alpha=0.55, density=True, lw=0)
            axt.axis("off")
            axt.text(
                0.02,
                0.98,
                f"{_cap(name)}, $\\rho$={np.corrcoef(xs, ys)[0, 1]:+.2f}",
                transform=axt.transAxes,
                va="top",
                ha="left",
                fontsize=7,
            )
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.savefig(OUT / "sample_clouds.pdf", bbox_inches="tight")
    fig.savefig(OUT / "sample_clouds.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="DSS-EU:default:0.4:0.4")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--no-latex", action="store_true", help="skip LaTeX typesetting")
    ap.add_argument("--backbone", default="mlpf", choices=["mlpf", "mlpgam"], help="encoder backbone")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="redraw from cached predictions instead of retraining; use for styling changes",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["joint", "independent", "fpqr shared grid", "fpqr per-target grid"],
        help=(
            "conditions to train and plot, in column order, or the single word "
            "'all' for every condition the chosen --backbone has (for an "
            "unattended server run building the complete metrics/conformal "
            "tables; the panel PDFs still render, just with one column per "
            "condition). Available: joint, joint laplace, independent, "
            "independent laplace, fpqr shared grid, fpqr per-target grid, fpqr "
            "shared + copula, fpqr per-target + copula, fpqr shared + clayton "
            "copula, fpqr per-target + clayton copula, fpqr shared + vine "
            "copula, fpqr per-target + vine copula (three targets only). Each "
            "adds a column, so more than five crowds the panel at thesis width "
            "-- fine for a server run's tables, less so for a figure meant to "
            "be read."
        ),
    )
    ap.add_argument(
        "--paths",
        type=int,
        default=0,
        help=(
            "sampled trajectories to overlay on each panel, 0 to disable. "
            "Parametric heads only: a quantile head carries no joint law to "
            "sample from without the copula of Section 6.5."
        ),
    )
    ap.add_argument(
        "--metric-days",
        type=int,
        default=60,
        help="days sampled for the three-tier metrics table; the joint scores are quadratic in ensemble size",
    )
    ap.add_argument(
        "--n-quantiles",
        type=int,
        default=21,
        help="FPQR quantile slots; 21 is the top of the thesis HPO range [7,9,15,21]",
    )
    ap.add_argument(
        "--entropy-weight",
        type=float,
        default=100.0,
        help=(
            "FPQR proposal entropy weight. Must be re-tuned with n_quantiles: the "
            "pinball term grows linearly in N while entropy is bounded by log(N), "
            "so 20 suits 9 slots and 200 suits 31. The clamp fraction is reported "
            "so a collapsed grid is visible rather than silently plotted."
        ),
    )
    args = ap.parse_args()

    # LaTeX typesetting when a toolchain is present, matplotlib's renderer
    # otherwise, so the figure still builds on a machine without TeX.
    try:
        configure(latex=not args.no_latex)
    except Exception as exc:  # noqa: BLE001
        log.warning("thesis style unavailable (%s); falling back to matplotlib defaults", exc)

    global OUT
    OUT = OUT_ROOT if args.backbone == "mlpf" else OUT_ROOT / args.backbone
    OUT.mkdir(parents=True, exist_ok=True)
    d = build_windows(args.scenario, args.lookback, args.horizon)
    targets = d["targets"]
    st = pd.to_datetime(d["stamps_te"].ravel())
    log.info("scenario     %s", args.scenario)
    log.info(
        "windows      lookback %s steps (%.0f h), horizon %s steps (%.0f h), stride 1",
        args.lookback,
        args.lookback / 2,
        args.horizon,
        args.horizon / 2,
    )
    log.info(
        "split        train %d  val %d  test %d windows", d["x_tr"].shape[0], d["x_val"].shape[0], d["x_te"].shape[0]
    )
    log.info("test period  %s to %s  (%d distinct days)", st.min(), st.max(), len(pd.Series(st.date).unique()))
    log.info("targets      %s", targets)

    ckpt_root = Path(tempfile.mkdtemp(prefix="fc_ex_"))
    path_rng = np.random.default_rng(args.seed)
    kw = {
        "epochs": args.epochs,
        "seed": args.seed,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "backbone": backbone_settings(),
    }
    # FPQR needs an entropy weight matched to its slot count or the proposal grid
    # collapses onto the tau clamp, and the panel would show that artifact rather
    # than the method.
    fpqr = {
        "n_quantiles": args.n_quantiles,
        "entropy_weight": args.entropy_weight,
        "num_cosines": 0,
        "rope_type": "tau_pos",
        "qvn_type": "monotone",
    }

    # The conditions available on either backbone. MLPGAM keeps an additive mean,
    # so its joint head is the additive one: projecting the latent to form the
    # mean would discard the per-feature decomposition the backbone exists to
    # give. MLPGAM has no joint Laplace head, so that variant is MLPF only and is
    # reported as unavailable rather than silently dropped.
    available = {
        "mlpf": {
            "joint": (MLPFMVNormalModel, MLPFMVNormalConfig, "normal", {}, None),
            "joint laplace": (MLPFMVLaplaceModel, MLPFMVLaplaceConfig, "laplace-sd", {}, None),
            "independent": (MLPFNormalModel, MLPFNormalConfig, "normal", {}, None),
            "independent laplace": (MLPFLaplaceModel, MLPFLaplaceConfig, "laplace-dispersion", {}, None),
            "fpqr shared grid": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, None),
            "fpqr per-target grid": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, None),
            "fpqr shared + copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "gaussian"),
            "fpqr per-target + copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "gaussian"),
            "fpqr shared + clayton copula": (
                MLPFFPQRModel,
                MLPFFPQRConfig,
                None,
                {"per_target_tau": False},
                "clayton",
            ),
            "fpqr per-target + clayton copula": (
                MLPFFPQRModel,
                MLPFFPQRConfig,
                None,
                {"per_target_tau": True},
                "clayton",
            ),
            "fpqr shared + vine copula": (
                MLPFFPQRModel,
                MLPFFPQRConfig,
                None,
                {"per_target_tau": False},
                "vine",
            ),
            "fpqr per-target + vine copula": (
                MLPFFPQRModel,
                MLPFFPQRConfig,
                None,
                {"per_target_tau": True},
                "vine",
            ),
        },
        "mlpgam": {
            "joint": (MLPGAMMVNormalModel, MLPGAMMVNormalConfig, "normal", {}, None),
            "independent": (MLPGAMNormalModel, MLPGAMNormalConfig, "normal", {}, None),
            "independent laplace": (MLPGAMLaplaceModel, MLPGAMLaplaceConfig, "laplace-dispersion", {}, None),
            "fpqr shared grid": (MLPGAMFPQRModel, MLPGAMFPQRConfig, None, {"per_target_tau": False}, None),
            "fpqr per-target grid": (MLPGAMFPQRModel, MLPGAMFPQRConfig, None, {"per_target_tau": True}, None),
            "fpqr shared + copula": (MLPGAMFPQRModel, MLPGAMFPQRConfig, None, {"per_target_tau": False}, "gaussian"),
            "fpqr per-target + copula": (
                MLPGAMFPQRModel,
                MLPGAMFPQRConfig,
                None,
                {"per_target_tau": True},
                "gaussian",
            ),
            "fpqr shared + clayton copula": (
                MLPGAMFPQRModel,
                MLPGAMFPQRConfig,
                None,
                {"per_target_tau": False},
                "clayton",
            ),
            "fpqr per-target + clayton copula": (
                MLPGAMFPQRModel,
                MLPGAMFPQRConfig,
                None,
                {"per_target_tau": True},
                "clayton",
            ),
            "fpqr shared + vine copula": (
                MLPGAMFPQRModel,
                MLPGAMFPQRConfig,
                None,
                {"per_target_tau": False},
                "vine",
            ),
            "fpqr per-target + vine copula": (
                MLPGAMFPQRModel,
                MLPGAMFPQRConfig,
                None,
                {"per_target_tau": True},
                "vine",
            ),
        },
    }[args.backbone]

    if args.variants == ["all"]:
        args.variants = list(available)

    missing = [v for v in args.variants if v not in available]
    if missing:
        raise SystemExit(f"{args.backbone} has no head for: {', '.join(missing)}")

    # Ingredients only, not a built cfg: each pass below builds its own cfg
    # instance fresh, so nothing a Lightning fit mutates in place on one pass
    # (however unlikely) can leak into the other's.
    specs = []
    for name in args.variants:
        cls, cfg_cls, family, extra, copula_family = available[name]
        extra = {**extra, **fpqr} if name.startswith("fpqr") else extra
        specs.append((name, cls, cfg_cls, extra, family, copula_family))

    # Predictions are deterministic given the seed, so they are cached. Redrawing
    # the panels for a styling change should not cost four model fits, which is
    # exactly what it cost before this existed.
    cache = OUT / "predictions.npz"
    cached = None
    if args.replot and cache.exists():
        blob = np.load(cache, allow_pickle=False)
        names = [str(n) for n in blob["model_names"]]
        # The cache is keyed by condition, so it only satisfies this run if it
        # holds every condition asked for. Without this check, adding a variant
        # and passing --replot would quietly redraw the previous set of columns.
        # A cache written before the correlation and paths were stored lacks
        # those keys, so membership of model_names alone is not enough.
        required = {f"{k}::{n}" for n in args.variants for k in ("q", "lv", "co", "q2", "lv2", "pa2", "co2")}
        if set(args.variants) <= set(names) and required <= set(blob.files) and "path_days" in blob.files:
            cached = {n: (blob[f"q::{n}"], blob[f"lv::{n}"], np.empty((0,)), blob[f"co::{n}"]) for n in args.variants}
            cached_paths = {
                n: (blob[f"q2::{n}"], blob[f"lv2::{n}"], blob[f"pa2::{n}"], blob[f"co2::{n}"]) for n in args.variants
            }
            path_days = blob["path_days"]
        else:
            absent = ", ".join(sorted(set(args.variants) - set(names))) or "the current cache format"
            log.info("cache at %s lacks %s; training instead", cache, absent)
    if cached is not None:
        preds, preds_paths = cached, cached_paths
        log.info("loaded cached predictions for %d models from %s", len(preds), cache)
    else:
        if args.replot and not cache.exists():
            log.info("no cache at %s; training instead", cache)

        # Pass 1: every condition, no sampled trajectories. This alone decides
        # the best/worst day and supplies the panels and dependence/correlation
        # references, none of which need a path. Quantiles cover the whole test
        # set here (needed to rank and plot an arbitrary day), which is cheap;
        # the array n_paths multiplies is the one that is not.
        preds = {}
        for name, cls, cfg_cls, extra, family, copula_family in specs:
            log.info("  training %s ...", name)
            preds[name] = fit_and_predict(
                name,
                cls,
                _cfg(cfg_cls, d, **kw, **extra),
                d,
                ckpt_root,
                family=family or "normal",
                n_paths=0,
                copula_family=copula_family,
            )

        truth = d["y_te"] * d["sd"] + d["mu"]
        medians_pass1 = {
            name: np.stack([median_of(q, lv, i) for i in range(truth.shape[2])], axis=-1)
            for name, (q, lv, _, _) in preds.items()
        }
        flat_pass1 = truth.reshape(-1, truth.shape[2])
        scale_pass1 = flat_pass1.std(0)
        day_err_pass1 = np.mean(
            [np.abs(m - truth).mean(axis=1) / scale_pass1 for m in medians_pass1.values()], axis=0
        ).mean(axis=1)
        starts_pass1 = pd.to_datetime(d["stamps_te"][:, 0])
        midnight_pass1 = np.flatnonzero((starts_pass1.hour == 0) & (starts_pass1.minute == 0))
        if midnight_pass1.size == 0:
            raise RuntimeError("no midnight-aligned forecast window in the test split")
        good = int(midnight_pass1[day_err_pass1[midnight_pass1].argmin()])
        bad = int(midnight_pass1[day_err_pass1[midnight_pass1].argmax()])
        metric_days = np.linspace(0, truth.shape[0] - 1, min(args.metric_days, truth.shape[0])).astype(int)

        # Every day any downstream table, DM test, conformal check, or cloud
        # panel will ever ask for, and nothing else -- this is the whole fix:
        # n_paths trajectories are sampled for this small set, not the entire
        # test set (~2900 windows on this scenario), which is what exhausted
        # memory the one time this ran at --paths 500 without it.
        path_days = np.union1d(metric_days, [good, bad])

        # Pass 2: refit (cheap; each condition trains in well under a minute)
        # restricted to path_days, so quantiles, correlation, and paths are all
        # sampled only for the days that will actually be scored or plotted.
        preds_paths = {}
        for name, cls, cfg_cls, extra, family, copula_family in specs:
            log.info("  training %s (paths) ...", name)
            preds_paths[name] = fit_and_predict(
                name,
                cls,
                _cfg(cfg_cls, d, **kw, **extra),
                d,
                ckpt_root,
                family=family or "normal",
                n_paths=args.paths,
                rng=path_rng,
                copula_family=copula_family,
                path_days=path_days,
            )

        payload = {"model_names": np.array(list(preds), dtype=object).astype(str), "path_days": path_days}
        for n, (q, lv, _pa, co) in preds.items():
            payload[f"q::{n}"] = q
            payload[f"lv::{n}"] = lv
            payload[f"co::{n}"] = co
        for n, (q2, lv2, pa2, co2) in preds_paths.items():
            payload[f"q2::{n}"] = q2
            payload[f"lv2::{n}"] = lv2
            payload[f"pa2::{n}"] = pa2
            payload[f"co2::{n}"] = co2
        np.savez_compressed(cache, **payload)
        log.info("cached predictions -> %s", cache)

    truth = d["y_te"] * d["sd"] + d["mu"]
    truth_paths = truth[path_days]
    metric_days = np.linspace(0, truth.shape[0] - 1, min(args.metric_days, truth.shape[0])).astype(int)
    metric_days_local = np.searchsorted(path_days, metric_days)

    # Rank days by error pooled over models and targets, each target scaled by its
    # own spread so P in kW does not swamp V in pu. Both days are then the same
    # day in every panel, which is what makes the columns comparable.
    flat = truth.reshape(-1, truth.shape[2])
    scale = flat.std(0)  # for ranking days, where a robust spread is wanted
    # NMPI and NRMSE are normalised by the full-test range of each target, the
    # library's default "range" rule. It is constant across days and panels, so
    # the best and worst day are on the same footing.
    scale_ref = np.ptp(flat, axis=0)
    medians = {
        name: np.stack([median_of(q, lv, i) for i in range(truth.shape[2])], axis=-1)
        for name, (q, lv, _, _) in preds.items()
    }
    day_err = np.mean([np.abs(m - truth).mean(axis=1) / scale for m in medians.values()], axis=0).mean(axis=1)

    # Only windows that begin at 00:00 are eligible. With a stride of one step a
    # window can start at any half hour, and a panel spanning 18:00 to 17:30
    # cannot carry an hour-of-day axis running 00 to 23.
    starts = pd.to_datetime(d["stamps_te"][:, 0])
    midnight = np.flatnonzero((starts.hour == 0) & (starts.minute == 0))
    if midnight.size == 0:
        raise RuntimeError("no midnight-aligned forecast window in the test split")
    good, bad = int(midnight[day_err[midnight].argmin()]), int(midnight[day_err[midnight].argmax()])
    log.info("midnight-aligned candidate days: %d", midnight.size)
    log.info("best day  index %d  normalised MAE %.4f", good, day_err[good])
    log.info("worst day index %d  normalised MAE %.4f", bad, day_err[bad])
    good_local = int(np.searchsorted(path_days, good))

    rows = []
    for label, day in (("best", good), ("worst", bad)):
        t_axis = pd.to_datetime(d["stamps_te"][day])
        pw, ph = panel(0.48)
        # squeeze=False keeps axes two-dimensional when a single condition is
        # selected, so the [row, column] indexing below holds for any variant count.
        fig, axes = plt.subplots(len(targets), len(preds), figsize=(pw * len(preds), ph * len(targets)), squeeze=False)
        for r, t in enumerate(targets):
            # One y range per (day, target), spanning the observed series and
            # every model's quantiles. Letting each panel autoscale would give
            # the four models different vertical scales, and comparing fan
            # widths across a row would then be meaningless.
            y_row = truth[day, :, r]
            all_q = np.stack([q[day, :, :, r] for q, _, _, _ in preds.values()])
            ylim = pad_limits(y_row, all_q)

            for c, (name, (q, lv, _, _)) in enumerate(preds.items()):
                y = truth[day, :, r]
                nrmse, cwe, crps = panel_metrics(y, q[day, :, :, r], lv, r, float(scale_ref[r]))
                rows.append(
                    {
                        "day": label,
                        "target": t,
                        "model": name,
                        "NRMSE": nrmse,
                        "CWE": cwe,
                        "CRPS": crps,
                        "n_levels": q.shape[1],
                    }
                )
                draw(axes[r, c], t_axis, y, q[day, :, :, r], lv, r, ylim=ylim)
                axes[r, c].text(0.02, 0.96, _cap(name), transform=axes[r, c].transAxes, va="top", ha="left", fontsize=7)
                if c == 0:
                    axes[r, c].set_ylabel(AXIS[t])

                one, ax1 = plt.subplots(figsize=panel(0.48))
                draw(ax1, t_axis, y, q[day, :, :, r], lv, r, ylim=ylim)
                ax1.set_ylabel(AXIS[t])
                # tight_layout first, so the legend is anchored against the axes
                # box as finally laid out. Anchoring before it leaves the legend
                # sitting on top of the x label, since the label is only given
                # its own space once the layout has run.
                one.tight_layout()
                place_legend_below(one, ax1, ncol=3, frameon=False)
                stem = f"{label}_{t}_{name.replace(' ', '_').replace('-', '_')}"
                one.savefig(OUT / f"{stem}.pdf", dpi=600, bbox_inches="tight")
                plt.close(one)

        handles, lbls = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f"overview_{label}_day.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    flat_truth = truth.reshape(-1, truth.shape[2])
    # The conditional mean the heads share, estimated by averaging their medians.
    backbone_mean = np.mean(list(medians.values()), axis=0)
    flat_residual = (truth - backbone_mean).reshape(-1, truth.shape[2])
    correlation_panel(
        preds,
        np.corrcoef(flat_truth, rowvar=False),
        np.corrcoef(flat_residual, rowvar=False),
        targets,
    )
    raw_dcor, residual_dcor = dependence_matrix(flat_truth), dependence_matrix(flat_residual)
    # preds_paths, not preds: this panel is measured on each condition's sampled
    # ensemble (dependence_panel skips any condition with an empty paths array),
    # and only preds_paths ever has one -- preds (pass 1) is fit with n_paths=0.
    dependence_panel(preds_paths, raw_dcor, residual_dcor, targets)
    # The same statistic with the shared conditional mean removed, so what is
    # left is the dependence each likelihood contributes on its own.
    dependence_panel(
        preds_paths,
        raw_dcor,
        residual_dcor,
        targets,
        centred=True,
        stem="dependence_matrices_centred",
    )
    # Midday, where the record's own dependence is strongest and a wrong joint
    # law is most visible. preds_paths/truth_paths/good_local: the day-based
    # queries below only ever ask for a day in path_days, so they read from the
    # restricted, path-sampled dict/truth and translate to its local position.
    cloud_panel(preds_paths, truth_paths, targets, day=good_local, step=truth.shape[1] // 2)

    # The rating is taken from the record rather than assumed, so the operational
    # tier screens against a level this feeder actually reaches.
    s_all = np.hypot(truth[..., 0], truth[..., 1])
    rated = float(np.quantile(s_all, 0.98))
    # The full test set's range, not truth_paths's -- NRMSE/NMPI are meant to be
    # constant across days and panels (see metrics_table's scale_ref docstring),
    # and truth_paths only covers the small subset of days paths were sampled for.
    full_scale_ref = np.ptp(truth.reshape(-1, truth.shape[2]), axis=0)
    full = metrics_table(preds_paths, truth_paths, targets, rated, metric_days_local, scale_ref=full_scale_ref)
    full.to_csv(OUT / "metrics_all_tiers.csv", index=False)
    print(f"\n=== all tiers, {len(metric_days)} days, rating {rated:.1f} kVA ===")
    for tier in ("marginal", "joint", "operational"):
        sub = full[full.tier == tier]
        if sub.empty:
            continue
        print(f"\n{tier}")
        print(sub.pivot_table(index=["metric", "target"], columns="model", values="value").round(4).to_string())

    # Chapter-6.tex's own convention (Evaluation Metrics, mirroring Chapter 7):
    # significance is assessed only on Winkler score, via Diebold-Mariano and
    # the Model Confidence Set. CWE, NMPI, and PICP above stay diagnostic and
    # never decide a comparison -- see significance.py's module docstring for
    # why CWE specifically cannot furnish a loss series for DM at all.
    point_losses = pointwise_losses(preds_paths, truth_paths, targets, rated, metric_days_local)
    point_losses.to_csv(OUT / "pointwise_losses.csv", index=False)
    ws_losses = {
        name: point_losses[(point_losses.model == name) & (point_losses.metric == "Winkler")]
        .sort_values("day")["value"]
        .to_numpy()
        for name in preds_paths
    }
    if len(ws_losses) >= 2 and len({len(v) for v in ws_losses.values()}) == 1:
        dm_table = pairwise_dm_table(ws_losses, h=truth_paths.shape[1])
        dm_table.to_csv(OUT / "winkler_dm_table.csv", index=False)
        print("\n=== Winkler-score significance (Diebold-Mariano, BH-corrected) ===")
        print(dm_table.to_string(index=False))

        mcs = mcs_ranking(ws_losses)
        print(f"\nModel Confidence Set (Winkler score, alpha=0.10): {sorted(mcs.included)}")
    else:
        print("\nWinkler-score significance skipped: need >=2 conditions with equal-length loss series.")

    # Chapter-6.tex's multivariate adaptive conformal calibration, applied on
    # top of every condition's own forecast, "max-norm" and "mahalanobis"
    # side by side -- see conformal_rows's docstring and COPULA_METHODS.md
    # section 6 for what the comparison is expected to show.
    conformal = conformal_rows(preds_paths, truth_paths, metric_days_local, alpha=0.1)
    if not conformal.empty:
        conformal.to_csv(OUT / "conformal_calibration.csv", index=False)
        print("\n=== multivariate conformal calibration, nominal coverage 90% ===")
        print(conformal.to_string(index=False))

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "forecast_examples.csv", index=False)
    for label in ("best", "worst"):
        print(f"\n=== {label} day ===")
        sub = table[table.day == label]
        for m in ("NRMSE", "CWE", "CRPS"):
            print(f"\n{m}")
            print(sub.pivot_table(index="target", columns="model", values=m).round(4).to_string())
    log.info("panels  %s", OUT)


if __name__ == "__main__":
    main()
