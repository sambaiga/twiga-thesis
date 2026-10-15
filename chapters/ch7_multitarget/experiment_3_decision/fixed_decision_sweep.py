"""Operational decisions across the DSS-UK and DSS-US penetration grid, with Experiment 1's heads.

Each scenario is one engine run with the four Experiment 1 heads (anchored
FPQR, QR, BNN, Laplace), trained exactly as in the fixed benchmark. Their
calibrated variants reuse the trained weights: max-norm for every head and the
six CQR variants (scaled/unscaled x marginal/pooled/Bonferroni) for the
quantile heads. ``--fpqr tuned`` swaps in FPQR's head-tuned settings for the
record (tuned at EV 0.4, PV 0.4 on the weak DSS-UK feeder and on DSS-US; the
stiff feeder uses the weak feeder's); the chapter uses the fixed FPQR. Every condition is scored by
:mod:`decision_scoring` through what the head exposes: quantile and Laplace
heads as event probabilities (marginal CDFs, plus Schaake-shuffled scenarios
for the joint events), BNN through its own weight-draw paths, and the max-norm
region as set-valued warnings at the same cost-loss ratios.

Event limits are fixed per network, not per scenario, so event rates move with
penetration, which is what the sweep measures:

* Thermal: the 98th percentile of apparent power in the network's no-DER
  scenario (EV 0, PV 0), i.e. its pre-DER peak headroom; exhausting it is a
  reinforcement trigger rather than a nameplate overload (DSS-US's rank-8
  transformer peaks at 65% of its 55 kVA nameplate even at EV 0.6, and DSS-UK
  records no nameplate). The day event is a sustained
  overload, at least two hours above the limit, in line with duration-based
  transformer loading guides: counting any single half-hour, it occurs on
  97-100% of DSS-UK days once EV reaches 0.2.
* Voltage (DSS-UK only): statutory limits apply at customer terminals, so the
  event is the feeder's lowest customer voltage outside 0.94-1.10 pu (ESQCR
  230 V +10%/-6%), reached through a busbar-to-customer drop model fitted on
  each fold's preceding days. The busbar itself never leaves the band, while
  customer under-voltage on the weak feeder rises with EV (0.4% of steps at
  EV 0.2, 8.8% at EV 0.6). DSS-US never breaches ANSI C84.1 Range A even at its
  lowest LV voltage, so it is scored on thermal events only.

Protocol (every scenario): monthly folds, a fixed four-month training window
plus one month of validation, one month of calibration and one month of test,
advanced by one month, as many folds as the record holds.

Networks: DSS-UK's weak and stiff feeders, and DSS-US rank 8 (the most loaded of
the seven SMART-DS transformers whose load grows steadily with both EV and PV).
SMART-DS assigns DER to a fresh random set of customers at each penetration level,
so most single transformers, including the most loaded (rank 0), do not see their
DER grow from one level to the next: rank 0 exports most at PV 0.2 and least at
PV 0.6, and its EV 0.2 record equals its no-EV record. Such a transformer is not a
penetration sweep; :func:`check_penetration_response` guards against one.

Stages: ``axes`` runs EV-only and PV-only levels on DSS-UK weak and stiff and
DSS-US rank 8; ``grid`` runs the remaining EV x PV points; ``all`` runs both.

Usage::

    uv run python experiment/ch7_multitarget/experiment_3_decision/fixed_decision_sweep.py --stage axes
    uv run python experiment/ch7_multitarget/experiment_3_decision/fixed_decision_sweep.py --quick \
        --records DSS-UK:weak --ev 0.4 --pv 0.4
    uv run python experiment/ch7_multitarget/experiment_3_decision/fixed_decision_sweep.py --stage axes \
        --fpqr tuned --tuned-hpo-root <experiments>/ch7_multitarget/multitarget_fixed_benchmark_v1/backbone_hpo
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parent / "experiment_1_benchmark")]

from twiga.core.utils import configure, get_logger  # noqa: E402
from twiga.distributions.ml.utils import get_sigma_from_quantile_levels  # noqa: E402
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec  # noqa: E402
from twiga.utils.data_loading import load_dataset  # noqa: E402

import chapters.ch7_multitarget.experiment_1_benchmark.fixed_benchmark as fb  # noqa: E402
import chapters.ch7_multitarget.experiment_3_decision.decision_scoring as ds  # noqa: E402
from twiga_thesis.paths import thesis_data_dir

log = get_logger(__name__)

RECORDS = ("DSS-UK:weak", "DSS-UK:stiff", "DSS-US:8")
LEVELS = (0.0, 0.2, 0.4, 0.6)
LIMIT_PERCENTILE = 0.98
# Overload tiers above the pre-DER headroom. 1.5x keeps one tier informative at high EV, where the
# 1.0x and 1.25x sustained events occur on nearly every day.
SEVERITIES = (1.0, 1.25, 1.5)
STATUTORY_BAND = (0.94, 1.10)
CUSTOMER_COLUMNS = ("p_kw", "q_kvar", "v_lv_pu", "v_cust_min_pu")
TARGETS = ("p_kw", "q_kvar", "v_lv_pu")
HEAD_KIND = {
    "fpqr_anchor": ds.HeadKind.QUANTILE,
    "qr": ds.HeadKind.QUANTILE,
    "bnn": ds.HeadKind.BNN,
    "laplace": ds.HeadKind.LAPLACE,
}
# Each record's head-tuned FPQR settings come from Experiment 1's tuning scenario.
TUNED_ON = {"DSS-UK:weak": fb.DSS_UK, "DSS-UK:stiff": fb.DSS_UK, "DSS-US:8": fb.DSS_US}
N_PATHS = 200
HISTORY_DAYS = 120
SEED = 20260906
DEFAULT_OUTPUT_ROOT = fb.DEFAULT_OUTPUT_ROOT / "fixed_decision_sweep"

PROTOCOL = {
    "cv_train_size": 5,
    "cv_train_size_mode": "fixed",
    "cv_val_size": 1,
    "cv_calib_size": 1,
    "cv_test_size": 1,
    "cv_folds": 10,
    "cv_stride": 1,
    "cv_window": "rolling",
}


def scenario_keys(stage: str, records: tuple[str, ...], ev: tuple[float, ...], pv: tuple[float, ...]) -> list[str]:
    """Scenario keys for a stage.

    Args:
        stage: ``"axes"`` (EV-only and PV-only levels), ``"grid"`` (the rest of
            the EV x PV grid) or ``"all"``.
        records: ``"<record>:<feeder or rank>"`` prefixes.
        ev: EV penetration levels.
        pv: PV penetration levels.

    Returns:
        Keys ``"<record>:<feeder>:<ev>:<pv>"``, in a stable order.
    """
    points = [(e, p) for e in ev for p in pv]
    on_axes = [(e, p) for e, p in points if e == 0.0 or p == 0.0]
    chosen = {"axes": on_axes, "grid": [x for x in points if x not in on_axes], "all": points}[stage]
    return [f"{record}:{e:g}:{p:g}" for record in records for e, p in chosen]


def check_penetration_response(record: str, levels: tuple[float, ...] = LEVELS) -> bool:
    """Whether a network's load responds steadily to penetration: up with EV, down with PV, mean and tail alike.

    Reads each level's raw record (no feature selection) and requires, at every step of ``levels``, the mean
    net load and the 98th-percentile apparent power to rise with EV, and the mean net load to fall with PV.
    Logs a warning naming the failing axis: the sweep's EV and PV trends mean nothing on a network where added
    penetration does not add DER.

    Args:
        record: Network prefix, e.g. ``"DSS-US:8"``.
        levels: Penetration levels of the sweep.

    Returns:
        ``True`` when both axes respond steadily.
    """
    by_ev = [scenario_record(f"{record}:{level:g}:0") for level in levels]
    by_pv = [scenario_record(f"{record}:0:{level:g}") for level in levels]
    mean_ev = [frame["p_kw"].mean() for frame in by_ev]
    tail_ev = [np.quantile(np.hypot(frame["p_kw"], frame["q_kvar"]), LIMIT_PERCENTILE) for frame in by_ev]
    mean_pv = [frame["p_kw"].mean() for frame in by_pv]
    steady_ev = bool(np.all(np.diff(mean_ev) > 0) and np.all(np.diff(tail_ev) > 0))
    steady_pv = bool(np.all(np.diff(mean_pv) < 0))
    if not (steady_ev and steady_pv):
        log.warning(
            "%s does not respond steadily to penetration: mean P by EV %s, apparent-power P98 by EV %s, "
            "mean P by PV %s",
            record,
            np.round(mean_ev, 3).tolist(),
            np.round(tail_ev, 3).tolist(),
            np.round(mean_pv, 3).tolist(),
        )
    return steady_ev and steady_pv


def decision_limits(record: str) -> ds.DecisionLimits:
    """Fixed thermal limit for a network: the 98th percentile of its no-DER apparent power (pre-DER headroom)."""
    data, *_ = load_dataset(f"{record}:0:0", data_path=str(thesis_data_dir()))
    rating = float(np.quantile(np.hypot(data["p_kw"], data["q_kvar"]), LIMIT_PERCENTILE))
    log.info("limits for %s: thermal %.3f (no-DER 98th percentile)", record, rating)
    return ds.DecisionLimits(rating=rating, target_index=(0, 1, 2), severities=SEVERITIES, voltage_band=STATUTORY_BAND)


def scenario_record(key: str) -> pd.DataFrame:
    """The scenario's raw record with every decision column, including customer voltage where recorded.

    The loader keeps only model inputs; the customer voltage defines an event and is
    never a model input, so it is read from the record directly.
    """
    record, feeder, ev, pv = key.split(":")
    raw = pd.read_parquet(thesis_data_dir() / f"{record}.parquet")
    if record == "DSS-US":
        trafo = raw.groupby("trafo")["utilisation"].max().sort_values(ascending=False).index[int(feeder)]
        raw = raw[raw["trafo"] == trafo]
    else:
        raw = raw[raw["feeder"] == feeder]
    raw = raw[np.isclose(raw["ev_pen"], float(ev)) & np.isclose(raw["pv_pen"], float(pv))]
    extra = [name for name in ("v_cust_min_pu", "kva_rating") if name in raw.columns]
    columns = ["timestamp", *TARGETS, *extra]
    return raw[columns].sort_values("timestamp").reset_index(drop=True)


def _past(record: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Rows of the last ``HISTORY_DAYS`` whole days before ``origin``, aligned to the forecast's start hour."""
    stamps = pd.to_datetime(record["timestamp"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize("UTC")
    past = record.loc[(stamps < origin).to_numpy()]
    days = min(HISTORY_DAYS, len(past) // horizon)
    return past.iloc[len(past) - days * horizon :]


def _recorded_customer_voltage(record: pd.DataFrame, stamps: np.ndarray) -> np.ndarray:
    """The recorded lowest customer voltage at each forecast step, ``(N, H)``: the voltage events' ground truth."""
    series = pd.Series(
        record["v_cust_min_pu"].to_numpy(dtype=float), index=pd.to_datetime(record["timestamp"], utc=True)
    )
    flat = pd.to_datetime(np.asarray(stamps).reshape(-1), unit="ns", utc=True)
    return series.reindex(flat).to_numpy(dtype=float).reshape(np.shape(stamps))


def _history(past: pd.DataFrame, horizon: int) -> np.ndarray:
    """The past rows as days of targets, oldest first: ``(K, H, D)``."""
    values = past[list(TARGETS)].to_numpy(dtype=float)
    return values.reshape(-1, horizon, len(TARGETS))


def _predictive(forecaster, raw, prepared: pd.DataFrame, kind: ds.HeadKind, method: str) -> ds.Predictive:
    """The fold's predictive object in original units, as the head exposes it."""
    loc = np.asarray(raw.loc, dtype=float)
    if kind is ds.HeadKind.QUANTILE:
        quantiles, levels = fb._per_cell_levels(raw)
        if method in fb.QUANTILE_CONFORMAL:
            conformal = forecaster.conformal[forecaster.model_type]
            lower, upper = conformal.generate_intervals(quantiles[:, 0], quantiles[:, -1])
            quantiles = quantiles.copy()
            quantiles[:, 0], quantiles[:, -1] = np.asarray(lower), np.asarray(upper)
        return ds.Predictive(kind=kind, loc=loc, quantiles=quantiles, levels=levels)
    if kind is ds.HeadKind.LAPLACE:
        return ds.Predictive(kind=kind, loc=loc, scale=np.asarray(raw.scale, dtype=float))
    scale = np.asarray(raw.scale, dtype=float) if getattr(raw, "scale", None) is not None else None
    features = torch.as_tensor(forecaster.data_pipeline.transform_features(prepared), dtype=torch.float32)
    out = forecaster.model.forecast(features)
    samples = np.asarray(out["samples"], dtype=float)
    n_draws, n_windows, horizon, n_targets = samples.shape
    flat = forecaster._inverse_scale_predictions(samples.reshape(n_draws * n_windows, horizon, n_targets))
    noise = forecaster._inverse_scale_predictions(np.asarray(out["scale_noise"], dtype=float), is_scale=True)
    return ds.Predictive(
        kind=kind,
        loc=loc,
        scale=scale,
        paths=flat.reshape(n_draws, n_windows, horizon, n_targets),
        noise_scale=np.broadcast_to(noise, loc.shape),
    )


def _daily_brier(
    origins: pd.DatetimeIndex, probabilities: dict[str, np.ndarray], events: dict[str, np.ndarray], form: str
) -> pd.DataFrame:
    """Brier loss per forecast day for every event: averaged over the day's steps for single-step events.

    Args:
        origins: Forecast origin of each window, ``(N,)``.
        probabilities: Event probabilities, ``(N, H)`` for step events and ``(N,)`` for day events.
        events: Observed outcomes with the same shapes.
        form: Scenario form the probabilities come from.

    Returns:
        Long frame with ``forecast_origin``, ``dependence``, ``event``, ``probability``, ``outcome`` and ``brier``.
    """
    frames = []
    for name, probability in probabilities.items():
        outcome = events[name].astype(float)
        loss = (probability - outcome) ** 2
        if loss.ndim == 2:
            loss, probability, outcome = loss.mean(axis=1), probability.mean(axis=1), outcome.mean(axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "forecast_origin": origins,
                    "dependence": form,
                    "event": name,
                    "probability": probability,
                    "outcome": outcome,
                    "brier": loss,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_decision_scorer(head: str, method: str, limits: ds.DecisionLimits, record: pd.DataFrame):
    """Per-fold scorer for one condition: the head's decisions scored in the form it produces.

    Args:
        head: Experiment 1 head name.
        method: ``"none"``, ``"max_norm"`` or a CQR variant of :data:`fixed_benchmark.QUANTILE_CONFORMAL`.
        limits: The network's fixed event limits.
        record: The scenario's raw record (:func:`scenario_record`), for the days
            before each test window and the customer-voltage drop model.

    Returns:
        A ``custom_scorer`` for :class:`~twiga.experiment.Condition`.
    """
    kind = HEAD_KIND[head]

    def scorer(forecaster, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)  # the BNN's weight draws: reproducible scores, and the same draws on a re-score
        prepared = forecaster.prepare_test_data(test_df)
        stamps, observed = forecaster._extract_ground_truth(prepared)
        observed = np.asarray(observed, dtype=float)
        prediction, _ = forecaster._predict_core(test_df=prepared, prepare_test_data=False)
        raw = next(iter(prediction.values()))
        origins = pd.to_datetime(np.asarray(stamps)[:, 0, 0], unit="ns", utc=True)
        horizon = observed.shape[1]
        past = _past(record, origins[0], horizon)
        fold_limits, v_past, v_test = limits, None, None
        if "v_cust_min_pu" in record.columns:
            fold_limits = dc_replace(limits, voltage_drop=ds.fit_voltage_drop(past, CUSTOMER_COLUMNS))
            v_past = past["v_cust_min_pu"].to_numpy(dtype=float).reshape(-1, horizon)
            v_test = _recorded_customer_voltage(record, np.asarray(stamps)[..., 0] if np.ndim(stamps) == 3 else stamps)
        history = _history(past, horizon)
        rates = ds.reference_rates(history, fold_limits, v_past)
        events = ds.events_from_values(observed, fold_limits, v_test)
        rows, per_day = [], []
        if method == "max_norm":
            pred = _predictive(forecaster, raw, prepared, kind, "none")
            sigma = (
                get_sigma_from_quantile_levels(np.asarray(raw.quantiles), np.asarray(raw.quantile_levels))
                if kind is ds.HeadKind.QUANTILE
                else np.asarray(raw.scale, dtype=float)
            )
            conformal = forecaster.conformal[forecaster.model_type]
            warnings, plausibility = ds.conformal_decisions(pred.loc, sigma, conformal, fold_limits)
            rows.append(ds.score_warnings(warnings, plausibility, events, rates).assign(dependence="region"))
            curve = pd.DataFrame()
        else:
            # Every way the head's forecast becomes joint scenarios (FORECAST_TYPE_PLAN.md): independent and
            # Schaake-coupled marginal draws for every head, and the BNN's own paths with per-cell or
            # day-coherent noise. All forms use N_PATHS paths, so their (fair) Brier scores compare.
            pred = _predictive(forecaster, raw, prepared, kind, method)
            template = ds.history_template(history, N_PATHS, rng)
            curves = []
            for form, paths in ds.scenario_forms(pred, N_PATHS, template, rng).items():
                probabilities = ds.probabilities_from_paths(paths, fold_limits, rng)
                scored, form_curve = ds.score_probabilities(probabilities, events, rates, N_PATHS)
                rows.append(scored.assign(dependence=form))
                curves.append(form_curve.assign(dependence=form))
                per_day.append(_daily_brier(origins, probabilities, events, form))
            curve = pd.concat(curves, ignore_index=True)
        metric_df = pd.concat(rows, ignore_index=True)
        rate_rows = [
            {"tier": "validity", "event": name, "metric": metric, "value": value}
            for name in events
            for metric, value in (("reference_rate", rates[name]), ("realised_rate", float(events[name].mean())))
        ]
        metric_df = pd.concat([metric_df, pd.DataFrame(rate_rows)], ignore_index=True)
        if not curve.empty:
            metric_df = pd.concat([metric_df, curve.assign(tier="value_curve", metric="relative_value")])
        # The day event per origin: the union with voltage where voltage is scored (DSS-UK), else the
        # base-severity sustained overload (DSS-US, scored on thermal events only).
        day_event = events.get("union_day", events[ds.overload_name(min(fold_limits.severities), "day")])
        pred_df = pd.DataFrame({"forecast_origin": origins, "realised_day_event": day_event.astype(float)})
        if per_day:
            # One Brier loss per forecast day, form and event: the series the DM test compares.
            pred_df = pd.concat(per_day, ignore_index=True).merge(pred_df, on="forecast_origin", how="left")
        return pred_df, metric_df.assign(condition=f"{head}_{method}")

    return scorer


def build_conditions(
    limits: ds.DecisionLimits,
    record: pd.DataFrame,
    fpqr_overrides: dict | None = None,
    with_conformal: bool = False,
) -> list[Condition]:
    """The four heads, scored through every scenario form they support; optionally their conformal variants.

    Args:
        limits: The network's fixed event limits.
        record: The scenario's raw record (:func:`scenario_record`).
        fpqr_overrides: FPQR's settings in place of its fixed ones (the head-tuned settings).
        with_conformal: Also run max-norm and the CQR variants on the same weights (dropped from the
            chapter, FORECAST_TYPE_PLAN.md).

    Returns:
        One condition per head and calibration.
    """
    conditions = []
    for head, kind in HEAD_KIND.items():
        spec = fb.HEADS[head]
        overrides = fpqr_overrides if head == "fpqr_anchor" and fpqr_overrides else spec["overrides"]
        common = {"group": "decision", "model_cls": spec["model_cls"], "overrides": dict(overrides)}
        conditions.append(
            Condition(
                name=f"{head}_none",
                description=f"{head}, native predictive",
                custom_scorer=make_decision_scorer(head, "none", limits, record),
                reserve_calib_window=True,
                # As Experiment 1's raw_export: keep the weights and save the raw forecasts, so the
                # forecast-type analyses (forecast_type_study.py) run on every scenario offline.
                keep_checkpoints=True,
                save_raw_predictions=True,
                **common,
            )
        )
        variants = dict(fb.MULTIVARIATE_CONFORMAL) if with_conformal else {}
        if with_conformal and kind is ds.HeadKind.QUANTILE:
            variants.update(fb.QUANTILE_CONFORMAL)
        conditions.extend(
            Condition(
                name=f"{head}_{method}",
                description=f"{head} with conformal={method}",
                custom_scorer=make_decision_scorer(head, method, limits, record),
                conformal_config=config,
                reuse_checkpoint_from=f"{head}_none",
                **common,
            )
            for method, config in variants.items()
        )
    return conditions


def make_spec(
    key: str,
    limits: ds.DecisionLimits,
    input_window_size: int,
    fpqr_overrides: dict | None = None,
    with_conformal: bool = False,
) -> ExperimentSpec:
    """One scenario's spec: Experiment 1's heads and training, with a calibration window."""
    dataset = {**fb._DATASET_COMMON, **fb._DSS_2019, "dataset_name": key, "input_window_size": input_window_size}
    return ExperimentSpec(
        **{
            **fb._SPEC_COMMON,
            "name": "Multitarget Fixed Decision Sweep",
            "output_prefix": "ch6_fixed_decision",
            "reference_conditions": {"decision": "fpqr_anchor_none"},
        },
        **PROTOCOL,
        conditions=build_conditions(limits, scenario_record(key), fpqr_overrides, with_conformal),
        datasets={key: dataset},
    )


def checkpoint_sources(sweep_root: Path, key: str) -> dict[str, Path]:
    """Each head's kept per-fold checkpoints for scenario ``key``, from the newest run under ``sweep_root`` holding it.

    Used by ``--rescore-from``: the heads reload these weights instead of training, so an earlier sweep can be
    scored again with the current scorer (which adds the per-day Brier losses the DM tests need).

    Args:
        sweep_root: An earlier sweep's experiment root (holds ``runs/``).
        key: Scenario key ``"<record>:<feeder>:<ev>:<pv>"``.

    Returns:
        ``{head: checkpoint directory}`` for every head whose checkpoints exist; empty if no run holds the scenario.
    """
    from chapters.ch7_multitarget.experiment_3_decision.decision_forecast_type_report import (  # noqa: PLC0415
        scenario_key,
    )

    for run in sorted((sweep_root / "runs").glob("*"), reverse=True):
        if scenario_key(run) != key:
            continue
        sources = {}
        for head in HEAD_KIND:
            model_type = fb.HEADS[head]["model_cls"].__name__.lower().removesuffix("config")
            for scenario_dir in (run / "conditions").glob("*"):
                path = scenario_dir / "decision" / f"{head}_none" / "checkpoints" / f"{head}_none" / model_type
                # Only a head that finished in the source run (its metrics exist) has a checkpoint for every fold.
                finished = (scenario_dir / "decision" / f"{head}_none" / "metrics.parquet").exists()
                if finished and path.is_dir() and any(path.glob("fold_*")):
                    sources[head] = path
        if sources:
            return sources
    return {}


def main() -> None:
    """Run the selected stage of the penetration sweep, one engine run per scenario."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["axes", "grid", "all"], default="axes")
    ap.add_argument("--records", nargs="*", default=list(RECORDS), choices=list(RECORDS))
    ap.add_argument("--ev", nargs="*", type=float, default=list(LEVELS))
    ap.add_argument("--pv", nargs="*", type=float, default=list(LEVELS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--quick", action="store_true", help="2 folds and 2 epochs: a smoke test")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument(
        "--with-conformal", action="store_true", help="also run the conformal variants (not used in the chapter)"
    )
    ap.add_argument(
        "--rescore-from",
        type=Path,
        default=None,
        help=(
            "an earlier sweep's experiment root (holds runs/): reload each head's kept checkpoints for the same "
            "scenario instead of training, and score again with the current scorer"
        ),
    )
    ap.add_argument(
        "--fpqr", choices=["fixed", "tuned"], default="fixed", help="FPQR at its fixed (default) or head-tuned settings"
    )
    ap.add_argument(
        "--tuned-hpo-root",
        type=Path,
        default=fb.DEFAULT_OUTPUT_ROOT / "fixed_benchmark" / fb.EXPERIMENT_SLUG / "backbone_hpo",
        help="backbone_hpo folder of Experiment 1's head-tuned run",
    )
    args = ap.parse_args()

    configure(level="INFO")
    cfg = fb.base_config(2 if args.quick else args.epochs, args.output_root)
    keys = scenario_keys(args.stage, tuple(args.records), tuple(args.ev), tuple(args.pv))
    limits = {record: decision_limits(record) for record in {key.rsplit(":", 2)[0] for key in keys}}
    for record in limits:
        check_penetration_response(record)
    log.info("%d scenarios: %s", len(keys), ", ".join(keys))
    tuned = {
        record: fb.tuned_overrides("fpqr_anchor", TUNED_ON[record], args.tuned_hpo_root)
        for record in limits
        if args.fpqr == "tuned"
    }
    for key in keys:
        record = key.rsplit(":", 2)[0]
        spec = make_spec(key, limits[record], args.lookback, tuned.get(record), args.with_conformal)
        if args.quick:
            spec = dc_replace(spec, cv_folds=2)
        if args.rescore_from is not None:
            sources = checkpoint_sources(args.rescore_from, key)
            for condition in spec.conditions:
                head = condition.name.removesuffix("_none")
                if condition.name.endswith("_none") and head in sources:
                    condition.reuse_checkpoint_dir = str(sources[head])
            missing = [h for h in HEAD_KIND if h not in sources]
            if missing:
                log.warning("%s: no kept checkpoints for %s; these heads train", key, ", ".join(missing))
        engine = ExperimentEngine(spec)
        try:
            engine.run(base_cfg=cfg, groups=["decision"], dataset_keys=[key], tracking_uri=None)
        except RuntimeError as error:  # every condition failed; the retry below handles it
            log.warning("%s: %s", key, error)
        if args.rescore_from is not None and getattr(engine, "last_run_root", None) is not None:
            retrain_failed(spec, cfg, key, Path(engine.last_run_root))


def retrain_failed(spec: ExperimentSpec, cfg, key: str, run_root: Path) -> None:
    """Train, in a run of their own, the heads whose reused checkpoints failed to load for scenario ``key``.

    A checkpoint fails to load when the current pipeline builds a different input layout from the source run's
    (a near-tie in the rank-fusion feature selection resolved the other way on a different machine). Those heads
    are trained afresh, so the scenario still has every head, each consistent with its own features.
    """
    status_file = run_root / "run_status.json"
    if not status_file.exists():
        return
    status = json.loads(status_file.read_text())
    failed = {c["condition"] for c in status["conditions"] if c.get("status") == "failed"}
    if not failed:
        return
    log.warning("%s: training %s afresh (reused checkpoints did not load)", key, ", ".join(sorted(failed)))
    for condition in spec.conditions:
        condition.reuse_checkpoint_dir = None
    retry = dc_replace(spec, conditions=[c for c in spec.conditions if c.name in failed])
    ExperimentEngine(retry).run(base_cfg=cfg, groups=["decision"], dataset_keys=[key], tracking_uri=None)


if __name__ == "__main__":
    main()
