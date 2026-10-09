"""Chapter 6, Experiment 3: operational decision-making assessment.

``Chapter-6.tex`` (``subsec:ch6-protocol``) names this directly: "the analysis
examines how explicitly modelling dependencies between active power, reactive
power, and voltage influences estimates of network-operational risk against
two distinct, dataset-specific decision criteria: voltage compliance and
hosting capacity under EN 50160 on the weak European substation of DSS-UK,
where voltage is the binding constraint, and transformer thermal loading
under ANSI C84.1 on the North American pole-transformer record of DSS-US,
where the thermal rating binds instead. In both cases, validation consists
of comparing the calibrated multivariate prediction region's
constraint-exceedance probability against the frequency with which the
constraint is actually violated in the held-out data."

Two decisions, two datasets, one comparison repeated on each:

  DSS-UK (weak feeder)  hosting-capacity union event, eqn:ch6-hosting-capacity
                         in Chapter-6.tex: Pr(union_t {S_t > S_rated} u
                         {V_t not in EN-50160 band}), across the forecast
                         horizon -- not per timestep, since a breach anywhere
                         in the horizon is what the criterion asks about.
  DSS-US                thermal-only exceedance, Pr(union_t {S_t > S_rated}),
                         no voltage term: Chapter 6's own §6.3 finding is that
                         voltage never binds on this record.

Both use :func:`forecast_examples.hosting_capacity_rows`, which evaluates the
union rather than the marginal per-timestep events
:func:`forecast_examples.metrics_table`'s operational tier reports.

The comparison
---------------
Two parametric baselines -- joint Gaussian (``MLPFMVNormal``, full covariance
across P, Q and V) against independent Gaussian (``MLPFNormal``, no
cross-target term), the same "head" pair ``multitarget_experiment.py``
compares for Experiment 1 -- plus every FPQR/copula condition
``copula_variants.py`` fits: shared/per-target grid (independent margins) and
each coupled through a Gaussian, Clayton, or three-target D-vine copula
(:func:`forecast_examples.fit_and_predict`'s ``copula_family``; see
``COPULA_METHODS.md``). The parametric pair isolates whether dependence was
modelled at all, holding the backbone and margin family fixed; the FPQR
family isolates which copula, if any, couples an otherwise-identical margin.
Both parametric heads sample directly from their own fitted covariance, so no
post-hoc copula step applies to them; the FPQR conditions draw trajectories
through whichever copula (if any) their name states, exactly as
``copula_variants.py`` does.

Protocol
--------
**Changed 2026-09-17, no longer ``Chapter-6.tex``'s literal figures.** DSS-UK
and DSS-US are one-year synthetic power-flow records (365 days at 30-minute
resolution, confirmed against the files), unlike the multi-year measured
MLVS-PT-3PHASE record these figures were originally set for. The chapter's
literal protocol (expanding window, training set of at least twelve months
in the first fold, four-month test window, origin advancing one month
between folds) gave only 2-3 folds here -- confirmed directly, same problem
``penetration_sweep.py``/``copula_variants.py`` hit on the same class of
one-year OpenDSS records. Now matches both of those scripts' fix exactly:
rolling (not expanding), ``split_freq="weeks"``, train/test/step = 17/4/4
weeks (~4/~1/~1 months). Verified: 10 folds on DSS-UK/DSS-US.
:func:`forecast_examples.build_backtest_folds` (backed by
:class:`~twiga.core.backtester.TimeBasedCV`, the same class
:class:`~twiga.experiment.ExperimentEngine` uses) now takes a ``split_freq``
argument for exactly this reason -- a week is a fixed 7 days, so ``--step``
equal to ``--test-size`` gives an *exact* non-overlap guarantee between
consecutive folds' test periods, where a calendar month (28-31 days) only
gives an approximate one. Pass ``--split-freq months --train-size 12
--test-size 4 --step 1 --expanding`` to reproduce the original, literal
(but 2-3-fold) chapter protocol.

**Gap, not yet closed**: unlike ``copula_variants.py``, the multivariate
adaptive conformal calibration step is not applied to the prediction region
evaluated here; every condition reports its native (uncalibrated) predictive
distribution, and the hosting-capacity/thermal union event is scored from
sampled trajectories, not from a calibrated interval, so there is no
box/ellipsoid to calibrate against in the first place -- the event this
experiment scores is a probability of union over the horizon, which
``MultivariateConformal`` does not produce. ``multitarget_experiment.py``'s
``conformal`` group demonstrates the calibration mechanism itself is wired
(``ConformalConfig(method="multivariate", ...)``); folding calibration into
this experiment's decision criterion would need a differently-shaped
evaluation and is a follow-on, not built here.

    uv run python experiment/ch7_multitarget/operational_decision.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from twiga.core.metrics import pairwise_dm_table  # noqa: E402
from twiga.core.utils import get_logger
from twiga.experiment import start_run

from chapters.ch7_multitarget.shared.forecast_examples import (  # noqa: E402
    _cfg,
    backbone_settings,
    build_backtest_folds,
    fit_and_predict,
    hosting_capacity_rows,
)
from chapters.ch7_multitarget.shared.tune_backbone import tune as tune_backbone  # noqa: E402

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
# Outside the repo, matching multitarget_experiment.py/fpqr_gating_ablation.py/
# copula_variants.py's output_dir convention -- see copula_variants.py's OUT
# comment for why.
OUT = script_dir.parent.parent.parent.parent / "experiments" / "ch7_multitarget" / "operational_decision"

VARIANTS = (
    "joint",
    "independent",
    "fpqr shared grid",
    "fpqr per-target grid",
    "fpqr shared + copula",
    "fpqr per-target + copula",
    "fpqr shared + clayton copula",
    "fpqr per-target + clayton copula",
    "fpqr shared + vine copula",
    "fpqr per-target + vine copula",
    "mlpgam fpqr shared grid",
    "mlpgam fpqr per-target grid",
    "nhits fpqr shared grid",
    "nhits fpqr per-target grid",
)

# FPQR settings shared by every FPQR condition, matching copula_variants.py.
FPQR_SETTINGS = {
    "n_quantiles": 21,
    "entropy_weight": 100.0,
    "num_cosines": 0,
    "rope_type": "tau_pos",
    # proposal_type/qvn_type: fpqr_gating_ablation.py's own settled winners
    # (proposal_type="softmax": "horizon worst"; qvn_type="standard": "best
    # MAE"), the same fix backbone_conformal_study.py's BACKBONES dict
    # applies -- this dict previously left qvn_type at "monotone" (not the
    # ablation's own winner) and proposal_type unset (left to each config
    # class's own default/HPO), exposing every "fpqr"-named condition here to
    # the same "HPO/default can pick a documented loser" risk that caused the
    # NHITS/DSS-UK interval collapse in backbone_conformal_study.py before
    # that script's own fix landed (2026-09-26).
    "qvn_type": "standard",
    "proposal_type": "softmax",
}

# EN 50160 (LV supply): 230 V +10% / -6%, i.e. 0.94-1.10 pu. ANSI C84.1
# Range A: 0.95-1.05 pu. DSS-US's decision is thermal-only (include_voltage
# is False for it below), so its voltage band is unused but kept for symmetry.
VOLTAGE_BANDS = {"DSS-UK": (0.94, 1.10), "DSS-US": (0.95, 1.05)}

# DSS-UK: weak feeder, mid penetration -- Chapter-6.tex's own decision
# criterion (voltage-binding hosting capacity) is defined on this feeder.
# DSS-US: rank 0, the most-loaded transformer, mid penetration -- the asset
# Chapter-6.tex's thermal-loading decision is defined against.
SCENARIOS = {
    "DSS-UK": "DSS-UK:weak:0.4:0.4",
    "DSS-US": "DSS-US:0:0.4:0.4",
}
INCLUDE_VOLTAGE = {"DSS-UK": True, "DSS-US": False}


def variant_registry():
    """Model class, config class, family, extra config and copula family per name.

    The two parametric baselines carry no ``extra`` config and no copula
    family (``None``, ``None``); the MLPF FPQR conditions mirror
    ``copula_variants.py``'s registry exactly, so the hosting-capacity/
    thermal criterion is computed for the same eight FPQR conditions that
    script's marginal/joint metrics and conformal calibration already cover.

    MLPGAM/NHiTS FPQR variants (grid only, no copula) are added alongside the
    MLPF ones so this script's own backbone coverage matches
    ``backbone_conformal_study.py``/``backbone_conformal_decision.py``'s
    three-backbone scope -- the copula-coupled variants stay MLPF-only since
    copula fitting has not been exercised against these other backbones in
    this script and doing so untested would be a bigger, separate change.
    """
    from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig, MLPFFPQRModel
    from twiga.models.nn.mlpfmvnormal_model import MLPFMVNormalConfig, MLPFMVNormalModel
    from twiga.models.nn.mlpfnormal_model import MLPFNormalConfig, MLPFNormalModel
    from twiga.models.nn.mlpgamfpqr_model import MLPGAMFPQRConfig, MLPGAMFPQRModel
    from twiga.models.nn.nhitsfpqr_model import NHITSFPQRConfig, NHITSFPQRModel

    return {
        "joint": (MLPFMVNormalModel, MLPFMVNormalConfig, "normal", {}, None),
        "independent": (MLPFNormalModel, MLPFNormalConfig, "normal", {}, None),
        "fpqr shared grid": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, None),
        "fpqr per-target grid": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, None),
        "fpqr shared + copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "gaussian"),
        "fpqr per-target + copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "gaussian"),
        "fpqr shared + clayton copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "clayton"),
        "fpqr per-target + clayton copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "clayton"),
        "fpqr shared + vine copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "vine"),
        "fpqr per-target + vine copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "vine"),
        "mlpgam fpqr shared grid": (MLPGAMFPQRModel, MLPGAMFPQRConfig, None, {"per_target_tau": False}, None),
        "mlpgam fpqr per-target grid": (MLPGAMFPQRModel, MLPGAMFPQRConfig, None, {"per_target_tau": True}, None),
        "nhits fpqr shared grid": (NHITSFPQRModel, NHITSFPQRConfig, None, {"per_target_tau": False}, None),
        "nhits fpqr per-target grid": (NHITSFPQRModel, NHITSFPQRConfig, None, {"per_target_tau": True}, None),
    }


def decision_rating(scenario, folds, quantile):
    """Apparent-power rating, from the whole record rather than per fold.

    A threshold recomputed per fold would move with the season, and the
    operational tier would then compare exceedance probabilities defined
    against different levels, which is not a comparison at all.
    """
    values = []
    for windows in folds:
        truth = windows["y_te"] * windows["sd"] + windows["mu"]
        values.append(np.hypot(truth[..., 0], truth[..., 1]).ravel())
    return float(np.quantile(np.concatenate(values), quantile))


def run_fold(windows, args, backbone, registry, include_voltage, voltage_band):
    """Fit both conditions on one fold and score the hosting-capacity/thermal union event."""
    fold = windows["fold"]
    root = Path(tempfile.mkdtemp(prefix=f"opdecision_{fold}_"))
    rng = np.random.default_rng(args.seed)
    settings = {
        "epochs": args.epochs,
        "seed": args.seed,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "backbone": backbone,
        "num_workers": args.num_workers,
    }

    preds = {}
    for name in args.variants:
        cls, cfg_cls, family, extra, copula_family = registry[name]
        extra = {**extra, **FPQR_SETTINGS} if "fpqr" in name else extra
        started = time.time()
        preds[name] = fit_and_predict(
            f"{name}_{fold}",
            cls,
            _cfg(cfg_cls, windows, **settings, **extra),
            windows,
            root,
            family=family or "normal",
            n_paths=args.paths,
            rng=rng,
            copula_family=copula_family,
        )
        log.info("    %-32s %6.1fs", name, time.time() - started)

    truth = windows["y_te"] * windows["sd"] + windows["mu"]
    days = np.arange(truth.shape[0])
    curve_rows: list[dict] = []
    table = hosting_capacity_rows(
        preds,
        truth,
        windows["targets"],
        args.rated,
        days,
        voltage_band=voltage_band,
        include_voltage=include_voltage,
        curve_sink=curve_rows,
    )
    table["fold"] = fold
    curve = pd.DataFrame(curve_rows)
    curve["fold"] = fold
    return table, curve


def family_significance(table: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Diebold-Mariano on the Brier score, joint against independent, per fold.

    Reuses ``twiga.core.metrics``'s
    ``pairwise_dm_table`` rather than a separate reimplementation, per that
    module's own chapter-agnostic design.
    """
    sub = table[table.metric == "Brier score"]
    if sub.empty:
        return pd.DataFrame()
    per_fold = sub.groupby(["model", "fold"])["value"].mean().unstack("fold")
    losses = {name: per_fold.loc[name].to_numpy() for name in per_fold.index}
    if len(losses) < 2:
        return pd.DataFrame()
    return pairwise_dm_table(losses, h=1, alpha=0.05, version="univariate")


def run_decision(key: str, args, out_dir: Path) -> pd.DataFrame:
    """Run one decision (DSS-UK hosting-capacity or DSS-US thermal) end to end."""
    scenario = SCENARIOS[key]
    include_voltage = INCLUDE_VOLTAGE[key]
    voltage_band = VOLTAGE_BANDS[key]

    folds = build_backtest_folds(
        scenario,
        args.lookback,
        args.horizon,
        train_months=args.train_size,
        test_months=args.test_size,
        step_months=args.step,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
        expanding=args.expanding,
        split_freq=args.split_freq,
    )
    if not folds:
        raise SystemExit(
            f"no folds for {scenario}; the record is shorter than "
            f"train_size + test_size at split_freq={args.split_freq!r}"
        )
    backbone = backbone_settings()
    registry = variant_registry()

    rated = decision_rating(scenario, folds, args.rated_q)
    # run_fold() reads args.rated (matching copula_variants.py's convention);
    # each decision has its own rating, computed from its own scenario/folds,
    # so this is set fresh per call rather than once in main().
    args.rated = rated

    label = "hosting-capacity (thermal or voltage)" if include_voltage else "thermal-only"
    log.info("=== %s: %s, %s ===", key, scenario, label)
    shape = "expanding" if args.expanding else "rolling"
    log.info(
        "protocol %s train %s %s, test %s %s, step %s; %d folds",
        shape,
        args.train_size,
        args.split_freq,
        args.test_size,
        args.split_freq,
        args.step,
        len(folds),
    )
    log.info("rating   %.1f kVA at the %.0f%% quantile of recorded apparent power", rated, args.rated_q * 100)

    tables = []
    curves = []
    for windows in folds:
        log.info(
            "fold %s (train %s, %d train / %d test windows)",
            windows["fold"],
            windows["train_span"],
            windows["x_tr"].shape[0],
            windows["x_te"].shape[0],
        )
        table, curve = run_fold(windows, args, backbone, registry, include_voltage, voltage_band)
        table["decision"] = key
        curve["decision"] = key
        tables.append(table)
        curves.append(curve)
        pd.concat(tables, ignore_index=True).to_csv(out_dir / f"{key.lower()}_per_fold.csv", index=False)
        pd.concat(curves, ignore_index=True).to_csv(out_dir / f"{key.lower()}_value_curve.csv", index=False)

    full = pd.concat(tables, ignore_index=True)
    full.to_csv(out_dir / f"{key.lower()}_per_fold.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out_dir / f"{key.lower()}_value_curve.csv", index=False)

    summary = full.groupby(["metric", "target", "model"])["value"].agg(["mean", "std", "min", "max"]).reset_index()
    summary.to_csv(out_dir / f"{key.lower()}_summary.csv", index=False)
    print(f"\n=== {key} summary (mean across {len(folds)} folds) ===")
    print(summary.pivot_table(index="metric", columns="model", values="mean").round(4).to_string())

    dm = family_significance(full)
    if not dm.empty:
        dm.to_csv(out_dir / f"{key.lower()}_significance.csv", index=False)
        print(f"\n=== {key}: joint vs independent, Brier score DM test ===")
        print(dm.to_string(index=False))

    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906, help="fixed; folds supply the variation")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes -- see copula_variants.py's --num-workers help for why 0 "
        "is the default here rather than NeuralModelConfig's own 8.",
    )
    parser.add_argument(
        "--split-freq",
        default="weeks",
        choices=["days", "weeks", "months", "years"],
        help="calendar unit --train-size/--test-size/--step are counted in -- see module "
        "docstring's 2026-09-17 note for why weeks, not Chapter-6.tex's literal months, is the "
        "default now",
    )
    parser.add_argument(
        "--train-size",
        "--train-months",
        dest="train_size",
        type=int,
        default=17,
        help="training window width, in --split-freq units; default 17 weeks (~4 months), not "
        "Chapter-6.tex's literal 12 months -- see module docstring",
    )
    parser.add_argument(
        "--test-size",
        "--test-months",
        dest="test_size",
        type=int,
        default=4,
        help="held-out window width, in --split-freq units; default 4 weeks (~1 month), not "
        "Chapter-6.tex's literal 4 months -- see module docstring",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="the origin's advance between folds, in --split-freq units. Default 4 weeks == "
        "--test-size, so consecutive folds' test periods do not overlap.",
    )
    parser.add_argument(
        "--expanding",
        action="store_true",
        default=False,
        help="grow the training span fold by fold rather than sliding a fixed width. Off "
        "(rolling) by default as of 2026-09-17 -- see module docstring; pass this flag to "
        "restore Chapter-6.tex's original expanding protocol",
    )
    parser.add_argument("--no-expanding", dest="expanding", action="store_false")
    parser.add_argument("--train-stride", type=int, default=1, help="training windows may overlap; augmentation")
    parser.add_argument("--test-stride", type=int, default=48, help="one day, so the test set does not overlap")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument("--rated-q", type=float, default=0.98, help="rating as a quantile of recorded apparent power")
    parser.add_argument(
        "--run-hpo",
        action="store_true",
        help="tune the shared MLPF backbone (tune_backbone.py's random search) before either "
        "decision runs, instead of reading backbone_params.json or, if that is also missing, "
        "silently falling back to a fixed default architecture. Matches multitarget_experiment.py's "
        "--run-hpo convention. Tuned once, on --hpo-scenario, since both decisions share one "
        "backbone by design (see tune_backbone.py's module docstring).",
    )
    parser.add_argument(
        "--hpo-scenario",
        default="DSS-EU:default:0.4:0.4",
        help="scenario tuned against when --run-hpo is passed, independent of --decisions",
    )
    parser.add_argument("--hpo-trials", type=int, default=16)
    parser.add_argument("--hpo-epochs", type=int, default=30, help="epochs per HPO trial, not per final condition")
    args = parser.parse_args()

    if args.run_hpo:
        tune_backbone(scenario=args.hpo_scenario, trials=args.hpo_trials, epochs=args.hpo_epochs, seed=args.seed)

    # runs/<run_id>/results/, mirroring ExperimentEngine's own layout (see
    # twiga.experiment.run_versioning) -- previously a flat OUT directory
    # both decisions wrote into directly, silently overwritten by the next
    # run with no record of which config/git commit produced it.
    run_root = start_run(
        OUT,
        meta={"decisions": args.decisions, "epochs": args.epochs, "seed": args.seed},
    )
    out_dir = run_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {key: run_decision(key, args, out_dir) for key in args.decisions}
    (out_dir / "operational_decision_config.json").write_text(
        json.dumps(
            {
                "decisions": args.decisions,
                "scenarios": {k: SCENARIOS[k] for k in args.decisions},
                "epochs": args.epochs,
                "seed": args.seed,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step": args.step,
                "split_freq": args.split_freq,
                "window": "expanding" if args.expanding else "rolling",
                "train_stride": args.train_stride,
                "test_stride": args.test_stride,
                "paths": args.paths,
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
