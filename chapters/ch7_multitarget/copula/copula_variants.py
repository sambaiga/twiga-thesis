"""Chapter 6, Experiment 2: the copula-based dependence modelling condition.

``Chapter-6.tex`` (``subsec:ch6-protocol``) names "copula-based dependence
modelling" as one of the design choices Experiment 2 evaluates, alongside the
proposal mechanism and monotonic value network that
``fpqr_gating_ablation.py`` already covers through
:class:`~twiga.experiment.ExperimentEngine`. This script covers the one piece
of Experiment 2 that engine cannot express: FPQR fits each target's margin
independently, and the two ``+ copula`` conditions here couple those margins
post hoc through a Gaussian copula fitted on validation rows
(:func:`twiga.distributions.nn.copula_sampling.fit_gaussian_copula`). Nothing
in ``twiga.experiment`` has a hook for a post-fit sampling step like that, so
it stays hand-rolled here rather than forced into a ``Condition``.

All eight FPQR-family variants (shared/per-target grid; shared/per-target
coupled through a Gaussian, Clayton, or three-target D-vine copula) are fit
together in this one script, on the same folds and seed, so the paired
comparison between a copula variant and its non-coupled counterpart is drawn
from one run rather than reconciled after the fact against
``fpqr_gating_ablation.py``'s separately-run "grid" group; the two are
expected to agree on the non-copula pair up to seed noise, and a large
disagreement would itself be worth reporting. See
``experiment/ch7_multitarget/COPULA_METHODS.md`` for the theory behind
each copula family and what is novel about each.

Protocol: **changed 2026-09-17, no longer ``Chapter-6.tex``'s literal
figures.** The chapter's stated protocol (expanding window, training set of
at least twelve months in the first fold, four-month test window, origin
advancing one month between folds) gave only 3 folds on DSS-EU at these
settings -- confirmed directly, and the same problem affects DSS-UK/DSS-US
(all three of the OpenDSS records with a penetration dial are close to
exactly one calendar year long, not long enough for a 12+4-month protocol to
yield many folds). Now matches ``penetration_sweep.py``'s fix exactly:
rolling (not expanding -- see that script's own note on why expanding
confounds season with training-set size across folds), ``split_freq="weeks"``,
train/test/step = 17/4/4 weeks (~4/~1/~1 months). Verified: 10 folds on
DSS-UK/DSS-US, 8 on DSS-EU. Fold boundaries come from
:func:`forecast_examples.build_backtest_folds`, backed by
:class:`~twiga.core.backtester.TimeBasedCV` (the same class
:class:`~twiga.experiment.ExperimentEngine` uses), which now takes a
``split_freq`` argument for exactly this reason -- a week is a fixed 7 days,
so ``--step`` equal to ``--test-size`` gives an *exact* non-overlap
guarantee between consecutive folds' test periods, where a calendar month
(28-31 days) only gives an approximate one. Every test window is scored at
daily, non-overlapping granularity, a post-fold evaluation detail
independent of how the fold boundaries were generated. Pass
``--split-freq months --train-size 12 --test-size 4 --step 1 --expanding``
to reproduce the original, literal (but 3-fold) chapter protocol.

**Gap closed**: Chapter 6's text (``subsec:ch6-evaluation``) states every
backbone is evaluated with the multivariate adaptive conformal calibration of
``sec:ch6-conformal`` applied on top of it. Every condition here is now
additionally calibrated by
:class:`~twiga.distributions.conformal.multivariate.MultivariateConformal`, both
``score_type="max-norm"`` (the baseline pooling rule) and
``score_type="mahalanobis"`` (using each condition's own fitted copula
correlation, identity for the two non-coupled variants). The test fold
supplies no separate calibration split, so its days are themselves halved
chronologically: the first half calibrates ``q_hat``, the second is scored,
which is a within-fold split rather than a leak, since no scored day was used
for calibration. See ``COPULA_METHODS.md`` section 6 for why "mahalanobis"
is expected to be valid but conservative, not a free efficiency win, except
where the correlation genuinely varies by condition.

    uv run python experiment/ch7_multitarget/copula_variants.py --epochs 100
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from twiga.core.metrics import diebold_mariano_from_losses, model_confidence_set
from twiga.core.utils import get_logger
from twiga.core.utils.paths import slugify
from twiga.experiment import start_run

log = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from experiment.common.significance import pairwise_dm_table  # noqa: E402
from forecast_examples import (  # noqa: E402
    _cfg,
    backbone_settings,
    build_backtest_folds,
    conformal_rows,
    fit_and_predict,
    metrics_table,
    pointwise_losses,
)
from tune_backbone import tune as tune_backbone  # noqa: E402

script_dir = Path(__file__).resolve().parent
# Outside the repo, matching multitarget_experiment.py/fpqr_gating_ablation.py's
# own output_dir convention -- previously script_dir / "copula_variants", the only
# hand-rolled Chapter 6 script not aligned with it. Checkpoints here already go
# to a throwaway tempfile.mkdtemp() directory regardless (see run_fold), so
# this move is purely about giving every Chapter 6 experiment's output one
# common location, not about keeping large artifacts out of git.
OUT = script_dir.parent.parent.parent.parent / "experiments" / "ch7_multitarget" / "copula_variants"

# The three canonical full scenario strings this script's protocol is
# verified against (module docstring: "Verified: 10 folds on DSS-UK/DSS-US,
# 8 on DSS-EU"), matching operational_decision.py's own SCENARIOS dict
# exactly (DSS-UK's weak feeder, DSS-US's rank-0/most-loaded transformer,
# both at the same 0.4/0.4 penetration default DSS-EU already uses here) --
# not penetration_sweep.py's RECORD_CHOICES, which omits the penetration
# suffix since that script sweeps it separately via --penetrations.
#
# "all" iterates this list alone, not MLVS_PT_3PHASE below: the three OpenDSS
# records share one verified weeks-based protocol (this module's own
# --train-size/--test-size/--step/--split-freq defaults), while
# MLVS-PT-3PHASE needs a different, also-verified months-based one (see
# MLVS_PT_3PHASE_ARGS below) -- looping the same global args over both
# families in one run would silently apply the wrong protocol to whichever
# one isn't the CLI defaults' target.
ALL_SCENARIOS = ["DSS-EU:default:0.4:0.4", "DSS-UK:weak:0.4:0.4", "DSS-US:0:0.4:0.4"]

# The measured (not power-flow-simulated) P/Q/V record at the Madeira
# substation -- no penetration dial, so no ":pen:pen" suffix, unlike
# ALL_SCENARIOS' three OpenDSS entries. build_backtest_folds()/fit_and_predict()
# are already dataset-agnostic (they just call load_dataset(scenario, ...)),
# so this scenario runs through the same code path; only the CV protocol
# needs to differ, per MLVS_PT_3PHASE_ARGS below.
MLVS_PT_3PHASE = "MLVS-PT-3PHASE"

# The verified fold recipe for MLVS-PT-3PHASE -- not re-derived here, but
# lifted directly from multitarget_experiment.py's and fpqr_gating_ablation.py's
# SPEC_MLVS_PT_3PHASE, which both independently arrive at the same numbers:
# train=12/test=3 months, stride=3 (== test_size, so consecutive folds' test
# windows do not overlap), rolling window -> confirmed 10 non-overlapping
# folds on the real 37.5-month record. This script's own --train-size 17
# --test-size 4 --step 4 --split-freq weeks defaults are verified only for
# the ~11-12-month OpenDSS records (ALL_SCENARIOS) and would silently apply
# an unverified protocol here -- pass this dict's values explicitly instead:
#
#   uv run python experiment/ch7_multitarget/copula_variants.py \
#       --dataset MLVS-PT-3PHASE --split-freq months --train-size 12 \
#       --test-size 3 --step 3 --no-expanding
#
# copula_variants.py has no separate cv_val_size/cv_calib_size reservation
# (build_backtest_folds carves its own 87.5%/12.5% train/val split from the
# already-windowed training block, and the multivariate-conformal step
# calibrates from a within-fold day-split -- see the module docstring's
# "Gap closed" note) so only train/test/step/split-freq/window need to
# transfer from the ExperimentSpec numbers above.
MLVS_PT_3PHASE_ARGS = {
    "split_freq": "months",
    "train_size": 12,
    "test_size": 3,
    "step": 3,
    "expanding": False,
}

# The FPQR family only. copula_family is None (independent margins),
# "gaussian", "clayton", or "vine" (three targets only) -- see
# forecast_examples.fit_and_predict's docstring and COPULA_METHODS.md.
VARIANTS = (
    "fpqr shared grid",
    "fpqr per-target grid",
    "fpqr shared + copula",
    "fpqr per-target + copula",
    "fpqr shared + clayton copula",
    "fpqr per-target + clayton copula",
    "fpqr shared + vine copula",
    "fpqr per-target + vine copula",
)


def variant_registry():
    """Model class, config class, family, extra config and copula family per name."""
    from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig, MLPFFPQRModel

    return {
        "fpqr shared grid": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, None),
        "fpqr per-target grid": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, None),
        "fpqr shared + copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "gaussian"),
        "fpqr per-target + copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "gaussian"),
        "fpqr shared + clayton copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "clayton"),
        "fpqr per-target + clayton copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "clayton"),
        "fpqr shared + vine copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": False}, "vine"),
        "fpqr per-target + vine copula": (MLPFFPQRModel, MLPFFPQRConfig, None, {"per_target_tau": True}, "vine"),
    }


def study_rating(scenario, folds, quantile):
    """Apparent-power rating, from the whole record rather than per fold.

    A threshold recomputed per fold would move with the season, and the
    operational tier would then compare exceedance probabilities defined against
    different levels, which is not a comparison at all.
    """
    values = []
    for windows in folds:
        truth = windows["y_te"] * windows["sd"] + windows["mu"]
        values.append(np.hypot(truth[..., 0], truth[..., 1]).ravel())
    return float(np.quantile(np.concatenate(values), quantile))


def run_fold(windows, args, backbone, registry):
    """Fit every condition on one fold and score all three tiers."""
    fold = windows["fold"]
    root = Path(tempfile.mkdtemp(prefix=f"copula_{fold}_"))
    rng = np.random.default_rng(args.seed)
    settings = {
        "epochs": args.epochs,
        "seed": args.seed,
        "horizon": args.horizon,
        "lookback": args.lookback,
        "backbone": backbone,
        "num_workers": args.num_workers,
    }
    fpqr = {
        "n_quantiles": args.n_quantiles,
        "entropy_weight": args.entropy_weight,
        "num_cosines": 0,
        "rope_type": "tau_pos",
        "qvn_type": "monotone",
    }

    preds = {}
    for name in args.variants:
        cls, cfg_cls, family, extra, copula_family = registry[name]
        extra = {**extra, **fpqr}
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
        log.info("    %-26s %6.1fs", name, time.time() - started)

    truth = windows["y_te"] * windows["sd"] + windows["mu"]
    # Every test window is scored: the fold is one month of non-overlapping days,
    # so there is nothing to subsample and no reason to.
    days = np.arange(truth.shape[0])
    table = metrics_table(preds, truth, windows["targets"], args.rated, days)
    table["fold"] = fold
    table["train_span"] = windows["train_span"]
    losses = pointwise_losses(preds, truth, windows["targets"], args.rated, days)
    losses["fold"] = fold
    conformal = conformal_rows(preds, truth, days, alpha=args.alpha)
    conformal["fold"] = fold
    return table, losses, conformal


def significance(losses, primary=("fpqr shared + copula", "fpqr shared grid"), metric="Brier score"):
    """Paired tests over per-forecast losses, for one contrast and for the family.

    The losses are one value per forecast day rather than one per fold. Both
    tests are asymptotic in the length of the series, and at the eight folds this
    design produces the normal approximation does not hold its size: simulated
    under a true null, Diebold-Mariano rejects at about 0.077 against a nominal
    0.05. Pooling the folds gives roughly 250 daily losses, where it does.

    Two tests, for two purposes. The primary contrast is fixed in advance and
    tested by Diebold-Mariano, which is a statement about one pair. The family is
    summarised by the model confidence set, which returns the conditions that
    cannot be separated from the best; testing every pair instead would inflate
    the false-positive rate with the number of conditions.

    Args:
        losses: per-forecast losses with day, model, metric and value columns.
        primary: the two conditions of the pre-registered comparison.
        metric: proper score to test. Any pointwise score serves.

    Returns:
        Mapping with the DM result and the confidence set, or a reason neither
        could be computed.
    """
    sub = losses[losses.metric == metric]
    if sub.empty:
        return {"note": f"no {metric} losses recorded"}
    wide = sub.pivot_table(index=["fold", "day"], columns="model", values="value").dropna(axis=1)
    n = int(wide.shape[0])
    out: dict = {"metric": metric, "observations": n}
    if n < 30:
        out["warning"] = f"{n} observations is too few for the asymptotics; treat the p-value as indicative"

    if all(name in wide.columns for name in primary):
        a, b = (wide[name].to_numpy() for name in primary)
        # Lower is better, so a negative mean difference favours the first named.
        out["primary"] = {
            "pair": list(primary),
            "mean_difference": float(np.mean(a - b)),
            # version="multivariate": these are one-step scalar losses (one
            # Brier/Winkler value per fold-day, no forecast-horizon axis), so
            # "univariate" and "multivariate" are mathematically identical
            # here -- "multivariate" always returns a genuine Python float,
            # while "univariate" returns a length-1 ndarray even for a single
            # step, which numpy>=2.0 refuses to convert with float() (only
            # exactly 0-dimensional arrays are allowed now, not just size-1
            # ones) -- confirmed as the exact cause of a TypeError here on
            # numpy 2.4.4: "only 0-dimensional arrays can be converted to
            # Python scalars".
            "dm_p_value": float(diebold_mariano_from_losses(a, b, version="multivariate")),
        }
    else:
        out["primary"] = {"note": f"{primary} not both present"}

    mcs = model_confidence_set(wide.to_numpy(), names=list(wide.columns), alpha=0.10)
    out["confidence_set"] = list(getattr(mcs, "included", getattr(mcs, "models", [])))
    return out


def family_significance(table: pd.DataFrame, metric: str = "Winkler") -> pd.DataFrame:
    """BH-corrected pairwise Diebold-Mariano across all four variants, per fold.

    Complements :func:`significance`'s single pre-registered pair (which tests
    one contrast against a strict null) with the full M(M-1)/2 picture across
    the FPQR family, using ``experiment.common.significance``'s
    ``pairwise_dm_table`` rather than a separate reimplementation -- that
    module is documented as chapter-agnostic given per-fold loss arrays of
    equal length per condition, which is exactly what this table provides
    once averaged across targets.

    Args:
        table: the concatenated per-fold metrics table (tier/metric/target/
            model/value/fold columns), as produced by ``metrics_table``.
        metric: which marginal metric to test. Winkler score is a proper
            scoring rule for a central interval; CWE is never used here, per
            ``significance.py``'s module docstring.

    Returns:
        One row per unordered pair of variants, BH-corrected, sorted by
        adjusted p-value.
    """
    sub = table[(table.tier == "marginal") & (table.metric == metric)]
    per_fold = sub.groupby(["model", "fold"])["value"].mean().unstack("fold")
    losses = {name: per_fold.loc[name].to_numpy() for name in per_fold.index}
    return pairwise_dm_table(losses, h=1, alpha=0.05, version="univariate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "--scenario",
        dest="dataset",
        nargs="+",
        default=["DSS-EU:default:0.4:0.4"],
        choices=[*ALL_SCENARIOS, MLVS_PT_3PHASE, "all"],
        help="full scenario string(s), or 'all' for every entry in ALL_SCENARIOS (not "
        f"{MLVS_PT_3PHASE!r} -- see ALL_SCENARIOS's own comment for why) -- matches the "
        "other Chapter 6 scripts' --dataset convention; --scenario is kept as an alias for "
        f"backward compatibility. {MLVS_PT_3PHASE!r} automatically applies its own verified "
        "months-based CV protocol (MLVS_PT_3PHASE_ARGS) in place of this script's weeks-based "
        "defaults, unless --split-freq/--train-size/--test-size/--step/--expanding are passed "
        "explicitly, which always win.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906, help="fixed; folds supply the variation")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (NeuralModelConfig's own default is 8). 0 by default "
        "here: every fold's windows are already fully materialised in memory (build_backtest_folds), "
        "so extra worker processes buy nothing but subprocess-spawn overhead -- and on Windows, "
        "spawn (not fork) re-imports this whole process's dependency chain (torch/sklearn/scipy) "
        "in every worker, which has been observed to raise MemoryError/'DataLoader worker exited "
        "unexpectedly' outright rather than just running slower. Raise this only if profiling shows "
        "data loading, not the model, is the bottleneck.",
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
    parser.add_argument("--n-quantiles", type=int, default=21)
    parser.add_argument("--entropy-weight", type=float, default=100.0)
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="miscoverage level for the multivariate conformal calibration step"
    )
    parser.add_argument(
        "--run-hpo",
        action="store_true",
        help="tune the shared MLPF backbone (tune_backbone.py's random search) before running any "
        "scenario, instead of reading backbone_params.json or, if that is also missing, silently "
        "falling back to a fixed default architecture. Matches multitarget_experiment.py's "
        "--run-hpo convention. Tuned once, on --hpo-scenario, since every scenario here shares one "
        "backbone by design (see tune_backbone.py's module docstring) -- not once per --dataset.",
    )
    parser.add_argument(
        "--hpo-scenario",
        default="DSS-EU:default:0.4:0.4",
        help="scenario tuned against when --run-hpo is passed, independent of --dataset",
    )
    parser.add_argument("--hpo-trials", type=int, default=16)
    parser.add_argument("--hpo-epochs", type=int, default=30, help="epochs per HPO trial, not per final condition")
    args = parser.parse_args()

    if args.run_hpo:
        tune_backbone(scenario=args.hpo_scenario, trials=args.hpo_trials, epochs=args.hpo_epochs, seed=args.seed)

    dataset_keys = list(ALL_SCENARIOS) if "all" in args.dataset else args.dataset
    for scenario in dataset_keys:
        run_scenario(scenario, _resolve_scenario_args(scenario, args, parser))


def _resolve_scenario_args(
    scenario: str,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Apply MLVS_PT_3PHASE_ARGS's verified CV protocol for the measured record.

    Only fills in a field when the caller left it at this script's own
    (OpenDSS-verified) argparse default -- an explicit --train-size/--test-size/
    --step/--split-freq/--expanding on the command line always wins, exactly
    like the module docstring's existing "pass these flags to reproduce the
    original literal protocol" convention for ALL_SCENARIOS.
    """
    if scenario != MLVS_PT_3PHASE:
        return args
    resolved = copy.copy(args)
    for name, value in MLVS_PT_3PHASE_ARGS.items():
        if getattr(args, name) == parser.get_default(name):
            setattr(resolved, name, value)
        else:
            log.info(
                "%s: keeping explicit --%s=%s over the verified default %s",
                MLVS_PT_3PHASE,
                name.replace("_", "-"),
                getattr(args, name),
                value,
            )
    return resolved


def run_scenario(scenario: str, args: argparse.Namespace) -> None:
    """Run the full copula-variants study for one scenario string, writing to its own versioned run."""
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
            f"no folds for {scenario!r}; the record is shorter than train_size + test_size at this split_freq"
        )
    backbone = backbone_settings()
    registry = variant_registry()

    # One rating for the whole study, from the entire record rather than per
    # fold. A threshold that moved with the fold would make the operational tier
    # incomparable between seasons, which is exactly the comparison being drawn.
    args.rated = study_rating(scenario, folds, args.rated_q)

    # runs/<run_id>/results/, mirroring ExperimentEngine's own layout (see
    # twiga.experiment.run_versioning) -- previously a flat OUT/slugify(scenario)
    # directory that a later run silently overwrote, with no record of which
    # config or git commit produced it. copula_config.json below still carries
    # the full config for readability; run_id/git_commit/timestamp here are
    # what let a discover_runs()-style reader find "the latest run of this
    # scenario" the same way it already does for the ExperimentEngine-based
    # multitarget_experiment.py/fpqr_gating_ablation.py runs.
    run_root = start_run(
        OUT / slugify(scenario),
        meta={
            "scenario": scenario,
            "epochs": args.epochs,
            "seed": args.seed,
            "folds": len(folds),
        },
    )
    out_dir = run_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("scenario %s", scenario)
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
    log.info("strides  train %s, test %s (one day)", args.train_stride, args.test_stride)
    log.info(
        "budget   %s epochs, %d conditions, %d fits", args.epochs, len(args.variants), len(folds) * len(args.variants)
    )
    log.info("rating   %.1f kVA at the %.0f%% quantile of recorded apparent power", args.rated, args.rated_q * 100)

    tables, loss_frames, conformal_frames = [], [], []
    for windows in folds:
        log.info(
            "fold %s (train %s, %d train / %d test windows)",
            windows["fold"],
            windows["train_span"],
            windows["x_tr"].shape[0],
            windows["x_te"].shape[0],
        )
        table, fold_losses, fold_conformal = run_fold(windows, args, backbone, registry)
        tables.append(table)
        loss_frames.append(fold_losses)
        conformal_frames.append(fold_conformal)
        # Written after every fold, so an interrupted run still leaves usable
        # results rather than nothing.
        pd.concat(tables, ignore_index=True).to_csv(out_dir / "copula_per_fold.csv", index=False)

    full = pd.concat(tables, ignore_index=True)
    full.to_csv(out_dir / "copula_per_fold.csv", index=False)
    losses = pd.concat(loss_frames, ignore_index=True)
    losses.to_csv(out_dir / "copula_losses.csv", index=False)
    conformal = pd.concat(conformal_frames, ignore_index=True)
    conformal.to_csv(out_dir / "copula_conformal.csv", index=False)

    # Mean and spread across folds. The spread is seasonal variation, and a gap
    # between two variants smaller than it is not a result.
    summary = (
        full.groupby(["tier", "metric", "target", "model"])["value"].agg(["mean", "std", "min", "max"]).reset_index()
    )
    summary.to_csv(out_dir / "copula_summary.csv", index=False)
    (out_dir / "copula_config.json").write_text(
        json.dumps(
            {
                "scenario": scenario,
                "epochs": args.epochs,
                "seed": args.seed,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step": args.step,
                "split_freq": args.split_freq,
                "window": "expanding" if args.expanding else "rolling",
                "train_stride": args.train_stride,
                "test_stride": args.test_stride,
                "folds": [f["fold"] for f in folds],
                "backbone": backbone,
                "rated_kva": args.rated,
                "paths": args.paths,
                "train_windows": [int(f["x_tr"].shape[0]) for f in folds],
                "test_windows": [int(f["x_te"].shape[0]) for f in folds],
            },
            indent=2,
        )
        + "\n"
    )

    tests = significance(losses)
    (out_dir / "copula_significance.json").write_text(json.dumps(tests, indent=2, default=str) + "\n")
    print(f"\n=== pre-registered pair + confidence set ===\n{json.dumps(tests, indent=2, default=str)}")

    family = family_significance(full)
    family.to_csv(out_dir / "copula_family_dm.csv", index=False)
    print(f"\n=== all-pairs Diebold-Mariano, BH-corrected ===\n{family.to_string(index=False)}")

    conformal_summary = (
        conformal.groupby(["model", "score_type"])[["coverage", "mean_width"]].mean().round(4).reset_index()
    )
    conformal_summary.to_csv(out_dir / "copula_conformal_summary.csv", index=False)
    print(f"\n=== multivariate conformal calibration, nominal coverage {1.0 - args.alpha:.0%} (mean across folds) ===")
    print(conformal_summary.to_string(index=False))

    for tier in ("marginal", "joint", "operational"):
        sub = summary[summary.tier == tier]
        if sub.empty:
            continue
        print(f"\n=== {tier} (mean across {len(folds)} folds) ===")
        print(sub.pivot_table(index=["metric", "target"], columns="model", values="mean").round(4).to_string())
    log.info(
        "wrote %s, %s and %s",
        out_dir / "copula_per_fold.csv",
        out_dir / "copula_summary.csv",
        out_dir / "copula_conformal.csv",
    )


if __name__ == "__main__":
    main()
