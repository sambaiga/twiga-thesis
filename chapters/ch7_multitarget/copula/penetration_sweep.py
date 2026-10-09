"""Does the advantage of modelling dependence widen as the coupling strengthens?

A single penetration level yields a number. If a joint likelihood helps because
it represents dependence, then its advantage should grow with the dependence,
and the penetration sweep is the axis along which that dependence is known to
change: Section 6.3 measures the recorded correlation between active and
reactive power moving from strongly positive at zero adoption to strongly
negative under generation.

This runs the backtest at each point of the sweep and regresses the operational
advantage on the measured coupling. A trend is evidence that the effect is the
one claimed; a flat line says the advantage, if any, is not about dependence.

    uv run python experiment/ch7_multitarget/penetration_sweep.py --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from twiga.core.utils import get_logger
from twiga.core.utils.paths import slugify
from twiga.experiment import start_run

from chapters.ch7_multitarget.copula.copula_variants import run_fold, variant_registry  # noqa: E402
from chapters.ch7_multitarget.shared.forecast_examples import backbone_settings, build_backtest_folds  # noqa: E402
from chapters.ch7_multitarget.shared.tune_backbone import tune as tune_backbone  # noqa: E402

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
# Outside the repo, matching multitarget_experiment.py/fpqr_gating_ablation.py/
# copula_variants.py's output_dir convention -- see copula_variants.py's OUT
# comment for why.
OUT = script_dir.parent.parent.parent.parent / "experiments" / "ch7_multitarget" / "sweep"

# The pair whose margins are identical by construction, so a difference between
# them is attributable to the coupling alone rather than to a differently fitted
# model. The joint density heads change the margins as well.
COUPLED = "fpqr shared + copula"
UNCOUPLED = "fpqr shared grid"


def coupling_strength(windows) -> float:
    """Absolute correlation between active and reactive power on the test rows."""
    truth = windows["y_te"] * windows["sd"] + windows["mu"]
    flat = truth.reshape(-1, truth.shape[-1])
    return abs(float(np.corrcoef(flat[:, 0], flat[:, 1])[0, 1]))


# The three OpenDSS power-flow records that expose a penetration dial (the
# other Chapter 6 records -- MLVS-PT, MLVS-PT-3PHASE, MLVS-UK -- are real,
# fixed-penetration measurements with no sweep to run). Each is close to
# exactly one calendar year long (DSS-EU ~11 months, DSS-UK/DSS-US 12 months
# -- verified directly against examples/data/*.parquet), which is why the
# default train/test/step window below is 17/4/4 *weeks*, not the chapter's
# literal 12/4 *months*: at 12/4 months an ~11-12 month record yields only 2
# rolling folds per penetration level, too few for the >=3-point trend
# regression below, and on DSS-EU specifically some penetration levels
# produce zero folds outright. Weeks (a fixed 7 days) rather than months (28-
# 31 days, variable) also make the non-overlap guarantee between consecutive
# folds' test periods exact rather than approximate, when --step equals
# --test-size, as the defaults do here. Verified: 17/4/4 weeks gives 10 folds
# per penetration level on DSS-UK/DSS-US, 8 on DSS-EU (its record is ~1 month
# shorter).
RECORD_CHOICES = ["DSS-EU:default", "DSS-UK:weak", "DSS-UK:stiff", "DSS-US:0"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "--record",
        dest="record",
        nargs="+",
        default=["DSS-EU:default"],
        choices=[*RECORD_CHOICES, "all"],
        help="record(s) and feeder, without the penetration suffix, or 'all' for every entry in "
        "RECORD_CHOICES -- matches the other Chapter 6 scripts' --dataset convention; --record is "
        "kept as an alias for backward compatibility",
    )
    parser.add_argument("--penetrations", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes -- see copula_variants.py's --num-workers help for why 0 "
        "is the default here rather than NeuralModelConfig's own 8 (run_fold is imported from there "
        "and reads this same flag).",
    )
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument(
        "--split-freq",
        default="weeks",
        choices=["days", "weeks", "months", "years"],
        help="calendar unit --train-size/--test-size/--step are counted in (passed straight to "
        "TimeBasedCV). 'weeks' (default) rather than Chapter-6.tex's literal 'months': a week is "
        "a fixed 7 days, so a train/test/step window measured in weeks partitions an ~11-12 month "
        "record exactly, with no overlap between consecutive folds' test periods, when --step "
        "equals --test-size. A calendar month varies 28-31 days, so the same non-overlap "
        "guarantee only holds approximately under 'months'.",
    )
    parser.add_argument(
        "--train-size",
        "--train-months",
        dest="train_size",
        type=int,
        default=17,
        help="training window width, in --split-freq units; default 17 weeks (~4 months) -- not "
        "Chapter-6.tex's literal 12 months, so an ~11-12 month record yields several rolling folds "
        "instead of 1-2 (see RECORD_CHOICES's comment above). --train-months kept as an alias for "
        "backward compatibility.",
    )
    parser.add_argument(
        "--test-size",
        "--test-months",
        dest="test_size",
        type=int,
        default=4,
        help="held-out window width, in --split-freq units; default 4 weeks (~1 month), same "
        "reason as --train-size. --test-months kept as an alias for backward compatibility.",
    )
    parser.add_argument(
        "--step",
        "--step-months",
        dest="step",
        type=int,
        default=4,
        help="the origin's advance between folds, in --split-freq units. Default 4 weeks == "
        "--test-size, so consecutive folds' test periods do not overlap -- keep --step equal to "
        "--test-size unless you specifically want overlapping folds (more folds, but no longer "
        "independent evidence for the coupling-vs-advantage regression below). Verified: at these "
        "defaults, DSS-UK/DSS-US give 10 folds per penetration level, DSS-EU gives 8 (its record "
        "is ~1 month shorter).",
    )
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=48)
    parser.add_argument("--rated-q", type=float, default=0.98)
    parser.add_argument("--n-quantiles", type=int, default=21)
    parser.add_argument("--entropy-weight", type=float, default=100.0)
    parser.add_argument("--variants", nargs="+", default=[COUPLED, UNCOUPLED])
    parser.add_argument(
        "--expanding",
        action="store_true",
        default=False,
        help="grow the training window fold-by-fold instead of sliding a fixed width. Off by "
        "default: this script's own claim is a trend in the *coupling-vs-advantage* relationship "
        "across folds, and expanding confounds that with a training-set-size effect (see "
        "build_backtest_folds's docstring) -- rolling keeps every fold's training size fixed so a "
        "difference between folds is attributable to season/coupling alone.",
    )
    parser.add_argument("--no-expanding", dest="expanding", action="store_false")
    parser.add_argument(
        "--run-hpo",
        action="store_true",
        help="tune the shared MLPF backbone (tune_backbone.py's random search) before the sweep "
        "runs, instead of reading backbone_params.json or, if that is also missing, silently "
        "falling back to a fixed default architecture. Matches multitarget_experiment.py's "
        "--run-hpo convention. Tuned once, on --hpo-scenario, since every penetration level and "
        "record here shares one backbone by design (see tune_backbone.py's module docstring).",
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

    records = list(RECORD_CHOICES) if "all" in args.record else args.record
    for record in records:
        run_record(record, args)


def run_record(record: str, args: argparse.Namespace) -> None:
    """Run the full penetration sweep for one base record, writing to its own versioned run."""
    # runs/<run_id>/results/, mirroring ExperimentEngine's own layout (see
    # twiga.experiment.run_versioning) -- previously a flat OUT/slugify(record)
    # directory that a later sweep silently overwrote.
    run_root = start_run(
        OUT / slugify(record),
        meta={"record": record, "epochs": args.epochs, "seed": args.seed, "penetrations": args.penetrations},
    )
    out_dir = run_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = backbone_settings()
    registry = variant_registry()
    rows, summary = [], []

    for pen in args.penetrations:
        scenario = f"{record}:{pen}:{pen}"
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
            log.warning("  %s: no folds, skipped", scenario)
            continue
        rho = coupling_strength(folds[0])
        # One rating per scenario, from that scenario's own record, since the
        # sweep changes the loading and a shared threshold would not screen the
        # same way at both ends of it.
        truth = np.concatenate([f["y_te"] * f["sd"] + f["mu"] for f in folds])
        args.rated = float(np.quantile(np.hypot(truth[..., 0], truth[..., 1]), args.rated_q))
        log.info("  %s  |rho| %.3f  rating %.1f kVA  %d folds", scenario, rho, args.rated, len(folds))

        tables = []
        for windows in folds:
            table, _losses = run_fold(windows, args, backbone, registry)
            table["penetration"] = pen
            table["coupling"] = rho
            tables.append(table)
        scenario_table = pd.concat(tables, ignore_index=True)
        rows.append(scenario_table)

        brier = scenario_table[(scenario_table.tier == "operational") & (scenario_table.metric == "Brier score")]
        wide = brier.pivot_table(index="fold", columns="model", values="value")
        if COUPLED in wide.columns and UNCOUPLED in wide.columns:
            # Positive advantage means the coupled head has the lower score.
            advantage = float((wide[UNCOUPLED] - wide[COUPLED]).mean())
            summary.append({"penetration": pen, "coupling": rho, "advantage": advantage})
            log.info("    advantage of coupling: %+.5f Brier", advantage)

    if not rows:
        raise SystemExit(f"no scenario produced folds for record {record!r}")
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "sweep_per_fold.csv", index=False)

    trend: dict = {"points": summary}
    if len(summary) >= 3:
        x = np.array([s["coupling"] for s in summary])
        y = np.array([s["advantage"] for s in summary])
        slope, intercept = np.polyfit(x, y, 1)
        trend |= {
            "slope_vs_coupling": float(slope),
            "intercept": float(intercept),
            "pearson": float(np.corrcoef(x, y)[0, 1]),
        }
        log.info("  advantage against coupling: slope %+.5f, r %+.3f", slope, trend["pearson"])
    else:
        trend["note"] = f"only {len(summary)} points; a trend needs at least three"
    (out_dir / "sweep_trend.json").write_text(json.dumps(trend, indent=2) + "\n")
    log.info("wrote %s and %s", out_dir / "sweep_per_fold.csv", out_dir / "sweep_trend.json")


if __name__ == "__main__":
    main()
