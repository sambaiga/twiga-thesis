r"""Chapter 6, Experiment 1b: does increasing DER penetration degrade FPQR forecast quality.

Runs through :class:`~twiga.experiment.ExperimentEngine`, the same machinery as
``backbone_conformal_study.py``/``multitarget_experiment.py`` -- this replaces
``penetration_sweep.py``, which hand-rolled its own backtest loop and tested a
now-retired question (whether a copula-coupled head's advantage over a plain
head grows with the recorded P/Q correlation; ``Chapter-6.tex`` no longer
makes a copula-vs-no-copula comparison at all). This script answers the
question the chapter text actually asks (``subsec:ch6-protocol``,
``chap6:exp-one``): "the study investigates the impact of increasing PV and
EV penetration on forecasting performance in order to assess the robustness
of probabilistic forecasts under evolving network conditions and increasing
DER adoption" -- one condition (plain FPQR/MLPF, the same simple
configuration ``backbone_conformal_study.py``'s ``"mlpf_none"`` uses, no
conformal calibration -- that is Experiment 2's question, not this one), run
across a penetration grid, varying one axis at a time with the other held at
zero, matching the chapter's own figures
(``fig:ch6-penetration-nrmse`` etc., "each varied one axis at a time with the
other held at zero").

Beyond the chapter's own text: EV-only is also run on DSS-UK/DSS-US, not
just DSS-EU. Chapter-6.tex names DSS-EU exclusively for this study, and
DSS-EU already supplies a native, denser (5-point) EV-only sweep, so this
doesn't fill a data gap -- it answers a genuinely different question, whether
the EV-degrades-forecasts trend seen on DSS-EU generalises to other networks
(UK stiff/weak feeders, US pole transformers) or is specific to the one
substation DSS-EU represents. Kept clearly separate in the output (``axis``
column values ``ev_dss-eu``/``ev_dss-uk``/``ev_dss-us``) so this addition is
never mistaken for the chapter's own single-dataset claim.

Why DSS-UK/DSS-US carry the PV-only axis, not DSS-EU
------------------------------------------------------
``DSS-EU``'s own OpenDSS scenario files
(``examples/data/PT_DSS_EV_*_PV_*.parquet``, merged by ``build_dss_eu.py``)
cover a full EV-only sweep (EV in {0, .2, .4, .6, .8} at PV=0) but only two
points on the PV-only axis (PV in {0, .8} at EV=0) -- the PV=.2/.4/.6-at-
EV=0 scenario files were simply never simulated, confirmed directly against
the file listing, not assumed. Two points cannot support a trend claim.
``DSS-UK``/``DSS-US`` (``examples/data/DSS-UK.parquet``/``DSS-US.parquet``)
each carry a full, already-simulated 4x4 (EV, PV) grid at
{0, .2, .4, .6} on both axes (confirmed directly against the files' own
``ev_pen``/``pv_pen`` columns), including a genuine 4-point PV-only sweep at
EV=0 that DSS-EU lacks. So: DSS-EU supplies the EV-only axis (its own
native strength, matching ``backbone_conformal_study.py``'s existing DSS-EU
condition and CV window unchanged), DSS-UK and DSS-US both supply the
PV-only axis (run on both, as two independent records, rather than picking
one arbitrarily) -- not because the chapter names DSS-UK/DSS-US for this
study (it names only DSS-EU), but because they are the only records that
actually contain the PV-only data DSS-EU is missing. This is stated plainly
in the chapter text as a data-availability substitution, not silently
presented as if DSS-EU itself produced the PV-only curve.

CV window: as close to the stated protocol as each record's real span allows
-------------------------------------------------------------------------------
``subsec:ch6-protocol`` states "a five-fold rolling-origin backtesting
protocol with an expanding training window... initial historical period to
at least 6 months, and the fixed future time window to 2 months" for the
DSS-EU penetration analysis. DSS-EU's real span is 334 days (~11 months,
confirmed against the parquet files) -- 5 folds at a 2-month test window and
no overlap needs 6 + 5x2 = 16 months, which the record cannot supply at all.
The closest achievable configuration that keeps both the stated fold count
and test width is 5 folds, 2-month test, 1-month stride (folds' test windows
overlap by a month, the same overlap trade-off ``backbone_conformal_study.py``'s
own ``SPEC_MLVS_PT_3PHASE`` already accepts for the same reason) -- this
derives (via the engine's ``cv_folds``-driven train-size formula, see
``backbone_conformal_study.py``'s own note on why ``cv_train_size`` alone is
not honored once ``cv_folds`` is set) to a 5-month initial training window,
one month short of the stated "at least 6". Stated plainly rather than
silently claimed as exact: DSS-EU's real span makes the literal 6-month
figure unreachable at 5 folds, the same structural finding already made for
DSS-UK/DSS-US's own winter-coverage limitation elsewhere in this chapter's
scripts. DSS-UK/DSS-US reuse ``backbone_conformal_study.py``'s own already-
verified weeks-based CV numbers unchanged (26-week derived train, 4-week
test/stride, 6 folds, expanding) -- this study borrows their data, not their
own chapter section's protocol, so there is no "stated protocol" of theirs to
match here; reusing already-verified numbers avoids inventing a third,
unverified CV configuration.

Usage::

    uv run python experiment/ch7_multitarget/penetration_study.py
    uv run python experiment/ch7_multitarget/penetration_study.py --axis ev
    uv run python experiment/ch7_multitarget/penetration_study.py --quick
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment_2_backbone_conformal"))
from backbone_conformal_study import BACKBONES, DATASETS, DSS_UK, DSS_US  # noqa: E402
import numpy as np
import pandas as pd
from twiga.core.utils import configure, get_logger
from twiga.experiment import Condition, ExperimentEngine, ExperimentSpec
from twiga.utils.run_forecast import PipelineConfig

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent

# ── EV-only axis: DSS-EU, its own native strength (see module docstring) ──────
EV_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
DSS_EU_EV_ONLY = {ev: f"DSS-EU:default:{ev}:0" for ev in EV_LEVELS}

# ── PV-only axis: DSS-UK and DSS-US, borrowed for the PV data DSS-EU lacks ────
# Both records' own (EV, PV) grid only reaches 0.6 (confirmed against the
# files), one step short of DSS-EU's own 0.8 ceiling -- kept as each record's
# own real range rather than padding with an unsimulated level.
UK_US_LEVELS = (0.0, 0.2, 0.4, 0.6)
DSS_UK_PV_ONLY = {pv: f"DSS-UK:weak:0:{pv}" for pv in UK_US_LEVELS}
DSS_US_PV_ONLY = {pv: f"DSS-US:0:0:{pv}" for pv in UK_US_LEVELS}

# ── EV-only axis on DSS-UK/DSS-US too, as a cross-dataset robustness check ────
# Not something Chapter-6.tex's own text asks for (it names DSS-EU only for
# the penetration analysis) -- DSS-EU already supplies a native, denser
# (5-point) EV-only sweep, so this doesn't fill a data gap the way the PV-only
# borrowing above does. It answers a different, genuine question instead:
# does the EV-degrades-forecasts trend seen on DSS-EU generalise to other
# networks (UK stiff/weak feeders, US pole transformers), or is it specific
# to the one substation DSS-EU represents.
DSS_UK_EV_ONLY = {ev: f"DSS-UK:weak:{ev}:0" for ev in UK_US_LEVELS}
DSS_US_EV_ONLY = {ev: f"DSS-US:0:{ev}:0" for ev in UK_US_LEVELS}

_DSS_EU_WINDOW = {k: v for k, v in DATASETS["DSS-EU:default:0.4:0.4"].items() if k != "dataset_name"}
_DSS_UK_WINDOW = {k: v for k, v in DATASETS[DSS_UK].items() if k != "dataset_name"}
_DSS_US_WINDOW = {k: v for k, v in DATASETS[DSS_US].items() if k != "dataset_name"}

DSS_EU_DATASETS: dict[str, dict] = {key: {"dataset_name": key, **_DSS_EU_WINDOW} for key in DSS_EU_EV_ONLY.values()}
# Built from the *values* of both level dicts as a list, not a merged dict:
# DSS-UK:weak:0:0.0 (PV-only baseline) and DSS-UK:weak:0.0:0 (EV-only
# baseline) are different strings for the identical (EV=0, PV=0) scenario --
# merging by the shared penetration-level key (0.0) would silently drop one
# of the two, leaving the other axis's baseline dataset key missing from the
# spec entirely.
DSS_UK_DATASETS: dict[str, dict] = {
    key: {"dataset_name": key, **_DSS_UK_WINDOW} for key in [*DSS_UK_PV_ONLY.values(), *DSS_UK_EV_ONLY.values()]
}
DSS_US_DATASETS: dict[str, dict] = {
    key: {"dataset_name": key, **_DSS_US_WINDOW} for key in [*DSS_US_PV_ONLY.values(), *DSS_US_EV_ONLY.values()]
}

# ── Single condition: plain FPQR/MLPF, no conformal (Experiment 2's question) ──
MLPF_CONFIG_CLS, MLPF_OVERRIDES = BACKBONES["mlpf"]
CONDITIONS: list[Condition] = [
    Condition(
        name="mlpf_none",
        group="penetration",
        description="MLPF FPQR (simple config), no conformal -- matches backbone_conformal_study.py's own condition",
        overrides=MLPF_OVERRIDES,
        model_cls=MLPF_CONFIG_CLS,
        metric_types=["point", "interval", "quantile"],
    )
]

_SPEC_COMMON: dict = {
    "name": "Penetration Study",
    "output_prefix": "ch6_penetration_study",
    "condition_cls": MLPF_CONFIG_CLS,
    "backbone_cls": MLPF_CONFIG_CLS,
    "conditions": CONDITIONS,
    "cv_val_size": 1,
    "cv_calib_size": 0,
    # num_workers=0: see backbone_conformal_study.py's own fixed_overrides
    # comment -- the default of 8 forks DataLoader workers after MPS/Metal
    # init and was observed hanging indefinitely under this same engine.
    "fixed_overrides": {"early_stop_patience": 100, "num_workers": 0},
    "reference_conditions": {"penetration": "mlpf_none"},
}

# 5 folds, 2-month test, 1-month stride, expanding -> derived train_size = 5
# months (see module docstring for why 6 is unreachable on this record).
SPEC_DSS_EU = ExperimentSpec(
    **{**_SPEC_COMMON, "cv_test_size": 2, "cv_val_size": 1},
    datasets=DSS_EU_DATASETS,
    cv_folds=5,
    cv_stride=1,
    cv_window="expanding",
    hpo_train_months=8,
    hpo_val_months=2,
)
# Weeks-based, identical to backbone_conformal_study.py's own DSS-UK/DSS-US
# spec (see module docstring for why those numbers, not a new derivation).
_DSS_WEEKS_COMMON = {
    **_SPEC_COMMON,
    "cv_split_freq": "weeks",
    "cv_train_size": 26,
    "cv_test_size": 4,
    "cv_val_size": 4,
    "cv_folds": 6,
    "cv_stride": 4,
    "cv_window": "expanding",
    "hpo_train_months": 4,
    "hpo_val_months": 1,
}
SPEC_DSS_UK = ExperimentSpec(**_DSS_WEEKS_COMMON, datasets=DSS_UK_DATASETS)
SPEC_DSS_US = ExperimentSpec(**_DSS_WEEKS_COMMON, datasets=DSS_US_DATASETS)

ENGINE_DSS_EU = ExperimentEngine(SPEC_DSS_EU)
ENGINE_DSS_UK = ExperimentEngine(SPEC_DSS_UK)
ENGINE_DSS_US = ExperimentEngine(SPEC_DSS_US)


def base_config(epochs: int, hpo_trials: int = 50, hpo_epochs: int = 100) -> PipelineConfig:
    """Shared PipelineConfig defaults; dataset/output_dir overridden per run."""
    output_dir = script_dir.parent.parent.parent.parent / "experiments" / "ch7_multitarget" / "penetration_study"
    return PipelineConfig(
        dataset_name=next(iter(DSS_EU_DATASETS)),
        data_path="examples/data",
        output_dir=str(output_dir),
        epochs=epochs,
        num_trials=hpo_trials,
        hpo_epochs=hpo_epochs,
        use_latex=False,
        num_threads=4,
        metrics=["rmse", "mae"],
    )


def run_axis(
    engine: ExperimentEngine, levels: dict[float, str], axis: str, cfg: PipelineConfig, skip_hpo: bool
) -> pd.DataFrame:
    """Run every penetration level of one axis through one engine, tagging the result with its level."""
    tables = []
    for level, dataset_key in levels.items():
        summary = engine.run(base_cfg=cfg, groups=["penetration"], dataset_keys=[dataset_key], skip_hpo=skip_hpo)
        if summary.empty:
            log.warning("axis=%s level=%.1f dataset=%s: empty summary, skipped", axis, level, dataset_key)
            continue
        summary["axis"] = axis
        summary["penetration"] = level
        summary["dataset"] = dataset_key
        tables.append(summary)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def trend(table: pd.DataFrame, metric: str, axis: str) -> None:
    """Print a one-line slope for one metric's mean value against penetration, per axis."""
    sub = table[(table.axis == axis) & (table.metric == metric)] if "metric" in table.columns else pd.DataFrame()
    if sub.empty or sub["penetration"].nunique() < 3:
        return
    value_col = "mean" if "mean" in sub.columns else "value"
    grouped = sub.groupby("penetration")[value_col].mean()
    x, y = grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float)
    slope, _intercept = np.polyfit(x, y, 1)
    log.info(
        "  axis=%s metric=%s  slope=%+.4f per unit penetration  (levels=%s)",
        axis,
        metric,
        slope,
        list(grouped.index),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", nargs="*", default=["ev", "pv"], choices=["ev", "pv"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="short run for a smoke test")
    ap.add_argument("--skip-hpo", dest="skip_hpo", action="store_true", default=True)
    ap.add_argument("--run-hpo", dest="skip_hpo", action="store_false")
    ap.add_argument("--hpo-trials", type=int, default=50)
    ap.add_argument("--hpo-epochs", type=int, default=100)
    args = ap.parse_args()

    configure(level="INFO")
    cfg = base_config(5 if args.quick else args.epochs, hpo_trials=args.hpo_trials, hpo_epochs=args.hpo_epochs)
    if args.quick:
        cfg = dc_replace(cfg, num_trials=1, hpo_epochs=5)
    skip_hpo = args.skip_hpo or args.quick

    tables = []
    if "ev" in args.axis:
        for engine, dataset_levels, label in (
            (ENGINE_DSS_EU, DSS_EU_EV_ONLY, "ev_dss-eu"),
            (ENGINE_DSS_UK, DSS_UK_EV_ONLY, "ev_dss-uk"),
            (ENGINE_DSS_US, DSS_US_EV_ONLY, "ev_dss-us"),
        ):
            levels = {0.0: dataset_levels[0.0], 0.4: dataset_levels[0.4]} if args.quick else dataset_levels
            tables.append(run_axis(engine, levels, label, cfg, skip_hpo))
    if "pv" in args.axis:
        for engine, dataset_levels, label in (
            (ENGINE_DSS_UK, DSS_UK_PV_ONLY, "pv_dss-uk"),
            (ENGINE_DSS_US, DSS_US_PV_ONLY, "pv_dss-us"),
        ):
            levels = {0.0: dataset_levels[0.0], 0.4: dataset_levels[0.4]} if args.quick else dataset_levels
            tables.append(run_axis(engine, levels, label, cfg, skip_hpo))

    full = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    if full.empty:
        raise SystemExit("no axis produced results")
    out = script_dir / "results_penetration_study.csv"
    full.to_csv(out, index=False)
    log.info("wrote %s", out)

    for axis in full["axis"].unique():
        for metric in ("nrmse", "winkler_score"):
            trend(full, metric, axis)


if __name__ == "__main__":
    main()
