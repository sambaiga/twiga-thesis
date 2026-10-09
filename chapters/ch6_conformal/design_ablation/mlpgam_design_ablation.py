"""MLPGAM Design Evaluation — Chapter 7, `Chapter-7.tex:481-493` (`chap7:exp-one`).

This is not a new experiment harness. `experiment/ch6_conformal/design_ablation/conditions.py`'s
``ABLATION_SPEC``/``ABLATION_ENGINE`` already implement every group Chapter 7's
Design Evaluation subsection describes except one: training mode, sigma loss,
monitor, Lasso strength, split-vs-residual-fitting CP, the coverage-level
(alpha) comparison, absolute-vs-signed score, and the
`two_stage_batch`/`best_config` reference points the reanalysis in
`Papers/MLPF_CRF (1)/EXPERIMENT_PLAN.md` already settled. The
`training_x_conformal` group there is the `ep_05 x signed-CRC` interaction
test that composability finding needs — it already exists
(`FACTORIAL_CONDITIONS`), not re-derived here.

The one real gap is feature gating: Chapter 7 lists "gating on vs. off
(otherwise identical architecture)" as a Design Evaluation sweep
(`Chapter-7.tex:489`), and no condition anywhere tests it —
`grep -n "gating" experiment/ch6_conformal/design_ablation/conditions.py` returns nothing. `MLPGAMConfig`
already has the fields this needs natively (`use_gate`, `lambda_gate`,
`gate_scale`, `warmup_epochs`, `delta` — `twiga/models/nn/mlpgam_model.py`), so
this only adds three ``Condition``s, following the `gate_none`/`gate_unconstrained`/
`gate_default` pattern originally built for MLPGAF's gating ablation, ported
here to MLPGAM's own gate fields, which is the backbone this chapter actually
uses.

This module imports design_ablation/conditions.py's conditions and dataset config rather
than duplicating them, extends them with the gating group, and adds the
significance-testing and dual-CWE reporting `design_ablation/conditions.py` itself never
did (Part C of the review plan): design-choice decisions run on Winkler score
only, via `significance.py` — never on CWE, which stays a diagnostic.

Usage::

    # Everything, including the new gating group:
    python experiment/ch6_conformal/design_ablation/mlpgam_design_ablation.py

    # Gating only:
    python experiment/ch6_conformal/design_ablation/mlpgam_design_ablation.py --group gating

    # After a run, print the Winkler-score significance table for one group:
    python experiment/ch6_conformal/design_ablation/mlpgam_design_ablation.py \
        --report gating --full-csv <path to *_full.csv>
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # chapter directory (shared.*)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twiga.core.metrics import mcs_ranking, pairwise_dm_table  # noqa: E402
from twiga.core.utils import configure, get_logger  # noqa: E402
from twiga.experiment import Condition, ExperimentEngine  # noqa: E402
from twiga.models.nn import MLPGAMConfig  # noqa: E402
from twiga.utils.run_forecast import PipelineConfig, setup_environment  # noqa: E402

from chapters.ch6_conformal.design_ablation.conditions import (  # noqa: E402
    ABLATION_CONDITIONS,
    ABLATION_SPEC,
    make_conformal_config,
)
from chapters.ch6_conformal.shared.metrics import INTERVAL_METRICS, POINT_METRICS, cwe_comparison  # noqa: E402
from chapters.ch6_conformal.shared.protocol import DATASETS, experiment_output_dir  # noqa: E402

log = get_logger(__name__)

OUTPUT_DIR = f"{experiment_output_dir('design_ablation')}/"

# The best_config reference settings (two_stage_batch training, hybrid-sqrt
# sigma loss, sigma_loss monitor, lasso=1e-6, CRC-90-sign) — gating is tested
# on top of this, matching "an otherwise identical architecture in which
# gating is disabled" (Chapter-7.tex:489), not against an arbitrary baseline.
_GATING_BASE_OVERRIDES = {
    "two_stage": True,
    "sigma_loss_fn": "hybrid_sqrt",
    "monitor": "sigma_loss",
    "lambda_lasso": 1e-6,
}


def _make_gating_condition(name: str, description: str, **gate_overrides: object) -> Condition:
    """A Design Evaluation condition varying only the MLPGAM gate fields.

    Args:
        name: Condition name, unique within the "gating" group.
        description: One-line description, shown in the summary table.
        **gate_overrides: MLPGAMConfig gate fields to override
            (``use_gate``, ``lambda_gate``, ``gate_scale``, ``warmup_epochs``,
            ``delta``). Fields not given keep the config's own defaults.

    Returns:
        A :class:`~twiga.experiment.Condition` in the "gating" group, on the
        CRC-90-sign conformal configuration `best_config` also uses.
    """
    return Condition(
        name=name,
        group="gating",
        description=description,
        overrides={**_GATING_BASE_OVERRIDES, **gate_overrides},
        conformal_config=make_conformal_config(method="residual-fitting", score_type="sign-res", alpha=0.10),
        metric_types=["point", "interval"],
    )


# MLPGAMConfig's own field defaults (mlpgam_model.py): use_gate=False,
# lambda_gate=1e-6, gate_scale=3.0, warmup_epochs=5, delta=0.05.
GATING_CONDITIONS = [
    _make_gating_condition(
        "gate_none",
        "No gating — features pass through unmodified (config default, true gate-absent baseline)",
        use_gate=False,
    ),
    _make_gating_condition(
        "gate_unconstrained",
        "Gating active, no sparsity penalty or warmup",
        use_gate=True,
        lambda_gate=0.0,
        warmup_epochs=0,
    ),
    _make_gating_condition(
        "gate_default",
        "Default gating — lambda_gate=1e-6, gate_scale=3.0, warmup=5 (reference)",
        use_gate=True,
    ),
]

DESIGN_CONDITIONS = [*ABLATION_CONDITIONS, *GATING_CONDITIONS]

DESIGN_SPEC = dc_replace(
    ABLATION_SPEC,
    name="MLPGAM Design Evaluation (Chapter 7)",
    output_prefix="ch7_design_ablation",
    condition_cls=ABLATION_SPEC.condition_cls,
    backbone_cls=MLPGAMConfig,
    conditions=DESIGN_CONDITIONS,
    reference_conditions={**ABLATION_SPEC.reference_conditions, "gating": "gate_default"},
)

DESIGN_ENGINE = ExperimentEngine(DESIGN_SPEC)


def significance_report(full_csv: str | Path, group: str, alpha: float = 0.05) -> dict[str, pd.DataFrame]:
    """Winkler-score significance, point-accuracy significance, and dual CWE, for one group.

    Reads the per-fold results `ExperimentEngine.run` writes (the
    ``*_full.csv`` alongside the summary) and reports exactly what decides a
    design choice in this chapter: a Diebold-Mariano/MCS test on Winkler
    score, never on CWE. CWE (marginal and stratified) is reported alongside
    as the diagnostic it is, not as the basis for the recommendation.

    Args:
        full_csv: Path to the per-fold results CSV (``condition``, ``group``,
            ``winkler_score``, ``mae``, ``cwe``, ``cwe_stratified`` columns).
        group: Which condition group to report (e.g. ``"gating"``,
            ``"training_mode"``, ``"training_x_conformal"``).
        alpha: Significance level for both the DM table and the MCS.

    Returns:
        A dict with keys ``"winkler_dm"``, ``"mae_dm"``, ``"mcs"`` (as a
        one-row frame summarising the included/eliminated sets), and ``"cwe"``.

    Raises:
        ValueError: If ``group`` has fewer than two conditions in the CSV, or
            folds are not aligned 1:1 across conditions (the per-fold arrays
            `significance.py` requires must be in matching order).
    """
    frame = pd.read_csv(full_csv)
    frame = frame[frame["group"] == group]
    conditions = sorted(frame["condition"].unique())
    if len(conditions) < 2:
        raise ValueError(f"group {group!r} has {len(conditions)} condition(s) in {full_csv}; need at least two")

    fold_col = "fold_start" if "fold_start" in frame.columns else "fold"
    pivot_winkler = frame.pivot_table(index=fold_col, columns="condition", values="winkler_score")
    pivot_mae = frame.pivot_table(index=fold_col, columns="condition", values="mae")
    if pivot_winkler.isna().any().any() or pivot_mae.isna().any().any():
        raise ValueError(f"folds are not aligned across all conditions in group {group!r} — check {full_csv}")

    winkler_losses = {c: pivot_winkler[c].to_numpy() for c in conditions}
    mae_losses = {c: pivot_mae[c].to_numpy() for c in conditions}

    return {
        "winkler_dm": pairwise_dm_table(winkler_losses, alpha=alpha),
        "mae_dm": pairwise_dm_table(mae_losses, alpha=alpha),
        "mcs": pd.DataFrame([mcs_ranking(winkler_losses, alpha=alpha).__dict__]),
        "cwe": cwe_comparison(frame, group_cols=["condition"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", default=["all"])
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Do not tune; require a matching fingerprinted HPO cache.",
    )
    parser.add_argument(
        "--hpo-cache-policy",
        choices=["reuse", "refresh", "error-if-missing"],
        default="reuse",
        help="Reuse matching HPO caches, always refresh, or fail when no matching cache exists.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help=(
            "Optuna trials for MLPGAM backbone HPO. Increase this to give TPE enough adaptive "
            "trials after its startup phase. Use --hpo-cache-policy refresh to rerun with a "
            "changed trial budget; cache fingerprints prevent silently reusing incompatible params."
        ),
    )
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--report", type=str, default=None, help="print the significance report for one group and exit")
    parser.add_argument("--full-csv", type=str, default=None, help="per-fold results CSV, required with --report")
    args = parser.parse_args()

    if args.report:
        if not args.full_csv:
            raise SystemExit("--report requires --full-csv")
        report = significance_report(args.full_csv, args.report)
        for key, table in report.items():
            print(f"\n--- {key} ---")
            print(table.to_string(index=False))
        return

    all_groups = sorted({c.group for c in DESIGN_CONDITIONS})
    selected_groups = all_groups if "all" in args.group else args.group
    selected_datasets = list(DATASETS) if "all" in args.dataset else args.dataset

    base_cfg = PipelineConfig(
        output_dir=OUTPUT_DIR,
        out_activation="Identity",
        epochs=args.epochs,
        split_freq="months",
        metrics=POINT_METRICS + INTERVAL_METRICS,
        log_level="INFO",
        log_file=f"{OUTPUT_DIR}run.log",
        use_latex=False,
        num_threads=1,
        num_trials=args.num_trials,
    )
    configure(level=base_cfg.log_level, log_file=base_cfg.log_file)
    setup_environment(base_cfg)

    engine = DESIGN_ENGINE if args.folds is None else ExperimentEngine(dc_replace(DESIGN_SPEC, cv_folds=args.folds))
    engine.run(
        base_cfg=base_cfg,
        groups=selected_groups,
        dataset_keys=selected_datasets,
        skip_hpo=args.skip_hpo,
        hpo_cache_policy=args.hpo_cache_policy,
    )


if __name__ == "__main__":
    main()
