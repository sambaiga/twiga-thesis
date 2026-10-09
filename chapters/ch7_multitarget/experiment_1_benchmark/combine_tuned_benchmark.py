r"""Combined Experiment 1 result set: the head-tuned FPQR beside the fixed QR, BNN and Laplace.

The chapter reports FPQR at its head-tuned settings and the other heads at their
fixed settings. This script copies the saved ``metrics.parquet`` and
``predictions.parquet`` of each condition into one experiment root that
``fixed_benchmark_report.py``, ``fixed_benchmark_tables.py`` and
``fixed_benchmark_chapter_figures.py`` read unchanged:

* ``benchmark/fpqr_anchor`` is the head-tuned FPQR (``head_tuned/fpqr_anchor_tuned``);
  the fixed FPQR moves to ``benchmark_fixed/`` for reference;
* every other benchmark condition is the fixed run's;
* on MLVS-PT, the FPQR conformal conditions come from the ``head_tuned_conformal``
  refit (``conformal/fpqr_anchor_refit`` is that refit's native forecast), the
  fixed FPQR's move to ``conformal_fixed/``, and the other heads' are the fixed run's.

Nothing in the source runs is modified.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/combine_tuned_benchmark.py \
        --fixed <experiments>/ch7_multitarget/multitarget_fixed_benchmark \
        --tuned <experiments>/ch7_multitarget/multitarget_fixed_benchmark_v1 \
        --refit <experiments>/ch7_multitarget/tuned_fpqr_refit/fixed_benchmark/multitarget_fixed_benchmark \
        --out <experiments>/ch7_multitarget/multitarget_benchmark_final
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

FILES = ("metrics.parquet", "predictions.parquet")
HEAD = "fpqr_anchor"


def copy_condition(source: Path, target: Path) -> None:
    """Copy one condition's saved metrics and predictions."""
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)


def newest_condition(root: Path, dataset: str, group: str, condition: str) -> Path | None:
    """The newest run's directory for one condition, or ``None``."""
    for run in sorted((root / "runs").glob("*"), reverse=True):
        path = run / "conditions" / dataset / group / condition
        if (path / "metrics.parquet").exists():
            return path
    return None


def combine(fixed: Path, tuned: Path, refit: Path | None, out: Path) -> None:
    """Write the combined result set under ``out/runs``, keeping the fixed run's layout.

    Raises:
        FileNotFoundError: if a dataset of the fixed run has no head-tuned FPQR.
    """
    for run in sorted((fixed / "runs").glob("*")):
        for dataset_dir in sorted((run / "conditions").glob("*")):
            if not (dataset_dir / "benchmark").exists():
                continue
            dataset = dataset_dir.name
            target = out / "runs" / run.name / "conditions" / dataset
            for condition in sorted((dataset_dir / "benchmark").glob("*")):
                group = "benchmark_fixed" if condition.name == HEAD else "benchmark"
                copy_condition(condition, target / group / condition.name)
            source = newest_condition(tuned, dataset, "head_tuned", f"{HEAD}_tuned")
            if source is None:
                raise FileNotFoundError(f"no head-tuned FPQR for {dataset} under {tuned}")
            copy_condition(source, target / "benchmark" / HEAD)
            print(f"{dataset}: FPQR <- {source}")
            if not (dataset_dir / "conformal").exists():
                continue
            for condition in sorted((dataset_dir / "conformal").glob("*")):
                group = "conformal_fixed" if condition.name.startswith(f"{HEAD}_") else "conformal"
                copy_condition(condition, target / group / condition.name)
            refit_name = f"{HEAD}_tuned_refit"
            refit_dir = None if refit is None else newest_condition(refit, dataset, "head_tuned_conformal", refit_name)
            if refit_dir is None:
                print(f"{dataset}: no head_tuned_conformal refit, so no calibrated FPQR rows")
                continue
            for condition in sorted(refit_dir.parent.glob(f"{HEAD}_tuned_*")):
                copy_condition(condition, target / "conformal" / condition.name.replace(f"{HEAD}_tuned_", f"{HEAD}_"))
            print(f"{dataset}: FPQR conformal <- {refit_dir.parent}")


def main() -> None:
    """Build the combined result set from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixed", type=Path, required=True, help="the fixed benchmark's experiment root")
    ap.add_argument("--tuned", type=Path, required=True, help="the head-tuned run's experiment root")
    ap.add_argument("--refit", type=Path, default=None, help="the head_tuned_conformal run's experiment root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    combine(args.fixed, args.tuned, args.refit, args.out)


if __name__ == "__main__":
    main()
