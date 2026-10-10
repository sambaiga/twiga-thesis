r"""Chapter 3: protocol comparison -- single train/test split vs expanding-window backtesting.

Runs the same five simple baselines (``conditions.CONDITIONS``: seasonal-naive, naive, random forest, CatBoost,
linear regression) under each of two backtesting protocols defined in ``protocol.py``:

  single_split   One historical training block, one held-out test block (``cv_folds=1``).
  expanding      ``protocol.EXPANDING_FOLDS``-fold expanding-window backtest, matching the legacy notebook's own
                 ``forecast_len``/``incremental_len`` sizes (see ``protocol.py``'s own docstring).

``analysis.py`` then compares model rankings and point-accuracy numbers across the two protocols -- the
"best practices" question this experiment answers is whether a single split gives a trustworthy accuracy estimate
relative to an expanding-window backtest, not which model wins.

Usage::

    uv run python chapters/ch3_methodology/best_practices/experiment.py --protocol single_split
    uv run python chapters/ch3_methodology/best_practices/experiment.py --protocol expanding
    uv run python chapters/ch3_methodology/best_practices/experiment.py --protocol all --quick
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path

import pandas as pd
from twiga.core.utils import configure, get_logger
from twiga.experiment import ExperimentEngine

from chapters.ch3_methodology.best_practices.conditions import CONDITIONS, POINT_METRICS
from chapters.ch3_methodology.best_practices.protocol import (
    CHAPTER_DATASETS,
    DEFAULT_OUTPUT_ROOT,
    base_config,
    dataset_spec,
    experiment_output_dir,
)

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent

PROTOCOLS = ("single_split", "expanding")


def build_engines(output_root: Path) -> dict[str, dict[str, ExperimentEngine]]:
    """One :class:`~twiga.experiment.ExperimentEngine` per (protocol, dataset) pair."""
    engines: dict[str, dict[str, ExperimentEngine]] = {}
    for protocol in PROTOCOLS:
        engines[protocol] = {}
        for dataset_key in CHAPTER_DATASETS:
            spec = dataset_spec(
                dataset_key,
                protocol,
                name=f"Best Practices: {protocol}",
                output_prefix=f"best_practices_{protocol}",
                condition_cls=CONDITIONS[0].model_cls,
                backbone_cls=CONDITIONS[0].model_cls,
                conditions=CONDITIONS,
                reference_conditions={"baselines": "seasonal_naive"},
            )
            engines[protocol][dataset_key] = ExperimentEngine(spec)
    return engines


def main() -> None:
    """CLI entry point: run the requested protocol(s), write one combined results CSV per protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "all"], default="all")
    parser.add_argument("--dataset", nargs="*", default=list(CHAPTER_DATASETS), choices=[*CHAPTER_DATASETS, "all"])
    parser.add_argument("--quick", action="store_true", help="short smoke run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    configure(level="INFO")
    protocols = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    dataset_keys = list(CHAPTER_DATASETS) if "all" in args.dataset else args.dataset
    engines = build_engines(args.output_root)

    for protocol in protocols:
        output_dir = experiment_output_dir(protocol, args.output_root)
        cfg = base_config(output_dir, metrics=POINT_METRICS)
        if args.quick:
            cfg = dc_replace(cfg, epochs=1, num_trials=1)
        summaries = []
        for dataset_key in dataset_keys:
            summaries.append(
                engines[protocol][dataset_key].run(
                    base_cfg=cfg,
                    groups=["baselines"],
                    dataset_keys=[dataset_key],
                    skip_hpo=True,
                    tracking_uri=None,
                )
            )
        summary = pd.concat(summaries, ignore_index=True)
        summary.insert(0, "protocol", protocol)
        out = output_dir / f"results_{protocol}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        log.info("protocol=%s summary written to %s", protocol, out)


if __name__ == "__main__":
    main()
