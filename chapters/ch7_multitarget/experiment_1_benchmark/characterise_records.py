"""Choose the lookback and the lags from the records rather than by assumption.

The experiment scripts have used a 96-step lookback, two days, for a 48-step
forecast. That was assumed, not measured. This runs twiga's signal
characterisation on each target of each record and reports what the data
actually asks for: the autoregressive order from the partial autocorrelation
cutoff, the seasonal periods, and the calendar-aligned lags that follow.

    uv run python experiment/ch7_multitarget/characterise_records.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import pandas as pd
from twiga.core.data.characterisation import CharacterisationConfig, SignalCharacteriser
from twiga.core.utils import get_logger
from twiga.utils.data_loading import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
OUT = script_dir / "characterisation.json"
STEPS_PER_DAY = 48


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["DSS-EU:default:0.4:0.4", "DSS-US:0:0.4:0.4"],
    )
    parser.add_argument("--max-rows", type=int, default=17520, help="one year at 30 minutes")
    args = parser.parse_args()

    characteriser = SignalCharacteriser(
        CharacterisationConfig(n_samples_per_day=STEPS_PER_DAY, ami_max_horizons=STEPS_PER_DAY * 2)
    )
    report: dict = {}

    for scenario in args.scenarios:
        data, targets, *_ = load_dataset(scenario, data_path="examples/data")
        report[scenario] = {}
        print(f"\n=== {scenario} ===")
        for target in targets:
            series = pd.Series(data[target].to_numpy()[: args.max_rows]).dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = characteriser.analyse(series)
            temporal = result.temporal
            entry = {
                "ar_order": int(temporal.ar_order),
                "seasonal_periods": [int(p) for p in temporal.seasonal_periods],
                "dominant_period": int(temporal.dominant_period) if temporal.dominant_period else None,
                "suggested_lags": [int(x) for x in temporal.suggested_lags],
                "stationarity": str(result.stationarity.verdict),
                "integration_order": int(result.stationarity.integration_order),
            }
            report[scenario][target] = entry
            print(
                f"  {target:10s} AR order {entry['ar_order']:4d}  dominant period {entry['dominant_period']}"
                f"  seasonal {entry['seasonal_periods'][:4]}"
            )
            print(f"  {'':10s} suggested lags {entry['suggested_lags'][:8]}  ({entry['stationarity']})")

    # The lookback must cover the longest lag any target asks for, since one
    # window feeds all three.
    longest = max(
        (max(e["suggested_lags"]) if e["suggested_lags"] else e["ar_order"])
        for scenario in report.values()
        for e in scenario.values()
    )
    report["implied_lookback"] = {
        "longest_suggested_lag": int(longest),
        "days": round(longest / STEPS_PER_DAY, 2),
        "current_lookback": 96,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nlongest lag any target asks for: {longest} steps ({longest / STEPS_PER_DAY:.2f} days)")
    print("current lookback: 96 steps (2.00 days)")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
