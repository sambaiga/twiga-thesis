"""Build examples/data/DSS-EU.parquet from the PT_DSS_EV_*_PV_*.parquet scenarios.

The nine scenario files are the OpenDSS time-series power-flow output for the
four-feeder Hybrid European MV-LV network model, the non-synthetic test
system of Koirala et al. (2020), "Non-Synthetic European Low Voltage Test
System" (IJEPES 118:105712, doi:10.1016/j.ijepes.2019.105712), simulated via
`load_opendss_data()` in PhD-Thesis-Anthony/src/utils/load_data.py. Each file
is one (EV penetration, PV penetration) scenario, filtered to a single
monitored transformer LV busbar -- there is no per-feeder dimension here,
unlike the earlier multi-feeder record this replaces (kept alongside as
DSS-EU_legacy_multifeeder.parquet).

Voltage is reported in volts (phase-to-neutral, ~230 V nominal); converted to
per-unit here since that is what load_dataset()'s DSS-EU branch and every
downstream metric (EN 50160 compliance, hosting-capacity checks) expects.
Ghi/Temperature are not in the scenario files themselves -- merged in from
MLVS-PT.parquet, the real weather record for the same Madeira location and
timezone, which fully covers the scenario files' date range.

Usage::

    uv run python experiment/ch7_multitarget/build_dss_eu.py
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from twiga.core.utils import get_logger

log = get_logger(__name__)
DATA_DIR = Path(__file__).resolve().parents[3] / "examples" / "data"
NOMINAL_VOLTAGE_V = 230.0
FEEDER = "default"  # single monitored busbar, no per-feeder dimension

SCENARIO_PATTERN = re.compile(r"PT_DSS_EV_([\d.]+)_PV_([\d.]+)\.parquet$")


def build() -> pd.DataFrame:
    """Concatenate the nine EV/PV scenario files into one DSS-EU-shaped frame."""
    weather = pd.read_parquet(DATA_DIR / "MLVS-PT.parquet", columns=["Ghi", "Temperature"])

    frames = []
    for path in sorted(DATA_DIR.glob("PT_DSS_EV_*_PV_*.parquet")):
        match = SCENARIO_PATTERN.search(path.name)
        if not match:
            continue
        ev_pen, pv_pen = float(match.group(1)), float(match.group(2))

        scenario = pd.read_parquet(path, columns=["P(kW)", "Q(kvar)", "V(V)"])
        scenario.index = scenario.index.tz_convert("UTC")
        scenario = scenario.join(weather, how="inner")
        scenario["feeder"] = FEEDER
        scenario["ev_pen"] = ev_pen
        scenario["pv_pen"] = pv_pen
        scenario = scenario.rename(columns={"P(kW)": "p_kw", "Q(kvar)": "q_kvar"})
        scenario["v_lv_pu"] = scenario.pop("V(V)") / NOMINAL_VOLTAGE_V
        frames.append(scenario.reset_index())

    data = pd.concat(frames, ignore_index=True)
    return data[["timestamp", "feeder", "ev_pen", "pv_pen", "p_kw", "q_kvar", "v_lv_pu", "Ghi", "Temperature"]]


def main() -> None:
    data = build()
    out = DATA_DIR / "DSS-EU.parquet"
    data.to_parquet(out, index=False)
    log.info("wrote %s rows to %s", f"{len(data):,}", out)
    log.info("  scenarios: %d", data.groupby(["ev_pen", "pv_pen"]).ngroups)
    log.info("  date range: %s to %s", data["timestamp"].min(), data["timestamp"].max())
    log.info("  v_lv_pu range: %.3f to %.3f", data["v_lv_pu"].min(), data["v_lv_pu"].max())


if __name__ == "__main__":
    main()
