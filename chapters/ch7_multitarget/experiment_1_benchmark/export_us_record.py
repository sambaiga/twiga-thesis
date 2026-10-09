"""Export the North American pole-transformer record for the twiga loaders.

The power-flow record is produced in the ``ml-energy-disaggregation`` repository
and carries a step index rather than a timestamp, and no weather column. Both
are needed before ``load_dataset`` can treat it like any other dataset, so this
script attaches them once and writes a parquet into ``examples/data``.

The irradiance column is the same plane-of-array profile that drove the
simulation, so it is the physical driver of the photovoltaic generation in the
record rather than a proxy fitted afterwards.

    uv run python -m chapters.ch7_multitarget.experiment_1_benchmark.export_us_record
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from twiga.core.utils import get_logger

from twiga_thesis.paths import external_data_dir, thesis_data_dir

log = get_logger(__name__)
ml_energy_repo = external_data_dir()
# The full year is preferred over the 90-day run: a record covering one season
# is a poor basis for a forecasting study read across the year.
record_dir = ml_energy_repo / "experiments/lv-multitarget/data"
RECORD = next(
    (
        record_dir / name
        for name in (
            "pole_transformer_pqv_17520_year.parquet",
            "pole_transformer_pqv_4320_d121_loaded.parquet",
        )
        if (record_dir / name).exists()
    ),
    record_dir / "pole_transformer_pqv_4320_d121_loaded.parquet",
)
SOLAR = ml_energy_repo / "resources/smart-ds/profiles/AUS_30.3459_-97.8095_15_180.csv"
OUT = thesis_data_dir() / "DSS-US.parquet"

# Day one of the year at half-hourly resolution. This constant sets two things
# at once, the calendar stamp and the offset into the solar profile, so a value
# that does not match the simulation's own start day both mislabels the dates and
# shifts the irradiance out of phase with the generation it drove. The 90-day
# record began on day 121 and the full year begins on day 1.
START = pd.Timestamp("2019-01-01", tz="UTC")
FREQ = pd.Timedelta(minutes=30)
STEPS_PER_DAY = 48


def build() -> pd.DataFrame:
    """Attach a timestamp and the driving irradiance to the recorded series."""
    data = pd.read_parquet(RECORD)
    data["timestamp"] = START + data["step"].to_numpy() * FREQ

    # The profile is 15-minute plane-of-array irradiance over a full year;
    # take every other value from the run's start day to match the record.
    poa = np.loadtxt(SOLAR)
    per_day = len(poa) // 365
    offset = (START.dayofyear - 1) * per_day
    half_hourly = poa[offset::2][: data["step"].max() + 1]
    data["Ghi"] = half_hourly[data["step"].to_numpy() % len(half_hourly)]

    # No temperature is recorded, so a daily cycle is not invented for it; the
    # US case study therefore has one exogenous driver where the European record
    # has two, which is a difference to report rather than paper over.
    data["utilisation"] = data["s_kva"] / data["kva_rating"]
    return data


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT, index=False)
    peak = data.groupby("trafo")["utilisation"].max().sort_values(ascending=False)
    log.info("wrote %s rows to %s", f"{len(data):,}", OUT)
    log.info("  transformers %d, scenarios %d", data["trafo"].nunique(), data.groupby(["ev_pen", "pv_pen"]).ngroups)
    log.info("  peak utilisation %.2f, median across transformers %.2f", peak.iloc[0], peak.median())
    log.info("  irradiance %.0f to %.0f", data["Ghi"].min(), data["Ghi"].max())


if __name__ == "__main__":
    main()
