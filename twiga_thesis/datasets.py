"""Dataset access for the two primary net-load datasets used across chapters.

See ``data/README.md`` for the full picture, including the Chapter-7-only
power-flow records, which are handled separately in ``twiga_thesis.paths``.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

import pandas as pd
from twiga.core.utils import get_logger

from twiga_thesis.paths import thesis_data_dir

__all__ = ["load_mlvs_pt", "fetch_sps_uk_raw", "build_sps_uk_parquet"]

log = get_logger(__name__)

SPS_UK_DOI = "10.5281/zenodo.5500457"
# This is the thesis's "SPS-UK" dataset (Chapter 2, Table 2.1); twiga's own
# data loader (twiga/utils/data_loading.py, dataset key "MLVS-UK") expects it
# as MLVS-UK.parquet with columns timestamp/NetLoad(MW)/Ghi/Temperature
# (and PV(MW) for the PV-MLVS-UK variant) -- the thesis name and the code's
# dataset key differ, which is why this was initially missed.
SPS_UK_FILES = (
    "weather_train_set4.csv",
    "pv_train_set4.csv",
    "pv_test_set4.csv",
    "demand_test_set4.csv",
    "demand_train_set4.csv",
)
SPS_UK_BASE_URL = "https://zenodo.org/api/records/5500457/files"


def fetch_sps_uk_raw(destination_dir: Path | None = None, force: bool = False) -> Path:
    """Download the public SPS-UK raw CSVs from Zenodo if not already present.

    Args:
        destination_dir: Directory to save the raw CSVs into. Defaults to
            ``data/SPS-UK_raw/`` under the repository root.
        force: Re-download even if a file already exists.

    Returns:
        The directory containing the downloaded (or already-present) CSVs.
    """
    destination_dir = destination_dir or (thesis_data_dir() / "SPS-UK_raw")
    destination_dir.mkdir(parents=True, exist_ok=True)

    for name in SPS_UK_FILES:
        target = destination_dir / name
        if target.exists() and not force:
            log.info("%s already present at %s", name, target)
            continue
        url = f"{SPS_UK_BASE_URL}/{name}/content"
        log.info("downloading %s (DOI %s) to %s", name, SPS_UK_DOI, target)
        urllib.request.urlretrieve(url, target)  # noqa: S310  # nosec

    return destination_dir


def build_sps_uk_parquet(raw_dir: Path | None = None, destination: Path | None = None) -> Path:
    """Assemble the raw SPS-UK CSVs into the parquet ``load_dataset`` expects.

    Follows Chapter 2's own description of the dataset (the only documented
    ground truth available): load demand and PV generation are recorded in
    MW at 30-minute resolution; net load is demand minus PV generation;
    weather (solar irradiance and temperature at six nearby stations,
    60-minute resolution) is linearly interpolated to the 30-minute
    resolution of PV and demand.

    One assumption is NOT specified in the thesis text and is made here
    explicitly rather than silently: the six weather stations' solar
    irradiance and temperature readings are averaged into single ``Ghi`` and
    ``Temperature`` columns, since ``load_dataset`` expects exactly one of
    each. **This averaging has not been verified against the parquet file
    actually used to produce the thesis's results** (that file could not be
    located in this project for comparison) -- treat the output of this
    function as an unverified reconstruction until checked against it.

    Args:
        raw_dir: Directory containing the raw CSVs (see
            :func:`fetch_sps_uk_raw`). Defaults to ``data/SPS-UK_raw/``.
        destination: Where to write the assembled parquet. Defaults to
            ``data/MLVS-UK.parquet`` (the filename ``load_dataset`` expects
            for its ``"MLVS-UK"``/``"PV-MLVS-UK"`` dataset keys).

    Returns:
        Path to the written parquet file.

    Raises:
        FileNotFoundError: If the raw CSVs are not present at ``raw_dir``.
    """
    raw_dir = raw_dir or (thesis_data_dir() / "SPS-UK_raw")
    destination = destination or (thesis_data_dir() / "MLVS-UK.parquet")

    missing = [name for name in SPS_UK_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing raw SPS-UK files in {raw_dir}: {missing}. Run fetch_sps_uk_raw() first.")

    weather = pd.read_csv(raw_dir / "weather_train_set4.csv", parse_dates=["datetime"])
    pv = pd.concat(
        [pd.read_csv(raw_dir / f"pv_{split}_set4.csv", parse_dates=["datetime"]) for split in ("train", "test")],
        ignore_index=True,
    ).sort_values("datetime")
    demand = pd.concat(
        [pd.read_csv(raw_dir / f"demand_{split}_set4.csv", parse_dates=["datetime"]) for split in ("train", "test")],
        ignore_index=True,
    ).sort_values("datetime")

    temp_cols = [c for c in weather.columns if c.startswith("temp_location")]
    solar_cols = [c for c in weather.columns if c.startswith("solar_location")]
    weather = weather.set_index("datetime")
    weather_30min = weather[temp_cols + solar_cols].resample("30min").interpolate("linear")
    weather_30min["Temperature"] = weather_30min[temp_cols].mean(axis=1)
    weather_30min["Ghi"] = weather_30min[solar_cols].mean(axis=1)
    weather_30min = weather_30min[["Temperature", "Ghi"]].reset_index()

    pv_demand = pd.merge(
        pv.rename(columns={"pv_power_mw": "PV(MW)"})[["datetime", "PV(MW)"]],
        demand.rename(columns={"demand_MW": "demand_MW"})[["datetime", "demand_MW"]],
        on="datetime",
        how="inner",
    )
    pv_demand["NetLoad(MW)"] = pv_demand["demand_MW"] - pv_demand["PV(MW)"]

    merged = pd.merge(pv_demand, weather_30min, on="datetime", how="inner")
    merged = merged.rename(columns={"datetime": "timestamp"})
    merged = merged[["timestamp", "NetLoad(MW)", "PV(MW)", "Ghi", "Temperature"]]

    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(destination, index=False)
    log.warning(
        "wrote %s -- UNVERIFIED reconstruction (weather station averaging is an undocumented "
        "assumption; see build_sps_uk_parquet's docstring)",
        destination,
    )
    return destination


def load_mlvs_pt(path: Path | None = None) -> Path:
    """Return the path to the MLVS-PT dataset, with a clear error if missing.

    MLVS-PT is an unpublished project dataset with no DOI, available from the
    author on reasonable request (see ``data/README.md``). This function does
    not download anything; it only resolves and validates the expected path.

    Args:
        path: Where to look for the parquet file. Defaults to
            ``data/MLVS-PT.parquet`` under the repository root.

    Returns:
        The validated path to the dataset.

    Raises:
        FileNotFoundError: If the dataset is not present at the expected
            location.
    """
    path = path or (thesis_data_dir() / "MLVS-PT.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"MLVS-PT not found at {path}. This is an unpublished dataset, available from the "
            "author on reasonable request -- see data/README.md. Once obtained, place it at "
            f"{path}."
        )
    return path
