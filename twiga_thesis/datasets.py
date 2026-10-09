"""Dataset access for the two primary net-load datasets used across chapters.

See ``data/README.md`` for the full picture, including the Chapter-7-only
power-flow records, which are handled separately in ``twiga_thesis.paths``.
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

from twiga.core.utils import get_logger

from twiga_thesis.paths import thesis_data_dir

__all__ = ["load_mlvs_pt", "fetch_sps_uk_raw"]

log = get_logger(__name__)

SPS_UK_DOI = "10.5281/zenodo.5500457"
# Confirmed via the Zenodo API (api/records/5500457): the record holds six raw
# CSVs (weather/PV/demand, each split train/test), not a single assembled
# file. There is no verified script in this project that merges them into the
# single net-load parquet `load_dataset` expects, so this function only
# fetches the raw files; assembling them into that schema is a documented gap
# (see data/README.md), not something this function silently papers over.
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
