"""Path resolution for data this repository does not (and should not) commit.

Two kinds of external data are involved in reproducing the thesis:

- The unpublished MLVS-PT measurement data (see ``data/README.md``): available
  from the author on reasonable request.
- Raw power-flow simulation records produced by the companion
  ``ml-energy-disaggregation`` simulator repository, used by the Chapter 7
  (multi-target) scripts. These are also not redistributed here.

Rather than hardcoding a path to either on one machine, scripts that need them
call :func:`external_data_dir`, which resolves the ``EXTERNAL_DATA_DIR``
environment variable and fails with a clear, actionable error if it is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["external_data_dir", "repo_root", "thesis_data_dir"]


def repo_root() -> Path:
    """Return this repository's root directory."""
    return Path(__file__).resolve().parent.parent


def thesis_data_dir() -> Path:
    """Return the repository's shared ``data/`` directory.

    See ``data/README.md`` for what lives here and how to obtain it (most of
    it is not committed; this just resolves the path robustly regardless of
    which chapter script is calling it).
    """
    return repo_root() / "data"


def external_data_dir() -> Path:
    """Return the root of a local ``ml-energy-disaggregation`` checkout.

    Chapter 7 (multi-target) scripts derive their specific record paths from
    this root, e.g. ``external_data_dir() / "experiments/lv-multitarget/data"``.

    Returns:
        The path from the ``EXTERNAL_DATA_DIR`` environment variable.

    Raises:
        FileNotFoundError: If the environment variable is unset, or set to a
            path that does not exist. The message explains what this
            directory should contain and how to obtain it.
    """
    value = os.environ.get("EXTERNAL_DATA_DIR")
    if not value:
        raise FileNotFoundError(
            "EXTERNAL_DATA_DIR is not set. This chapter needs raw power-flow "
            "simulation records produced by the companion ml-energy-disaggregation "
            "simulator repository; these are not redistributed in this repository. "
            "See data/README.md for how to obtain them, then set EXTERNAL_DATA_DIR "
            "to the root of a local ml-energy-disaggregation checkout "
            "(e.g. `export EXTERNAL_DATA_DIR=/path/to/ml-energy-disaggregation`)."
        )
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"EXTERNAL_DATA_DIR is set to {path}, but that path does not exist.")
    return path
