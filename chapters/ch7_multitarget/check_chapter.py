"""Check that the chapter quotes the numbers the analyses actually produced.

Numbers were transcribed by hand from console output into the chapter prose, and
drifted twice: once when the distance-correlation estimator moved into twiga and
reseeded its subsample, and once when the North American record was extended from
seven days to ninety. Both times the text kept a stale value that read as
plausible. This script closes that loop.

Run after any edit to the chapter or re-run of the analysis:

    uv run python experiment/ch7_multitarget/check_chapter.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from twiga.core.utils import get_logger

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
CHAPTER = script_dir.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Chapters" / "Chapter-6-mult.tex"

# Analyses whose emitted values the chapter quotes. Each is optional: a source
# that has not been run yet is reported as skipped rather than failing the run,
# so the check is usable while the chapter is still being drafted.
SOURCES: dict[str, Path] = {
    "dependence": script_dir / "dependence_cost_reported.json",
    "residual": script_dir / "residual_dependence.json",
}

# Each entry is the source, the phrase the chapter uses and the key whose value
# must appear in it. The phrase is matched loosely so rewording does not break
# the check; only the number is asserted.
CLAIMS: tuple[tuple[str, str, str, str], ...] = (
    ("dependence", "median European error", r"median of \$(-?\d+)\\,\\%\$ on\s+the European record", "eu_median_err"),
    ("dependence", "median US error", r"and \$(-?\d+)\\,\\%\$ on the pole transformers", "us_median_err"),
    ("dependence", "worst European error", r"reaching \$(-?\d+)\\,\\%\$ and", "eu_worst_err"),
    ("dependence", "worst US error", r"reaching \$-?\d+\\,\\%\$ and \$(-?\d+)\\,\\%\$", "us_worst_err"),
    ("dependence", "conditional median", r"leaves a median of \$(-?\d+)\\,\\%\$", "eu_cond_median"),
    ("dependence", "bootstrap estimate", r"block bootstrap gives \$(-?\d+)\\,\\%\$", "eu_boot_err"),
    ("dependence", "bootstrap lower", r"interval of \$\[(-?\d+),\s*-?\d+\]", "eu_boot_lo"),
    ("dependence", "bootstrap upper", r"interval of \$\[-?\d+,\s*(-?\d+)\]", "eu_boot_hi"),
    ("dependence", "near-zero European cells", r"In the \$(\d+)\$ European", "eu_near_zero_cells"),
    ("dependence", "near-zero US cells", r"and \$(\d+)\$ transformer scenarios", "us_near_zero_cells"),
    ("dependence", "European distance correlation", r"distance correlation is \$([\d.]+)\$", "eu_near_zero_dcor"),
    (
        "dependence",
        "US distance correlation",
        r"distance correlation is \$[\d.]+\$ and \$([\d.]+)\$",
        "us_near_zero_dcor",
    ),
    ("dependence", "worst-cell joint", r"recorded probability is \$([\d.]+)\\,\\%\$", "eu_worst_joint_pct"),
    ("dependence", "worst-cell independent", r"against \$([\d.]+)\\,\\%\$ under independence", "eu_worst_indep_pct"),
    ("residual", "residual mean correlation", r"target pairs of \$([\d.]+)\$, against", "residual_mean_abs"),
    ("residual", "raw mean correlation", r"against \$([\d.]+)\$ in the record", "raw_mean_abs"),
    ("residual", "raw P-Q", r"from \$\+([\d.]+)\$ in the record to", "raw.pq"),
    ("residual", "residual P-Q", r"in the record to \$(-[\d.]+)\$ in the residuals", "residual_joint.pq"),
    ("residual", "raw Q-V", r"strengthens from \$\+([\d.]+)\$ to", "raw.qv"),
    ("residual", "residual Q-V", r"strengthens from \$\+[\d.]+\$ to \$\+([\d.]+)\$", "residual_joint.qv"),
    ("residual", "asserted P-V", r"correlations of \$(-[\d.]+)\$ between active power and voltage", "asserted.pv"),
    (
        "residual",
        "asserted Q-V",
        r"and \$\+([\d.]+)\$ between reactive power and voltage",
        "asserted.qv",
    ),
    (
        "residual",
        "residual P-V restated",
        r"against residual values of \$(-[\d.]+)\$ and",
        "residual_joint.pv",
    ),
    (
        "residual",
        "residual Q-V restated",
        r"against residual values of \$-[\d.]+\$ and \$\+([\d.]+)\$",
        "residual_joint.qv",
    ),
    ("residual", "asserted P-Q", r"head asserts \$(-[\d.]+)\$ between active", "asserted.pq"),
    (
        "residual",
        "residual P-Q restated",
        r"where the residual value is \$(-[\d.]+)\$",
        "residual_joint.pq",
    ),
    ("residual", "asserted P-Q day min", r"ranging from \$(-[\d.]+)\$ to", "asserted_pq_min"),
    ("residual", "asserted P-Q day max", r"ranging from \$-[\d.]+\$ to \$\+([\d.]+)\$", "asserted_pq_max"),
)


def lookup(values: dict, key: str) -> float:
    """Read a possibly nested key, written with dots, from an analysis report."""
    node = values
    for part in key.split("."):
        node = node[part]
    return float(node)


def main() -> int:
    available = {}
    for name, path in SOURCES.items():
        if path.exists():
            available[name] = json.loads(path.read_text())
        else:
            log.warning("  skipped %s: no report at %s", name, path)
    if not available:
        log.warning("no analysis reports found; run dependence_cost.py and residual_dependence.py first")
        return 2
    tex = CHAPTER.read_text(encoding="utf-8")

    failures, missing, checked = [], [], 0
    for source, name, pattern, key in CLAIMS:
        if source not in available:
            continue
        checked += 1
        match = re.search(pattern, tex)
        if match is None:
            missing.append(name)
            continue
        found, expected = float(match.group(1)), lookup(available[source], key)
        if abs(found - expected) > 1e-9:
            failures.append(f"  {name}: chapter says {match.group(1)}, analysis gives {expected:g}")

    for name in missing:
        print(f"  NOT FOUND in the chapter: {name}")
    for line in failures:
        print(line)
    if failures or missing:
        print(f"\n{len(failures)} mismatched, {len(missing)} not found")
        return 1
    print(f"all {checked} reported numbers match the analyses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
