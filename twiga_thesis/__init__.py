"""Shared, chapter-agnostic utilities for reproducing the thesis's results and figures.

Chapter-specific experiment, analysis, and figure code lives under ``chapters/``, organized by
final thesis chapter number. This package holds only what multiple chapters share: path
resolution for data this repository does not commit (``paths``), dataset access helpers
(``datasets``), and the thesis's matplotlib figure style (``plotting``).
"""

from twiga_thesis.paths import external_data_dir, repo_root, thesis_data_dir

__all__ = ["external_data_dir", "repo_root", "thesis_data_dir"]
