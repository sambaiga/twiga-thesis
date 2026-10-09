r"""Plots for Chapter 6's Experiment 3 (operational_decision.py).

The experiment's point is not forecast accuracy for its own sake but whether
better-calibrated forecasts translate into better operational decisions
(hosting-capacity/thermal exceedance). Two metrics carry that claim directly:
Brier skill score (probabilistic forecast quality, relative to climatology)
and peak relative economic value (the decision-relevant payoff of acting on
the forecast). ``plot_headline`` puts these two side by side, one model per
row, so the "does better calibration turn into better decisions" question is
answered by a single figure rather than left to be inferred from separate
per-metric tables. The remaining metrics (reliability, resolution, event-rate
error, Pr(event) predicted/realised) are diagnostics for *why* the headline
numbers look the way they do, plotted afterward via ``plot_metric``.

Reads every ``<decision>_per_fold.csv`` and ``<decision>_value_curve.csv``
written by either ``operational_decision.py`` (under ``operational_decision/``,
the original, copula-focused MLPF-only source) or ``backbone_conformal_decision.py``
(under ``backbone_conformal_decision/``, the current backbone x conformal-method
source -- pass ``--source backbone_conformal_decision``). Both scripts write
the same ``{decision}_per_fold.csv``/``{decision}_value_curve.csv`` shape (one
row per fold/metric/model, or per fold/model/cost-loss-ratio for the curve),
so this reader and every plot below is source-agnostic. The value-curve file
only exists for runs made after ``curve_sink`` was wired into
``hosting_capacity_rows``/``_operational_rows`` (``forecast_examples.py``) and,
for ``backbone_conformal_decision.py``, into its own ``custom_scorer``; older
per_fold-only runs still produce every other figure, just not
``01_value_curve.pdf``.

Output: ``<source>/figures/00_headline_value_vs_skill.pdf``,
``<source>/figures/01_value_curve.pdf``,
``<source>/figures/<metric_slug>_operational_decision.pdf``

Usage::

    uv run python experiment/ch7_multitarget/plot_operational_decision.py
    uv run python experiment/ch7_multitarget/plot_operational_decision.py \\
        --source backbone_conformal_decision
"""

import argparse
from pathlib import Path
import re

import arviz as az
from matplotlib import pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (registers the "science" style)
import seaborn as sns
from twiga.core.utils import get_logger

az.style.use(["science", "arviz-doc"])
log = get_logger(__name__)

script_dir = Path(__file__).parent
_EXPERIMENTS_ROOT = script_dir.parent.parent.parent.parent / "experiments" / "ch7_multitarget"

DATA_DIR: Path
OUT_DIR: Path


def _latest_run_results(runs_root: Path) -> Path:
    """Most recent ``runs/<run_id>/results/`` under ``runs_root``.

    Both source scripts write each invocation to its own ``runs/<run_id>/``
    (see ``twiga.experiment.run_versioning``) rather than a flat directory a
    later run would silently overwrite. Run IDs sort lexicographically
    (``YYYYMMDD_HHMMSS_<git-hash>``), so the last one alphabetically is also
    the most recent one chronologically.
    """
    run_dirs = sorted((runs_root / "runs").glob("*/results"))
    if not run_dirs:
        raise FileNotFoundError(f"No runs/*/results found under {runs_root}. Run the source script first.")
    return run_dirs[-1]


# Direction shown in each metric's title, not used for sorting -- the bar
# chart shows every model side by side rather than ranking them. Matches
# ch6_analyse_results.py's OPERATIONAL_HIGHER_BETTER/OPERATIONAL_LOWER_BETTER.
HIGHER_BETTER = {"Brier skill score", "Resolution", "Peak relative economic value", "cwe_stratified"}
LOWER_BETTER = {"Brier score", "Reliability", "Event rate error (%)", "rmse", "mae", "winkler_score", "nmpi"}


def _slug(metric: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")


def load_per_fold() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(DATA_DIR.glob("*_per_fold.csv"))]
    if not frames:
        raise FileNotFoundError(f"No *_per_fold.csv found under {DATA_DIR}. Run operational_decision.py first.")
    return pd.concat(frames, ignore_index=True)


def load_value_curve() -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in sorted(DATA_DIR.glob("*_value_curve.csv"))]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def plot_value_curve(data: pd.DataFrame, ratio_range: tuple[float, float] = (0.05, 0.95)) -> sns.axisgrid.FacetGrid:
    """Relative economic value against cost-loss ratio, the diagram Richardson (2000) uses.

    A single "peak value" number collapses the curve to one point and hides
    which users the forecast actually serves; different operators sit at
    different cost-loss ratios (a cheap, reversible curtailment action versus
    an expensive network reinforcement), so the curve, not the peak, is the
    claim this experiment needs to support. Averaged across folds (mean +/- sd
    band) at each ratio, one line per model, faceted by decision.

    ``ratio_range`` trims the plotted ratios away from (0, 1): the cost-loss
    value score's denominator shrinks near both edges, so finite-sample noise
    there can dwarf the interior of the curve on a shared y-axis -- the same
    reason Richardson-style value diagrams are conventionally read over the
    interior of the range rather than at the extremes. The underlying CSV
    still has the full (0.01, 0.99) sweep; only the plot is trimmed.
    """
    sub = data[(data["cost_loss_ratio"] >= ratio_range[0]) & (data["cost_loss_ratio"] <= ratio_range[1])]
    # Clip the y-axis too, not only the x-range: even inside ratio_range a few
    # noisy folds/models can spike far below the informative part of the
    # curve (all curves are bounded above by 1; there is no comparable lower
    # bound, so the display clip is one-sided). This is a display choice
    # only -- the underlying CSV and computed means are untouched.
    y_display_floor = -2.0
    g = sns.relplot(
        data=sub,
        x="cost_loss_ratio",
        y="value",
        hue="model",
        col="decision",
        kind="line",
        errorbar="sd",
        height=3.5,
        aspect=1.2,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("cost-loss ratio  $\\alpha = C/L$", "relative economic value  $V(\\alpha)$")
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flat:
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
        ax.set_xlim(*ratio_range)
        ax.set_ylim(bottom=max(ax.get_ylim()[0], y_display_floor), top=1.05)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False, title="model")
    g.figure.tight_layout()
    return g


def plot_headline(data: pd.DataFrame) -> plt.Figure:
    """Brier skill score and peak relative economic value, side by side, one row per model.

    Skill score answers "is the forecast any better than climatology"; economic
    value answers "does that translate into a better decision outcome." Same
    model order on both axes (sorted by economic value) makes it visually
    direct whether the two move together.
    """
    headline_metrics = ["Brier skill score", "Peak relative economic value"]
    sub = data[data["metric"].isin(headline_metrics)]
    decisions = sorted(sub["decision"].unique())

    fig, axes = plt.subplots(len(decisions), 2, figsize=(7, 2.5 * len(decisions)), squeeze=False)
    for row, decision in enumerate(decisions):
        dsub = sub[sub["decision"] == decision]
        order = (
            dsub[dsub["metric"] == "Peak relative economic value"]
            .groupby("model")["value"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        for col, metric in enumerate(headline_metrics):
            ax = axes[row, col]
            msub = dsub[dsub["metric"] == metric]
            sns.barplot(data=msub, y="model", x="value", order=order, errorbar="sd", color="C0", ax=ax)
            ax.set_title(f"{decision}: {metric}" if col == 0 else metric)
            ax.set_xlabel(metric)
            ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_metric(data: pd.DataFrame, metric: str) -> sns.axisgrid.FacetGrid:
    sub = data[data["metric"] == metric]
    direction = " (higher better)" if metric in HIGHER_BETTER else " (lower better)" if metric in LOWER_BETTER else ""
    g = sns.catplot(
        data=sub,
        y="model",
        x="value",
        col="decision",
        sharex=False,
        margin_titles=True,
        height=2.5,
        aspect=1.4,
        kind="bar",
        errorbar="sd",
        color="C0",
    )
    g.set_axis_labels(metric + direction, "")
    g.set_titles(col_template="{col_name}")
    g.figure.tight_layout()
    return g


def main() -> None:
    global DATA_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        default="operational_decision",
        choices=["operational_decision", "backbone_conformal_decision"],
        help="which script's run tree to read/plot",
    )
    args = parser.parse_args()

    DATA_DIR = _latest_run_results(_EXPERIMENTS_ROOT / args.source)
    OUT_DIR = DATA_DIR / "figures"
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    log.info("reading %s, writing figures to %s", DATA_DIR, OUT_DIR)

    data = load_per_fold()

    fig = plot_headline(data)
    out = OUT_DIR / "00_headline_value_vs_skill.pdf"
    fig.savefig(out, dpi=480, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)

    curve = load_value_curve()
    if curve is not None:
        g = plot_value_curve(curve)
        out = OUT_DIR / "01_value_curve.pdf"
        g.savefig(out, dpi=480, bbox_inches="tight")
        plt.close(g.figure)
        log.info("wrote %s", out)
    else:
        log.warning(
            "no *_value_curve.csv under %s; skipping the value-curve plot (rerun operational_decision.py)", DATA_DIR
        )

    for metric in sorted(data["metric"].unique()):
        g = plot_metric(data, metric)
        out = OUT_DIR / f"{_slug(metric)}_operational_decision.pdf"
        g.savefig(out, dpi=480, bbox_inches="tight")
        plt.close(g.figure)
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
