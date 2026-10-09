"""Analysis of the fixed-parameter decision sweep: per scenario, then across the penetration grid.

Reads every scenario of :mod:`fixed_decision_sweep` (the newest run that holds
it) and writes, under ``<experiment-root>/analysis``:

Per scenario (``analysis/<scenario>/``), following the Chapter 6 benchmark
report's conventions (fold-level losses, DM with Benjamini-Hochberg adjustment
and the 95% MCS through :mod:`experiment.common.result_analysis`, a
``metric_table`` ranking, a ``conclusion.md``):

* ``probability_<event>.csv``: Brier, fair Brier, Brier skill, reliability and
  resolution for every head that issues a probability, with the DM/MCS verdict
  on fold-level Brier;
* ``warning_<event>.csv``: relative value, Peirce skill, hit rate and false-alarm
  ratio of the warnings at each cost-loss ratio, for every head including the
  conformal regions, with the DM/MCS verdict on fold-level expense;
* ``dependence.csv``: what the Schaake-shuffle copula changes against independent
  margins, per head and event;
* ``rates.csv``: realised and reference event rates (degenerate events, never or
  always occurring, are flagged: skill and value are undefined for them);
* figures: relative economic value curves per event.

Across scenarios (``analysis/``):

* ``sweep_summary.csv``: one row per scenario, condition and event;
* ``wilcoxon_<event>.csv``: a Wilcoxon signed-rank test across scenarios of each
  condition's Brier score against the reference (FPQR with the copula), so the
  conclusion does not rest on any one scenario;
* ``skill_grid_<event>.pdf``: Brier skill over EV x PV per network and head.

Usage::

    uv run python experiment/ch7_multitarget/experiment_3_decision/fixed_decision_report.py \
        --experiment-root <output-root>/fixed_benchmark/multitarget_fixed_decision_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parents[1])]

from experiment.common.result_analysis import dm_and_mcs, metric_table  # noqa: E402
import fixed_decision_sweep as sweep  # noqa: E402
from thesis_figures import style as S, tokens as T  # noqa: E402

HEADS = {"fpqr_anchor": "FPQR", "qr": "QR", "bnn": "BNN", "laplace": "Laplace"}
# CQR variants as "CQR (<score>, <coverage>)", e.g. quantile_unscaled_marginal -> "CQR (unscaled, marginal)".
METHODS = {
    "none": "",
    "max_norm": " + max-norm",
    **{
        name: " + CQR ({}, {})".format(*name.removeprefix("quantile_").split("_"))
        for name in sweep.fb.QUANTILE_CONFORMAL
    },
}
COLOUR = {"fpqr_anchor": T.PALETTE[2], "qr": T.PALETTE[0], "bnn": T.PALETTE[4], "laplace": T.PALETTE[5]}
LINESTYLE = {"none": "-", "max_norm": "--", **dict.fromkeys(sweep.fb.QUANTILE_CONFORMAL, ":")}
# Primary route to an event per condition: the copula for quantile and Laplace
# heads, the BNN's own paths, the region for max-norm. "independent" is the ablation.
PRIMARY = {"schaake", "native", "region"}
REFERENCE = "FPQR"
DECISION_RATIO = 0.1
MIN_FOLDS_FOR_DM = 3
PROBABILITY_METRICS = {
    "brier_score": "lower",
    "brier_skill_score": "higher",
    "reliability": "lower",
    "resolution": "higher",
}
WARNING_METRICS = {"expense": "lower", "relative_value": "higher", "peirce_skill_score": "higher"}
NETWORK_LABEL = {"DSS-UK:weak": "DSS-UK weak", "DSS-UK:stiff": "DSS-UK stiff", "DSS-US:8": "DSS-US rank 8"}
SKILL_CMAP = LinearSegmentedColormap.from_list("skill", [T.DANGER, T.WHITE, T.PRIMARY])


# ── Loading ───────────────────────────────────────────────────────────────────


def newest_scenario_dirs(experiment_root: Path) -> dict[str, Path]:
    """Each scenario's condition directory, from the newest run that holds it."""
    found: dict[str, Path] = {}
    for run in sorted((experiment_root / "runs").glob("*"), reverse=True):
        for scenario_dir in sorted((run / "conditions").glob("*")):
            if any(scenario_dir.glob("decision/*/metrics.parquet")):
                found.setdefault(scenario_dir.name, scenario_dir)
    return found


def label(condition: str, dependence: str) -> str:
    """Display label of a condition and its route to the event, e.g. ``FPQR + CQR`` or ``QR (indep.)``."""
    head, method = next((h, condition[len(h) + 1 :]) for h in HEADS if condition.startswith(f"{h}_"))
    text = f"{HEADS[head]}{METHODS.get(method, ' + ' + method)}"
    return f"{text} (indep.)" if dependence == "independent" else text


def load_scenario(scenario_dir: Path) -> pd.DataFrame:
    """Every condition's metric rows for one scenario, with a display ``series`` label."""
    frames = [pd.read_parquet(path) for path in sorted(scenario_dir.glob("decision/*/metrics.parquet"))]
    rows = pd.concat(frames, ignore_index=True)
    rows["dependence"] = rows["dependence"].fillna("")
    rows["series"] = [label(c, d) for c, d in zip(rows["condition"], rows["dependence"], strict=True)]
    return rows


def scenario_fields(key: str) -> dict[str, object]:
    """Network, EV and PV of a scenario key such as ``DSS-UK:weak:0.4:0.4``."""
    record, feeder, ev, pv = key.split(":")
    return {"network": f"{record}:{feeder}", "ev": float(ev), "pv": float(pv)}


def fold_frame(rows: pd.DataFrame, tier: str, event: str, ratio: float | None = None) -> pd.DataFrame:
    """One row per (series, fold) with each metric as a column, for the primary routes."""
    scoped = rows[(rows["tier"] == tier) & (rows["event"] == event) & rows["dependence"].isin(PRIMARY)]
    if ratio is not None:
        scoped = scoped[np.isclose(scoped["cost_loss"], ratio)]
    wide = scoped.pivot_table(index=["series", "Folds"], columns="metric", values="value", aggfunc="mean")
    return wide.reset_index().rename(columns={"series": "condition"})


# ── Per scenario ──────────────────────────────────────────────────────────────


def verdict(dm: pd.DataFrame, model_set, table: pd.DataFrame, metric: str, lower: bool) -> str:
    """One line: the best series on ``metric``, its standing against the reference, and the MCS."""
    # metric_table reports each metric's fold mean under the metric's own name.
    means = table.set_index("condition")[metric] if metric in table else pd.Series(dtype=float)
    if means.empty:
        return f"no {metric} to compare"
    best = means.idxmin() if lower else means.idxmax()
    included = ", ".join(sorted(map(str, model_set.included)))
    if best == REFERENCE or REFERENCE not in means.index:
        return f"**{best}** has the best mean `{metric}`; MCS: {included}."
    pair = dm[
        ((dm.condition_a == best) & (dm.condition_b == REFERENCE))
        | ((dm.condition_a == REFERENCE) & (dm.condition_b == best))
    ]
    status = "not significantly different from"
    if not pair.empty and bool(pair["significant"].iloc[0]):
        status = "significantly better than"
    p_adj = float(pair["p_adjusted"].iloc[0]) if not pair.empty else float("nan")
    return (
        f"**{best}** has the best mean `{metric}` and is {status} **{REFERENCE}** "
        f"(DM p_adj={p_adj:.3f}); MCS: {included}."
    )


def compare(frame: pd.DataFrame, metrics: dict[str, str], decide_on: str) -> tuple[pd.DataFrame, str]:
    """``metric_table`` ranking plus the DM/MCS verdict on ``decide_on``."""
    present = [m for m in metrics if m in frame.columns and frame[m].notna().any()]
    if frame["condition"].nunique() < 2 or decide_on not in present:
        return pd.DataFrame(), "fewer than two comparable conditions"
    table = metric_table(
        frame,
        metric_cols=present,
        lower_better={m for m in present if metrics[m] == "lower"},
        higher_better={m for m in present if metrics[m] == "higher"},
        target_values={},
    )
    usable = frame.dropna(subset=[decide_on])
    # Two fold differences cannot support the DM variance estimate (p collapses to 0).
    if usable["condition"].nunique() < 2 or usable["Folds"].nunique() < MIN_FOLDS_FOR_DM:
        return table, f"too few folds for a DM test on `{decide_on}`"
    dm, _, model_set = dm_and_mcs(usable, metric=decide_on)
    return table, verdict(dm, model_set, table, decide_on, metrics[decide_on] == "lower")


def event_rates(rows: pd.DataFrame) -> pd.DataFrame:
    """Realised and reference rates per event (fold mean), with a degeneracy flag."""
    rates = rows[rows["tier"] == "validity"].groupby(["event", "metric"])["value"].mean().unstack()
    rates["degenerate"] = (rates["realised_rate"] <= 0.0) | (rates["realised_rate"] >= 1.0)
    return rates.reset_index()


def dependence_effect(rows: pd.DataFrame) -> pd.DataFrame:
    """Brier score and skill with the copula against independent margins, per condition and event."""
    scoped = rows[(rows["tier"] == "probability") & rows["dependence"].isin(["schaake", "independent"])]
    scoped = scoped[scoped["metric"].isin(["brier_score", "brier_skill_score"])]
    table = scoped.groupby(["condition", "event", "metric", "dependence"])["value"].mean().unstack("dependence")
    table["copula_minus_independent"] = table["schaake"] - table["independent"]
    return table.reset_index()


def plot_value_curves(rows: pd.DataFrame, event: str, out: Path) -> None:
    """Relative economic value against cost-loss ratio for each probability head (fold mean)."""
    curves = rows[(rows["tier"] == "value_curve") & (rows["event"] == event)]
    if curves.empty or not np.isfinite(curves["value"]).any():
        return  # a degenerate event has no defined value at any ratio
    fig, ax = plt.subplots(figsize=S.panel(0.6, aspect=0.7))
    for (condition, _), group in curves.groupby(["condition", "series"]):
        head = next(h for h in HEADS if condition.startswith(f"{h}_"))
        method = condition[len(head) + 1 :]
        mean = group.groupby("cost_loss")["value"].mean()
        ax.plot(
            mean.index,
            mean.to_numpy(),
            color=COLOUR[head],
            ls=LINESTYLE.get(method, "-"),
            lw=1.3,
            label=label(condition, "schaake"),
        )
    ax.axhline(0.0, color=T.TEXT_MUTED, lw=0.7)
    ax.set_xlabel("Cost-loss ratio $C/L$")
    ax.set_ylabel("Relative economic value")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 0.99)
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.5, 1.0)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"value_curve_{event}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_rows(key: str, event: str, prob: pd.DataFrame, warn: pd.DataFrame, degenerate: bool) -> list[dict]:
    """One row per series for the cross-scenario summary: mean Brier, skill and value at the decision ratio."""
    # Skill is undefined for every fold of a degenerate event, and the pivot then drops its column.
    columns = ["brier_score", "brier_skill_score"]
    means = None
    if not prob.empty:
        means = prob.reindex(columns=["condition", *columns]).groupby("condition")[columns].mean()
    values = warn.groupby("condition")["relative_value"].mean() if "relative_value" in warn else pd.Series()
    series_names = set(prob.get("condition", pd.Series(dtype=str))) | set(warn.get("condition", pd.Series(dtype=str)))
    rows = []
    for series in sorted(series_names):
        scored = means is not None and series in means.index
        rows.append(
            {
                "scenario": key,
                **scenario_fields(key),
                "event": event,
                "series": series,
                "brier_score": float(means.loc[series, "brier_score"]) if scored else np.nan,
                "brier_skill_score": float(means.loc[series, "brier_skill_score"]) if scored else np.nan,
                "relative_value": float(values.get(series, np.nan)),
                "degenerate": degenerate,
            }
        )
    return rows


def report_scenario(key: str, scenario_dir: Path, out: Path) -> pd.DataFrame:
    """Tables, tests, figures and conclusion for one scenario; returns its summary rows."""
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    rows = load_scenario(scenario_dir)
    rates = event_rates(rows)
    rates.to_csv(out / "rates.csv", index=False)
    dependence_effect(rows).to_csv(out / "dependence.csv", index=False)
    degenerate = set(rates.loc[rates["degenerate"], "event"])
    lines = [f"# Decision experiment, {key} ({scenario_dir.parent.parent.name})", ""]
    lines += ["Fold-level losses; DM with Benjamini-Hochberg adjustment; MCS at alpha=0.05.", ""]
    lines += ["## Event rates", "", rates.round(4).to_markdown(index=False), ""]
    summary = []
    for event in sorted(rows.loc[rows["tier"] == "probability", "event"].unique()):
        lines.append(f"## {event}" + (" (degenerate: skill and value undefined)" if event in degenerate else ""))
        lines.append("")
        prob = fold_frame(rows, "probability", event)
        table, text = compare(prob, PROBABILITY_METRICS, "brier_score")
        if not table.empty:
            table.to_csv(out / f"probability_{event}.csv", index=False)
            lines += ["Probability tier:", "", table.round(4).to_markdown(index=False), "", f"- {text}", ""]
        warn = fold_frame(rows, "warning", event, DECISION_RATIO)
        table_w, text_w = compare(warn, WARNING_METRICS, "expense")
        if not table_w.empty:
            table_w.to_csv(out / f"warning_{event}.csv", index=False)
            lines += [f"Warning tier at C/L = {DECISION_RATIO}:", "", table_w.round(4).to_markdown(index=False), ""]
            lines += [f"- {text_w}", ""]
        plot_value_curves(rows, event, figures)
        summary += summary_rows(key, event, prob, warn, event in degenerate)
    (out / "conclusion.md").write_text("\n".join(lines) + "\n")
    return pd.DataFrame(summary)


# ── Across scenarios ──────────────────────────────────────────────────────────


def wilcoxon_table(summary: pd.DataFrame, event: str) -> pd.DataFrame:
    """Each series' Brier score against the reference across scenarios (Wilcoxon signed-rank, BH-adjusted)."""
    scoped = summary[(summary["event"] == event) & ~summary["degenerate"]]
    wide = scoped.pivot_table(index="scenario", columns="series", values="brier_score")
    if REFERENCE not in wide:
        return pd.DataFrame()
    rows = []
    for series in wide.columns.drop(REFERENCE):
        pair = wide[[REFERENCE, series]].dropna()
        if len(pair) < 2:
            continue
        diff = pair[series] - pair[REFERENCE]
        p = wilcoxon(diff, zero_method="zsplit").pvalue if np.any(diff != 0) else 1.0
        rows.append(
            {
                "series": series,
                "scenarios": len(pair),
                "median_brier_difference": float(diff.median()),
                "share_better_than_reference": float((diff < 0).mean()),
                "p_value": float(p),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_adjusted"] = multipletests(table["p_value"], method="fdr_bh")[1]
        table["significant"] = table["p_adjusted"] < 0.05
    return table


def plot_skill_grid(summary: pd.DataFrame, event: str, out: Path) -> None:
    """Brier skill over EV x PV, rows the heads (with the copula), columns the networks."""
    scoped = summary[(summary["event"] == event) & summary["series"].isin(HEADS.values())]
    if scoped.empty:
        return
    networks = [n for n in NETWORK_LABEL if n in set(scoped["network"])]
    heads = [h for h in HEADS.values() if h in set(scoped["series"])]
    levels = sweep.LEVELS
    fig, axes = plt.subplots(len(heads), len(networks), figsize=S.panel(1.0, aspect=0.32 * len(heads)), squeeze=False)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    for i, head in enumerate(heads):
        for j, network in enumerate(networks):
            ax = axes[i, j]
            cells = scoped[(scoped["series"] == head) & (scoped["network"] == network)]
            grid = np.full((len(levels), len(levels)), np.nan)
            for _, row in cells.iterrows():
                if not row["degenerate"]:
                    grid[levels.index(row["ev"]), levels.index(row["pv"])] = row["brier_skill_score"]
            ax.imshow(np.clip(grid, -1.0, 1.0), cmap=SKILL_CMAP, norm=norm, origin="lower")
            for (r, c), value in np.ndenumerate(grid):
                if np.isfinite(value):
                    ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=6, color=T.TEXT_DARK)
            ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels], fontsize=7)
            ax.set_yticks(range(len(levels)), [f"{v:g}" for v in levels], fontsize=7)
            ax.tick_params(which="minor", left=False, bottom=False, top=False, right=False)
            ax.tick_params(which="major", length=0)
            if i == 0:
                ax.set_title(NETWORK_LABEL[network], fontsize=9)
            if j == 0:
                ax.set_ylabel("EV penetration", fontsize=8)
                ax.text(-0.42, 0.5, head, rotation=90, ha="center", va="center", fontsize=9, transform=ax.transAxes)
            if i == len(heads) - 1:
                ax.set_xlabel("PV penetration", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"skill_grid_{event}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Report every scenario, then the sweep across scenarios."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment-root", type=Path, required=True, help="folder holding the sweep's runs/")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()
    S.configure(latex=not args.no_latex)
    analysis = args.experiment_root / "analysis"
    summaries = []
    for name, scenario_dir in newest_scenario_dirs(args.experiment_root).items():
        frames = [pd.read_parquet(p, columns=["dataset"]) for p in scenario_dir.glob("decision/*/metrics.parquet")]
        key = str(pd.concat(frames)["dataset"].iloc[0])
        print(f"== {key}")
        summaries.append(report_scenario(key, scenario_dir, analysis / name))
    if not summaries:
        print("no scenarios found")
        return
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(analysis / "sweep_summary.csv", index=False)
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for event in sorted(summary["event"].unique()):
        table = wilcoxon_table(summary, event)
        if not table.empty:
            table.to_csv(analysis / f"wilcoxon_{event}.csv", index=False)
        plot_skill_grid(summary, event, figures)
    print(f"written under {analysis}")


if __name__ == "__main__":
    main()
