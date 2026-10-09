r"""Chapter 6, Experiment 1 tables from ``fixed_benchmark.py`` runs, one set per dataset.

For every dataset found under the experiment's ``runs/`` (the newest run that
holds it), writes LaTeX tables and a CSV of every number behind them to
``<experiment-root>/analysis/<dataset>/``, in the layout of ``Chapter-6.tex``'s
own Experiment 1 tables:

  ch6_benchmark.tex             per-target NRMSE, CRPS, WS and stratified CWE
                                for every benchmark head, fold-then-mean, with
                                an average rank across folds, targets and the
                                four metrics (tab:ch6-benchmark)
  ch6_benchmark_joint.tex       marginal PICP per target, exact joint coverage
                                of P, Q and V, energy score and thermal-event
                                Brier skill (tab:ch6-benchmark-joint)
  ch6_benchmark_calibrated.tex  joint coverage and WS before and after max-norm
                                and CQR-marginal calibration; only where the
                                run has a conformal group (MLVS-PT-3PHASE)

Significance markers (benchmark table):
  bold          the head is in the 95% Model Confidence Set for that metric and
                target (NRMSE, CRPS, WS; fold-level losses, as Chapters 5 and 7
                test them, via twiga_thesis.analysis); for stratified CWE,
                which is a diagnostic rather than a proper score, bold marks the
                highest value and carries no test
  up/down arrow the head is significantly better / worse than anchored FPQR
                (Diebold-Mariano, Benjamini-Hochberg adjusted, 5%)

Stratified CWE is the two-sided version (coverage gate within one standard
error of nominal on either side; see fixed_benchmark_analysis.py), scored per
fold and averaged.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/fixed_benchmark_tables.py \\
        --experiment-root <experiments>/ch7_multitarget/multitarget_fixed_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parents[2]))
from chapters.ch7_multitarget.experiment_1_benchmark.fixed_benchmark_analysis import (  # noqa: E402
    load_condition,
    two_sided_cwe,
)
from twiga_thesis.analysis import dm_and_mcs  # noqa: E402

REFERENCE = "fpqr_anchor"
# Row order and labels: the proposed configuration first, the baselines, then
# the two FPQR variants whose native interval is not at the nominal 90% level.
HEADS = {
    "fpqr_anchor": r"\ac{FPQR}",
    "qr": r"\ac{QR}",
    "bnn": r"\ac{BNN}",
    "laplace": "Laplace",
    "fpqr": r"\ac{FPQR} (learned outer levels)$^{\ddagger}$",
    "fpqr_ch5": r"\ac{FPQR} (Chapter~\ref{chap:chapter-5} configuration)$^{\ddagger}$",
}
MAIN = list(HEADS)[:4]
# head_tuned group (fixed_benchmark.py): the same heads with their own settings
# tuned and the backbone pinned. Shown beside the fixed heads when a run holds them.
TUNED = {
    "fpqr_anchor_tuned": r"\ac{FPQR} (head-tuned)",
    "qr_tuned": r"\ac{QR} (head-tuned)",
    "bnn_tuned": r"\ac{BNN} (head-tuned)",
}
LABELS = {**HEADS, **TUNED}
# Each record names P, Q and V differently; the tables use the role.
ROLE = {"LP": "P", "LQ": "Q", "LV": "V", "p_kw": "P", "q_kvar": "Q", "v_lv_pu": "V"}
ROLE_LABEL = {"P": r"\ac{P}", "Q": r"\ac{Q}", "V": r"\ac{V}"}
DATASET_LABEL = {
    "mlvs-pt": r"\ac{MLVS-PT}",
    "dss-eu": r"\ac{DSS-EU}",
    "dss-uk": r"\ac{DSS-UK}",
    "dss-us": r"\ac{DSS-US}",
}
METRICS = {"nrmse": r"\ac{NRMSE}", "crps": r"\ac{CRPS}", "winkler_score": r"\ac{WS}", "cwe": r"\ac{CWE}"}
LOWER_IS_BETTER = {"nrmse": True, "crps": True, "winkler_score": True, "cwe": False}
# Every conformal variant the benchmark runs, as the chapter names them: max-norm on every
# head, CQR (unscaled, then scaled) per target, with a Bonferroni level, or pooled.
CALIBRATION = {
    "native": None,
    "max-norm": "max_norm",
    "CQR marginal": "quantile_unscaled_marginal",
    "CQR Bonferroni": "quantile_unscaled_bonferroni",
    "CQR pooled": "quantile_unscaled_pooled",
    "CQR marginal, scaled": "quantile_scaled_marginal",
    "CQR Bonferroni, scaled": "quantile_scaled_bonferroni",
    "CQR pooled, scaled": "quantile_scaled_pooled",
}


def dataset_label(slug: str) -> str:
    return next((label for prefix, label in DATASET_LABEL.items() if slug.startswith(prefix)), slug)


def ordered_targets(metrics: pd.DataFrame) -> list[str]:
    """The record's own target names, ordered P, Q, V."""
    names = [t for t in metrics["target"].dropna().unique() if t in ROLE]
    return sorted(names, key=lambda t: "PQV".index(ROLE[t]))


def per_fold(metrics: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Fold-level NRMSE, CRPS, WS and two-sided stratified CWE per target."""
    point = metrics[metrics.metric_type == "point"].groupby(["Folds", "target"])["nrmse"].mean()
    ws = metrics[metrics.metric_type == "interval"].groupby(["Folds", "target"])["winkler_score"].mean()
    window = metrics[(metrics.metric_type == "window") & (metrics.target != "all")]
    crps = window.groupby(["Folds", "target"])["crps"].mean()
    cwe = pd.concat(
        {fold: two_sided_cwe(g).set_index("target")["cwe_s_two_sided"] for fold, g in preds.groupby("Folds")},
        names=["Folds", "target"],
    ).rename("cwe")
    return pd.concat([point, crps, ws, cwe], axis=1).reset_index()


def drop_degenerate(fold_frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Drop heads whose ``metric`` is NaN in every fold, so one collapsed head cannot empty the DM/MCS pivot.

    Chapter 7's ``drop_degenerate_conditions``, applied before every significance test.
    """
    alive = fold_frame.groupby("condition")[metric].apply(lambda s: s.notna().any())
    return fold_frame[fold_frame["condition"].isin(alive[alive].index)]


def significance(
    folds: pd.DataFrame, targets: list[str], compared: list[str] | None = None
) -> tuple[dict, dict, dict, dict]:
    """Fold-level DM and MCS per (metric, target), as Chapters 5 and 7 test them.

    One loss per head and fold, through ``twiga_thesis.analysis.dm_and_mcs``
    (DM with Benjamini-Hochberg adjustment, MCS at alpha=0.05). The compared heads form
    one test family, the same one the report's significance figures use; the adjustment
    depends on the family, so adding the reference rows to it would make the compared
    heads' pairs easier to pass on the reference rows' very small p-values. Each other
    head is instead tested against anchored FPQR alone. Returns MCS membership, each
    head's DM direction against anchored FPQR, the adjusted-p matrices and the DM tables
    (both for the compared family).
    """
    in_mcs, versus, matrices, tables = {}, {}, {}, {}
    frame = folds.rename(columns={"model": "condition"})
    compared = compared or sorted(frame["condition"].unique())
    others = [h for h in frame["condition"].unique() if h not in compared]
    for target in targets:
        scoped = frame[frame["target"] == target]
        for metric in ("nrmse", "crps", "winkler_score"):
            key = (metric, target)
            family = scoped[scoped["condition"].isin(compared)]
            dm, p_matrix, model_set = dm_and_mcs(drop_degenerate(family, metric), metric=metric)
            in_mcs[key], matrices[key], tables[key] = set(map(str, model_set.included)), p_matrix, dm
            rows = list(dm.itertuples(index=False))
            if others:
                # The reference rows' bold marks the full-field MCS; their markers come from a
                # family of their own with anchored FPQR.
                _, _, full_set = dm_and_mcs(drop_degenerate(scoped, metric), metric=metric)
                in_mcs[key] |= {h for h in map(str, full_set.included) if h in others}
                own = scoped[scoped["condition"].isin([REFERENCE, *others])]
                own_dm, _, _ = dm_and_mcs(drop_degenerate(own, metric), metric=metric)
                rows += [r for r in own_dm.itertuples(index=False) if {r.condition_a, r.condition_b} & set(others)]
            for row in rows:
                pair = {row.condition_a, row.condition_b}
                if REFERENCE in pair and row.significant:
                    other = (pair - {REFERENCE}).pop()
                    versus[(key, other)] = "up" if row.winner == other else "down"
    return in_mcs, versus, matrices, tables


def average_rank(folds: pd.DataFrame) -> pd.Series:
    """Mean rank per head across folds, targets and the four metrics (1 is best)."""
    ranks = []
    for metric, lower in LOWER_IS_BETTER.items():
        r = folds.groupby(["Folds", "target"])[metric].rank(ascending=lower, method="average")
        ranks.append(r.rename("rank").to_frame().assign(model=folds["model"]))
    return pd.concat(ranks).groupby("model")["rank"].mean()


def joint_metrics(metrics: pd.DataFrame, preds: pd.DataFrame, targets: list[str]) -> pd.Series:
    """Marginal PICP per target, exact joint coverage, energy score and pooled thermal Brier skill."""
    interval = preds[preds.metric_type == "interval"].copy()
    interval["hit"] = (interval.Actual >= interval.lower) & (interval.Actual <= interval.upper)
    marginal = interval.groupby("target")["hit"].mean().reindex(targets)
    marginal.index = [ROLE[t] for t in marginal.index]
    joint = interval.pivot_table(index=["Folds", "timestamp"], columns="target", values="hit").dropna().all(axis=1)
    out = {**marginal.to_dict(), "joint": joint.mean()}
    window = metrics[(metrics.metric_type == "window") & (metrics.target == "all")]
    if not window.empty:
        base = window["event_rate_realised"].mean()
        brier = window["brier"].mean()
        out["energy"] = window["energy_score"].mean()
        out["brier"] = brier
        out["brier_skill"] = 1.0 - brier / (base * (1.0 - base)) if 0.0 < base < 1.0 else np.nan
    return pd.Series(out)


def _cell(value: float, digits: int, bold: bool, arrow: str | None = None) -> str:
    text = f"{value:.{digits}f}"
    text = rf"\B{text}" if bold else text
    if arrow:
        text += r"$^{\uparrow}$" if arrow == "up" else r"$^{\downarrow}$"
    return text


def benchmark_table(summary, ranks, in_mcs, versus, targets, label, slug) -> str:
    """The per-target benchmark table, in Chapter-6.tex's own style."""
    digits = {"nrmse": 3, "crps": 3, "winkler_score": 2, "cwe": 3}
    best_cwe = summary.groupby("target")["cwe"].max()
    header_groups = " & ".join(rf"\multicolumn{{4}}{{c}}{{{ROLE_LABEL[ROLE[t]]}}}" for t in targets)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(targets)))
    sub = " & ".join(" & ".join(METRICS.values()) for _ in targets)
    caption = (
        rf"Multi-target probabilistic forecasting benchmark on {label} at fixed hyperparameters, per target, "
        r"averaged over the rolling-origin folds. Every method shares one fixed \ac{MLPF} backbone and training "
        r"budget, and every interval is scored at the nominal 90\% level. Lower \ac{NRMSE}, \ac{CRPS}, and \ac{WS} "
        r"are better; higher stratified \ac{CWE} is better. Bold marks membership of the 95\% \acl{MCS} "
        r"(stratified \ac{CWE}, a diagnostic rather than a proper score, bolds its highest value). "
        r"$^{\uparrow}$/$^{\downarrow}$ mark a method significantly better/worse than \ac{FPQR} "
        r"(\ac{DM} test on fold-level losses, Benjamini-Hochberg adjusted, 5\%). "
        r"Rank averages each method's rank over folds, targets, "
        r"and the four metrics. $^{\ddagger}$Learned outer quantile levels: the native interval is not at the "
        r"nominal 90\% level, so \ac{WS} and stratified \ac{CWE} are not comparable with the other rows."
    )
    lines = [
        r"\begin{table*}[ht]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \begin{adjustbox}{width=\textwidth,center}",
        r"    \begin{tabular}{l" + "cccc" * len(targets) + r"c} \toprule",
        rf"    & {header_groups} & \\ {cmid}",
        rf"    Model & {sub} & Rank \\",
        r"    \midrule",
    ]
    present = set(summary["model"])
    order = [h for h in [*MAIN, *[t for t in TUNED if t in present], "fpqr", "fpqr_ch5"] if h in present]
    compared = [h for h in order if h in MAIN or h in TUNED]
    for head in order:
        head_label = LABELS[head]
        if head in ("fpqr", next((t for t in TUNED if t in present), None)):
            lines.append(r"    \midrule")
        cells = []
        for target in targets:
            row = summary[(summary["model"] == head) & (summary["target"] == target)].iloc[0]
            for metric in METRICS:
                if metric == "cwe":
                    cells.append(_cell(row[metric], digits[metric], np.isclose(row[metric], best_cwe[target])))
                else:
                    bold = head in in_mcs.get((metric, target), set())
                    cells.append(_cell(row[metric], digits[metric], bold, versus.get(((metric, target), head))))
        cells.append(
            _cell(ranks[head], 2, np.isclose(ranks[head], ranks.loc[[h for h in compared if h in present]].min()))
        )
        lines.append(f"    {head_label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        rf"    \label{{tab:ch6-benchmark-{slug}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def joint_table(joint: pd.DataFrame, label: str, slug: str) -> str:
    """Marginal PICP, joint coverage, energy score and Brier skill for the main heads."""
    has_scores = "energy" in joint
    caption = (
        rf"Marginal \ac{{PICP}} per target and exact joint coverage (nominal 90\%) on {label}. Joint coverage is "
        r"the fraction of test instances where \ac{P}, \ac{Q}, and \ac{V} are simultaneously within their own "
        r"interval. The energy score (lower is better) scores the joint predictive distribution of all three "
        r"targets; Brier skill (higher is better) scores the probability that apparent power exceeds its rating, "
        r"against the base rate. Bold marks the coverage closest to nominal and the best score."
    )
    cols = ["P", "Q", "V", "joint"] + (["energy", "brier_skill"] if has_scores else [])
    head = r"\ac{PICP} (\ac{P}) & \ac{PICP} (\ac{Q}) & \ac{PICP} (\ac{V}) & Joint coverage"
    head += r" & Energy score & Brier skill" if has_scores else ""
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \begin{adjustbox}{width=\textwidth,center}",
        r"    \begin{tabular}{l" + "c" * len(cols) + r"} \toprule",
        rf"    Model & {head} \\",
        r"    \midrule",
    ]
    rows = [h for h in [*MAIN, *TUNED] if h in joint.index]
    main = joint.loc[rows]
    best = {c: (main[c] - 0.9).abs().idxmin() for c in ("P", "Q", "V", "joint")}
    if has_scores:
        best["energy"] = main["energy"].idxmin()
        best["brier_skill"] = main["brier_skill"].idxmax()
    for h in rows:
        digits = {"energy": 3}
        cells = [_cell(main.loc[h, c], digits.get(c, 3), h == best[c]) for c in cols]
        lines.append(f"    {LABELS[h]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        rf"    \label{{tab:ch6-benchmark-joint-{slug}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def calibrated_table(calibrated: pd.DataFrame, label: str, slug: str) -> str:
    """Joint coverage and WS per target before and after calibration."""
    caption = (
        rf"Joint coverage (nominal 90\%) and \ac{{WS}} per target before and after conformal calibration on "
        rf"{label}. Calibration rescales each method's own trained intervals; no method is retrained. \ac{{CQR}} "
        r"applies only to the quantile methods. Bold marks the lowest \ac{WS} per target and the joint coverage "
        r"closest to nominal."
    )
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \begin{adjustbox}{width=\textwidth,center}",
        r"    \begin{tabular}{llcccc} \toprule",
        r"    Model & Calibration & Joint coverage & \ac{WS} (\ac{P}) & \ac{WS} (\ac{Q}) & \ac{WS} (\ac{V}) \\",
        r"    \midrule",
    ]
    best_ws = calibrated[["P", "Q", "V"]].min()
    closest = (calibrated["joint"] - 0.9).abs().min()
    previous = None
    for (head, method), row in calibrated.iterrows():
        if previous is not None and head != previous:
            lines.append(r"    \midrule")
        previous = head
        cells = [_cell(row["joint"], 3, np.isclose(abs(row["joint"] - 0.9), closest))]
        cells += [_cell(row[t], 2, np.isclose(row[t], best_ws[t])) for t in ("P", "Q", "V")]
        lines.append(f"    {HEADS[head] if method == 'native' else ''} & {method} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        rf"    \label{{tab:ch6-benchmark-calibrated-{slug}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def newest_dataset_dirs(experiment_root: Path) -> dict[str, Path]:
    """Each dataset's condition directory, from the newest run that holds it."""
    found: dict[str, Path] = {}
    for run in sorted((experiment_root / "runs").glob("*"), reverse=True):
        for dataset_dir in sorted((run / "conditions").glob("*")):
            if any(dataset_dir.glob("benchmark/*/metrics.parquet")):
                found.setdefault(dataset_dir.name, dataset_dir)
    return found


def find_condition(experiment_root: Path, dataset_name: str, group: str, condition: str) -> Path | None:
    """The newest run's directory for one condition, so a group run on its own is read beside the others."""
    for run in sorted((experiment_root / "runs").glob("*"), reverse=True):
        path = run / "conditions" / dataset_name / group / condition
        if (path / "metrics.parquet").exists():
            return path
    return None


def build(dataset_dir: Path, out: Path, experiment_root: Path | None = None, include_tuned: bool = False) -> dict:
    """Every number and table for one dataset; returns the frames for printing and the figures script."""
    label, slug = dataset_label(dataset_dir.name), dataset_dir.name.split("_")[0]
    # A run may hold only some heads (e.g. a raw_export rerun without the FPQR reference variants).
    loaded = {
        head: load_condition(dataset_dir / "benchmark" / head)
        for head in HEADS
        if (dataset_dir / "benchmark" / head / "metrics.parquet").exists()
    }
    # The head-tuned heads join the comparison only on request: they change the test family
    # (ranks, MCS, DM markers) of the fixed-parameter benchmark the chapter reports.
    if experiment_root is not None and include_tuned:
        for name in TUNED:
            path = find_condition(experiment_root, dataset_dir.name, "head_tuned", name)
            if path is not None:
                loaded[name] = load_condition(path)
    targets = ordered_targets(next(iter(loaded.values()))[0])
    folds = pd.concat([per_fold(m, p).assign(model=h) for h, (m, p) in loaded.items()], ignore_index=True)
    summary = folds.groupby(["model", "target"])[list(METRICS)].mean().reset_index()
    compared = [h for h in [*MAIN, *TUNED] if h in loaded]
    ranks = average_rank(folds[folds["model"].isin(compared)])
    references = [h for h in ("fpqr", "fpqr_ch5") if h in loaded]
    if references:
        ranks = pd.concat([ranks, average_rank(folds).loc[references]])
    in_mcs, versus, matrices, dm_tables = significance(folds, targets, compared)
    joint = pd.DataFrame({h: joint_metrics(m, p, targets) for h, (m, p) in loaded.items()}).T

    rows = {}
    for head in MAIN:
        for method, suffix in CALIBRATION.items():
            group, name = ("benchmark", head) if suffix is None else ("conformal", f"{head}_{suffix}")
            # A head calibrated from a refit (conformal/<head>_refit) takes its native row from
            # that refit too, so each head's rows share one set of weights.
            if suffix is None and (dataset_dir / "conformal" / f"{head}_refit" / "metrics.parquet").exists():
                group, name = "conformal", f"{head}_refit"
            if not (dataset_dir / group / name / "metrics.parquet").exists():
                continue
            metrics, preds = load_condition(dataset_dir / group / name)
            ws = metrics[metrics.metric_type == "interval"].groupby("target")["winkler_score"].mean()
            rows[(head, method)] = {
                **{ROLE[t]: v for t, v in ws.items()},
                "joint": joint_metrics(metrics, preds, targets)["joint"],
            }
    calibrated = pd.DataFrame(rows).T if any(m != "native" for _, m in rows) else pd.DataFrame()

    out.mkdir(parents=True, exist_ok=True)
    (out / "ch6_benchmark.tex").write_text(benchmark_table(summary, ranks, in_mcs, versus, targets, label, slug))
    (out / "ch6_benchmark_joint.tex").write_text(joint_table(joint, label, slug))
    summary.merge(ranks.rename("rank"), left_on="model", right_index=True).to_csv(
        out / "ch6_benchmark_numbers.csv", index=False
    )
    joint.to_csv(out / "ch6_benchmark_joint_numbers.csv")
    if not calibrated.empty:
        (out / "ch6_benchmark_calibrated.tex").write_text(calibrated_table(calibrated, label, slug))
        calibrated.to_csv(out / "ch6_benchmark_calibrated_numbers.csv")
    pd.DataFrame([{"metric": m, "target": t, "mcs": ", ".join(sorted(s))} for (m, t), s in in_mcs.items()]).to_csv(
        out / "ch6_benchmark_mcs.csv", index=False
    )
    for (metric, target), matrix in matrices.items():
        matrix.to_csv(out / f"ch6_dm_pvalues_{metric}_{ROLE[target]}.csv")
    return {
        "targets": targets,
        "summary": summary,
        "ranks": ranks,
        "in_mcs": in_mcs,
        "versus": versus,
        "matrices": matrices,
        "dm_tables": dm_tables,
        "folds": folds,
        "loaded": loaded,
        "joint": joint,
        "calibrated": calibrated,
    }


def main() -> None:
    """Build and write every dataset's tables; print the numbers behind them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, required=True, help="folder holding the fixed benchmark's runs/")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    for name, dataset_dir in newest_dataset_dirs(args.experiment_root).items():
        result = build(dataset_dir, args.experiment_root / "analysis" / name, args.experiment_root)
        summary = result["summary"].assign(target=lambda d: d["target"].map(ROLE))
        print(f"\n################ {name}  ({dataset_dir.parent.parent.name})")
        print(summary.pivot(index="model", columns="target", values=list(METRICS)).round(3).to_string())
        print("\nrank:", result["ranks"].round(2).to_dict())
        print("MCS:", {f"{m}/{ROLE[t]}": sorted(s) for (m, t), s in result["in_mcs"].items()})
        print(
            "vs anchored FPQR (main heads):",
            {f"{k[0][0]}/{ROLE[k[0][1]]}/{k[1]}": v for k, v in result["versus"].items() if k[1] in MAIN},
        )
        print(result["joint"].round(3).to_string())
        if not result["calibrated"].empty:
            print(result["calibrated"].round(3).to_string())
    print(f"\nwritten under {args.experiment_root / 'analysis'}")


if __name__ == "__main__":
    main()
