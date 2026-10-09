r"""Chapter 7 architectural evaluation: backbone, feature gating, and horizon-wise calibration.

Analyses the three hypotheses of ``Chapter-7.tex`` (``chap7:exp-two``) from saved results, retraining nothing. Two
input modes:

``--study-runs`` (current)
    Run directories of ``experiment_1_architectural/architectural_study.py``: one spec covering ``backbone`` (seeded),
    ``gating``, ``gating_noise`` and ``calibration``. Seeds are averaged per fold before testing; the seed spread is
    reported as the retraining noise floor; gate faithfulness comes from the per-fold ``gate_*`` columns the study's
    scorer writes, and the noise stress test compares each gating condition on ``<dataset>+noise`` with ``<dataset>``.
    Output: ``<output-root>/analysis/architectural_study``.
``--gating-run`` / ``--calibration-run`` (legacy)
    The earlier ``architectural_evaluation`` results plus two design-ablation runs. Output:
    ``<output-root>/analysis/architectural``.

Point accuracy is decided on per-fold MAE and interval quality on per-fold Winkler score (DM test with
Benjamini-Hochberg adjustment, MCS at alpha = 0.05); CWE, PICP and NMPI are diagnostics.

Usage::

    uv run python experiment/ch6_conformal/shared/ch7_architectural_analysis.py \
        --study-runs <architectural_study runs/...> <...> --copy-to-images
    uv run python experiment/ch6_conformal/shared/ch7_architectural_analysis.py \
        --gating-run <runs/...> --calibration-run <runs/...>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir.parents[1]))

from twiga.core.utils import slugify  # noqa: E402

from chapters.ch6_conformal.shared import ch7_analyse_results as analysis  # noqa: E402
from chapters.ch6_conformal.shared.ch7_analyse_results import load_full  # noqa: E402
from chapters.ch6_conformal.shared.protocol import CHAPTER_DATASETS, DEFAULT_OUTPUT_ROOT  # noqa: E402
from twiga_thesis.analysis import dm_and_mcs, fold_level, metric_table  # noqa: E402
from twiga_thesis.plotting import line_style as LS, style as S, tokens as T  # noqa: E402
from twiga_thesis.plotting.comparison_plots import (  # noqa: E402
    head_label,
    plot_dm_significance_matrix,
    thesis_palette,
)

IMAGES_DIR = script_dir.parents[2] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-7"
SLUG = {"MLVS-PT": "pt", "MLVS-UK": "uk"}
NOMINAL_PICP = 0.90
METRICS = ["nrmse", "mae", "winkler_score", "cwe_stratified", "picp", "nmpi"]
LOWER = {"nrmse", "mae", "winkler_score", "nmpi"}
HIGHER = {"cwe_stratified"}
BACKBONE = ["mlpf_point", "mlpgam_point", "mlpf_scp", "mlpgam_scp"]
INTERVAL_BACKBONE = ["mlpf_scp", "mlpgam_scp"]
GATING = ["gate_none", "gate_unconstrained", "gate_default"]
GATING_STUDY = ["gate_none", "gate_unconstrained", "gate_default", "gate_l1e-4", "gate_l1e-3", "gate_l1e-2"]
GATE_SUMMARY = [
    "gate_rho_spearman",
    "gate_rho_pearson",
    "gate_mean",
    "gate_signal_mean",
    "gate_noise_mean",
    "gate_noise_auc",
]
STUDY_PREFIX = "ch7_architectural_study"
# conf_crc and conf_crc_signed only in runs made after they were added; filtered to the conditions present
CALIBRATION = ["conf_scp_pooled", "conf_scp", "conf_crc", "conf_crc_signed"]
LABELS = {
    "mlpf_point": "MLPF (point)",
    "mlpgam_point": "MLPGAM (point)",
    "mlpf_scp": "MLPF + SCP",
    "mlpgam_scp": "MLPGAM + SCP",
    "gate_none": "No gate",
    "gate_unconstrained": r"Gate, $\lambda_{\mathrm{gate}}=0$",
    "gate_default": "Gate (proposed)",
    "gate_l1e-4": r"Gate, $\lambda_{\mathrm{gate}}=10^{-4}$",
    "gate_l1e-3": r"Gate, $\lambda_{\mathrm{gate}}=10^{-3}$",
    "gate_l1e-2": r"Gate, $\lambda_{\mathrm{gate}}=10^{-2}$",
    "mlpgam_crc": "CRC",
    "mlpgam_qr": "QR",
    "mlpgam_bnn": "BNN",
    "mlpgam_laplace": "Laplace",
    "conf_scp": "Horizon-wise SCP",
    "conf_scp_pooled": "Pooled SCP",
    "conf_crc": "CRC",
    "conf_crc_signed": "CRC, signed",
}
HORIZON = 48


def _label(condition: str) -> str:
    return LABELS.get(condition, head_label(condition))


def _tex_label(condition: str) -> str:
    """Display label for a plot: escaped for LaTeX unless it already contains math."""
    label = _label(condition)
    return label if "$" in label else S.tex_safe(label)


def _feature_label(name: str) -> str:
    """``past__NetLoad(kW)_lag_96`` -> ``Past: net load (lag 96)``."""
    branch, feature = name.split("__", 1)
    target = feature.split("_lag_")[0].split("_rolling_")[0]
    pretty = {
        f"{target}_lag_96": "net load (lag 96)",
        f"{target}_rolling_mean_win_96": "net load (96-step mean)",
        "hour_cosin": "hour of day (cos)",
        "yweek_cos": "week of year (cos)",
        "yweek_sin": "week of year (sin)",
        "yday_cos": "day of year (cos)",
        "Ghi": "GHI",
    }.get(feature)
    if pretty is None:
        pretty = f"noise {feature.split('_')[1]}" if feature.startswith("noise_") else "net load"
    return f"{branch.capitalize()}: {pretty}"


def run_design_group(run_dir: Path, dataset: str, group: str) -> pd.DataFrame:
    """One design-ablation group's per-window rows from a run's ``ch7_design_ablation_full.csv``."""
    full = pd.read_csv(run_dir / "results" / "ch7_design_ablation_full.csv")
    return full[(full["dataset"] == dataset) & (full["group"] == group)]


def summarise(
    fold_df: pd.DataFrame, conditions: list[str], decisive: str, out: Path, name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, pd.DataFrame]:
    """Metric table, DM table, adjusted-p matrix and MCS on ``decisive``, restricted to ``conditions``."""
    scoped = fold_df[fold_df["condition"].isin(conditions)]
    metrics = [m for m in METRICS if scoped[m].notna().all()]
    table = metric_table(
        scoped.dropna(subset=metrics),
        metric_cols=metrics,
        lower_better=LOWER & set(metrics),
        higher_better=HIGHER & set(metrics),
        target_values={"picp": NOMINAL_PICP},
    )
    dm, p_matrix, mcs = dm_and_mcs(scoped.dropna(subset=[decisive]), metric=decisive)
    table.to_csv(out / f"{name}_metrics.csv", index=False)
    # Plain fold mean/SD per condition for every metric (NaN where undefined, e.g. intervals for point models).
    stats = scoped.groupby("condition")[METRICS].agg(["mean", "std"])
    stats.columns = [f"{m}_{k}" for m, k in stats.columns]
    stats.to_csv(out / f"{name}_stats.csv")
    dm.to_csv(out / f"{name}_dm_{decisive}.csv", index=False)
    pd.DataFrame({"condition": sorted(mcs.included)}).to_csv(out / f"{name}_mcs_{decisive}.csv", index=False)
    return table, dm, p_matrix, mcs, scoped


def relative_effect(fold_df: pd.DataFrame, reference: str, metric: str) -> pd.DataFrame:
    """Each condition's mean ``metric`` against ``reference``'s, as a percentage of the reference's mean.

    Negative is better for a lower-is-better metric. ``folds_better`` counts the folds in which the condition's
    per-fold value is below the reference's (of ``n_folds``).
    """
    wide = fold_df.pivot(index="Folds", columns="condition", values=metric)
    means = wide.mean()
    return pd.DataFrame(
        {
            "mean": means,
            "pct_vs_reference": (means / means[reference] - 1.0) * 100.0,
            "folds_better": wide.lt(wide[reference], axis=0).sum(),
            "n_folds": len(wide),
        }
    ).drop(index=reference)


def retraining_noise(fold_df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Gap between a point model and its conformalised twin, which share an architecture but are trained separately.

    The conformal wrapper leaves the point forecast untouched, so any difference in ``metric`` between
    ``<backbone>_point`` and ``<backbone>_scp`` is retraining variability (separate fits, same configuration).
    It is the noise floor against which the backbone effect should be read.
    """
    means = fold_df.groupby("condition")[metric].mean()
    rows = [
        {"pair": f"{b}_point vs {b}_scp", "kind": "retraining", "a": means[f"{b}_point"], "b": means[f"{b}_scp"]}
        for b in ("mlpf", "mlpgam")
    ]
    rows.append(
        {"pair": "mlpgam_scp vs mlpf_scp", "kind": "backbone", "a": means["mlpgam_scp"], "b": means["mlpf_scp"]}
    )
    rows.append(
        {"pair": "mlpgam_point vs mlpf_point", "kind": "backbone", "a": means["mlpgam_point"], "b": means["mlpf_point"]}
    )
    out = pd.DataFrame(rows)
    out["pct_a_vs_b"] = (out["a"] / out["b"] - 1.0) * 100.0
    return out


def horizon_profile(
    run_dir: Path, dataset: str, conditions: list[str], alpha: float = 0.10, group: str = "conformal"
) -> pd.DataFrame:
    """Empirical coverage and mean interval width by lead time, from saved per-timestamp predictions.

    A window is one calendar day of ``HORIZON`` steps starting at 00:00 (checked); the lead time is the
    step's position within its (fold, day) window.
    """
    rows = []
    for condition in conditions:
        path = run_dir / "conditions" / slugify(dataset) / group / condition / "predictions.parquet"
        preds = pd.read_parquet(path)
        # Engine runs also keep the point rows; a condition with a custom scorer stores its interval rows as "custom".
        if "metric_type" in preds.columns and preds["metric_type"].isin(["interval", "custom"]).any():
            preds = preds[preds["metric_type"].isin(["interval", "custom"])]
        preds = preds.dropna(subset=["lower", "upper"])
        preds = preds.sort_values(["Folds", "timestamp"])
        preds["day"] = preds["timestamp"].dt.normalize()
        grouped = preds.groupby(["Folds", "day"])
        if not (grouped.size() == HORIZON).all():
            raise ValueError(f"{dataset}/{condition}: not every (fold, day) window has {HORIZON} steps")
        if not (grouped["timestamp"].min() == grouped["timestamp"].min().dt.normalize()).all():
            raise ValueError(f"{dataset}/{condition}: windows do not start at 00:00")
        preds["h"] = grouped.cumcount() + 1
        preds["covered"] = (preds["Actual"] >= preds["lower"]) & (preds["Actual"] <= preds["upper"])
        preds["width"] = preds["upper"] - preds["lower"]
        profile = preds.groupby("h").agg(coverage=("covered", "mean"), width=("width", "mean")).reset_index()
        profile["condition"] = condition
        profile["dataset"] = dataset
        profile["target"] = 1.0 - alpha
        rows.append(profile)
    return pd.concat(rows, ignore_index=True)


DATASET_LABELS = {"MLVS-PT": "MLVS-PT", "MLVS-UK": "SPS-UK"}
RELATIVE = {"nrmse", "mae", "winkler_score"}  # error metrics: per-fold best / value (1 = best in that fold)


SYMBOLS = {
    "nrmse": r"$\mathrm{NRMSE}$",
    "mae": r"$\mathrm{MAE}$",
    "winkler_score": r"$\mathrm{WS}$",
    "cwe_stratified": r"Stratified $\mathrm{CWE}$",
    "picp": r"$\mathrm{PICP}$",
    "nmpi": r"$\mathrm{NMPI}$",
}
RELATIVE_SYMBOLS = {
    "nrmse": r"$\mathrm{NRMSE}^{\star}/\mathrm{NRMSE}$",
    "mae": r"$\mathrm{MAE}^{\star}/\mathrm{MAE}$",
    "winkler_score": r"$\mathrm{WS}^{\star}/\mathrm{WS}$",
}


def plot_dataset_comparison(frames: dict[str, pd.DataFrame], conditions: list[str], out: Path, name: str) -> None:
    """Chapter 5's cross-dataset bars, one single-axes figure per metric: datasets on x, one bar per condition.

    Stratified CWE, PICP and NMPI are on a [0, 1] scale and are plotted as they are. The error metrics (NRMSE, MAE,
    Winkler score) are shown as a paired relative score ``m* / m``, where ``m*`` is the best (lowest) value among
    the compared conditions in the same fold: 1 is the best condition of that fold and 0.9 is 10% worse. Pairing
    within the fold removes the seasonal fold-to-fold scale. Bars are the median over the ten folds with one
    standard deviation, as in Chapter 5. No title: the thesis caption carries it; the y-axis carries the symbol.
    """
    rows = []
    for dataset, frame in frames.items():
        frame = frame[frame["condition"].isin(conditions)]
        for metric in METRICS:
            sub = frame[["condition", "Folds", metric]].dropna()
            if sub.empty:
                continue
            values = sub[metric]
            if metric in RELATIVE:
                values = sub.groupby("Folds")[metric].transform("min") / values
            rows.extend(
                {"dataset": DATASET_LABELS[dataset], "condition": c, "metric": metric, "value": v}
                for c, v in zip(sub["condition"], values, strict=True)
            )
    data = pd.DataFrame(rows)
    colors = dict(zip(conditions, thesis_palette(len(conditions)), strict=True))
    datasets = list(dict.fromkeys(data["dataset"]))
    for metric in METRICS:
        sub = data[data["metric"] == metric]
        present = [c for c in conditions if c in set(sub["condition"])]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        width = 0.8 / len(present)
        for k, condition in enumerate(present):
            stats = (
                sub[sub["condition"] == condition].groupby("dataset")["value"].agg(["median", "std"]).reindex(datasets)
            )
            x = np.arange(len(datasets)) - 0.4 + width * (k + 0.5)
            ax.bar(
                x,
                stats["median"],
                width=width * 0.95,
                yerr=stats["std"],
                color=colors[condition],
                capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "#666666"},
                label=_tex_label(condition),
            )
        if metric == "picp":
            ax.axhline(NOMINAL_PICP, color="gray", ls=":", lw=1.0)
        ax.set_ylabel(RELATIVE_SYMBOLS[metric] if metric in RELATIVE else SYMBOLS[metric])
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([S.tex_safe(d) for d in datasets])
        ax.set_ylim(0, 1.05)
        S.format_axes(ax)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(present), 2), frameon=False, fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}_{metric}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_significance(p_matrix: pd.DataFrame, dataset: str, name: str, out: Path) -> None:
    """Chapter 6's binary BH-adjusted DM matrix, relabelled with this section's condition names."""
    fig = plot_dm_significance_matrix(p_matrix, dataset, name)
    if fig is None:
        return
    labels = [_tex_label(c) for c in p_matrix.index]
    ax = fig.axes[0]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}_significance_{SLUG[dataset]}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


HORIZON_STYLE = {
    "conf_scp_pooled": (T.HIGHLIGHT, "--"),
    "conf_scp": (T.INFO, "-."),
    "conf_crc": (T.PRIMARY, "-"),
    "conf_crc_signed": (T.SECONDARY, ":"),
}


def plot_horizon(profile: pd.DataFrame, out: Path) -> None:
    """Coverage and mean interval width by time of day, one single-axes figure per (quantity, dataset).

    In the compact thesis line-plot style (``twiga_thesis.plotting.line_style``): 0.40 text width,
    0.85 pt lines, y-axis from the plotted data rounded to round ticks, legend below the axes.
    """
    with plt.rc_context(LS.COMPACT_RC):
        for dataset in dict.fromkeys(profile["dataset"]):
            unit = {"MLVS-PT": "kW", "MLVS-UK": "MW"}.get(dataset, "")
            for quantity, ylabel in (("coverage", "Empirical coverage"), ("width", f"Mean interval width ({unit})")):
                fig, ax = LS.new_panel()
                plotted = []
                for condition in [c for c in CALIBRATION if c in set(profile["condition"])]:
                    sub = profile[(profile["dataset"] == dataset) & (profile["condition"] == condition)]
                    color, ls = HORIZON_STYLE[condition]
                    ax.plot((sub["h"] - 1) / 2.0, sub[quantity], color=color, ls=ls, label=_tex_label(condition))
                    plotted.append(sub[quantity].to_numpy())
                if quantity == "coverage":
                    LS.reference_line(ax, NOMINAL_PICP, label="Nominal")
                    plotted.append([NOMINAL_PICP])
                LS.hour_axis(ax)
                LS.set_y_axis(ax, *plotted, unit_interval=quantity == "coverage", from_zero=quantity == "width")
                ax.set_ylabel(ylabel)
                LS.finish(ax)
                LS.legend_below(ax, ncol=3)
                LS.save(fig, out / f"calibration_{quantity}_{SLUG[dataset]}")


PENALTY_STEPS = [
    ("gate_unconstrained", "0"),
    ("gate_l1e-4", "$10^{-4}$"),
    ("gate_l1e-3", "$10^{-3}$"),
    ("gate_l1e-2", "$10^{-2}$"),
]


def _relative_to_ungated(fold_df: pd.DataFrame, metric: str = "winkler_score") -> pd.DataFrame:
    """Per fold, the percent change of ``metric`` against the ungated model of the same input set."""
    wide = fold_df.pivot(index="Folds", columns="condition", values=metric)
    return (wide.div(wide["gate_none"], axis=0) - 1.0) * 100.0


def plot_gating_penalty(clean: pd.DataFrame, noisy: pd.DataFrame | None, dataset: str, out: Path) -> None:
    """Change in WS against the ungated model as the gate penalty grows, on curated and noise-augmented inputs.

    Each line is relative to the ungated model of its own input set (0 = no gate), so a point below zero is a gate
    setting that improves on the ungated model. The tuned gate is drawn apart, since its penalty is set by the
    hyperparameter search rather than fixed. Markers are the mean over the ten folds; bands one standard error.
    Compact thesis line-plot style (``twiga_thesis.plotting.line_style``).
    """
    with plt.rc_context(LS.COMPACT_RC):
        fig, ax = LS.new_panel()
        x = np.arange(len(PENALTY_STEPS))
        tuned_x = len(PENALTY_STEPS) + 0.6
        sets = [("Curated inputs", clean, LS.FORECAST_COLOR, "o")]
        if noisy is not None and not noisy.empty:
            sets.append(("With noise covariates", noisy, LS.OBSERVED_COLOR, "s"))
        plotted = [[0.0]]
        for label, frame, color, marker in sets:
            rel = _relative_to_ungated(frame)
            steps = [c for c, _ in PENALTY_STEPS]
            mean, se = rel[steps].mean(), rel[steps].std(ddof=1) / np.sqrt(rel[steps].notna().sum())
            ax.plot(x, mean.to_numpy(), color=color, marker=marker, ms=3, lw=LS.LINE_WIDTH, label=label)
            ax.fill_between(x, (mean - se).to_numpy(), (mean + se).to_numpy(), color=color, alpha=0.15, lw=0)
            plotted += [(mean - se).to_numpy(), (mean + se).to_numpy()]
            if "gate_default" in rel:
                t_mean = rel["gate_default"].mean()
                t_se = rel["gate_default"].std(ddof=1) / np.sqrt(rel["gate_default"].notna().sum())
                ax.errorbar(
                    tuned_x,
                    t_mean,
                    yerr=t_se,
                    color=color,
                    marker=marker,
                    mfc="white",
                    ms=3.5,
                    capsize=1.5,
                    lw=LS.LINE_WIDTH,
                    mew=LS.LINE_WIDTH,
                )
                plotted.append([t_mean - t_se, t_mean + t_se])
        LS.reference_line(ax, 0.0)
        ax.axvline(len(PENALTY_STEPS) - 0.2, color=LS.SEPARATOR_COLOR, lw=LS.LINE_WIDTH)
        ax.set_xticks([*x, tuned_x])
        ax.set_xticklabels(
            [lbl for _, lbl in PENALTY_STEPS] + ["Tuned"], rotation=35, ha="right", rotation_mode="anchor"
        )
        ax.set_xlim(-0.4, tuned_x + 0.4)
        ax.set_xlabel(r"Gate penalty $\lambda_{\mathrm{gate}}$")
        ax.set_ylabel(r"$\Delta\mathrm{WS}$ against no gate (\%)")
        LS.set_y_axis(ax, *plotted)
        LS.finish(ax)
        LS.legend_below(ax, ncol=2)
        LS.save(fig, out / f"gating_penalty_{SLUG[dataset]}")


def load_study(runs: list[Path]) -> pd.DataFrame:
    """Per-window rows of every study run, concatenated (later runs win for a repeated condition)."""
    # A repeated (dataset, group, condition) is taken whole from the last run that has it. De-duplicating row by row
    # would keep an earlier run's rows on days the later run does not score (a fold's first test day, which a
    # condition that reuses a checkpoint cannot forecast) and so mix two trainings into one condition.
    frames = [pd.read_csv(run / "results" / f"{STUDY_PREFIX}_full.csv").assign(_run=k) for k, run in enumerate(runs)]
    full = pd.concat(frames, ignore_index=True)
    key = ["dataset", "group", "condition"]
    latest = full.groupby(key)["_run"].transform("max")
    return full[full["_run"] == latest].drop(columns="_run").reset_index(drop=True)


def collapse_seeds(fold_df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average ``<name>__s<seed>`` conditions per fold; report the spread of fold-averaged metrics across seeds.

    Returns:
        ``(collapsed, spread)``: one row per (condition, fold) averaged over seeds, and one row per condition with
        the mean and standard deviation across seeds of the fold-averaged metric, and the number of seeds.
    """
    frame = fold_df.copy()
    frame["seed"] = frame["condition"].str.extract(r"__s(\d+)$", expand=False)
    frame["condition"] = frame["condition"].str.replace(r"__s\d+$", "", regex=True)
    collapsed = frame.groupby(["condition", "Folds"], as_index=False)[metrics].mean()
    per_seed = frame.groupby(["condition", "seed"])[metrics].mean()
    spread = per_seed.groupby("condition").agg(["mean", "std"])
    spread.columns = [f"{m}_{k}" for m, k in spread.columns]
    spread["n_seeds"] = per_seed.groupby("condition").size()
    return collapsed, spread


def gate_table(fold_df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Fold mean and standard deviation of the gate statistics for each gated condition."""
    cols = [c for c in GATE_SUMMARY if c in fold_df.columns]
    scoped = fold_df[fold_df["condition"].isin(conditions)].dropna(subset=cols, how="all")
    table = scoped.groupby("condition")[cols].agg(["mean", "std"])
    table.columns = [f"{m}_{k}" for m, k in table.columns]
    return table


def noise_degradation(clean: pd.DataFrame, noisy: pd.DataFrame, metric: str = "winkler_score") -> pd.DataFrame:
    """Per condition, the change in ``metric`` from the clean to the noise-augmented data, paired by fold."""
    wide_clean = clean.pivot(index="Folds", columns="condition", values=metric)
    wide_noisy = noisy.pivot(index="Folds", columns="condition", values=metric)
    common = wide_clean.columns.intersection(wide_noisy.columns)
    pct = (wide_noisy[common] / wide_clean[common] - 1.0) * 100.0
    return pd.DataFrame(
        {
            "clean_mean": wide_clean[common].mean(),
            "noisy_mean": wide_noisy[common].mean(),
            "pct_change": (wide_noisy[common].mean() / wide_clean[common].mean() - 1.0) * 100.0,
            "pct_change_median_fold": pct.median(),
            "folds_worse": (pct > 0).sum(),
            "n_folds": pct.notna().sum(),
        }
    )


def plot_noise_degradation(degradation: dict[str, pd.DataFrame], conditions: list[str], out: Path) -> None:
    """Percent change in Winkler score from clean to noise-augmented inputs; datasets on x, a bar per condition."""
    datasets = list(degradation)
    colors = dict(zip(conditions, thesis_palette(len(conditions)), strict=True))
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    width = 0.8 / len(conditions)
    for k, condition in enumerate(conditions):
        values = [degradation[d].loc[condition, "pct_change"] for d in datasets]
        x = np.arange(len(datasets)) - 0.4 + width * (k + 0.5)
        ax.bar(x, values, width=width * 0.95, color=colors[condition], label=_tex_label(condition))
    ax.axhline(0, color="gray", lw=1.0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([S.tex_safe(DATASET_LABELS[d]) for d in datasets])
    ax.set_ylabel(r"$\Delta\mathrm{WS}$ (\%)")
    S.format_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gating_noise_degradation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _gate_feature_label(name: str) -> str:
    """``past__NetLoad(kW)_lag_96`` -> ``Net load, lag 96 (history)``; noise features as ``Noise 1 (covariates)``."""
    branch, feature = name.split("__", 1)
    target = feature.split("_lag_")[0].split("_rolling_")[0]
    pretty = {
        f"{target}_lag_96": "Net load, lag 96",
        f"{target}_rolling_mean_win_96": "Net load, rolling mean 96",
        "hour_cosin": "Hour of day",
        "yweek_cos": "Week of year (cos)",
        "yweek_sin": "Week of year (sin)",
        "yday_cos": "Day of year (cos)",
        "Ghi": "GHI",
    }.get(feature)
    if pretty is None:
        pretty = f"Noise {int(feature.split('_')[1]) + 1}" if feature.startswith("noise_") else "Net load"
    return pretty


GATE_SETTINGS = {"gate_default": "tuned", "gate_l1e-2": r"$\lambda_{\mathrm{gate}}=10^{-2}$"}
BRANCH_NAMES = {"past": "History", "future": "Covariates"}


def _feature_key(feature: str) -> tuple[int, str]:
    """Column order: net load and its lags, GHI, calendar features, then injected noise."""
    order = ["net load", "lag", "rolling", "ghi", "day of year", "week of year", "hour"]
    label = _gate_feature_label(f"past__{feature}").lower()
    rank = next((k for k, token in enumerate(order) if token in label), len(order))
    if feature.startswith("noise_"):
        rank = len(order) + 1
    if label == "net load":
        rank = -1
    return rank, label


def plot_gate_values(noisy_fold: pd.DataFrame, dataset: str, out: Path) -> None:
    """Gate value per input feature on the noise-augmented data, as a branch-by-feature heatmap.

    Rows are the two branches under the tuned gate and the strongest penalty, columns the input features (injected
    noise last), and each cell is the fold-mean post-sigmoid gate value, annotated; a feature absent from a branch is
    left blank. Whether a setting separates real from noise features reads along each row. Full text width.
    """
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    present = [c for c in GATE_SETTINGS if c in set(noisy_fold["condition"])]
    scoped = noisy_fold[noisy_fold["condition"].isin(present)]
    # The results table holds every dataset's feature columns; keep the ones this dataset has values for.
    gate_cols = [c for c in scoped.columns if c.startswith("gate__") and scoped[c].notna().any()]
    features = sorted({c.split("__", 2)[2] for c in gate_cols}, key=_feature_key)
    rows, labels = [], []
    for branch in ("past", "future"):
        for condition in present:
            frame = noisy_fold.loc[noisy_fold["condition"] == condition]
            rows.append([frame.get(f"gate__{branch}__{f}", pd.Series(dtype=float)).mean() for f in features])
            labels.append(f"{BRANCH_NAMES[branch]}, {GATE_SETTINGS[condition]}")
    values = np.array(rows, dtype=float)
    cmap = LinearSegmentedColormap.from_list("gate", ["#FFFFFF", T.PRIMARY])
    cmap.set_bad("#FFFFFF")
    fig, ax = plt.subplots(figsize=S.panel(1.0, aspect=0.36))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] > 0.55 else T.TEXT_DARK
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    n_noise = sum(f.startswith("noise_") for f in features)
    if n_noise:
        ax.axvline(len(features) - n_noise - 0.5, color=T.HIGHLIGHT, lw=1.0)
        ax.text(
            len(features) - n_noise / 2 - 0.5,
            -0.75,
            "Injected noise",
            ha="center",
            va="bottom",
            fontsize=8,
            color=T.HIGHLIGHT,
        )
    if len(present) > 1:
        ax.axhline(len(present) - 0.5, color=T.TEXT_MUTED, lw=1.0)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([S.tex_safe(_gate_feature_label(f"past__{f}")) for f in features], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.minorticks_off()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.015)
    bar.set_label(r"Gate value $\eta$")
    bar.outline.set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_values_{SLUG[dataset]}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def study_main(args: argparse.Namespace) -> None:
    """Analyse the architectural study's runs (``--study-runs``)."""
    out = args.output_root / "analysis" / "architectural_study"
    out.mkdir(parents=True, exist_ok=True)
    full = load_study(args.study_runs)
    gate_cols = [c for c in full.columns if c.startswith("gate_")]
    frames: dict[str, dict[str, pd.DataFrame]] = {"backbone": {}, "gating": {}, "calibration": {}}
    degradations: dict[str, pd.DataFrame] = {}
    profiles = []
    for dataset in args.datasets:
        print(f"\n######## {dataset}")
        # H1: backbone, seeds averaged per fold
        seeded = fold_level(full, dataset, "backbone", metric_cols=METRICS)
        if not seeded.empty:
            fold_df, spread = collapse_seeds(seeded, METRICS)
            spread.to_csv(out / f"backbone_{dataset}_seed_spread.csv")
            print("seed spread (retraining noise)\n", spread[["mae_mean", "mae_std", "n_seeds"]].round(4).to_string())
            _, dm_mae, p_mae, mcs_mae, scoped = summarise(fold_df, BACKBONE, "mae", out, f"backbone_{dataset}")
            print("backbone MAE DM\n", dm_mae.round(4).to_string(index=False), "\nMCS:", sorted(mcs_mae.included))
            intervals = fold_df[fold_df["condition"].isin(INTERVAL_BACKBONE)]
            _, dm_ws, _, mcs_ws, _ = summarise(
                intervals, INTERVAL_BACKBONE, "winkler_score", out, f"backbone_{dataset}_interval"
            )
            print("backbone WS DM\n", dm_ws.round(4).to_string(index=False), "\nMCS:", sorted(mcs_ws.included))
            frames["backbone"][dataset] = scoped
            plot_significance(p_mae, dataset, "backbone_mae", out)

        # H2: gating, with faithfulness, and the noise stress test
        gate_fold = fold_level(full, dataset, "gating", metric_cols=METRICS + gate_cols)
        if not gate_fold.empty:
            g_table, g_dm, g_p, g_mcs, g_scoped = summarise(
                gate_fold, GATING_STUDY, "winkler_score", out, f"gating_{dataset}"
            )
            print("gating WS DM\n", g_dm.round(4).to_string(index=False), "\nMCS:", sorted(g_mcs.included))
            relative_effect(g_scoped, "gate_default", "winkler_score").to_csv(out / f"gating_{dataset}_relative_ws.csv")
            gate_table(gate_fold, GATING_STUDY).to_csv(out / f"gating_{dataset}_faithfulness.csv")
            frames["gating"][dataset] = g_scoped
            plot_significance(g_p, dataset, "gating_ws", out)
            noisy_key = f"{dataset}+noise"
            noisy_fold = fold_level(full, noisy_key, "gating_noise", metric_cols=METRICS + gate_cols)
            if not noisy_fold.empty:
                _, n_dm, n_p, n_mcs, _ = summarise(
                    noisy_fold, GATING_STUDY, "winkler_score", out, f"gating_noise_{dataset}"
                )
                print(
                    "gating under noise WS DM\n", n_dm.round(4).to_string(index=False), "\nMCS:", sorted(n_mcs.included)
                )
                faith = gate_table(noisy_fold, GATING_STUDY)
                faith.to_csv(out / f"gating_noise_{dataset}_faithfulness.csv")
                print("gate faithfulness under noise\n", faith.round(3).to_string())
                degr = noise_degradation(gate_fold, noisy_fold)
                degr.to_csv(out / f"gating_noise_{dataset}_degradation.csv")
                degradations[dataset] = degr
                plot_gate_values(noisy_fold, dataset, out)
                print("WS change from clean to noise-augmented inputs\n", degr.round(3).to_string())
                plot_significance(n_p, dataset, "gating_noise_ws", out)
            plot_gating_penalty(gate_fold, noisy_fold if not noisy_fold.empty else None, dataset, out)

        # H3: calibration
        cal_fold = fold_level(full, dataset, "calibration", metric_cols=METRICS)
        calibration = [c for c in CALIBRATION if not cal_fold.empty and c in set(cal_fold["condition"])]
        if calibration:
            _, c_dm, _, c_mcs, c_scoped = summarise(
                cal_fold, calibration, "winkler_score", out, f"calibration_{dataset}"
            )
            print("calibration WS DM\n", c_dm.round(4).to_string(index=False), "\nMCS:", sorted(c_mcs.included))
            frames["calibration"][dataset] = c_scoped
            # Each condition's predictions from the last run given that contains it (the same rule load_study uses).
            for condition in calibration:
                rel = Path("conditions") / slugify(dataset) / "calibration" / condition / "predictions.parquet"
                run = next(r for r in reversed(args.study_runs) if (r / rel).exists())
                profiles.append(horizon_profile(run, dataset, [condition], group="calibration"))

    for name, conditions in (("backbone", BACKBONE), ("gating", GATING_STUDY), ("calibration", CALIBRATION)):
        if frames[name]:
            plot_dataset_comparison(frames[name], conditions, out, name)
    if degradations:
        plot_noise_degradation(degradations, GATING_STUDY, out)
    if profiles:
        profile = pd.concat(profiles, ignore_index=True)
        profile.to_csv(out / "calibration_horizon_profile.csv", index=False)
        plot_horizon(profile, out)
    copy_figures(out, args)


def copy_figures(out: Path, args: argparse.Namespace) -> None:
    """Copy every figure PDF in ``out`` to the thesis image folder as ``fig_ch7_arch_<name>.pdf``."""
    if not args.copy_to_images:
        return
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in out.glob("*.pdf"):
        shutil.copy(pdf, IMAGES_DIR / f"fig_ch7_arch_{pdf.name}")


def legacy_main(args: argparse.Namespace) -> None:
    """Analyse the earlier backbone results and design-ablation runs (``--gating-run``/``--calibration-run``)."""
    analysis.RESULTS_ROOT = args.output_root.resolve()
    out = args.output_root / "analysis" / "architectural"
    out.mkdir(parents=True, exist_ok=True)
    report: list[pd.DataFrame] = []
    frames: dict[str, dict[str, pd.DataFrame]] = {"backbone": {}, "gating": {}, "calibration": {}}

    for dataset in args.datasets:
        print(f"\n######## {dataset}")
        # 1. backbone
        full = load_full("architectural_evaluation", dataset)
        fold_df = fold_level(full, dataset, "architectural_evaluation", metric_cols=METRICS)
        table, dm_mae, p_mae, mcs_mae, scoped = summarise(fold_df, BACKBONE, "mae", out, f"backbone_{dataset}")
        print("backbone point (MAE DM)\n", dm_mae.round(4).to_string(index=False), "\nMCS:", sorted(mcs_mae.included))
        intervals = fold_df[fold_df["condition"].isin(INTERVAL_BACKBONE)]
        _, dm_ws, p_ws, mcs_ws, _ = summarise(
            intervals, INTERVAL_BACKBONE, "winkler_score", out, f"backbone_{dataset}_interval"
        )
        print("backbone interval (WS DM)\n", dm_ws.round(4).to_string(index=False), "\nMCS:", sorted(mcs_ws.included))
        noise = retraining_noise(scoped)
        noise.to_csv(out / f"backbone_{dataset}_retraining_noise.csv", index=False)
        print("retraining noise vs backbone effect (MAE)\n", noise.round(4).to_string(index=False))
        frames["backbone"][dataset] = scoped
        plot_significance(p_mae, dataset, "backbone_mae", out)

        # 2. gating
        gate_full = run_design_group(args.gating_run, dataset, "gating")
        gate_fold = fold_level(gate_full, dataset, "gating", metric_cols=METRICS)
        g_table, g_dm, g_p, g_mcs, g_scoped = summarise(gate_fold, GATING, "winkler_score", out, f"gating_{dataset}")
        print("gating\n", g_table.round(4).to_string(index=False), "\n", g_dm.round(4).to_string(index=False))
        print("MCS:", sorted(g_mcs.included))
        eff = relative_effect(g_scoped, "gate_default", "winkler_score")
        eff.to_csv(out / f"gating_{dataset}_relative_ws.csv")
        print(eff.round(3).to_string())
        frames["gating"][dataset] = g_scoped
        plot_significance(g_p, dataset, "gating_ws", out)

        # 3. calibration
        cal_full = run_design_group(args.calibration_run, dataset, "conformal")
        cal_fold = fold_level(cal_full, dataset, "conformal", metric_cols=METRICS)
        legacy_calibration = ["conf_scp", "conf_scp_pooled"]
        c_table, c_dm, _, c_mcs, c_scoped = summarise(
            cal_fold, legacy_calibration, "winkler_score", out, f"calibration_{dataset}"
        )
        print("calibration\n", c_table.round(4).to_string(index=False), "\n", c_dm.round(4).to_string(index=False))
        print("MCS:", sorted(c_mcs.included))
        frames["calibration"][dataset] = c_scoped
        relative_effect(c_scoped, "conf_scp", "winkler_score").to_csv(out / f"calibration_{dataset}_relative_ws.csv")
        report.append(horizon_profile(args.calibration_run, dataset, legacy_calibration))

    for name, conditions in (("backbone", BACKBONE), ("gating", GATING), ("calibration", CALIBRATION)):
        plot_dataset_comparison(frames[name], conditions, out, name)
    profile = pd.concat(report, ignore_index=True)
    profile.to_csv(out / "calibration_horizon_profile.csv", index=False)
    summary = (
        profile.assign(abs_err=(profile["coverage"] - profile["target"]).abs())
        .groupby(["dataset", "condition"])
        .agg(
            coverage_h1=("coverage", "first"),
            coverage_h48=("coverage", "last"),
            coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"),
            mean_abs_coverage_error=("abs_err", "mean"),
            width_h1=("width", "first"),
            width_h48=("width", "last"),
            width_mean=("width", "mean"),
        )
    )
    summary.to_csv(out / "calibration_horizon_summary.csv")
    print("\nhorizon profile summary\n", summary.round(4).to_string())
    plot_horizon(profile, out)

    faith = pd.read_csv(args.output_root / "mlpgam_gate_analysis" / "rho_faith_all_datasets.csv")
    faith.to_csv(out / "gate_faithfulness.csv", index=False)
    print("\ngate faithfulness\n", faith.round(4).to_string(index=False))

    copy_figures(out, args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--study-runs", type=Path, nargs="+", help="architectural_study run directories")
    ap.add_argument("--gating-run", type=Path, help="legacy: design-ablation run dir containing the gating group")
    ap.add_argument("--calibration-run", type=Path, help="legacy: run dir containing conf_scp_pooled")
    ap.add_argument("--datasets", nargs="+", default=list(CHAPTER_DATASETS))
    ap.add_argument("--copy-to-images", action="store_true", help=f"copy figures to {IMAGES_DIR}")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    S.configure(latex=not args.no_latex)
    if args.study_runs:
        study_main(args)
    elif args.gating_run and args.calibration_run:
        legacy_main(args)
    else:
        ap.error("pass --study-runs, or both --gating-run and --calibration-run")


if __name__ == "__main__":
    main()
