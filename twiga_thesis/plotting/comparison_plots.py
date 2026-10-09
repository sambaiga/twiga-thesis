"""Shared comparison figures and labels for experiment analysis.

The plotting functions use the thesis figure style and accept chapter-owned
metric sets, labels, and reference conditions. Figures are returned to the
caller so each analysis script can choose its own output path and caption.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Literal

from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from twiga.experiment.analysis import MINIMIZE_METRICS

from twiga_thesis.plotting import style, tokens

# Only genuine acronyms/special-cased tokens need an entry; every other
# underscore-separated token is title-cased generically by display_name.
# Chapters extend this via display_name's ``extra_acronyms`` rather than
# mutating this dict, so one chapter's condition names never leak into
# another's labels by accident.
ACRONYM_TOKENS = {
    "mlpqf": "MLPQF",
    "nhits": "NHITS",
    "qr": "QR",
    "bnn": "BNN",
    "fpqr": "FPQR",
    "qvn": "QVN",
    "rope": "RoPE",
    "kuma": "Kumaraswamy",
    "sg": "SG",
    "nq": "NQ",
    "kl": "KL",
    "w1": "W1",
    "relu": "ReLU",
    "umnn": "UMNN",
    "sqda": "SQDA",
    "emb": "Emb",
    "mlpgam": "MLPGAM",
    "mlpf": "MLPF",
    "crc": "CRC",
    "scp": "SCP",
    "mae": "MAE",
    "nrmse": "NRMSE",
    "picp": "PICP",
    "nmpi": "NMPI",
    "cwe": "CWE",
}


def thesis_palette(count: int) -> list[tuple[float, float, float]]:
    """Return distinct categorical colors based on the thesis palette tokens.

    The ten house colors are used unchanged first. Larger comparisons add
    lighter variants of those colors instead of silently cycling duplicate
    hues, while keeping the same thesis palette as their visual source.
    """
    if count < 0:
        raise ValueError("Palette size must not be negative")
    base_colors = [to_rgb(color) for color in tokens.PALETTE]
    colors = []
    for index in range(count):
        base_color = base_colors[index % len(base_colors)]
        variation = index // len(base_colors)
        if variation == 0:
            colors.append(base_color)
            continue
        hue, lightness, saturation = colorsys.rgb_to_hls(*base_color)
        adjusted_lightness = lightness + (1 - lightness) * min(0.55, variation * 0.3)
        colors.append(colorsys.hls_to_rgb(hue, adjusted_lightness, saturation))
    return colors


def display_name(name: str, extra_acronyms: dict[str, str] | None = None) -> str:
    """Condition/group name -> thesis-cased display label.

    Known acronyms (``ACRONYM_TOKENS``, plus any ``extra_acronyms`` a caller
    passes for its own condition names) get their thesis casing; every other
    ``_``-separated token is title-cased (numbers passed through unchanged),
    so every label starts with a capital letter without hand-listing every
    condition name from every chapter in one shared dict.
    """
    tokens_map = ACRONYM_TOKENS if extra_acronyms is None else {**ACRONYM_TOKENS, **extra_acronyms}
    tokens = name.split("_")
    labels = [tokens_map.get(token.lower(), token if token.isdigit() else token.capitalize()) for token in tokens]
    return " ".join(labels)


def display_metric(metric: str, metric_display: dict[str, str] | None = None) -> str:
    """Metric column name -> thesis-cased display label, falling back to :func:`display_name`."""
    if metric_display and metric in metric_display:
        return metric_display[metric]
    return display_name(metric)


def head_label(
    condition: str,
    backbone_prefixes: tuple[str, ...] = (),
    extra_acronyms: dict[str, str] | None = None,
) -> str:
    """Display label for one benchmark head, stripping a redundant backbone prefix.

    ``backbone_prefixes`` lets a caller strip a condition-name prefix that
    its own figure/caption already states elsewhere (e.g. Chapter 5's
    ``benchmark_nhits`` group prefixes every condition with ``nhits_``).
    Falls through to :func:`display_name` either way.
    """
    for prefix in backbone_prefixes:
        if condition.startswith(prefix):
            condition = condition[len(prefix) :]
            if condition == "mlpqf":
                return "FPQR"
            break
    return display_name(condition, extra_acronyms=extra_acronyms)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, rng=None) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    rng = rng or np.random.default_rng()
    bootstrap_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(bootstrap_means, alpha)), float(np.quantile(bootstrap_means, 1 - alpha))


def plot_forest(
    fold_df: pd.DataFrame,
    dataset: str,
    group: str,
    reference: str,
    metric_cols: list[str],
    metric_display: dict[str, str] | None = None,
) -> dict[str, plt.Figure]:
    """Create one standalone bootstrap-confidence-interval figure per metric.

    The reference condition is drawn in the palette's primary teal; every
    other condition in muted grey, so the comparison the plot exists to make
    -- everything else against the reference -- is the one colour marks,
    rather than every condition competing for attention equally.

    Returns one full-width (:data:`~thesis_figures.style.TEXTWIDTH_IN`),
    single-panel figure per metric rather than one metric per subplot in a
    shared grid: a thesis figure environment takes one image at its own
    caption, not a pre-composed panel grid, and a fixed multi-panel grid
    forces every metric's plot to the same cramped size regardless of how
    many conditions it lists. Height instead scales with the number of
    conditions (the same per-row convention already used by
    ``plot_dm_significance_matrix``/Chapter 5's own ``plot_computation_time``/
    ``plot_good_bad_day``), so a 3-condition comparison and a 15-condition
    one are each sized to their own content instead of sharing one grid cell.

    ``metric_cols`` and ``metric_display`` are the caller's own metric set
    and display-name overrides; nothing here defaults to any one chapter's
    metrics.

    Args:
        fold_df: One row per (condition, Folds), as returned by ``fold_level``.
        dataset: Dataset identifier for API symmetry with other comparisons.
        group: Condition group identifier for API symmetry with other comparisons.
        reference: Reference condition to highlight.
        metric_cols: Metric columns to plot, one figure each.
        metric_display: Optional metric-name -> display-label overrides.

    Returns:
        ``{metric: Figure}``, one entry per ``metric_cols``. Save with
        :func:`save_forest_figures`.
    """
    rng = np.random.default_rng(42)
    rows = []
    for metric in metric_cols:
        for condition, condition_rows in fold_df.groupby("condition"):
            # NaN-safe: a metric like Brier skill score/relative economic
            # value is undefined (0/0) whenever a fold's realised event rate
            # is exactly 0 or 1, so a condition's own column can be a mix of
            # real values and NaN rather than either all-present or
            # all-absent. `values.mean()`/`bootstrap_ci` on an
            # unfiltered array both propagate a single NaN into every
            # statistic, silently blanking the whole row from the plot.
            values = condition_rows[metric].to_numpy()
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            ci_lower, ci_upper = bootstrap_ci(values, rng=rng)
            rows.append(
                {"metric": metric, "condition": condition, "mean": values.mean(), "lo": ci_lower, "hi": ci_upper}
            )
    data = pd.DataFrame(rows)

    n_conditions = fold_df["condition"].nunique()
    height = 0.35 * n_conditions + 1.2
    figures: dict[str, plt.Figure] = {}
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(style.TEXTWIDTH_IN, height))
        metric_data = data[data["metric"] == metric].sort_values("mean")
        ref_mean = metric_data.loc[metric_data["condition"] == reference, "mean"]
        if not ref_mean.empty:
            ax.axvline(ref_mean.iloc[0], color=tokens.TEXT_MUTED, lw=0.7, ls="--", zorder=0)
        for row_index, (_, row) in enumerate(metric_data.iterrows()):
            color = tokens.PRIMARY if row["condition"] == reference else tokens.TEXT_MUTED
            ax.errorbar(
                row["mean"],
                row_index,
                xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                fmt="o",
                color=color,
                ms=5,
            )
        ax.set_yticks(range(len(metric_data)))
        ax.set_yticklabels([style.tex_safe(display_name(condition)) for condition in metric_data["condition"]])
        ax.set_xlabel(style.tex_safe(display_metric(metric, metric_display)))
        # No in-figure title -- a caption carries the dataset/group/reference
        # (matching :func:`plot_dm_significance_matrix`'s own convention).
        xlim = style.pad_limits(metric_data["lo"].to_numpy(), metric_data["hi"].to_numpy())
        if xlim is not None:
            ax.set_xlim(*xlim)
        style.format_axes(ax)
        fig.tight_layout()
        figures[metric] = fig
    return figures


def plot_metric_bars(
    fold_df: pd.DataFrame,
    dataset: str,
    group: str,
    reference: str,
    metric_cols: list[str],
    metric_display: dict[str, str] | None = None,
    lower_better: set[str] | None = None,
    higher_better: set[str] | None = None,
    target_values: dict[str, float] | None = None,
    backbone_prefixes: tuple[str, ...] = (),
    extra_acronyms: dict[str, str] | None = None,
) -> dict[str, plt.Figure]:
    """Bar chart (``sns.catplot``, matching Chapter 5's own bar-chart convention), one figure per metric.

    :func:`plot_forest`'s dot/bootstrap-CI convention reads well for a
    *ranking* comparison, but for a plain condition-by-condition value
    comparison (e.g. CWE stratified) a bar chart is the more familiar,
    thesis-standard presentation -- the same ``sns.catplot(kind="bar",
    estimator="median", errorbar="sd", capsize=...)`` recipe Chapter 5's own
    ``plot_primary_metric_comparison``/``plot_normalized_comparison`` use,
    applied here per (dataset, metric) instead of faceted across datasets,
    since this function is already scoped to one dataset by its caller.

    The reference condition is still highlighted in the palette's primary
    teal, every other condition in muted grey -- the same visual role
    :func:`plot_forest` gives it, just carried over to a bar's fill colour
    instead of a dot's.

    Returns:
        ``{metric: Figure}``, one entry per ``metric_cols``. Save with
        :func:`save_forest_figures` (pass ``kind="bar"``).
    """
    lower_better = set() if lower_better is None else lower_better
    higher_better = set() if higher_better is None else higher_better
    target_values = {"picp": 0.90} if target_values is None else target_values
    n_conditions = fold_df["condition"].nunique()
    height = 0.35 * n_conditions + 1.2
    figures: dict[str, plt.Figure] = {}
    for metric in metric_cols:
        if metric not in fold_df.columns:
            continue
        data = fold_df[["condition", metric]].replace([np.inf, -np.inf], np.nan).dropna(subset=[metric]).copy()
        if data.empty:
            continue
        data["Condition"] = data["condition"].map(
            lambda condition: style.tex_safe(head_label(condition, backbone_prefixes, extra_acronyms))
        )
        condition_values = data.groupby("Condition")[metric].median()
        if metric in target_values:
            order = (condition_values - target_values[metric]).abs().sort_values().index
        else:
            if metric in lower_better:
                direction = "min"
            elif metric in higher_better:
                direction = "max"
            else:
                direction = "min" if metric.upper() in MINIMIZE_METRICS else "max"
            order = condition_values.sort_values(ascending=direction == "min").index

        colors = dict(zip(order, thesis_palette(len(order)), strict=True))
        if len(order) <= 8:
            facet_grid = sns.catplot(
                data=data,
                x="Condition",
                y=metric,
                order=order,
                hue="Condition",
                hue_order=order,
                palette=colors,
                dodge=False,
                legend=False,
                estimator="median",
                errorbar="sd",
                kind="bar",
                capsize=0.4,
                err_kws={"color": tokens.TEXT_MUTED, "linewidth": 0.7},
                height=2.6,
                aspect=style.TEXTWIDTH_IN / 2.6,
            )
            axis = facet_grid.ax
            axis.tick_params(axis="x", labelrotation=45)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
                if label.get_text() == style.tex_safe(head_label(reference, backbone_prefixes, extra_acronyms)):
                    label.set_fontweight("bold")
        else:
            facet_grid = sns.catplot(
                data=data,
                y="Condition",
                x=metric,
                order=order,
                hue="Condition",
                hue_order=order,
                palette=colors,
                dodge=False,
                legend=False,
                estimator="median",
                errorbar="sd",
                kind="bar",
                capsize=0.4,
                err_kws={"color": tokens.TEXT_MUTED, "linewidth": 0.7},
                height=height,
                aspect=style.TEXTWIDTH_IN / height,
            )
            axis = facet_grid.ax
            for label in axis.get_yticklabels():
                if label.get_text() == style.tex_safe(head_label(reference, backbone_prefixes, extra_acronyms)):
                    label.set_fontweight("bold")
        ax = facet_grid.ax
        if len(order) <= 8:
            facet_grid.set_axis_labels("", style.tex_safe(display_metric(metric, metric_display)))
        else:
            facet_grid.set_axis_labels(style.tex_safe(display_metric(metric, metric_display)), "")
        # No in-figure title -- a caption carries the dataset/group/reference
        # (matching plot_forest's/plot_dm_significance_matrix's own convention).
        # No pad_limits() here, unlike plot_forest: a bar's length is only a
        # valid visual proxy for its value when the axis is zero-anchored
        # (seaborn's own "bar" kind already does this by default), so
        # epsilon-padding away from zero -- correct for a dot's *position* --
        # would instead truncate every bar and misrepresent the comparison.
        style.format_axes(ax)
        facet_grid.figure.tight_layout()
        figures[metric] = facet_grid.figure
    return figures


def save_metric_figures(
    figures: dict[str, plt.Figure], fig_dir: Path, stem: str, dpi: int = 600, kind: str = "forest"
) -> None:
    """Save and close a collection of per-metric figures.

    Saved as ``{stem}_{kind}_{metric}.pdf`` -- ``kind`` defaults to
    ``"forest"`` for :func:`plot_forest`'s own dot/CI figures; pass
    ``kind="bar"`` for :func:`plot_metric_bars`'s bar-chart figures instead,
    so the filename never claims a chart type the file does not contain.

    ``dpi=600``: publication quality for any rasterised element (markers,
    error caps) the vector PDF backend still rasterises; text and lines stay
    vector regardless.
    """
    for metric, fig in figures.items():
        fig.savefig(fig_dir / f"{stem}_{kind}_{metric}.pdf", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_metric_bar_grid(
    fold_frame: pd.DataFrame,
    dataset: str,
    group: str,
    reference: str,
    metric_cols: list[str],
    metric_display: dict[str, str] | None = None,
    lower_better: set[str] | None = None,
    higher_better: set[str] | None = None,
    target_values: dict[str, float] | None = None,
    backbone_prefixes: tuple[str, ...] = (),
    extra_acronyms: dict[str, str] | None = None,
    columns: int = 3,
    sharey: bool = True,
) -> plt.Figure | None:
    """Create a faceted vertical median-plus-SD bar figure for all requested metrics.

    ``sharey=False`` gives every metric facet its own y-axis, needed when the metrics
    differ by orders of magnitude (e.g. NRMSE next to the Winkler score).

    The layout follows the thesis comparison convention: one metric per facet,
    conditions on the x-axis, one condition color per bar, and one shared
    legend below the figure. The reference condition is bolded on the x-axis;
    all conditions remain colored so the figure can compare the full design
    rather than reducing the plot to a reference-versus-rest contrast.
    """
    if columns < 1:
        raise ValueError("columns must be positive")
    metric_display = {} if metric_display is None else metric_display
    lower_better = set() if lower_better is None else lower_better
    higher_better = set() if higher_better is None else higher_better
    target_values = {"picp": 0.90} if target_values is None else target_values
    available_metrics = [metric for metric in metric_cols if metric in fold_frame.columns]
    if not available_metrics or "condition" not in fold_frame.columns:
        return None

    rows = []
    for metric in available_metrics:
        metric_frame = fold_frame[["condition", metric]].replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        if metric_frame.empty:
            continue
        median_values = metric_frame.groupby("condition")[metric].median()
        if metric in target_values:
            order = (median_values - target_values[metric]).abs().sort_values().index
        elif metric in lower_better:
            order = median_values.sort_values().index
        elif metric in higher_better:
            order = median_values.sort_values(ascending=False).index
        else:
            order = median_values.sort_values().index
        for condition in order:
            condition_rows = metric_frame.loc[metric_frame["condition"] == condition, metric]
            rows.extend(
                {
                    "condition": condition,
                    "condition_label": head_label(condition, backbone_prefixes, extra_acronyms),
                    "metric": metric,
                    "metric_label": display_metric(metric, metric_display),
                    "value": value,
                }
                for value in condition_rows
            )

    data = pd.DataFrame(rows)
    if data.empty:
        return None
    condition_order = list(dict.fromkeys(data["condition_label"]))
    metric_order = [
        display_metric(metric, metric_display) for metric in available_metrics if metric in data["metric"].values
    ]
    palette = dict(zip(condition_order, thesis_palette(len(condition_order)), strict=True))
    facet_grid = sns.catplot(
        data=data,
        x="condition_label",
        y="value",
        hue="condition_label",
        hue_order=condition_order,
        order=condition_order,
        col="metric_label",
        col_order=metric_order,
        col_wrap=min(columns, len(metric_order)),
        estimator="median",
        errorbar="sd",
        kind="bar",
        capsize=0.4,
        palette=palette,
        legend=True,
        height=2.6,
        aspect=1.05,
        sharey=sharey,
    )
    facet_grid.set_axis_labels("", "")
    facet_grid.set_titles(col_template="{col_name}")
    for axis in facet_grid.axes.flat:
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
            if label.get_text() == style.tex_safe(head_label(reference, backbone_prefixes, extra_acronyms)):
                label.set_fontweight("bold")
        style.format_axes(axis)
    sns.move_legend(
        facet_grid,
        "lower center",
        ncol=min(len(condition_order), 5),
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        title="Condition:",
    )
    facet_grid.figure.tight_layout()
    return facet_grid.figure


def plot_forest_grid(
    fold_df: pd.DataFrame,
    dataset: str,
    group: str,
    reference: str,
    metric_cols: list[str],
    metric_display: dict[str, str] | None = None,
    backbone_prefixes: tuple[str, ...] = (),
    extra_acronyms: dict[str, str] | None = None,
    n_columns: int = 3,
) -> plt.Figure:
    """Create a compact multi-metric forest overview on a shared figure canvas.

    Individual metric axes use percentile-bootstrap confidence intervals,
    highlight the reference condition, and omit undefined metric values.
    Dataset and group are accepted so callers can retain a uniform comparison
    API; the surrounding report supplies figure context through its caption.
    """
    available_metrics = [metric for metric in metric_cols if metric in fold_df.columns]
    if not available_metrics:
        raise ValueError("At least one requested metric must be present in the fold frame")
    if n_columns < 1:
        raise ValueError("n_columns must be positive")

    random_generator = np.random.default_rng(42)
    results = []
    for metric in available_metrics:
        for condition, condition_frame in fold_df.groupby("condition"):
            values = condition_frame[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            lower, upper = bootstrap_ci(values, rng=random_generator)
            results.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "mean": values.mean(),
                    "lower": lower,
                    "upper": upper,
                }
            )

    plot_data = pd.DataFrame(results)
    column_count = min(n_columns, len(available_metrics))
    row_count = (len(available_metrics) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(style.TEXTWIDTH_IN, 2.2 * row_count),
        squeeze=False,
    )
    for metric_index, metric in enumerate(available_metrics):
        axis = axes[metric_index // column_count, metric_index % column_count]
        metric_frame = plot_data.loc[plot_data["metric"] == metric].sort_values("mean")
        reference_mean = metric_frame.loc[metric_frame["condition"] == reference, "mean"]
        if not reference_mean.empty:
            axis.axvline(reference_mean.iloc[0], color=tokens.TEXT_MUTED, linewidth=0.7, linestyle="--", zorder=0)
        for row_index, row in enumerate(metric_frame.itertuples(index=False)):
            color = tokens.PRIMARY if row.condition == reference else tokens.TEXT_MUTED
            axis.errorbar(
                row.mean,
                row_index,
                xerr=[[row.mean - row.lower], [row.upper - row.mean]],
                fmt="o",
                color=color,
                markersize=5,
            )
        axis.set_yticks(range(len(metric_frame)))
        axis.set_yticklabels(
            [
                style.tex_safe(head_label(condition, backbone_prefixes, extra_acronyms))
                for condition in metric_frame["condition"]
            ]
        )
        axis.set_xlabel(style.tex_safe(display_metric(metric, metric_display)))
        limits = style.pad_limits(metric_frame["lower"].to_numpy(), metric_frame["upper"].to_numpy())
        if limits is not None:
            axis.set_xlim(*limits)
        style.format_axes(axis)

    for empty_index in range(len(available_metrics), row_count * column_count):
        axes[empty_index // column_count, empty_index % column_count].set_axis_off()
    figure.tight_layout()
    return figure


def plot_dm_significance_matrix(
    p_matrix: pd.DataFrame, dataset: str, group: str, alpha: float = 0.05
) -> plt.Figure | None:
    """Plot pairwise DM significance as a binary, thesis-styled matrix.

    Cells with adjusted ``p < alpha`` are marked significant; all other
    values, including undefined diagonal entries, are marked not significant.
    White crosses identify the diagonal, and the colorbar names both classes.
    The figure has no title so its thesis caption can carry the context.

    Args:
        p_matrix: Square matrix of adjusted p-values, with condition names as
            both its index and columns.
        dataset: Dataset label used by the caller for its caption or filename.
        group: Experiment group label used by the caller for its caption or
            filename.
        alpha: Significance threshold. Values equal to ``alpha`` are not
            significant.

    Returns:
        A Matplotlib figure, or ``None`` when fewer than two conditions are
        present.
    """
    import seaborn as sns

    conditions = list(p_matrix.index)
    if len(conditions) < 2:
        return None

    fig, ax = plt.subplots(figsize=(0.6 * len(conditions) + 2.2, 0.6 * len(conditions) + 1.8))
    p_values = p_matrix.to_numpy(dtype=float, copy=True)
    p_values_image = np.where(p_values < alpha, 0, 1).astype(int)
    cmap = sns.light_palette("#525252", as_cmap=True)
    image = ax.imshow(p_values_image, cmap=cmap, vmin=0, vmax=1)
    ax.plot(range(len(conditions)), range(len(conditions)), "wx")

    labels = [style.tex_safe(head_label(condition)) for condition in conditions]
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(labels)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    colorbar.ax.set_yticklabels([r"$p < \alpha$", r"$p \geq \alpha$"])
    colorbar.set_label(rf"BH-adjusted DM test ($\alpha = {alpha:g}$)")

    style.format_axes(ax)
    return fig


def plot_good_bad_day(
    representative: pd.DataFrame, dataset: str, group: str, metric: str = "nrmse"
) -> plt.Figure | None:
    """Plot the best-to-worst representative value per condition as a dumbbell.

    The story this figure needs to tell is the *gap* between a condition's
    best and worst day, not the two absolute values side by side -- a
    dumbbell's connecting line makes that gap the thing the eye reads first,
    the way two independently-scaled bars do not.

    Args:
        representative: Best/worst-instance table, as
            :func:`twiga.experiment.analysis.flag_representative_instances`
            (or its ``*_representative.csv`` output) produces -- one row per
            (dataset, group, condition, kind in {"best", "worst"}).
        dataset: Dataset name to scope to.
        group: Condition group name to scope to.
        metric: Column to plot. Defaults to Chapter 5's own ``"nrmse"``; pass
            a different column for a chapter whose representative-instance
            table is ranked on a different metric.

    Returns:
        The dumbbell figure, or ``None`` if there is nothing to plot for
        this (dataset, group) or ``metric`` is not a ranked column.
    """
    group_rows = representative[(representative["dataset"] == dataset) & (representative["group"] == group)]
    if group_rows.empty or metric not in group_rows.columns:
        return None
    aggregated = group_rows.groupby(["condition", "kind"])[metric].mean().unstack("kind")
    if "best" not in aggregated.columns or "worst" not in aggregated.columns:
        return None
    # Sorted by the gap (worst - best), not the raw worst value: the figure's
    # own point is day-to-day robustness, so the most consistent condition
    # (smallest gap) anchors the plot, not merely whichever has the lowest
    # worst-day error.
    gap = (aggregated["worst"] - aggregated["best"]).sort_values()
    order = gap.index

    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH_IN, 0.4 * len(order) + 1.5))
    y = np.arange(len(order))
    for row_index, condition in enumerate(order):
        ax.plot(
            [aggregated.loc[condition, "best"], aggregated.loc[condition, "worst"]],
            [row_index, row_index],
            color=tokens.TEXT_MUTED,
            lw=1.0,
            zorder=1,
        )
    ax.scatter(
        [aggregated.loc[condition, "best"] for condition in order],
        y,
        color=tokens.PRIMARY,
        s=30,
        zorder=2,
        label="Best Day",
    )
    ax.scatter(
        [aggregated.loc[condition, "worst"] for condition in order],
        y,
        color=tokens.HIGHLIGHT,
        s=30,
        zorder=2,
        label="Worst Day",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([style.tex_safe(display_name(condition)) for condition in order])
    ax.set_xlabel(display_metric(metric).upper() if len(metric) <= 5 else display_metric(metric))
    ax.legend(loc="lower right")
    # No in-figure title -- a caption carries the dataset/group (matching
    # :func:`plot_dm_significance_matrix`'s own convention).
    xlim = style.pad_limits(aggregated["best"].to_numpy(), aggregated["worst"].to_numpy())
    if xlim is not None:
        ax.set_xlim(*xlim)
    style.format_axes(ax)
    fig.tight_layout()
    return fig


def plot_normalized_comparison(
    full: pd.DataFrame,
    group: str,
    metrics: list[tuple[str, bool]] | None = None,
    legend_title: str = "Prob-Head:",
    metric_display: dict[str, str] | None = None,
) -> plt.Figure | None:
    """Compare conditions on metrics normalized within each dataset and metric.

    A metric like Winkler score is not directly comparable across datasets
    that differ in scale by an order of magnitude or more, so plotting the
    raw values on one shared axis makes the smaller-scale dataset look
    trivially "better". Each metric is min-max normalised **within its own
    (dataset, metric) group** to a skill-like score, ``(worst - value) /
    (worst - best)``, so 0 is always the worst-performing condition on that
    dataset and 1 the best -- the same direction a stratified coverage-width
    metric's own [0, 1] scale already reads in, so every panel shares one
    y-axis interpretation without implying the raw values are on the same
    scale.

    ``metrics`` is a list of ``(metric_col, lower_is_better)`` pairs,
    defaulting to Chapter 5's own ``[("winkler_score", True), ("crps",
    True), ("cwe_stratified", False)]`` -- pass a different list to reuse
    this function for a chapter whose metric set differs.
    """
    metrics = metrics if metrics is not None else [("winkler_score", True), ("crps", True), ("cwe_stratified", False)]
    metric_names = [name for name, _ in metrics if name in full.columns]

    group_rows = full[full["group"] == group]
    if group_rows.empty or not metric_names:
        return None
    fold_df = group_rows.groupby(["dataset", "condition", "Folds"], as_index=False)[metric_names].mean()

    rows = []
    for dataset, dataset_rows in fold_df.groupby("dataset"):
        for metric, lower_better in metrics:
            if metric not in metric_names:
                continue
            per_condition = dataset_rows.groupby("condition")[metric].mean()
            value_min, value_max = per_condition.min(), per_condition.max()
            if value_max == value_min:
                continue
            for condition, value in per_condition.items():
                if lower_better:
                    normalised_score = (value_max - value) / (value_max - value_min)
                else:
                    normalised_score = (value - value_min) / (value_max - value_min)
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": head_label(condition),
                        "metric": display_metric(metric, metric_display) + " (Normalised)",
                        "value": normalised_score,
                    }
                )
    data = pd.DataFrame(rows)
    if data.empty:
        return None

    facet_grid = sns.catplot(
        data=data,
        y="value",
        x="dataset",
        hue="condition",
        col="metric",
        estimator="median",
        errorbar="sd",
        kind="bar",
        capsize=0.4,
        palette=thesis_palette(data["condition"].nunique()),
        err_kws={"color": ".4", "linewidth": 0.7},
        height=2.6,
        aspect=1.0,
        legend=True,
    )
    facet_grid.set_axis_labels("", "Normalised score (0 = worst, 1 = best)")
    facet_grid.set_titles(col_template="{col_name}")
    facet_grid.set(ylim=(0, 1))
    sns.move_legend(facet_grid, "lower center", ncol=4, bbox_to_anchor=(0.5, -0.18), frameon=False, title=legend_title)
    facet_grid.figure.tight_layout()
    return facet_grid.figure


def plot_primary_metric_comparison(
    full: pd.DataFrame,
    group: str,
    primary_metric_cols: list[str],
    metric_display: dict[str, str] | None = None,
    log_metrics: set[str] | None = None,
    legend_title: str = "Prob-Head:",
    backbone_prefixes: tuple[str, ...] = (),
    extra_acronyms: dict[str, str] | None = None,
) -> plt.Figure | None:
    """Compare raw fold metrics across datasets with one independently scaled panel per metric.

    Bars show the median across folds with standard-deviation error bars.
    Each metric gets its own y-axis, and configured metrics can use a log
    axis when their positive values span substantially different scales.
    """
    metric_display = {} if metric_display is None else metric_display
    log_metrics = set() if log_metrics is None else log_metrics
    available_metrics = [metric for metric in primary_metric_cols if metric in full.columns]
    if not available_metrics:
        return None

    scoped = full.loc[full["group"] == group]
    if scoped.empty:
        return None
    fold_frame = scoped.groupby(["dataset", "condition", "Folds"], as_index=False)[available_metrics].mean()
    long_frame = fold_frame.melt(
        id_vars=["dataset", "condition", "Folds"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )
    long_frame["condition_label"] = long_frame["condition"].map(
        lambda condition: head_label(condition, backbone_prefixes, extra_acronyms)
    )
    long_frame["metric_label"] = long_frame["metric"].map(lambda metric: display_metric(metric, metric_display))
    hue_order = list(dict.fromkeys(long_frame["condition_label"]))
    metric_order = [display_metric(metric, metric_display) for metric in available_metrics]

    facet_grid = sns.catplot(
        data=long_frame,
        y="value",
        x="dataset",
        hue="condition_label",
        hue_order=hue_order,
        col="metric_label",
        col_order=metric_order,
        sharey=False,
        margin_titles=True,
        height=2.5,
        aspect=1.25,
        legend=True,
        estimator="median",
        errorbar="sd",
        kind="bar",
        capsize=0.4,
        err_kws={"color": tokens.TEXT_MUTED, "linewidth": 0.5},
        palette=thesis_palette(len(hue_order)),
    )
    facet_grid.set_axis_labels("Dataset", "")
    facet_grid.set_titles(col_template="{col_name}")
    for metric in log_metrics:
        if metric not in available_metrics:
            continue
        metric_label = display_metric(metric, metric_display)
        metric_values = long_frame.loc[long_frame["metric"] == metric, "value"]
        if (metric_values.dropna() <= 0).any():
            raise ValueError(f"Log-scaled metric {metric!r} must contain only positive values")
        for axis in facet_grid.axes.flat:
            if axis.get_title() == metric_label:
                axis.set_yscale("log")
    legend_columns = max(1, len(hue_order))
    sns.move_legend(
        facet_grid,
        "lower center",
        ncol=legend_columns,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        title=legend_title,
    )
    for axis in facet_grid.axes.flat:
        style.format_axes(axis)
    facet_grid.figure.tight_layout()
    return facet_grid.figure
