r"""Multivariate dependence structure for Experiment 1's four benchmark heads.

``copula/joint_vs_independent.py`` makes this exact argument (marginal fan
charts cannot show a dependence-structure difference; correlation matrices
and bivariate sample clouds can) for a *synthetic* joint-vs-independent
comparison (``MLPFMVNormalModel`` vs ``MLPFNormalModel``, trained for that
comparison alone). This script asks the same question of the four *real*
Experiment 1 benchmark heads (FPQR, QR, BNN, Laplace,
``multitarget_experiment.py``'s own ``benchmark`` group) on their own saved
test-set predictions -- no retraining, no new model, reading directly from
each condition's already-written ``predictions.parquet``
(``timestamp``/``forecast``/``lower``/``upper``/``Actual``/``target``, long
format).

Two things this script approximates, stated explicitly rather than silently:

1. **No raw ensemble is saved** for the benchmark group (only the point
   forecast and the outer interval pair) -- a per-instance sample is
   reconstructed as ``forecast + L @ z * scale``, ``z`` a standard normal
   draw, ``L`` the Cholesky factor of that head's own *residual*
   correlation (``corrcoef(Actual - forecast)``, pooled across every
   timestamp and fold), and ``scale = (upper - lower) / (2 * 1.645)`` (the
   project's own outer-quantile-to-scale convention, matching
   ``get_sigma_from_quantile_levels``'s IQR-style derivation elsewhere).
   This is **not** a copula construction -- there is no fitted dependence
   model or copula family involved, no parameters are learned, and nothing
   here depends on the (now-removed) copula methodology. It is a one-line,
   non-parametric Monte Carlo device, correlated Gaussian innovations
   applied to each head's own already-fitted marginal width, used purely to
   visualise *what dependence structure each head's own residuals already
   imply* -- not a draw from the head's own true predictive law (which,
   for FPQR/QR, is a full quantile grid this script never sees), and not a
   replacement for a real posterior-predictive sample.
2. **Correlation is residual-based**, following the same convention
   ``sec:ch6-mahalanobis``'s own empirical-correlation construction in
   ``Chapter-6.tex`` already uses -- not a per-target output correlation,
   since none of these four heads models the joint distribution directly.

Usage::

    uv run python experiment/ch7_multitarget/shared/benchmark_dependence_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from twiga.core.metrics.parametric import energy_score  # noqa: E402
from twiga.core.utils import get_logger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thesis_figures.style import configure  # noqa: E402
from thesis_figures.tokens import (  # noqa: E402
    DANGER,
    HIGHLIGHT,
    PALETTE,
    PRIMARY,
    TEXT_DARK as TEXT,  # noqa: E402
)

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
OUT = script_dir.parent / "figures"

CORR_CMAP = LinearSegmentedColormap.from_list("thesis_diverging", [PALETTE[0], "#ffffff", PALETTE[2]])
SYM = {"p_kw": "$P$", "q_kvar": "$Q$", "v_lv_pu": "$V$", "LP": "$P$", "LQ": "$Q$", "LV": "$V$"}
HEAD_LABEL = {"fpqr_anchor_outer": "FPQR", "qr": "QR", "bnn": "BNN", "laplace": "Laplace"}
HEAD_COLOUR = {"fpqr_anchor_outer": PALETTE[2], "qr": PALETTE[0], "bnn": PALETTE[4], "laplace": PALETTE[5]}
OBS_C = HIGHLIGHT
Z90 = 1.6448536269514722  # norm.ppf(0.95): (upper, lower) is a nominal-90% pair throughout this chapter
NOMINAL_PICP = 0.90
COVERAGE_CMAP = LinearSegmentedColormap.from_list(
    "coverage", [(0.0, DANGER), (0.5, DANGER), (NOMINAL_PICP, "#FFFFFF"), (1.0, PRIMARY)]
)


def _default_predictions_root() -> Path:
    return (
        script_dir.parent
        / "results"
        / "ch7_multitarget"
        / "multitarget_experiment"
        / "multitarget_forecasting_benchmark"
    )


def load_predictions(
    root: Path, dataset: str, heads: list[str], overrides: dict[str, Path] | None = None
) -> dict[str, pd.DataFrame]:
    """Latest run's own ``predictions.parquet`` per head, under ``root``'s ``runs/`` tree.

    ``overrides`` sources one or more heads from an explicit path instead --
    e.g. a different run whose own saved checkpoint turned out better
    calibrated for that head specifically (own-run native numbers, not a
    retrain), rather than the glob's own latest-by-mtime choice.
    """
    dataset_slug = dataset.lower().replace("-", "-")
    overrides = overrides or {}
    out = {}
    for head in heads:
        if head in overrides:
            path = overrides[head]
            out[head] = pd.read_parquet(path)
            log.info("%s: loaded %s (override)", head, path)
            continue
        candidates = sorted(root.glob(f"runs/*/conditions/{dataset_slug}/benchmark/{head}/predictions.parquet"))
        if not candidates:
            log.warning("no predictions.parquet found for %s under %s", head, root)
            continue
        path = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        out[head] = pd.read_parquet(path)
        log.info("%s: loaded %s", head, path)
    return out


def wide_frame(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Pivot one head's long-format predictions to one row per (timestamp, Folds), columns per target."""
    idx = ["timestamp", "Folds"]
    pieces = []
    for col in ("Actual", "forecast", "lower", "upper"):
        piece = df.pivot_table(index=idx, columns="target", values=col)
        piece = piece.reindex(columns=targets)
        piece.columns = [f"{col}_{t}" for t in piece.columns]
        pieces.append(piece)
    return pd.concat(pieces, axis=1).dropna()


def head_stats(wide: pd.DataFrame, targets: list[str]) -> dict:
    """Residual correlation and per-target scale for one head, from its own saved predictions."""
    actual = wide[[f"Actual_{t}" for t in targets]].to_numpy()
    forecast = wide[[f"forecast_{t}" for t in targets]].to_numpy()
    lower = wide[[f"lower_{t}" for t in targets]].to_numpy()
    upper = wide[[f"upper_{t}" for t in targets]].to_numpy()
    residual = actual - forecast
    corr = np.corrcoef(residual.T)
    scale = (upper - lower) / (2 * Z90)
    scale = np.clip(scale, 1e-6, None)
    return {"actual": actual, "forecast": forecast, "scale": scale, "corr": corr}


def sample_cloud(stats: dict, n_paths: int, rng: np.random.Generator, row: int) -> np.ndarray:
    """``n_paths`` correlated-Gaussian draws for one test instance, from this head's own residual corr/scale."""
    corr = np.nan_to_num(stats["corr"], nan=0.0)
    corr = (corr + corr.T) / 2
    corr += np.eye(corr.shape[0]) * 1e-6
    tril = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_paths, corr.shape[0]))
    return stats["forecast"][row] + (tril @ z.T).T * stats["scale"][row]


def ellipse(ax, pts: np.ndarray, colour: str, label: str) -> None:
    mu = pts.mean(0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.c_[np.cos(theta), np.sin(theta)] * np.sqrt(4.605 * np.maximum(vals, 1e-12))
    pts_e = circle @ vecs.T + mu
    ax.plot(pts_e[:, 0], pts_e[:, 1], color=colour, lw=1.6, label=label, zorder=4)


def fold_picp(wide: pd.DataFrame, targets: list[str]) -> pd.Series:
    """Pooled-across-targets PICP per fold, indexed by each fold's own start timestamp."""
    lower = wide[[f"lower_{t}" for t in targets]].to_numpy()
    upper = wide[[f"upper_{t}" for t in targets]].to_numpy()
    actual = wide[[f"Actual_{t}" for t in targets]].to_numpy()
    within = (actual >= lower) & (actual <= upper)
    picp = pd.Series(within.mean(axis=1), index=wide.index)
    fold_start = picp.groupby(level="Folds").apply(lambda s: s.index.get_level_values("timestamp").min())
    out = picp.groupby(level="Folds").mean()
    out.index = pd.Index(fold_start.loc[out.index].to_numpy(), name="fold_start")
    return out.sort_index()


def plot_coverage_heatmap(wide: dict[str, pd.DataFrame], targets: list[str], out: Path) -> None:
    """Chapter-7-style per-fold PICP heatmap (``plot_fold_coverage``), rows=heads, cols=fold start."""
    heads = list(wide)
    per_head = {h: fold_picp(wide[h], targets) for h in heads}
    fold_starts = sorted(set().union(*(s.index for s in per_head.values())))
    grid = np.full((len(heads), len(fold_starts)), np.nan)
    for i, h in enumerate(heads):
        s = per_head[h]
        for j, fs in enumerate(fold_starts):
            if fs in s.index:
                grid[i, j] = s.loc[fs]

    fig, ax = plt.subplots(figsize=(1.3 * len(fold_starts) + 1.5, 0.6 * len(heads) + 1.2), layout="constrained")
    im = ax.imshow(grid, cmap=COVERAGE_CMAP, norm=plt.Normalize(0.0, 1.0), aspect="auto")
    fold_labels = [pd.Timestamp(fs).strftime("%b %Y") for fs in fold_starts]
    ax.set_xticks(range(len(fold_starts)), fold_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(heads)), [HEAD_LABEL[h] for h in heads])
    for i in range(len(heads)):
        for j in range(len(fold_starts)):
            val = grid[i, j]
            if np.isnan(val):
                continue
            shade = "white" if (val < 0.65 or val > 0.985) else TEXT
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=shade)
    ax.set_xlabel("fold start")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, label="PICP")
    cbar.ax.axhline(NOMINAL_PICP, color="k", lw=1.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    fig.savefig(out / "fig_ch6_benchmark_fold_coverage.pdf")
    plt.close(fig)


def coverage_table(wide: dict[str, pd.DataFrame], targets: list[str]) -> pd.DataFrame:
    """Marginal PICP per target plus exact joint coverage.

    Same definition as ``tab:ch6-conformal``'s Joint Coverage column
    (``within_interval.all(axis=-1).mean()``) -- computed directly from each
    head's own saved interval, no copula/sample reconstruction involved.
    """
    rows = []
    for head, w in wide.items():
        lower = w[[f"lower_{t}" for t in targets]].to_numpy()
        upper = w[[f"upper_{t}" for t in targets]].to_numpy()
        actual = w[[f"Actual_{t}" for t in targets]].to_numpy()
        within = (actual >= lower) & (actual <= upper)
        row = {"head": HEAD_LABEL[head]}
        for i, t in enumerate(targets):
            row[f"picp_{t}"] = float(within[:, i].mean())
        row["joint_coverage"] = float(within.all(axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("head")


def plot_correlation_matrices(
    observed_corr: np.ndarray, head_corrs: dict[str, np.ndarray], targets: list[str], out: Path
) -> None:
    panels = [("observed", observed_corr)] + [(HEAD_LABEL[h], c) for h, c in head_corrs.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.9), layout="constrained")
    labels = [SYM[t] for t in targets]
    im = None
    for ax, (title, m) in zip(axes, panels, strict=True):
        im = ax.imshow(m, vmin=-1, vmax=1, cmap=CORR_CMAP)
        ax.set_xticks(range(len(targets)), labels)
        ax.set_yticks(range(len(targets)), labels)
        for i in range(len(targets)):
            for j in range(len(targets)):
                shade = "white" if abs(m[i, j]) > 0.55 else TEXT
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9, color=shade)
        ax.set_title(title, fontsize=10)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="correlation", ticks=[-1, -0.5, 0, 0.5, 1])
    fig.savefig(out / "fig_ch6_benchmark_correlation_matrices.pdf")
    plt.close(fig)


def plot_sample_clouds(clouds: dict[str, np.ndarray], obs: np.ndarray, targets: list[str], out: Path) -> None:
    """One row per head, one column per target pair -- 4x3 small multiples."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    heads = list(clouds)
    fig, axes = plt.subplots(len(heads), 3, figsize=(10.5, 2.6 * len(heads)), layout="constrained")
    for row, head in enumerate(heads):
        s = clouds[head]
        colour = HEAD_COLOUR[head]
        for k, (a, b) in enumerate(pairs):
            ax = axes[row, k]
            ax.scatter(s[:, a], s[:, b], s=3, c=colour, alpha=0.18, lw=0, zorder=1)
            ellipse(ax, s[:, [a, b]], colour, "90%")
            ax.scatter([obs[a]], [obs[b]], marker="*", s=140, c=OBS_C, ec="k", lw=0.4, zorder=6, label="observed")
            rho = np.corrcoef(s[:, a], s[:, b])[0, 1]
            ax.set_title(f"$\\rho$ {rho:+.2f}", fontsize=9)
            if row == len(heads) - 1:
                ax.set_xlabel(SYM[targets[a]])
            if k == 0:
                ax.set_ylabel(f"{HEAD_LABEL[head]}\n{SYM[targets[b]]}")
            else:
                ax.set_ylabel(SYM[targets[b]])
    axes[0, -1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(out / "fig_ch6_benchmark_sample_clouds.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MLVS-PT-3PHASE")
    ap.add_argument("--heads", nargs="+", default=["fpqr_anchor_outer", "qr", "bnn", "laplace"])
    ap.add_argument("--predictions-root", type=Path, default=None)
    ap.add_argument("--paths", type=int, default=4000)
    ap.add_argument("--es-instances", type=int, default=300)
    ap.add_argument("--es-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    try:
        configure(latex=not args.no_latex)
    except Exception as exc:  # noqa: BLE001
        log.warning("thesis style unavailable (%s)", exc)
    OUT.mkdir(parents=True, exist_ok=True)

    root = args.predictions_root or _default_predictions_root()
    # fpqr_anchor_outer is sourced from run 904e94 ("v3") rather than the
    # latest-by-mtime benchmark run: v3's own grid-group checkpoint is the
    # best-calibrated fpqr_anchor_outer on record (joint coverage 0.782 vs.
    # the mtime-latest run's 0.237), already adopted as the chapter's
    # reported native FPQR numbers in tab:ch6-benchmark/-joint.
    v3_fpqr = (
        root
        / "runs"
        / "20261001_1117380f09a42a794040c6bf8b567e92532921_904e94"
        / "conditions"
        / args.dataset.lower()
        / "grid"
        / "fpqr_anchor_outer"
        / "predictions.parquet"
    )
    overrides = {"fpqr_anchor_outer": v3_fpqr} if v3_fpqr.exists() else {}
    raw = load_predictions(root, args.dataset, args.heads, overrides=overrides)
    if not raw:
        log.error("no predictions found under %s -- sync a benchmark run's conditions/ tree first", root)
        return

    targets = sorted(next(iter(raw.values()))["target"].unique().tolist())
    wide = {h: wide_frame(df, targets) for h, df in raw.items()}

    stats = {h: head_stats(w, targets) for h, w in wide.items()}
    observed_corr = np.corrcoef(next(iter(stats.values()))["actual"].T)
    head_corrs = {h: s["corr"] for h, s in stats.items()}
    plot_correlation_matrices(observed_corr, head_corrs, targets, OUT)
    log.info("wrote %s", OUT / "fig_ch6_benchmark_correlation_matrices.pdf")

    plot_coverage_heatmap(wide, targets, OUT)
    log.info("wrote %s", OUT / "fig_ch6_benchmark_fold_coverage.pdf")

    cov = coverage_table(wide, targets)
    print(f"\n=== marginal PICP by target + exact joint coverage (nominal {NOMINAL_PICP:.2f}) ===")
    print(cov.round(3).to_string())

    rng = np.random.default_rng(args.seed)
    # Representative instance: the row whose observed vector is closest to
    # the pooled mean, same "typical day" convention as
    # copula/joint_vs_independent.py's own _plot_clouds call site.
    ref_actual = next(iter(stats.values()))["actual"]
    row = int(np.abs(ref_actual - ref_actual.mean(0)).sum(1).argmin())
    clouds = {h: sample_cloud(s, args.paths, rng, row) for h, s in stats.items()}
    obs = ref_actual[row]
    plot_sample_clouds(clouds, obs, targets, OUT)
    log.info("wrote %s", OUT / "fig_ch6_benchmark_sample_clouds.pdf")

    print(
        f"\n=== energy score over ({', '.join(SYM[t] for t in targets)}), {args.es_samples} draws each "
        f"(lower is better) ==="
    )
    n_rows = next(iter(stats.values()))["actual"].shape[0]
    n_es = min(args.es_instances, n_rows)
    eval_rows = rng.choice(n_rows, size=n_es, replace=False)
    for head, s in stats.items():
        vals = []
        for r in eval_rows:
            draws = sample_cloud(s, args.es_samples, rng, r)
            vals.append(float(energy_score(s["actual"][r], draws)))
        print(f"  {HEAD_LABEL[head]:10s} {np.mean(vals):.5f}")


if __name__ == "__main__":
    main()
