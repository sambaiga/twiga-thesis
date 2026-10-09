r"""What each forecast type gives for multi-target P/Q/V: single-target probabilities and combined-target scenarios.

Reads the raw calibration- and test-window forecasts that ``fixed_benchmark.py --group raw_export`` saves and uses
each head through what it outputs (``FORECAST_TYPE_PLAN.md``):

* quantile heads (FPQR, QR): a quantile grid per target and step, completed with exponential tails;
* Laplace: a closed-form distribution per target and step;
* BNN: native scenarios, one weight draw shared by every target and lead time, plus its observation noise.

**A. Single target.** The probability that each target leaves its usual range -- P or Q above the 90th
percentile of its calibration-window observations, V below the 10th -- read from the head's own marginal (the
CDF for the quantile and Laplace heads, the share of native scenarios for the BNN). Scored with the Brier score,
its skill against the 10% base rate, and the Murphy reliability and resolution.

**B. Combined targets.** Scenarios of the whole day, ``M`` per window, in three forms:

* ``independent`` -- draws from each marginal on its own (no dependence across targets or lead times);
* ``schaake`` -- the same draws reordered to follow the ranks of ``M`` recent calibration days (an empirical
  copula: the joint pattern of past observations);
* ``native`` -- the BNN's own scenarios.

Scored on apparent power :math:`S = \sqrt{P^2 + Q^2}` (CRPS, a nonlinear combination no single-target forecast
gives), the energy score of (P, Q, V) per step, and the variogram score (p = 0.5) across targets and lead times
per window. The energy and variogram scores use targets standardised by their calibration-window spread. All
three are fair (unbiased) ensemble estimators, so ensembles of different size compare.

**Significance.** As everywhere in the thesis, the Diebold-Mariano test with Benjamini-Hochberg adjustment and the
Model Confidence Set (``twiga_thesis.analysis.dm_and_mcs``), here on per-window losses (one score per
forecast day), which gives hundreds of paired observations per record instead of a handful of folds: between heads
per target for A, between scenario forms per head for B.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/forecast_type_study.py \
        --run <experiments>/ch7_multitarget/forecast_type_study/fixed_benchmark/multitarget_fixed_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parents[1])]

from twiga.core.metrics.decision import probability_scores  # noqa: E402
from twiga.distributions.conformal.multitarget import Predictive  # noqa: E402
from twiga.distributions.scenarios import schaake_shuffle  # noqa: E402

from twiga_thesis.analysis import dm_and_mcs  # noqa: E402

ALPHA = 0.1
TARGETS = ("P", "Q", "V")
FAMILY = {"fpqr_anchor": "quantile", "qr": "quantile", "bnn": "normal", "laplace": "laplace"}
LABEL = {"fpqr_anchor": "FPQR", "qr": "QR", "bnn": "BNN", "laplace": "Laplace"}
# Tail event per target: above the calibration 90th percentile for P and Q, below the 10th for V.
EVENT = {"P": ("above", 0.9), "Q": ("above", 0.9), "V": ("below", 0.1)}
SEED = 20261007


def load_folds(condition_dir: Path) -> dict[int, tuple[dict, dict]]:
    """Each fold's saved (reference, test) arrays.

    The reference window supplies the event limits and the Schaake template: the calibration window where one is
    reserved (MLVS-PT), otherwise the validation window (the simulated records), both just before the test.
    """
    folds = {}
    for test in sorted((condition_dir / "raw").glob("fold_*_test.npz")):
        fold = int(test.stem.split("_")[1])
        reference = next(
            (path for w in ("calib", "val") if (path := test.with_name(f"fold_{fold:02d}_{w}.npz")).exists()), None
        )
        if reference is None:
            raise FileNotFoundError(f"no calibration or validation window saved beside {test}")
        folds[fold] = (dict(np.load(reference)), dict(np.load(test)))
    return folds


# ── Scores ────────────────────────────────────────────────────────────────────


def pair_mean_abs(x: np.ndarray) -> np.ndarray:
    """Mean of |X_i - X_j| over ordered pairs i != j along axis 0, from the sorted members."""
    m = x.shape[0]
    xs = np.sort(x, axis=0)
    weights = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (x.ndim - 1))
    return 2.0 * (weights * xs).sum(axis=0) / (m * (m - 1))


def crps_ensemble(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair CRPS of an ensemble ``x`` (M, ...) for observations ``y`` (...)."""
    return np.abs(x - y).mean(axis=0) - 0.5 * pair_mean_abs(x)


def energy_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair energy score over the last axis: ``x`` (M, ..., D), ``y`` (..., D)."""
    m = x.shape[0]
    first = np.linalg.norm(x - y, axis=-1).mean(axis=0)
    pairs = np.zeros(first.shape)
    for i in range(m):
        pairs += np.linalg.norm(x[i] - x, axis=-1).sum(axis=0)
    return first - 0.5 * pairs / (m * (m - 1))


def variogram_score(x: np.ndarray, y: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Variogram score of order ``p`` over the flattened margins: ``x`` (M, N, K), ``y`` (N, K); per window."""
    out = np.empty(y.shape[0])
    for n in range(y.shape[0]):
        obs = np.abs(y[n][:, None] - y[n][None, :]) ** p
        ens = (np.abs(x[:, n, :, None] - x[:, n, None, :]) ** p).mean(axis=0)
        out[n] = np.triu((obs - ens) ** 2, k=1).sum()
    return out


# ── Scenarios ─────────────────────────────────────────────────────────────────


def marginal_draws(pred: Predictive, m: int, rng: np.random.Generator) -> np.ndarray:
    """``m`` independent draws per cell from each marginal, ``(M, N, H, D)``."""
    return np.stack([pred.ppf(rng.uniform(size=pred.loc.shape)) for _ in range(m)])


def native_draws(arrays: dict, m: int, rng: np.random.Generator) -> np.ndarray:
    """``m`` of the BNN's own scenarios, each with its observation noise added, ``(M, N, H, D)``."""
    samples = arrays["samples"]
    idx = rng.choice(samples.shape[0], size=m, replace=samples.shape[0] < m)
    noise = np.broadcast_to(arrays.get("scale_noise", 0.0), samples.shape[1:])
    return samples[idx] + rng.normal(size=(m, *samples.shape[1:])) * noise


def scenario_sets(head: str, pred: Predictive, test: dict, calib: dict, m: int, rng) -> dict[str, np.ndarray]:
    """The scenario forms a head supports: independent and Schaake for every marginal, native for the BNN."""
    independent = marginal_draws(pred, m, rng)
    template = calib["actual"][-m:]  # the most recent calibration days, (M, H, D)
    sets = {"independent": independent, "schaake": schaake_shuffle(independent, template)}
    if "samples" in test:
        sets["native"] = native_draws(test, m, rng)
    return sets


# ── Analyses ──────────────────────────────────────────────────────────────────


def single_target(
    head: str, pred: Predictive, test: dict, calib: dict, rng, fold: int = 0
) -> tuple[list[dict], list[dict]]:
    """Analysis A: each target's tail-event probability from what the head outputs.

    Returns:
        Fold-level scores (Brier and its decomposition) and per-window Brier scores, the loss the DM test uses.
    """
    rows, window_rows = [], []
    y = test["actual"]
    native = native_draws(test, 200, rng) if "samples" in test else None
    for d, target in enumerate(TARGETS):
        side, level = EVENT[target]
        limit = float(np.quantile(calib["actual"][..., d], level))
        event = (y[..., d] > limit) if side == "above" else (y[..., d] < limit)
        if native is not None:
            share = (native[..., d] > limit).mean(axis=0) if side == "above" else (native[..., d] < limit).mean(axis=0)
            prob, ensemble, source = share, native.shape[0], "native"
        else:
            cdf = pred.cdf(np.full(pred.loc.shape, limit))[..., d]
            prob, ensemble, source = (1.0 - cdf) if side == "above" else cdf, None, "marginal"
        scores = probability_scores(
            prob, event.astype(float), reference_rate=1.0 - level if side == "above" else level, ensemble_size=ensemble
        )
        rows.append({"head": head, "target": target, "source": source, "rate": float(event.mean()), **scores})
        per_window = ((prob - event) ** 2).mean(axis=-1)
        window_rows += [
            {"head": head, "target": target, "window": f"{fold:02d}-{n:04d}", "brier_score": float(v)}
            for n, v in enumerate(per_window)
        ]
    return rows, window_rows


def combined_targets(
    head: str, pred: Predictive, test: dict, calib: dict, m: int, rng, fold: int = 0
) -> tuple[list[dict], list[dict]]:
    """Analysis B: apparent-power CRPS, energy and variogram scores for each scenario form.

    Returns:
        Fold means and per-window scores (the CRPS and energy score averaged over the window's lead times), the
        losses the DM test uses.
    """
    y = test["actual"]
    spread = calib["actual"].reshape(-1, y.shape[-1]).std(axis=0)
    s_obs = np.hypot(y[..., 0], y[..., 1])
    rows, window_rows = [], []
    for form, x in scenario_sets(head, pred, test, calib, m, rng).items():
        z, z_obs = x / spread, y / spread
        per_window = {
            "crps_apparent_power": crps_ensemble(np.hypot(x[..., 0], x[..., 1]), s_obs).mean(axis=-1),
            "energy_score": energy_score(z, z_obs).mean(axis=-1),
            "variogram_score": variogram_score(z.reshape(m, z.shape[1], -1), z_obs.reshape(z.shape[1], -1)),
        }
        rows.append({"head": head, "form": form, **{k: float(v.mean()) for k, v in per_window.items()}})
        window_rows += [
            {
                "head": head,
                "form": form,
                "window": f"{fold:02d}-{n:04d}",
                **{k: float(v[n]) for k, v in per_window.items()},
            }
            for n in range(y.shape[0])
        ]
    return rows, window_rows


COMBINED_METRICS = ("crps_apparent_power", "energy_score", "variogram_score")


def significance(single_windows: pd.DataFrame, combined_windows: pd.DataFrame) -> pd.DataFrame:
    """DM (Benjamini-Hochberg) and MCS on per-window losses: heads per target (A), scenario forms per head (B)."""
    rows = []

    def add(frame: pd.DataFrame, analysis: str, scope: str, metric: str, condition: str) -> None:
        if frame[condition].nunique() < 2:
            return
        table, _, mcs = dm_and_mcs(frame, metric=metric, fold_column="window", condition_column=condition)
        members = sorted(getattr(mcs, "included", getattr(mcs, "superior_set", [])) or [])
        for r in table.itertuples(index=False):
            rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "metric": metric,
                    "condition_a": r.condition_a,
                    "condition_b": r.condition_b,
                    "p_adjusted": r.p_adjusted,
                    "significant": r.significant,
                    "winner": r.winner,
                    "windows": frame["window"].nunique(),
                    "mcs": ", ".join(map(str, members)),
                }
            )

    for target, g in single_windows.groupby("target"):
        add(g, "A", target, "brier_score", "head")
    for head, g in combined_windows.groupby("head"):
        for metric in COMBINED_METRICS:
            add(g, "B", head, metric, "form")
    return pd.DataFrame(rows)


FORM_LABEL = {"independent": "independent draws", "schaake": "Schaake-coupled", "native": "BNN scenarios"}
PANEL_LABEL = ("P", "Q", "V", "S = sqrt(P^2 + Q^2)")


def plot_scenarios(dataset_dir: Path, heads: list[str], day: str, members: int, out: Path) -> None:
    """Sample paths of one forecast window per head: a row per scenario form, a column per P, Q, V and S.

    The window is the one issued on ``day``; ``members`` paths are drawn thin, the observations on top.
    """
    import matplotlib  # noqa: PLC0415
    import matplotlib.dates  # noqa: PLC0415

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415

    from twiga_thesis.plotting import tokens as T  # noqa: PLC0415

    rng = np.random.default_rng(SEED)
    target_day = pd.Timestamp(day, tz="UTC")
    colours = {"independent": T.PALETTE[5], "schaake": T.PALETTE[0], "native": T.PALETTE[2]}
    for head in heads:
        folds = load_folds(dataset_dir / "raw_export" / f"{head}_raw")
        issued = {fold: pd.to_datetime(test["timestamps"][:, 0, 0], utc=True) for fold, (_, test) in folds.items()}
        fold = next((f for f, origins in issued.items() if (origins.normalize() == target_day).any()), None)
        if fold is None:
            raise ValueError(f"no {head} window issued on {day}")
        calib, test = folds[fold]
        origins = issued[fold]
        n = int(np.flatnonzero(origins.normalize() == target_day)[0])
        pred = Predictive.from_raw(test, FAMILY[head], ALPHA)
        sets = scenario_sets(head, pred, test, calib, min(members, calib["actual"].shape[0]), rng)
        y = test["actual"][n]
        hours = pd.to_datetime(test["timestamps"][n, :, 0], utc=True)
        fig, axes = plt.subplots(len(sets), 4, figsize=(11, 2.3 * len(sets)), sharex=True, squeeze=False)
        for r, (form, x) in enumerate(sets.items()):
            paths = x[:, n]
            series = [paths[..., 0], paths[..., 1], paths[..., 2], np.hypot(paths[..., 0], paths[..., 1])]
            observed = [y[:, 0], y[:, 1], y[:, 2], np.hypot(y[:, 0], y[:, 1])]
            for c, (path, obs) in enumerate(zip(series, observed, strict=True)):
                ax = axes[r, c]
                ax.plot(hours, path.T, color=colours[form], lw=0.4, alpha=0.5)
                ax.plot(hours, obs, "o", ms=2.2, mfc="white", mec=T.TEXT_DARK, mew=0.6)
                if r == 0:
                    ax.set_title(PANEL_LABEL[c], fontsize=9)
                if c == 0:
                    ax.set_ylabel(FORM_LABEL[form], fontsize=8)
                ax.tick_params(labelsize=7)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
        for ax in axes[-1]:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H"))
        fig.suptitle(f"{LABEL[head]}: window issued {origins[n]:%d %b %Y %H:%M} (fold {fold})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"scenarios_{head}.pdf", bbox_inches="tight")
        fig.savefig(out / f"scenarios_{head}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


UNITS = {"mlvs": ("P (kW)", "Q (kvar)", "V (V)"), "dss": ("P (kW)", "Q (kvar)", "V (pu)")}
FAN = ((0.05, 0.95), (0.10, 0.90), (0.25, 0.75))
FAN_ALPHA = (0.18, 0.28, 0.42)
DRAWN_PATHS = 25


def plot_outputs(dataset_dir: Path, heads: list[str], day: str, out: Path, latex: bool = True) -> None:
    """Each head's own output for one forecast window: a fan for the quantile and Laplace heads, weight-draw curves.

    The quantile heads' fan is read from their predicted quantile grid, the Laplace fan from its closed-form
    distribution, both at 5-95%, 10-90% and 25-75% with the median. The BNN shows ``DRAWN_PATHS`` of its weight
    draws without the added observation noise, each a whole-day forecast of the model, over its 5-95% band
    (noise included). One thesis panel per (head, target): ``outputs_<head>_<p|q|v>.pdf``.
    """
    from matplotlib import pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    from twiga_thesis.plotting import line_style as LS  # noqa: PLC0415

    LS.configure(latex=latex)
    units = UNITS["mlvs" if dataset_dir.name.startswith("mlvs") else "dss"]
    target_day = pd.Timestamp(day, tz="UTC")
    pct = r"\%" if latex else "%"
    for head in heads:
        folds = load_folds(dataset_dir / "raw_export" / f"{head}_raw")
        issued = {fold: pd.to_datetime(t["timestamps"][:, 0, 0], utc=True) for fold, (_, t) in folds.items()}
        fold = next((f for f, o in issued.items() if (o.normalize() == target_day).any()), None)
        if fold is None:
            raise ValueError(f"no {head} window issued on {day}")
        _, test = folds[fold]
        n = int(np.flatnonzero(issued[fold].normalize() == target_day)[0])
        pred = Predictive.from_raw(test, FAMILY[head], ALPHA)
        times = pd.Series(pd.to_datetime(test["timestamps"][n, :, 0], utc=True).tz_convert(None))
        for d, target in enumerate(TARGETS):
            fig, ax = LS.new_panel()
            y = test["actual"][n, :, d]
            handles, labels = [], []
            if "samples" in test:
                lower = pred.ppf(0.05)[n, :, d]
                upper = pred.ppf(0.95)[n, :, d]
                ax.fill_between(times, lower, upper, color=LS.BAND_COLOR, lw=0)
                draws = test["samples"][:DRAWN_PATHS, n, :, d]
                ax.plot(times, draws.T, color=LS.FORECAST_COLOR, lw=0.45, alpha=0.55)
                ax.plot(times, test["loc"][n, :, d], color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
                handles += [
                    Patch(color=LS.BAND_COLOR),
                    Line2D([], [], color=LS.FORECAST_COLOR, lw=0.45),
                    Line2D([], [], color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH),
                ]
                labels += [f"90{pct} interval", "Weight draws", "Mean"]
            else:
                for (lo, hi), alpha in zip(FAN, FAN_ALPHA, strict=True):
                    ax.fill_between(
                        times, pred.ppf(lo)[n, :, d], pred.ppf(hi)[n, :, d], color=LS.FORECAST_COLOR, alpha=alpha, lw=0
                    )
                ax.plot(times, pred.ppf(0.5)[n, :, d], color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)
                handles += [Patch(color=LS.FORECAST_COLOR, alpha=a) for a in FAN_ALPHA]
                handles += [Line2D([], [], color=LS.FORECAST_COLOR, lw=LS.LINE_WIDTH)]
                labels += [f"{int(round(100 * (hi - lo)))}{pct}" for lo, hi in FAN] + ["Median"]
            ax.plot(times, y, "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=3, **LS.MARKER)
            handles += [Line2D([], [], marker="o", ls="", mec=LS.OBSERVED_COLOR, mfc="white", **LS.MARKER)]
            labels += ["Observed"]
            LS.time_axis(ax, times)
            ax.set_ylabel(units[d])
            LS.finish(ax)
            LS.legend_below(ax, handles, labels, ncol=min(len(labels), 3))
            LS.save(fig, out / f"outputs_{head}_{target.lower()}")


COUPLED_PATHS = 4


def plot_copula(dataset_dir: Path, heads: list[str], day: str, out: Path, latex: bool = True) -> None:
    """What the Schaake copula changes, for one forecast window: the same fan, independent against coupled paths.

    The 90% band is the head's own marginal output and is identical in both panels. ``COUPLED_PATHS`` paths are drawn
    in the thesis forecast style (teal paths over the interval band, hollow orange observations): on the left each
    half-hour is drawn on its own, on the right the same draws are reordered to follow the ranks of recent observed
    days. One thesis panel per (head, target, form):
    ``copula_<head>_<p|q|v>_<independent|schaake>.pdf``.
    """
    from matplotlib import pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    from twiga_thesis.plotting import line_style as LS  # noqa: PLC0415

    LS.configure(latex=latex)
    units = UNITS["mlvs" if dataset_dir.name.startswith("mlvs") else "dss"]
    target_day = pd.Timestamp(day, tz="UTC")
    pct = r"\%" if latex else "%"
    for head in heads:
        folds = load_folds(dataset_dir / "raw_export" / f"{head}_raw")
        issued = {fold: pd.to_datetime(t["timestamps"][:, 0, 0], utc=True) for fold, (_, t) in folds.items()}
        fold = next((f for f, o in issued.items() if (o.normalize() == target_day).any()), None)
        if fold is None:
            raise ValueError(f"no {head} window issued on {day}")
        calib, test = folds[fold]
        n = int(np.flatnonzero(issued[fold].normalize() == target_day)[0])
        pred = Predictive.from_raw(test, FAMILY[head], ALPHA)
        members = min(50, calib["actual"].shape[0])
        rng = np.random.default_rng(SEED)
        independent = marginal_draws(pred, members, rng)
        coupled = schaake_shuffle(independent, calib["actual"][-members:])
        times = pd.Series(pd.to_datetime(test["timestamps"][n, :, 0], utc=True).tz_convert(None))
        # The same members in both panels: their values at every half-hour are identical, only the pairing differs.
        chosen = np.linspace(0, members - 1, COUPLED_PATHS).astype(int)
        for d, target in enumerate(TARGETS):
            lower, upper = pred.ppf(0.05)[n, :, d], pred.ppf(0.95)[n, :, d]
            # One y-range for the pair of panels: band, observations and the drawn paths of both forms.
            span = [
                test["actual"][n, :, d],
                lower,
                upper,
                independent[chosen, n, :, d].ravel(),
                coupled[chosen, n, :, d].ravel(),
            ]
            for form, draws in (("independent", independent), ("schaake", coupled)):
                fig, ax = LS.new_panel()
                ax.fill_between(times, lower, upper, color=LS.BAND_COLOR, lw=0)
                for member in chosen:
                    ax.plot(times, draws[member, n, :, d], color=LS.FORECAST_COLOR, lw=0.6)
                ax.plot(times, test["actual"][n, :, d], "o", mec=LS.OBSERVED_COLOR, mfc="white", zorder=3, **LS.MARKER)
                LS.time_axis(ax, times)
                LS.set_y_axis(ax, *span)
                ax.set_ylabel(units[d])
                LS.finish(ax)
                handles = [
                    Patch(color=LS.BAND_COLOR),
                    Line2D([], [], color=LS.FORECAST_COLOR, lw=0.6),
                    Line2D([], [], marker="o", ls="", mec=LS.OBSERVED_COLOR, mfc="white", **LS.MARKER),
                ]
                labels = [f"90{pct} interval", "Sample paths", "Observed"]
                LS.legend_below(ax, handles, labels, ncol=3)
                LS.save(fig, out / f"copula_{head}_{target.lower()}_{form}")


def main() -> None:
    """Run analyses A and B for every exported head of one record."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="experiment root of the raw_export run")
    ap.add_argument("--dataset", default="mlvs-pt-3phase")
    ap.add_argument("--heads", nargs="*", default=list(FAMILY))
    ap.add_argument("--members", type=int, default=50, help="scenarios per window for analysis B")
    ap.add_argument("--plot-day", default=None, help="only draw sample paths of the window issued on this day")
    ap.add_argument("--outputs-day", default=None, help="only draw each head's own output for the window of this day")
    ap.add_argument("--copula-day", default=None, help="only draw independent against coupled paths for this day")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    dataset_dir = sorted((args.run / "runs").glob(f"*/conditions/{args.dataset}"))[-1]
    out = args.run / "analysis" / "forecast_type_study" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.copula_day:
        plot_copula(dataset_dir, args.heads, args.copula_day, out, latex=not args.no_latex)
        print(f"written under {out}")
        return
    if args.outputs_day:
        plot_outputs(dataset_dir, args.heads, args.outputs_day, out, latex=not args.no_latex)
        print(f"written under {out}")
        return
    if args.plot_day:
        plot_scenarios(dataset_dir, args.heads, args.plot_day, 30, out)
        print(f"written under {out}")
        return
    rng = np.random.default_rng(SEED)
    single, combined, single_windows, combined_windows = [], [], [], []
    for head in args.heads:
        for fold, (calib, test) in load_folds(dataset_dir / "raw_export" / f"{head}_raw").items():
            pred = Predictive.from_raw(test, FAMILY[head], ALPHA)
            fold_rows, window_rows = single_target(head, pred, test, calib, rng, fold)
            single += [r | {"Folds": fold} for r in fold_rows]
            single_windows += window_rows
            members = min(args.members, calib["actual"].shape[0])
            fold_rows, window_rows = combined_targets(head, pred, test, calib, members, rng, fold)
            combined += [r | {"Folds": fold} for r in fold_rows]
            combined_windows += window_rows
            print(f"{head} fold {fold} done", flush=True)
    single_df, combined_df = pd.DataFrame(single), pd.DataFrame(combined)
    single_df.to_csv(out / "single_target_folds.csv", index=False)
    combined_df.to_csv(out / "combined_targets_folds.csv", index=False)
    windows_a, windows_b = pd.DataFrame(single_windows), pd.DataFrame(combined_windows)
    windows_a.to_csv(out / "single_target_windows.csv", index=False)
    windows_b.to_csv(out / "combined_targets_windows.csv", index=False)
    tests = significance(windows_a, windows_b)
    tests.to_csv(out / "dm_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nA. single-target tail events (fold means)")
    cols = ["brier_score", "fair_brier_score", "brier_skill_score", "reliability", "resolution", "rate"]
    print(
        single_df.groupby(["head", "target", "source"])[[c for c in cols if c in single_df]].mean().round(4).to_string()
    )
    print("\nB. combined targets (fold means; lower is better)")
    print(
        combined_df.groupby(["head", "form"])[["crps_apparent_power", "energy_score", "variogram_score"]]
        .mean()
        .round(4)
        .to_string()
    )
    print("\nDM (Benjamini-Hochberg) on per-window losses")
    print(tests.round(4).to_string(index=False))
    print(f"\nwritten under {out}")


if __name__ == "__main__":
    main()
