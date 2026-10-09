r"""Offline multi-target conformal study on saved raw forecasts: every score x coupling against the native interval.

Reads the per-fold raw calibration- and test-window forecasts that ``fixed_benchmark.py --group raw_export`` saves
(``raw/fold_XX_{calib,test}.npz``), calibrates each method of
:mod:`twiga.distributions.conformal.multitarget` on the calibration window, and scores its intervals on the test
window, per target and jointly:

* PICP per target and joint coverage (all of P, Q and V inside at once), nominal 90%;
* NMPI, the median interval width over the observed range;
* the Winkler score at alpha = 0.1;
* two-sided CWE stratified by lead time, as the benchmark tables compute it, at the per-target level each
  method aims for: 90% for the native interval and the marginal coupling, :math:`1 - \alpha / D` (about 96.7%)
  for the joint couplings, which must over-cover each target to cover all three at once (``cwe_target``; the
  90% value is kept as ``cwe``);
* inflation spread, the largest minus the smallest per-target PICP: how far one target is over-covered to rescue
  another, the failure of a shared (max) threshold;
* tail balance, the share of a target's misses below its interval (0.5 when misses are symmetric).

Methods: the head's native interval, then CRF and DCP for every head and CQR (raw and width-scaled) for the
quantile heads, each with the marginal, Bonferroni, max and rank-max couplings. ``crf`` + ``max`` is the max-norm
construction the chapter used.

Writes ``conformal_study_folds.csv`` (per fold), ``conformal_study_summary.csv`` (fold means), a coverage-and-width
figure per head and, for one day, a native-versus-conformal forecast panel per method and target.

Usage::

    uv run python experiment/ch7_multitarget/experiment_1_benchmark/conformal_study.py \
        --run <experiments>/ch7_multitarget/conformal_study/fixed_benchmark/multitarget_fixed_benchmark \
        --heads fpqr_anchor
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parents[1])]

from twiga.core.metrics.interval import conditional_cwe  # noqa: E402
from twiga.distributions.conformal.multitarget import MultiTargetConformal, Predictive  # noqa: E402

from chapters.ch7_multitarget.experiment_1_benchmark.fixed_benchmark_analysis import (  # noqa: E402
    N_STRATA,
    _window_hits,
)
from twiga_thesis.plotting import line_style as LS, style as S, tokens as T  # noqa: E402

ALPHA = 0.1
TARGETS = ("P", "Q", "V")
UNITS = {"mlvs": ("P (kW)", "Q (kvar)", "V (V)"), "dss": ("P (kW)", "Q (kvar)", "V (pu)")}
FAMILY = {"fpqr_anchor": "quantile", "qr": "quantile", "bnn": "normal", "laplace": "laplace"}
LABEL = {"fpqr_anchor": "FPQR", "qr": "QR", "bnn": "BNN", "laplace": "Laplace"}
COUPLINGS = ("marginal", "bonferroni", "max", "rank_max")
COUPLING_LABEL = {"marginal": "marginal", "bonferroni": "Bonferroni", "max": "max", "rank_max": "rank-max"}
SCORE_LABEL = {"crf": "CRF", "cqr": "CQR", "cqr_scaled": "CQR scaled", "dcp": "DCP"}


def methods(head: str) -> list[tuple[str, str]]:
    """(score, coupling) pairs that apply to a head: CQR only where there is a quantile grid."""
    scores = ["crf", "dcp"] + (["cqr", "cqr_scaled"] if FAMILY[head] == "quantile" else [])
    return [(s, c) for s in scores for c in COUPLINGS]


def method_name(score: str | None, coupling: str | None) -> str:
    """Display name, e.g. ``DCP rank-max``; ``native`` for the head's own interval."""
    return "native" if score is None else f"{SCORE_LABEL[score]} {COUPLING_LABEL[coupling]}"


def load_folds(condition_dir: Path) -> dict[int, tuple[dict, dict]]:
    """Each fold's saved (calibration, test) arrays."""
    folds = {}
    for calib in sorted((condition_dir / "raw").glob("fold_*_calib.npz")):
        fold = int(calib.stem.split("_")[1])
        test = calib.with_name(f"fold_{fold:02d}_test.npz")
        folds[fold] = (dict(np.load(calib)), dict(np.load(test)))
    return folds


def interval_frame(arrays: dict, lower: np.ndarray, upper: np.ndarray, fold: int) -> pd.DataFrame:
    """Long frame in the benchmark's prediction layout, one row per (window, step, target)."""
    stamps = arrays["timestamps"][..., 0]  # (N, H)
    n, h, d = arrays["actual"].shape
    return pd.DataFrame(
        {
            "Folds": fold,
            "forecast_origin": np.repeat(stamps[:, 0], h * d),
            "timestamp": np.repeat(stamps.reshape(-1), d),
            "target": np.tile(TARGETS[:d], n * h),
            "Actual": arrays["actual"].reshape(-1),
            "forecast": arrays["loc"].reshape(-1),
            "lower": lower.reshape(-1),
            "upper": upper.reshape(-1),
            "metric_type": "interval",
        }
    )


def stratified_cwe(frame: pd.DataFrame, alpha: float) -> pd.Series:
    """Two-sided CWE stratified by lead time per target, at nominal miscoverage ``alpha``.

    The benchmark's ``two_sided_cwe`` with ``alpha`` as a parameter: the coverage tolerance is one standard
    error of a lead-time stratum's coverage at that level, inflated by within-window clustering.
    """
    out = {}
    for target, g in frame.groupby("target"):
        rows, hits = _window_hits(g)
        n_windows, m = hits.shape
        tolerances = []
        for stratum in np.array_split(np.arange(m), N_STRATA):
            h = hits[:, stratum]
            p, k = h.mean(), h.shape[1]
            clustered = p not in (0.0, 1.0) and k >= 2 and n_windows >= 2
            deff = max(1.0, h.mean(axis=1).var(ddof=1) / (p * (1 - p) / k)) if clustered else 1.0
            tolerances.append(np.sqrt(alpha * (1 - alpha) * deff / (n_windows * k)))
        tol = float(np.median(tolerances))
        y, lo, hi, lead = (rows[c].to_numpy(float) for c in ("Actual", "lower", "upper", "lead"))
        out[target] = conditional_cwe(
            y, lo, hi, strata=lead, alpha=alpha, n_bins=N_STRATA, min_count=1, tol=tol, tol_over=tol
        )
    return pd.Series(out)


def target_alpha(method: str, d: int = 3) -> float:
    """Per-target miscoverage a method aims for: alpha for native and marginal, alpha / D for joint couplings."""
    return ALPHA if method == "native" or method.endswith("marginal") else ALPHA / d


def fold_metrics(frame: pd.DataFrame, method: str = "native") -> pd.DataFrame:
    """Per-target PICP, NMPI, Winkler score, tail balance and two-sided stratified CWE for one fold."""
    rows = []
    for target, g in frame.groupby("target"):
        y, lo, hi = g["Actual"].to_numpy(), g["lower"].to_numpy(), g["upper"].to_numpy()
        below, above = y < lo, y > hi
        miss = below | above
        rows.append(
            {
                "target": target,
                "picp": float((~miss).mean()),
                "nmpi": float(np.median(hi - lo) / (y.max() - y.min())),
                "ws": float(np.mean((hi - lo) + (2 / ALPHA) * ((lo - y) * below + (y - hi) * above))),
                "tail_below": float(below.sum() / miss.sum()) if miss.any() else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("target")
    hit = (frame["Actual"] >= frame["lower"]) & (frame["Actual"] <= frame["upper"])
    joint = hit.groupby([frame["forecast_origin"], frame["timestamp"]]).all().mean()
    return out.assign(
        cwe=stratified_cwe(frame, ALPHA), cwe_target=stratified_cwe(frame, target_alpha(method)), joint=float(joint)
    ).reset_index()


def run_head(condition_dir: Path, head: str) -> tuple[pd.DataFrame, dict]:
    """Every method's fold metrics for one head, and its test intervals per (method, fold) for the day panels."""
    folds = load_folds(condition_dir)
    family = FAMILY[head]
    rows, intervals = [], {}
    for fold, (calib, test) in folds.items():
        pred_cal = Predictive.from_raw(calib, family, ALPHA)
        pred_test = Predictive.from_raw(test, family, ALPHA)
        candidates = {method_name(None, None): pred_test.bounds()}
        for score, coupling in methods(head):
            mtc = MultiTargetConformal(score, coupling, ALPHA).calibrate(pred_cal, calib["actual"])
            candidates[method_name(score, coupling)] = mtc.intervals(pred_test)
        for name, (lower, upper) in candidates.items():
            frame = interval_frame(test, lower, upper, fold)
            intervals[(name, fold)] = frame
            rows.append(fold_metrics(frame, name).assign(head=head, method=name, Folds=fold))
    return pd.concat(rows, ignore_index=True), intervals


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Fold means per (head, method, target), plus joint coverage and inflation spread per (head, method)."""
    columns = ["picp", "nmpi", "ws", "cwe", "cwe_target", "tail_below"]
    summary = folds.groupby(["head", "method", "target"])[columns].mean()
    per_method = folds.groupby(["head", "method"]).agg(joint=("joint", "mean"))
    picp = summary["picp"].unstack("target")
    per_method["spread"] = picp.max(axis=1) - picp.min(axis=1)
    per_method["cwe_mean"] = summary["cwe_target"].groupby(level=["head", "method"]).mean()
    return summary.reset_index().merge(per_method.reset_index(), on=["head", "method"])


def rank_methods(summary: pd.DataFrame, min_joint: float = 0.87) -> pd.DataFrame:
    """Per head, the methods ranked by mean stratified CWE at their own target level.

    Joint couplings must reach near-nominal joint coverage (``min_joint``) to be ranked; the native interval and
    the marginal coupling make no joint claim and are ranked on per-target CWE alone, marked ``joint_claim``.
    """
    wide = summary.pivot_table(index=["head", "method"], columns="target", values=["cwe_target", "picp", "ws"])
    wide.columns = [f"{metric}_{target}" for metric, target in wide.columns]
    per_method = summary.drop_duplicates(["head", "method"]).set_index(["head", "method"])[
        ["joint", "spread", "cwe_mean"]
    ]
    table = per_method.join(wide)
    table["joint_claim"] = [not (m == "native" or m.endswith("marginal")) for _, m in table.index]
    keep = ~table["joint_claim"] | (table["joint"] >= min_joint)
    return table[keep].sort_values(["head", "cwe_mean"], ascending=[True, False])


def plot_head(summary: pd.DataFrame, head: str, out: Path) -> None:
    """Per method: per-target PICP (markers) and joint coverage (bar); NMPI and stratified CWE beside it."""
    frame = summary[summary["head"] == head]
    order = ["native"] + [method_name(s, c) for s, c in methods(head)]
    x = np.arange(len(order))
    fig, axes = plt.subplots(3, 1, figsize=S.panel(1.0, 1.0), sharex=True)
    colours = dict(zip(TARGETS, (T.PALETTE[0], T.PALETTE[2], T.PALETTE[4]), strict=True))
    joint = frame.drop_duplicates("method").set_index("method")["joint"].reindex(order)
    axes[0].bar(x, joint, color=T.PALETTE[5], width=0.6, label="Joint coverage", lw=0)
    for target in TARGETS:
        values = frame[frame["target"] == target].set_index("method").reindex(order)
        axes[0].plot(x, values["picp"], "o", ms=3.5, color=colours[target], label=f"PICP ({target})")
        axes[1].plot(x, values["nmpi"], "o-", ms=3, lw=0.8, color=colours[target], label=target)
        axes[2].plot(x, values["cwe"], "o-", ms=3, lw=0.8, color=colours[target], label=target)
    axes[0].axhline(1 - ALPHA, color=T.TEXT_DARK, lw=0.8, ls=":")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Coverage")
    axes[1].set_ylabel("NMPI")
    axes[2].set_ylabel("CWE (stratified)")
    axes[2].set_xticks(x, order, rotation=60, ha="right", fontsize=7)
    for ax in axes:
        LS.finish(ax)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=7)
    LS.save(fig, out / f"conformal_study_{head}")


def plot_day(intervals: dict, head: str, methods_shown: list[str], day: str, fold: int, units, out: Path) -> None:
    """One forecast panel per (method, target) on one day, one y-range per target across the methods."""
    date = pd.Timestamp(day, tz="UTC")
    frames = {}
    for name in methods_shown:
        frame = intervals[(name, fold)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        keep = frame["timestamp"].dt.normalize() == date
        frames[name] = frame[keep]
    for t_index, target in enumerate(TARGETS):
        span = pd.concat([f[f["target"] == target][["Actual", "lower", "upper"]] for f in frames.values()])
        for name, frame in frames.items():
            day_frame = frame[frame["target"] == target].sort_values("timestamp")
            fig = LS.forecast_panel(day_frame, units[t_index])
            LS.set_y_axis(fig.axes[0], span["Actual"], span["lower"], span["upper"])
            slug = name.lower().replace(" ", "_").replace("-", "")
            LS.save(fig, out / f"conformal_day_{head}_{slug}_{target.lower()}")


def main() -> None:
    """Run the study for the chosen heads of one exported record."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="experiment root of the raw_export run")
    ap.add_argument("--dataset", default="mlvs-pt-3phase")
    ap.add_argument("--heads", nargs="*", default=list(FAMILY))
    ap.add_argument("--day", default="2021-09-12", help="day for the forecast panels")
    ap.add_argument("--day-fold", type=int, default=8)
    ap.add_argument(
        "--show",
        nargs="*",
        default=None,
        help="methods drawn as day panels (default: native, CRF max/Bonferroni/rank-max, DCP and CQR rank-max)",
    )
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()

    LS.configure(latex=not args.no_latex)
    dataset_dir = next(p for p in sorted((args.run / "runs").glob(f"*/conditions/{args.dataset}"), reverse=True))
    out = args.run / "analysis" / "conformal_study"
    out.mkdir(parents=True, exist_ok=True)
    units = UNITS["mlvs" if args.dataset.startswith("mlvs") else "dss"]
    all_folds = []
    for head in args.heads:
        folds, intervals = run_head(dataset_dir / "raw_export" / f"{head}_raw", head)
        all_folds.append(folds)
        summary = summarise(folds)
        plot_head(summary, head, out)
        available = {"native", *(method_name(s, c) for s, c in methods(head))}
        shown = ["native", "CRF max", "CRF Bonferroni", "CRF rank-max", "DCP rank-max", "CQR rank-max"]
        shown = [m for m in (args.show or shown) if m in available]
        plot_day(intervals, head, shown, args.day, args.day_fold, units, out)
    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(out / "conformal_study_folds.csv", index=False)
    summary = summarise(folds)
    summary.to_csv(out / "conformal_study_summary.csv", index=False)
    pd.set_option("display.width", 220)
    per_method = summary.drop_duplicates(["head", "method"]).set_index(["head", "method"])[
        ["joint", "spread", "cwe_mean"]
    ]
    ranked = rank_methods(summary)
    ranked.to_csv(out / "conformal_study_ranking.csv")
    print(ranked.round(3).to_string())
    print("\nall methods:")
    print(per_method.round(3).to_string())
    print(f"written under {out}")


if __name__ == "__main__":
    main()
