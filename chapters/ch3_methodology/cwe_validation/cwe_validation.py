"""Validating the reduced CWE on net-load data.

The soundness analysis in ``coverage_term_analysis.py`` uses a homoscedastic
Gaussian target. That is the right place to start --- a metric that fails there
cannot be rescued --- but it is not net-load. Net-load crosses zero, is strongly
heteroscedastic, and has a diurnal structure driven by PV that couples the
conditional mean and the conditional variance. This module asks whether the
conclusions survive those features, in three stages of increasing realism,
following the experimental protocol of the metrics chapter.

    A  synthetic net-load     structure is realistic, ground truth is known
    B  real data, MLVS-PT     ground truth unknown, calibrated width identifiable
    C  real data, ranking     do the metrics agree on which forecaster is better?

Stage A is the ideal-forecaster test transplanted onto a series with net-load
pathologies. Stage B is its real-data analogue: the true conditional distribution
is unavailable, but the *calibrated* interval width is identifiable from held-out
coverage, so we can still ask whether each metric is optimised there. Stage C is
the question that matters in practice --- whether a ranking by CWE agrees with a
ranking by a proper score.

Run:  uv run python experiment/chapter-3-cwe-validation/cwe_validation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from twiga.core.metrics.interval import cwe_score, interval_coverage, nmpi, winkler_score

from chapters.ch3_methodology.cwe_validation.metric_definitions import cwc_score, gaussian_crps

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA
# The quantile grid twiga fits by default, np.arange(0.1, 1, 0.1) in
# twiga.distributions.nn.quantile, augmented with 0.05 and 0.95 because the nine
# defaults cannot form the 90% interval this section reports. Using the library's
# own grid keeps the forecasts here comparable with those elsewhere in the thesis.
QGRID = np.array([0.05, *np.arange(0.1, 1.0, 0.1).round(1), 0.95])
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
OUT = Path(__file__).resolve().parent


# ── the reduced metric, as defined in the thesis ──────────────────────────────
def kernel(x: float) -> float:
    """f(x) = 2x/(1+x^2): unity at x=1, symmetric under x -> 1/x."""
    return 0.0 if x <= 0 else float(2.0 * x / (1.0 + x**2))


def reduced_cwe(picp: float, nmpi_v: float, err: float, ref: float, alpha: float = ALPHA) -> float:
    """Harmonic mean of the miscoverage and width kernels, weighted by accuracy."""
    fm = kernel((1.0 - picp) / alpha)
    fr = kernel(nmpi_v / ref) if ref > 0 else 0.0
    lam = float(np.clip(1.0 - err, 0.0, 1.0))
    return 0.0 if (fm + fr) == 0 else lam * 2.0 * fm * fr / (fm + fr)


def score_all(
    y: np.ndarray,
    lo: np.ndarray,
    up: np.ndarray,
    mu: np.ndarray,
    alpha: float = ALPHA,
    q: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one interval forecast under every metric considered in the chapter."""
    scale = float(y.max() - y.min())
    # Reference from the forecast RESIDUALS. Using the spread of y itself compares
    # the interval width against the size of the (largely explainable) net-load
    # signal rather than against the irreducible uncertainty.
    resid = y - mu
    kappa_a = float(np.quantile(resid, 1 - alpha / 2) - np.quantile(resid, alpha / 2))
    ref = kappa_a / scale
    picp = float(interval_coverage(y, lo, up))
    nv = float(nmpi(lo, up, scale=scale))
    err = float(np.mean(np.abs(y - mu))) / scale
    # Original CWE keeps its published defaults but with the alpha-tied reference,
    # which is the most favourable configuration for it.
    orig = float(np.ravel(cwe_score(nv, picp, err, ref, alpha=alpha))[0])
    # CWC exactly as the chapter defines it: eta = 30, following Jensen et al.
    cwc = float(np.ravel(cwc_score(nv, picp, alpha=alpha))[0])
    out = {
        "picp": picp,
        "nmpi": nv,
        "winkler": float(winkler_score(y, lo, up, alpha=alpha)),
        "cwe_reduced": reduced_cwe(picp, nv, err, ref, alpha),
        "cwe_original": orig,
        "cwc": cwc,
    }
    if q is not None:
        out["crps"] = crps_from_quantiles(y, q)
    return out


def crps_from_quantiles(y: np.ndarray, q: np.ndarray, levels: np.ndarray = QGRID) -> float:
    r"""CRPS estimated from a quantile grid.

    Uses :math:`\mathrm{CRPS} = 2\int_0^1 \rho_\tau \, d\tau`, so the mean pinball
    loss over a uniform grid of levels, doubled, estimates it. Lower is better.

    The grid is coarse and integrates only over ``[0.05, 0.95]``, so the estimate is
    biased low against the closed form by a nearly constant factor. That factor does
    not affect the ranking this is used for, and the estimate remains minimised at
    the correct dispersion.
    """
    diff = y[:, None] - q
    pinball = np.maximum(levels * diff, (levels - 1.0) * diff)
    # Trapezoidal, not a plain mean: the grid is not uniformly spaced once 0.05 and
    # 0.95 are added, so averaging would over-weight the tails.
    return float(2.0 * np.trapezoid(pinball.mean(axis=0), levels))


def stratified_cwe(
    y: np.ndarray, lo: np.ndarray, up: np.ndarray, mu: np.ndarray, by: np.ndarray, n_strata: int = 5
) -> float:
    """Reduced CWE scored inside strata of ``by``, then averaged.

    ``R`` is computed once over the whole sample and shared, so widths remain
    comparable across strata; recomputing it per stratum rescales NMPI differently
    in each bin and makes them incommensurable.
    """
    scale = float(y.max() - y.min())
    out = []
    for idx in np.array_split(np.argsort(by), n_strata):
        ys, los, ups, mus = y[idx], lo[idx], up[idx], mu[idx]
        res = ys - mus
        kap = float(np.quantile(res, 1 - ALPHA / 2) - np.quantile(res, ALPHA / 2))
        out.append(
            reduced_cwe(
                float(interval_coverage(ys, los, ups)),
                float(nmpi(los, ups, scale=scale)),
                float(np.mean(np.abs(ys - mus))) / scale,
                kap / scale,
            )
        )
    return float(np.mean(out))


# ── A. synthetic net-load with known ground truth ─────────────────────────────
def make_synthetic_netload(days: int = 730, seed: int = 0) -> dict[str, np.ndarray]:
    """A net-load series with the features that matter: PV, zero crossings, heteroscedasticity.

    Demand is a double-peaked diurnal profile modulated annually. PV is a
    clear-sky bell scaled by a stochastic clarity factor, so it is zero at night
    and highly variable at midday. Net-load is their difference and takes both
    signs. The conditional standard deviation rises with PV output, which is the
    dependence that makes a single global interval width inadequate.
    """
    rng = np.random.default_rng(seed)
    n = days * 48
    t = np.arange(n)
    hod = (t % 48) / 48.0
    doy = (t // 48) % 365

    demand = (
        3.0
        + 0.6 * np.sin(2 * np.pi * (doy / 365.0) - 1.2)
        + 0.9 * np.exp(-(((hod - 0.33) / 0.06) ** 2))
        + 1.3 * np.exp(-(((hod - 0.79) / 0.07) ** 2))
    )
    clear_sky = np.clip(np.sin(np.pi * (hod - 0.25) / 0.5), 0, None) ** 1.4
    seasonal_pv = 0.65 + 0.35 * np.sin(2 * np.pi * (doy / 365.0) - 1.6)
    clarity = rng.beta(5.0, 1.6, n)
    pv = 4.2 * clear_sky * seasonal_pv * clarity

    mu = demand - pv
    # Conditional sd: a floor plus a term proportional to PV output.
    sigma = 0.16 + 0.42 * clear_sky * seasonal_pv
    y = mu + rng.normal(0.0, sigma)
    return {"y": y, "mu": mu, "sigma": sigma, "pv": pv, "hod": hod}


def experiment_a(seed: int = 0) -> pd.DataFrame:
    """Ideal-forecaster sweep on synthetic net-load. Truth is at s = 1."""
    d = make_synthetic_netload(seed=seed)
    y, mu, sigma = d["y"], d["mu"], d["sigma"]
    z = stats.norm.ppf(1 - ALPHA / 2)
    factors = np.linspace(0.30, 2.00, 69)

    rows = []
    for s in factors:
        half = z * sigma * s
        lo, up = mu - half, mu + half
        rows.append(
            {
                "factor": s,
                **score_all(y, lo, up, mu),
                # Closed form: the forecast family is Gaussian by construction here,
                # so the CRPS reference does not need to be approximated.
                "crps": float(np.mean(gaussian_crps(mu, sigma * s, y))),
                # Stratified by hour of day: the conditioning variable is available
                # at forecast time and tracks the PV cycle that drives the
                # heteroscedasticity.
                "cwe_stratified": stratified_cwe(y, lo, up, mu, by=d["hod"]),
            }
        )
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 78}\nA. SYNTHETIC NET-LOAD  (duck curve, zero crossings, heteroscedastic)\n{'=' * 78}")
    print(f"   n = {len(y):,} half-hourly points; {100 * np.mean(y < 0):.1f}% of net-load values are negative")
    print(f"   conditional sd varies {sigma.min():.2f}-{sigma.max():.2f}; truth at factor = 1.000\n")
    print(f"   {'metric':<18}{'orientation':<14}{'optimum factor':<18}{'verdict'}")
    for name, better in (
        ("crps", "min"),
        ("winkler", "min"),
        ("cwe_reduced", "max"),
        ("cwe_stratified", "max"),
        ("cwe_original", "max"),
        ("cwc", "max"),
    ):
        v = df[name].to_numpy()
        opt = float(df["factor"].to_numpy()[int(np.argmin(v) if better == "min" else np.argmax(v))])
        verdict = "sound" if abs(opt - 1.0) <= 0.026 else f"unsound ({100 * (opt - 1):+.0f}%)"
        print(f"   {name:<18}{'lower better' if better == 'min' else 'higher better':<14}{opt:<18.3f}{verdict}")
    return df


# ── B. real data: is the metric optimised at the calibrated width? ────────────
def _fit_intervals(dataset: str = "MLVS-PT") -> dict[str, np.ndarray] | None:
    """Fit a quantile-regression CatBoost and return pooled held-out forecasts.

    The protocol is the one this chapter prescribes rather than a single split:
    ten-fold back-testing cross-validation with an expanding window, an initial
    historical period of twelve months, a fixed four-month evaluation window, and
    a two-month step. Held-out predictions are pooled across folds to give one
    evaluation set. Using a single train/test split here would validate the metric
    with the very methodology the chapter argues against.

    A deliberately simple probabilistic model, matching the model family used for
    the deterministic experiments, so that the metric is assessed on forecasts of
    the kind the thesis actually produces.
    """
    fp = DATA / f"{dataset}.parquet"
    if not fp.exists():
        print(f"   [skip] {fp} not found")
        return None
    df = pd.read_parquet(fp).sort_values("timestamp").reset_index(drop=True)
    target = next(c for c in df.columns if c.startswith("NetLoad"))
    ok = np.isfinite(df[target].astype(float).to_numpy())
    df = df.loc[ok].reset_index(drop=True)
    s_arr = df[target].astype(float).to_numpy()

    lags = [1, 2, 3, 48, 49, 96, 336]
    n = len(s_arr)
    idx = np.arange(max(lags), n)
    x = np.column_stack([s_arr[idx - lag] for lag in lags])
    hod = (idx % 48) / 48.0
    dow = ((idx // 48) % 7) / 7.0
    x = np.column_stack([x, np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod), np.sin(2 * np.pi * dow)])
    for col in ("Ghi", "Temperature"):
        if col in df.columns:
            x = np.column_stack([x, df[col].astype(float).to_numpy()[idx]])
    y = s_arr[idx]

    # Half-hourly data: 48 observations per day.
    initial, window, step = 365 * 48, 122 * 48, 61 * 48
    n_folds = min(10, max(1, 1 + (len(y) - initial - window) // step))

    from catboost import CatBoostRegressor

    out: dict[str, list[np.ndarray]] = {"y": [], "lo": [], "mu": [], "up": [], "hod": [], "q": []}
    for k in range(n_folds):
        train_end = initial + k * step
        test_end = min(train_end + window, len(y))
        if test_end <= train_end:
            break
        preds = {}
        for q in QGRID:
            m = CatBoostRegressor(
                loss_function=f"Quantile:alpha={q}", iterations=300, depth=6, learning_rate=0.08, verbose=0
            )
            m.fit(x[:train_end], y[:train_end])
            preds[q] = m.predict(x[train_end:test_end])
        out["y"].append(y[train_end:test_end])
        out["lo"].append(preds[ALPHA / 2])
        out["mu"].append(preds[0.5])
        out["up"].append(preds[1 - ALPHA / 2])
        out["hod"].append(hod[train_end:test_end])
        out["q"].append(np.column_stack([preds[q] for q in QGRID]))
    print(f"   {n_folds} expanding-window folds, pooled")
    res = {k: np.concatenate(v) for k, v in out.items()}

    # CatBoost fits each quantile level independently, so the levels are not
    # guaranteed to be ordered and in practice frequently are not. A quantile
    # function must be non-decreasing, so the rows are sorted before use; without
    # this the pinball and CRPS estimates are not evaluating a distribution at all.
    crossed = float((np.diff(res["q"], axis=1) < 0).any(axis=1).mean())
    res["q"] = np.sort(res["q"], axis=1)
    res["lo"], res["up"] = res["q"][:, 0], res["q"][:, -1]
    res["mu"] = res["q"][:, len(QGRID) // 2]
    print(f"   quantile crossing repaired by sorting: {100 * crossed:.1f}% of rows affected")
    return res


def experiment_b(dataset: str = "MLVS-PT") -> pd.DataFrame | None:
    """Width-perturbation sweep on real data.

    The true conditional distribution is unknown, so ``s = 1`` is no longer the
    reference. What *is* identifiable is the multiplier that achieves nominal
    coverage on held-out data; a sound metric should be optimised there.
    """
    print(f"\n{'=' * 78}\nB. REAL DATA --- {dataset}, quantile-regression CatBoost\n{'=' * 78}")
    f = _fit_intervals(dataset)
    if f is None:
        return None
    y, lo, up, mu = f["y"], f["lo"], f["up"], f["mu"]

    factors = np.linspace(0.30, 2.50, 89)
    rows = []
    for s in factors:
        lo_s, up_s = mu - (mu - lo) * s, mu + (up - mu) * s
        rows.append(
            {
                "factor": s,
                **score_all(y, lo_s, up_s, mu),
                "cwe_stratified": stratified_cwe(y, lo_s, up_s, mu, by=(up_s - lo_s), n_strata=24),
            }
        )
    df = pd.DataFrame(rows)

    # The empirical reference: the multiplier whose coverage is closest to nominal.
    s_cal = float(df["factor"].to_numpy()[int(np.argmin(np.abs(df["picp"].to_numpy() - NOMINAL)))])
    print(f"   n_test = {len(y):,};  raw model PICP = {interval_coverage(y, lo, up):.3f} against nominal {NOMINAL:.2f}")
    print(f"   calibrated width multiplier s* = {s_cal:.3f}  (the multiplier attaining nominal coverage)\n")
    print(f"   {'metric':<16}{'optimum s':<14}{'deviation from s*'}")
    for name, better in (
        ("winkler", "min"),
        ("cwe_reduced", "max"),
        ("cwe_stratified", "max"),
        ("cwe_original", "max"),
        ("cwc", "max"),
    ):
        v = df[name].to_numpy()
        opt = float(df["factor"].to_numpy()[int(np.argmin(v) if better == "min" else np.argmax(v))])
        print(f"   {name:<16}{opt:<14.3f}{100 * (opt - s_cal) / s_cal:+.1f}%")
    df.attrs["s_cal"] = s_cal
    return df


# ── C. real data: do the metrics rank forecasters the same way? ───────────────
def experiment_c(dataset: str = "MLVS-PT") -> pd.DataFrame | None:
    """Rank a panel of forecasters by each metric and compare against the interval score."""
    print(f"\n{'=' * 78}\nC. RANKING AGREEMENT --- {dataset}\n{'=' * 78}")
    f = _fit_intervals(dataset)
    if f is None:
        return None
    y, lo, up, mu = f["y"], f["lo"], f["up"], f["mu"]

    # A panel spanning over-confidence, calibration and over-conservatism, plus a
    # constant-width forecaster that is marginally calibrated but conditionally wrong.
    const_half = float(np.quantile(np.abs(y - mu), NOMINAL))
    panel: dict[str, tuple[np.ndarray, np.ndarray]] = {
        f"scaled x{s:.2f}": (mu - (mu - lo) * s, mu + (up - mu) * s) for s in (0.6, 0.8, 1.0, 1.2, 1.6)
    }
    panel["constant width"] = (mu - const_half, mu + const_half)

    rows = []
    for name, (lo_p, up_p) in panel.items():
        sc = score_all(y, lo_p, up_p, mu)
        sc["cwe_stratified"] = stratified_cwe(y, lo_p, up_p, mu, by=(up_p - lo_p), n_strata=24)
        rows.append({"forecaster": name, **sc})
    df = pd.DataFrame(rows)

    print(
        f"   {'forecaster':<18}{'PICP':<9}{'NMPI':<9}{'Winkler':<11}"
        f"{'CWE red.':<11}{'CWE strat.':<12}{'CWE orig.':<11}{'CWC':<9}"
    )
    for _, r in df.iterrows():
        print(
            f"   {r['forecaster']:<18}{r['picp']:<9.3f}{r['nmpi']:<9.3f}{r['winkler']:<11.4f}"
            f"{r['cwe_reduced']:<11.3f}{r['cwe_stratified']:<12.3f}{r['cwe_original']:<11.3f}{r['cwc']:<9.3f}"
        )

    best_w = df.loc[df["winkler"].idxmin(), "forecaster"]
    print(f"\n   interval score prefers: {best_w}")
    for name in ("cwe_reduced", "cwe_stratified", "cwe_original", "cwc"):
        pick = df.loc[df[name].idxmax(), "forecaster"]
        rho = stats.spearmanr(-df["winkler"], df[name]).statistic
        print(f"   {name:<16} prefers {pick:<18} rank correlation with interval score = {rho:+.3f}")
    return df


if __name__ == "__main__":
    a = experiment_a()
    a.to_csv(OUT / "s3523_netload_sweep.csv", index=False)
    b = experiment_b()
    if b is not None:
        b.to_csv(OUT / "results_real_sweep.csv", index=False)
    c = experiment_c()
    if c is not None:
        c.to_csv(OUT / "results_real_ranking.csv", index=False)
    print(f"\nwrote results_*.csv to {OUT}")


# ── 0. the Gaussian protocol, as stated in the thesis ─────────────────────────
def gaussian_sweep(alpha: float = ALPHA, n: int = 200_000, seed: int = 0) -> pd.DataFrame:
    r"""Ideal-forecaster sweep on ``Q = N(0,1)``, the protocol Section 3.5.2 states.

    Kept separate from the net-load stages because the text describes this exact
    construction, and because it is the case in which every metric has a known
    reference behaviour. For a zero-mean target the residual and target
    references of :func:`score_all` coincide, so the definition needs no special
    handling here.
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    z = stats.norm.ppf(1 - alpha / 2)
    sigmas = np.linspace(0.30, 2.00, 171)

    rows = []
    for s in sigmas:
        lo, up = np.full_like(y, -z * s), np.full_like(y, z * s)
        rows.append(
            {
                "sigma": s,
                **score_all(y, lo, up, np.zeros_like(y), alpha=alpha),
                "crps": float(np.mean(gaussian_crps(0.0, s, y))),
            }
        )
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 78}\n0. GAUSSIAN PROTOCOL  (Q = N(0,1); truth at sigma = 1.000)\n{'=' * 78}")
    print(f"   {'metric':<16}{'orientation':<15}{'optimum sigma':<17}{'verdict'}")
    for name, better in (("crps", "min"), ("winkler", "min"), ("cwe_reduced", "max"), ("cwc", "max")):
        v = df[name].to_numpy()
        opt = float(df["sigma"].to_numpy()[int(np.argmin(v) if better == "min" else np.argmax(v))])
        verdict = "sound" if abs(opt - 1.0) <= 0.011 else f"unsound ({100 * (opt - 1):+.0f}%)"
        print(f"   {name:<16}{'lower better' if better == 'min' else 'higher better':<15}{opt:<17.3f}{verdict}")
    return df


def cwc_eta_sensitivity(alpha: float = ALPHA, n: int = 200_000, seed: int = 0) -> pd.DataFrame:
    r"""How far CWC's optimum moves as its free parameter is varied.

    The forecasts are held fixed; only ``eta`` changes. The optimum ranges over
    roughly twenty percentage points of dispersion, and approaches soundness only
    as ``eta`` grows without bound --- the limit in which the coverage term stops
    being a reward that width can offset and becomes a constraint. This is the
    supporting evidence for the argument in Section 3.5.2.
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    z = stats.norm.ppf(1 - alpha / 2)
    scale = float(y.max() - y.min())
    sigmas = np.linspace(0.30, 2.00, 341)

    picp, widths = [], []
    for s in sigmas:
        lo, up = np.full_like(y, -z * s), np.full_like(y, z * s)
        picp.append(float(interval_coverage(y, lo, up)))
        widths.append(float(nmpi(lo, up, scale=scale)))
    picp_arr, width_arr = np.array(picp), np.array(widths)

    rows = []
    for eta in (5, 10, 20, 30, 50, 100, 200):
        v = cwc_score(width_arr, picp_arr, alpha=alpha, eta=float(eta))
        opt = float(sigmas[int(np.argmax(v))])
        rows.append({"eta": eta, "optimum_sigma": opt, "deviation_pct": 100 * (opt - 1)})
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 78}\nCWC SENSITIVITY TO ITS FREE PARAMETER\n{'=' * 78}")
    print(f"   {'eta':>6}   {'optimum sigma':>14}   {'deviation':>10}")
    for _, r in df.iterrows():
        print(f"   {int(r['eta']):>6}   {r['optimum_sigma']:>14.3f}   {r['deviation_pct']:>+9.1f}%")
    return df


def reference_scales(seed: int = 0) -> pd.DataFrame:
    """The three width scales the metrics section quotes, from the actual series.

    Written to CSV so that the illustration in ``twiga_thesis.plotting`` reports the same
    numbers as the text rather than recomputing them from a simplified profile.
    """
    d = make_synthetic_netload(seed=seed)
    y, mu, sigma = d["y"], d["mu"], d["sigma"]
    z = stats.norm.ppf(1 - ALPHA / 2)
    resid = y - mu
    df = pd.DataFrame(
        [
            {"quantity": "kappa_target", "value": float(np.quantile(y, 1 - ALPHA / 2) - np.quantile(y, ALPHA / 2))},
            {
                "quantity": "kappa_residual",
                "value": float(np.quantile(resid, 1 - ALPHA / 2) - np.quantile(resid, ALPHA / 2)),
            },
            {"quantity": "median_interval_width", "value": float(np.median(2 * z * sigma))},
        ]
    )
    print("\nReference scales:")
    for _, r in df.iterrows():
        print(f"   {r['quantity']:<24}{r['value']:.3f}")
    return df


def contrasting_cases(dataset: str = "MLVS-PT", alpha: float = ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two forecasts that differ only in how the interval width is chosen.

    The first is the model's own conditionally varying interval, rescaled to attain
    nominal coverage. The second replaces that width by a single constant, again
    chosen to attain nominal coverage. The two therefore have the same marginal
    coverage and nearly the same average width, and differ only in whether the
    width responds to the conditional scale. Returned as a day of forecasts for
    plotting, and a table of what each metric says about the pair.
    """
    f = _fit_intervals(dataset)
    if f is None:
        raise FileNotFoundError(dataset)
    y, lo, up, mu, hod, qmat = f["y"], f["lo"], f["up"], f["mu"], f["hod"], f["q"]

    # Scale the conditional interval to nominal coverage so that the two cases are
    # matched on coverage and the only difference is conditional responsiveness.
    factors = np.linspace(0.5, 2.5, 201)
    picps = [float(interval_coverage(y, mu - (mu - lo) * s, mu + (up - mu) * s)) for s in factors]
    s_star = float(factors[int(np.argmin(np.abs(np.array(picps) - (1 - alpha))))])
    good_lo, good_up = mu - (mu - lo) * s_star, mu + (up - mu) * s_star
    half = float(np.quantile(np.abs(y - mu), 1 - alpha))
    bad_lo, bad_up = mu - half, mu + half

    rows = []
    for name, (l_, u_) in (("conditional", (good_lo, good_up)), ("constant width", (bad_lo, bad_up))):
        sf = (u_ - l_) / np.clip(up - lo, 1e-3, None)
        q_s = mu[:, None] + (qmat - mu[:, None]) * sf[:, None]
        sc = score_all(y, l_, u_, mu, alpha=alpha, q=q_s)
        sc["cwe_stratified"] = stratified_cwe(y, l_, u_, mu, by=hod, n_strata=24)
        rows.append({"case": name, **sc})
    table = pd.DataFrame(rows)

    # Coverage by hour, which is where the two cases part company.
    hh = (hod * 48).astype(int) // 2
    cov = pd.DataFrame(
        {
            "hour": np.arange(24),
            "conditional": [float(np.mean((y >= good_lo)[hh == k] & (y <= good_up)[hh == k])) for k in range(24)],
            "constant": [float(np.mean((y >= bad_lo)[hh == k] & (y <= bad_up)[hh == k])) for k in range(24)],
        }
    )
    print(f"\n  Contrasting cases on {dataset} (interval scaled by {s_star:.3f} to reach nominal):")
    for _, r in table.iterrows():
        print(
            print(
                f"    {r['case']:<16} PICP {r['picp']:.3f}  NMPI {r['nmpi']:.3f}  "
                f"WS {r['winkler']:.3f}  CRPS {r['crps']:.4f}"
            )
        )
    return table, cov


def per_fold_metrics(dataset: str = "MLVS-PT", alpha: float = ALPHA) -> pd.DataFrame:
    """Metrics computed separately on each back-testing fold.

    Reporting per fold rather than pooled shows how far the conclusion depends on
    the evaluation period, which is the reason this chapter recommends back-testing
    over a single split. Both contrasting cases of :func:`contrasting_cases` are
    scored on every fold, so the comparison is repeated ten or eight times rather
    than once.
    """
    from twiga.core.metrics.point import nrmse

    fp = DATA / f"{dataset}.parquet"
    df = pd.read_parquet(fp).sort_values("timestamp").reset_index(drop=True)
    target = next(c for c in df.columns if c.startswith("NetLoad"))
    df = df.loc[np.isfinite(df[target].astype(float).to_numpy())].reset_index(drop=True)
    s_arr = df[target].astype(float).to_numpy()

    lags = [1, 2, 3, 48, 49, 96, 336]
    idx = np.arange(max(lags), len(s_arr))
    x = np.column_stack([s_arr[idx - lag] for lag in lags])
    hod = (idx % 48) / 48.0
    dow = ((idx // 48) % 7) / 7.0
    x = np.column_stack([x, np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod), np.sin(2 * np.pi * dow)])
    for col in ("Ghi", "Temperature"):
        if col in df.columns:
            x = np.column_stack([x, df[col].astype(float).to_numpy()[idx]])
    y_all = s_arr[idx]

    initial, window, step = 365 * 48, 122 * 48, 61 * 48
    n_folds = min(10, max(1, 1 + (len(y_all) - initial - window) // step))
    from catboost import CatBoostRegressor

    rows = []
    for k in range(n_folds):
        tr, te = initial + k * step, min(initial + k * step + window, len(y_all))
        preds = {}
        for q in QGRID:
            m = CatBoostRegressor(
                loss_function=f"Quantile:alpha={q}", iterations=300, depth=6, learning_rate=0.08, verbose=0
            )
            m.fit(x[:tr], y_all[:tr])
            preds[q] = m.predict(x[tr:te])
        qm = np.sort(np.column_stack([preds[q] for q in QGRID]), axis=1)
        y, mu, h = y_all[tr:te], qm[:, len(QGRID) // 2], hod[tr:te]
        lo, up = qm[:, 0], qm[:, -1]

        # Match each case to nominal coverage within the fold.
        fac = np.linspace(0.5, 2.5, 201)
        pic = [float(interval_coverage(y, mu - (mu - lo) * s, mu + (up - mu) * s)) for s in fac]
        s_star = float(fac[int(np.argmin(np.abs(np.array(pic) - (1 - alpha))))])
        half = float(np.quantile(np.abs(y - mu), 1 - alpha))

        for case, (l_, u_) in (
            ("conditional", (mu - (mu - lo) * s_star, mu + (up - mu) * s_star)),
            ("constant", (mu - half, mu + half)),
        ):
            sf = (u_ - l_) / np.clip(up - lo, 1e-3, None)
            sc = score_all(y, l_, u_, mu, alpha=alpha, q=mu[:, None] + (qm - mu[:, None]) * sf[:, None])
            rows.append(
                {
                    "dataset": dataset,
                    "fold": k + 1,
                    "case": case,
                    "nrmse": float(nrmse(y, mu, scale_method="max")),
                    "ws": sc["winkler"],
                    "crps": sc["crps"],
                    "cwe": sc["cwe_reduced"],
                    "cwe_strat": stratified_cwe(y, l_, u_, mu, by=h, n_strata=24),
                    "cwc": sc["cwc"],
                }
            )
    return pd.DataFrame(rows)
