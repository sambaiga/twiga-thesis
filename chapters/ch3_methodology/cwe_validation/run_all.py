"""Full validation sweep: both datasets, multiple seeds, and figures.

Extends ``netload_validation`` from a single-dataset demonstration to the
complete set reported in the metrics section:

  1. synthetic net-load soundness across seeds (stability of the optimum)
  2. reference/stratification factorial across seeds
  3. real-data width sweep and ranking on MLVS-PT and MLVS-UK
  4. figures for the tables

Run:  uv run python experiment/chapter-3-cwe-validation/run_all.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cwe_validation import (  # noqa: E402
    ALPHA,
    NOMINAL,
    _fit_intervals,
    make_synthetic_netload,
    reduced_cwe,
    score_all,
    stratified_cwe,
)
from twiga.core.metrics.interval import interval_coverage, nmpi, winkler_score  # noqa: E402

OUT = Path(__file__).resolve().parent
SEEDS = range(5)


def factorial_reference_stratification() -> pd.DataFrame:
    """Reference (target vs residual) x stratification, across seeds."""
    z = stats.norm.ppf(1 - ALPHA / 2)
    factors = np.linspace(0.70, 1.60, 91)
    rows = []
    for seed in SEEDS:
        d = make_synthetic_netload(seed=seed)
        y, mu, sigma, hod = d["y"], d["mu"], d["sigma"], d["hod"]
        scale = float(y.max() - y.min())
        for kfrom in ("target", "residual"):
            for label, by, ns in (("none", np.zeros_like(y), 1), ("hour of day", hod, 24)):
                vals = []
                for s in factors:
                    lo, up = mu - z * sigma * s, mu + z * sigma * s
                    out = []
                    for idx in np.array_split(np.argsort(by), ns):
                        ys, los, ups, mus = y[idx], lo[idx], up[idx], mu[idx]
                        base = ys - mus if kfrom == "residual" else ys
                        kap = float(np.quantile(base, 1 - ALPHA / 2) - np.quantile(base, ALPHA / 2))
                        out.append(
                            reduced_cwe(
                                float(interval_coverage(ys, los, ups)),
                                float(nmpi(los, ups, scale=scale)),
                                float(np.mean(np.abs(ys - mus))) / scale,
                                kap / scale,
                            )
                        )
                    vals.append(float(np.mean(out)))
                rows.append(
                    {
                        "seed": seed,
                        "reference": kfrom,
                        "stratified": label,
                        "optimum": float(factors[int(np.argmax(vals))]),
                    }
                )
        # interval score reference
        w = [float(winkler_score(y, mu - z * sigma * s, mu + z * sigma * s, alpha=ALPHA)) for s in factors]
        rows.append(
            {
                "seed": seed,
                "reference": "interval score",
                "stratified": "---",
                "optimum": float(factors[int(np.argmin(w))]),
            }
        )
    df = pd.DataFrame(rows)
    piv = df.groupby(["reference", "stratified"])["optimum"].agg(["mean", "std", "min", "max"]).reset_index()
    print(f"\n{'=' * 78}\nFACTORIAL: reference x stratification, {len(SEEDS)} seeds (truth = 1.000)\n{'=' * 78}")
    print(f"  {'reference':<18}{'stratified':<16}{'mean':<10}{'std':<10}{'range'}")
    for _, r in piv.iterrows():
        print(
            f"  {r['reference']:<18}{r['stratified']:<16}{r['mean']:<10.3f}"
            f"{r['std']:<10.4f}[{r['min']:.3f}, {r['max']:.3f}]"
        )
    return df


def real_datasets() -> pd.DataFrame:
    """Width sweep and ranking on both measured datasets."""
    allrows = []
    for ds in ("MLVS-PT", "MLVS-UK"):
        print(f"\n{'=' * 78}\nREAL DATA --- {ds}\n{'=' * 78}")
        f = _fit_intervals(ds)
        if f is None:
            continue
        y, lo, up, mu, qmat = f["y"], f["lo"], f["up"], f["mu"], f["q"]
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
        sw = pd.DataFrame(rows)
        s_cal = float(sw["factor"].to_numpy()[int(np.argmin(np.abs(sw["picp"].to_numpy() - NOMINAL)))])
        print(f"  n_test = {len(y):,}, raw PICP = {interval_coverage(y, lo, up):.3f}, s* = {s_cal:.3f}")
        for name, better in (("winkler", "min"), ("cwe_reduced", "max"), ("cwe_stratified", "max")):
            v = sw[name].to_numpy()
            opt = float(sw["factor"].to_numpy()[int(np.argmin(v) if better == "min" else np.argmax(v))])
            print(f"    {name:<16}optimum {opt:.3f}   ({100 * (opt - s_cal) / s_cal:+.1f}% from s*)")
        sw.insert(0, "dataset", ds)
        sw.to_csv(OUT / f"s3524_width_sweep_{ds}.csv", index=False)

        const_half = float(np.quantile(np.abs(y - mu), NOMINAL))
        # Carry the scale factor explicitly rather than recovering it by dividing
        # widths: a near-zero raw width makes that ratio explode.
        panel = {f"scaled x{s:.2f}": s for s in (0.6, 0.8, 1.0, 1.2, 1.6)}
        panel["constant width"] = None  # a single width, so no single factor
        prows = []
        for name, factor in panel.items():
            if factor is None:
                half = np.full_like(y, const_half)
                scale_f = 2.0 * half / np.clip(up - lo, 1e-3, None)
            else:
                scale_f = np.full_like(y, float(factor))
            q_scaled = mu[:, None] + (qmat - mu[:, None]) * scale_f[:, None]
            lo_p, up_p = q_scaled[:, 0], q_scaled[:, -1]
            sc = score_all(y, lo_p, up_p, mu, q=q_scaled)
            sc["cwe_stratified"] = stratified_cwe(y, lo_p, up_p, mu, by=(up_p - lo_p), n_strata=24)
            prows.append({"dataset": ds, "forecaster": name, **sc})
        rk = pd.DataFrame(prows)
        best_w = rk.loc[rk["winkler"].idxmin(), "forecaster"]
        print(f"\n  {'forecaster':<18}{'PICP':<9}{'NMPI':<9}{'WS':<11}{'CWE':<10}{'CWE strat':<11}{'CWC':<9}")
        for _, r in rk.iterrows():
            print(
                f"  {r['forecaster']:<18}{r['picp']:<9.3f}{r['nmpi']:<9.3f}{r['winkler']:<11.3f}"
                f"{r['cwe_reduced']:<10.3f}{r['cwe_stratified']:<11.3f}{r['cwc']:<9.3f}"
            )
        print(f"  interval score prefers: {best_w}")
        for n in ("cwe_reduced", "cwe_stratified", "cwc"):
            rho = stats.spearmanr(-rk["winkler"], rk[n]).statistic
            print(f"    {n:<16}prefers {rk.loc[rk[n].idxmax(), 'forecaster']:<18}rho = {rho:+.3f}")
        allrows.append(rk)
    return pd.concat(allrows) if allrows else pd.DataFrame()


if __name__ == "__main__":
    fac = factorial_reference_stratification()
    fac.to_csv(OUT / "s3523_netload_factorial.csv", index=False)
    rk = real_datasets()
    if len(rk):
        rk.to_csv(OUT / "s3524_ranking_measured.csv", index=False)
    print(f"\nwrote results to {OUT}")
