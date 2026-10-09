"""Does CWE need a *graded* coverage term, and can it have one for free?

The reduced formulation proposed earlier gates on calibration:

    CWE = lambda * sharpness(r)   if a calibration test passes, else 0

That removes all seven hyperparameters, but it also removes PICP as a scored
component: two forecasts that both pass the gate are indistinguishable on
coverage. Since combining coverage, width and accuracy was the entire point of
the metric, this asks whether a graded coverage term can be restored without
reintroducing free parameters.

Candidate: apply the *same* parameter-free kernel used for width,

    f(x) = 2x / (1 + x^2),      f(1) = 1,   f(x) = f(1/x),

to a coverage ratio. Two ratios are available:

    c = PICP / (1 - alpha)          coverage ratio
    m = (1 - PICP) / alpha          miscoverage ratio

Both equal 1 for a calibrated forecast. They are not equivalent: c is confined
to [0, 1/(1-alpha)] ~ [0, 1.11] at alpha=0.10, so f(c) is nearly constant over
the admissible range, whereas m ranges over [0, 1/alpha] = [0, 10] and inherits
the kernel's full dynamic range. m is also the quantity the Kupiec test is
built on.

Variants compared:
    A  gated      lambda * sharpness(r), zero if the calibration test fails
    B  product    lambda * cov(m) * sharpness(r)
    C  harmonic   lambda * 2*cov(m)*sharpness(r) / (cov(m) + sharpness(r))
    D  ratio-c    lambda * cov(c) * sharpness(r)      -- expected to be flat
    E  original   the implemented cwe_score with the alpha-tied reference

Run:  uv run python experiment/chapter-3-cwe-validation/coverage_term_analysis.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from twiga.core.metrics.interval import cwe_score, interval_coverage, nmpi, winkler_score

ALPHA = 0.10
N = 200_000


# ── parameter-free kernel ─────────────────────────────────────────────────────
def kernel(x: np.ndarray | float) -> np.ndarray | float:
    """f(x) = 2x/(1+x^2): peaks at 1, symmetric under x -> 1/x, zero at 0."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 2.0 * x / (1.0 + x**2), 0.0)


def components(picp: float, nmpi_v: float, ref: float, alpha: float) -> tuple[float, float, float]:
    """Return (miscoverage ratio m, coverage ratio c, width ratio r)."""
    m = (1.0 - picp) / alpha
    c = picp / (1.0 - alpha)
    r = nmpi_v / ref if ref > 0 else 0.0
    return m, c, r


def variants(picp: float, nmpi_v: float, err: float, ref: float, alpha: float, passes_gate: bool) -> dict[str, float]:
    """Score one (PICP, NMPI, error) triple under every candidate formulation."""
    m, c, r = components(picp, nmpi_v, ref, alpha)
    lam = float(np.clip(1.0 - err, 0.0, 1.0))
    sharp = float(kernel(r))
    cov_m = float(kernel(m))
    cov_c = float(kernel(c))
    harm = 0.0 if (cov_m + sharp) == 0 else 2.0 * cov_m * sharp / (cov_m + sharp)
    return {
        "A_gated": lam * sharp if passes_gate else 0.0,
        "B_product": lam * cov_m * sharp,
        "C_harmonic": lam * harm,
        "D_ratio_c": lam * cov_c * sharp,
        "E_original": float(np.ravel(cwe_score(nmpi_v, picp, err, ref, alpha=alpha))[0]),
    }


# ── 1. soundness: is each variant optimised by the ideal forecaster? ──────────
def soundness(alpha: float = ALPHA, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(N)
    z = stats.norm.ppf(1 - alpha / 2)
    kappa = float(np.quantile(y, 1 - alpha / 2) - np.quantile(y, alpha / 2))
    scale = float(y.max() - y.min())
    ref = kappa / scale

    sigmas = np.linspace(0.30, 2.00, 171)
    rows: dict[str, list[float]] = {}
    wink: list[float] = []
    for s in sigmas:
        lo, up = np.full_like(y, -z * s), np.full_like(y, z * s)
        picp = float(interval_coverage(y, lo, up))
        nv = float(nmpi(lo, up, scale=scale))
        err = float(np.mean(np.abs(y))) / scale
        # Gate: Kupiec proportion-of-failures at the 5% level.
        n_exc = int(np.sum((y < lo) | (y > up)))
        pi = n_exc / len(y)
        if 0 < pi < 1:
            llr = -2 * (
                (len(y) - n_exc) * np.log(1 - alpha)
                + n_exc * np.log(alpha)
                - ((len(y) - n_exc) * np.log(1 - pi) + n_exc * np.log(pi))
            )
            gate = bool(stats.chi2.sf(llr, 1) > 0.05)
        else:
            gate = False
        for k, v in variants(picp, nv, err, ref, alpha, gate).items():
            rows.setdefault(k, []).append(v)
        wink.append(float(winkler_score(y, lo, up, alpha=alpha)))

    print(f"\n{'=' * 74}\n1. SOUNDNESS  (truth at sigma = 1.000; higher is better for all variants)\n{'=' * 74}")
    print(f"  {'Winkler (reference, lower better)':<40} optimum sigma = {sigmas[int(np.argmin(wink))]:.3f}")
    for k, v in rows.items():
        arr = np.asarray(v)
        opt = sigmas[int(np.argmax(arr))]
        flag = "SOUND" if abs(opt - 1.0) < 0.02 else f"unsound ({100 * (opt - 1):+.0f}%)"
        print(f"  {k:<40} optimum sigma = {opt:.3f}   {flag}")
    return sigmas, rows, ref, scale


# ── 2. discriminability: can each variant tell 0.90 from 0.94 coverage? ───────
def discriminability(ref: float, alpha: float = ALPHA) -> None:
    print(f"\n{'=' * 74}\n2. DISCRIMINABILITY  (identical width, coverage varied)\n{'=' * 74}")
    print("  Does the variant separate forecasts that differ ONLY in coverage?")
    print(f"\n  {'PICP':<8}{'m':<8}{'A_gated':<12}{'B_product':<12}{'C_harmonic':<12}{'D_ratio_c':<12}")
    for picp in (0.82, 0.86, 0.90, 0.94, 0.98):
        v = variants(picp, ref, 0.0, ref, alpha, passes_gate=True)  # width exactly nominal
        m = (1 - picp) / alpha
        print(
            f"  {picp:<8.2f}{m:<8.2f}{v['A_gated']:<12.4f}{v['B_product']:<12.4f}"
            f"{v['C_harmonic']:<12.4f}{v['D_ratio_c']:<12.4f}"
        )
    print("\n  A_gated is constant by construction: coverage is not scored, only gated.")


# ── 3. conditional test: heteroscedastic target, marginally identical ─────────
def conditional(alpha: float = ALPHA, seeds: int = 6) -> None:
    print(f"\n{'=' * 74}\n3. CONDITIONAL CALIBRATION  (A is correct; B is constant-width)\n{'=' * 74}")
    print(f"  {'seed':<6}{'winkler picks':<16}{'A_gated':<11}{'B_product':<11}{'C_harmonic':<11}{'agrees?'}")
    agree = {"A_gated": 0, "B_product": 0, "C_harmonic": 0}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        n = 100_000
        s = rng.uniform(0.3, 3.0, n)
        y = rng.normal(0.0, s)
        z = stats.norm.ppf(1 - alpha / 2)
        scale = float(y.max() - y.min())
        kappa = float(np.quantile(y, 1 - alpha / 2) - np.quantile(y, alpha / 2))
        ref = kappa / scale

        # A: conditionally correct. B: one constant width, matched to marginal coverage.
        wa = z * s
        wb = np.full(n, float(np.quantile(np.abs(y), 1 - alpha)))
        out = {}
        for name, half in (("A", wa), ("B", wb)):
            lo, up = -half, half
            picp = float(interval_coverage(y, lo, up))
            nv = float(nmpi(lo, up, scale=scale))
            err = float(np.mean(np.abs(y))) / scale
            out[name] = {
                "w": float(winkler_score(y, lo, up, alpha=alpha)),
                **variants(picp, nv, err, ref, alpha, passes_gate=True),
            }
        w_pick = "A" if out["A"]["w"] < out["B"]["w"] else "B"
        line = f"  {seed:<6}{w_pick + ' (correct)':<16}"
        for k in ("A_gated", "B_product", "C_harmonic"):
            pick = "A" if out["A"][k] > out["B"][k] else "B"
            if pick == w_pick:
                agree[k] += 1
            line += f"{pick:<11}"
        print(line)
    print(f"\n  agreement with Winkler over {seeds} seeds: " + ", ".join(f"{k}={v}/{seeds}" for k, v in agree.items()))


if __name__ == "__main__":
    sigmas, rows, ref, scale = soundness()
    discriminability(ref)
    conditional()


# ── 4. stratified: does the graded variant survive the conditional test? ──────
def stratified(alpha: float = ALPHA, seeds: int = 6, n_strata: int = 5) -> None:
    """Score inside strata of the conditioning variable, then average.

    The conditional defect is a property of aggregating widths BEFORE comparing
    them to the reference; it is orthogonal to how coverage is treated. This
    checks that the graded formulation recovers the right answer once scored
    within strata, exactly as the gated one does.
    """
    print(f"\n{'=' * 74}\n4. STRATIFIED SCORING  (same target, scored within quintiles)\n{'=' * 74}")
    print(f"  {'seed':<6}{'winkler':<10}{'B_product':<24}{'C_harmonic':<24}")
    print(f"  {'':<6}{'picks':<10}{'A / B / picks':<24}{'A / B / picks':<24}")
    agree = {"B_product": 0, "C_harmonic": 0}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        n = 100_000
        s = rng.uniform(0.3, 3.0, n)
        y = rng.normal(0.0, s)
        z = stats.norm.ppf(1 - alpha / 2)
        scale = float(y.max() - y.min())  # shared R, computed once
        order = np.argsort(s)
        bins = np.array_split(order, n_strata)

        wa = z * s
        wb = np.full(n, float(np.quantile(np.abs(y), 1 - alpha)))
        scores = {"A": {}, "B": {}}
        wink = {}
        for name, half in (("A", wa), ("B", wb)):
            lo, up = -half, half
            wink[name] = float(winkler_score(y, lo, up, alpha=alpha))
            acc: dict[str, list[float]] = {}
            for idx in bins:
                ys, los, ups = y[idx], lo[idx], up[idx]
                kappa_b = float(np.quantile(ys, 1 - alpha / 2) - np.quantile(ys, alpha / 2))
                ref_b = kappa_b / scale
                picp_b = float(interval_coverage(ys, los, ups))
                nv_b = float(nmpi(los, ups, scale=scale))
                err_b = float(np.mean(np.abs(ys))) / scale
                for k, v in variants(picp_b, nv_b, err_b, ref_b, alpha, True).items():
                    acc.setdefault(k, []).append(v)
            scores[name] = {k: float(np.mean(v)) for k, v in acc.items()}

        w_pick = "A" if wink["A"] < wink["B"] else "B"
        line = f"  {seed:<6}{w_pick:<10}"
        for k in ("B_product", "C_harmonic"):
            pick = "A" if scores["A"][k] > scores["B"][k] else "B"
            if pick == w_pick:
                agree[k] += 1
            line += f"{scores['A'][k]:.3f} / {scores['B'][k]:.3f} / {pick:<7}"
        print(line)
    print("\n  agreement with Winkler: " + ", ".join(f"{k}={v}/{seeds}" for k, v in agree.items()))


stratified()
