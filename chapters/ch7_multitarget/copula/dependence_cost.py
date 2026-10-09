r"""What it costs to assume the targets are independent.

Section 6.3 argues that a triple of well-calibrated marginals need not describe
a realisable network state. This script measures the size of that error on the
power-flow records, without fitting anything: for each scenario the empirical
joint distribution of :math:`(P, Q, V)` is compared against the *same* marginals
with the dependence destroyed by permutation. Any difference in the resulting
constraint-violation probability is attributable to dependence alone, and it
bounds what a forecaster with perfect marginals but no coupling could achieve.

Three records are covered:

``b1_measured``
    European feeders, household demand from the SMART-DS residential pool, so
    reactive power is measured rather than synthesised from a power factor.
``b1_synth``
    The same feeders with reactive power synthesised from an assumed power
    factor. Kept for contrast: it shows how much the assumption distorts the
    dependence, which is the third limitation the section reports.
``b2``
    North American pole transformers, where the binding constraint is thermal
    rather than voltage and aggregation is an order of magnitude lower.

Alongside the permutation, each cell is summarised by Pearson correlation and by
the distance correlation of :func:`twiga.core.stats.distance_correlation`, which
is zero only under independence. Pearson passes through zero where the coupling
changes character, so the two together show both how strong the dependence is
and how badly a linear summary represents it.

Run:  uv run python experiment/ch7_multitarget/dependence_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from twiga.core.stats import distance_correlation
from twiga.core.utils import get_logger

from twiga_thesis.paths import external_data_dir

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
ml_energy_repo = external_data_dir() / "experiments/lv-multitarget/data"

# Permutations per cell. The estimate is a mean over permutations, so this
# controls its standard error rather than any bias.
N_PERM = 200
SEED = 0

# European secondary substations are rated 800 kVA in this network model, but the
# recorded feeders never approach that, so the screening threshold is set from the
# record itself: the level a planner would treat as the top of normal operation.
B1_QUANTILE = 0.995
# US pole transformers are small enough that utilisation is the natural trigger.
# 0.7 is the loading at which a planner opens a reinforcement study rather than
# the nameplate itself. The 90-day record does reach 1.05 utilisation, but only
# three of its 480 cells carry an event at the nameplate, too few to describe a
# distribution of errors; 26 carry one at 0.7.
B2_UTILISATION = 0.70
V_LOWER_PU = 0.95


def _rates_conditional(p, q, v, strata, s_limit, v_limit, rng, n_perm=None):
    """Violation probabilities with the dependence removed *within* strata.

    An unrestricted permutation confounds genuine coupling with dependence
    induced by shared drivers such as time of day, which a model conditioning on
    those drivers would already represent. Permuting inside strata removes that
    component and leaves the conditional dependence, which is the quantity a
    forecaster would actually have to learn.
    """
    n_perm = n_perm or N_PERM
    s = np.hypot(p, q)
    joint_th = float(np.mean(s > s_limit))
    joint_un = float(np.mean((s > s_limit) | (v < v_limit)))
    groups = [np.flatnonzero(strata == k) for k in np.unique(strata)]
    ind_th, ind_un = [], []
    for _ in range(n_perm):
        qq, vv = q.copy(), v.copy()
        for idx in groups:
            qq[idx] = rng.permutation(q[idx])
            vv[idx] = rng.permutation(v[idx])
        s_i = np.hypot(p, qq)
        ind_th.append(np.mean(s_i > s_limit))
        ind_un.append(np.mean((s_i > s_limit) | (vv < v_limit)))
    return joint_th, float(np.mean(ind_th)), joint_un, float(np.mean(ind_un))


def block_bootstrap(p, q, s_limit, rng, block=48, reps=300):
    """Percentage error from independence, with a moving-block confidence interval.

    Half-hourly series are strongly autocorrelated, so resampling individual
    steps would give an interval far too narrow. Daily blocks preserve the
    within-day structure that drives the exceedance.
    """
    n = len(p)
    if n <= block * 2:
        return np.nan, np.nan, np.nan
    errs = []
    for _ in range(reps):
        starts = rng.integers(0, n - block, size=max(n // block, 1))
        idx = np.concatenate([np.arange(st, st + block) for st in starts])
        pj, qj = p[idx], q[idx]
        joint = np.mean(np.hypot(pj, qj) > s_limit)
        indep = np.mean(np.hypot(pj, rng.permutation(qj)) > s_limit)
        if joint > 0:
            errs.append((indep - joint) / joint * 100.0)
    if not errs:
        return np.nan, np.nan, np.nan
    return float(np.mean(errs)), *(float(x) for x in np.percentile(errs, [2.5, 97.5]))


def _rates(p, q, v, s_limit, v_limit, rng):
    """Violation probabilities under the joint record and under permuted margins.

    Permuting one target against the others preserves every marginal exactly
    and destroys only the dependence, which is what isolates its contribution.
    """
    s = np.hypot(p, q)
    joint_th = float(np.mean(s > s_limit))
    joint_un = float(np.mean((s > s_limit) | (v < v_limit)))
    ind_th, ind_un = [], []
    for _ in range(N_PERM):
        s_i = np.hypot(p, rng.permutation(q))
        ind_th.append(np.mean(s_i > s_limit))
        ind_un.append(np.mean((s_i > s_limit) | (rng.permutation(v) < v_limit)))
    return joint_th, float(np.mean(ind_th)), joint_un, float(np.mean(ind_un))


def _cells(df, group, p_col, q_col, v_col, s_limit_of, v_limit, strata_of=None):
    rng = np.random.default_rng(SEED)
    rows = []
    for key, g in df.groupby(group):
        p = g[p_col].to_numpy(float)
        q = g[q_col].to_numpy(float)
        v = g[v_col].to_numpy(float)
        if len(p) < 100:
            continue
        limit = s_limit_of(g)
        jt, it, ju, iu = _rates(p, q, v, limit, v_limit, rng)
        if strata_of is None:
            ct = np.nan
        else:
            _, ct, _, _ = _rates_conditional(p, q, v, strata_of(g), limit, v_limit, rng, n_perm=40)
        boot, lo, hi = block_bootstrap(p, q, limit, rng)
        rows.append(
            dict(
                zip(group, key if isinstance(key, tuple) else (key,), strict=False),
                n=len(p),
                r_pq=np.corrcoef(p, q)[0, 1],
                dcor_pq=distance_correlation(p, q, random_state=SEED),
                r_absp_q=np.corrcoef(np.abs(p), q)[0, 1],
                r_pv=np.corrcoef(p, v)[0, 1],
                thermal_joint=jt,
                thermal_indep=it,
                thermal_cond=ct,
                boot_err=boot,
                boot_lo=lo,
                boot_hi=hi,
                union_joint=ju,
                union_indep=iu,
            )
        )
    out = pd.DataFrame(rows)
    for kind in ("thermal", "union"):
        j, i = out[f"{kind}_joint"], out[f"{kind}_indep"]
        out[f"{kind}_err_pct"] = np.where(j > 0, (i - j) / j * 100.0, np.nan)
    j = out["thermal_joint"]
    out["cond_err_pct"] = np.where(j > 0, (out["thermal_cond"] - j) / j * 100.0, np.nan)
    return out


def b1(measured: bool) -> pd.DataFrame:
    """European feeders, one row per feeder and penetration pair."""
    local_path = script_dir.parent / "data" / "substation_pqv_17520.parquet"
    path = ml_energy_repo / "substation_pqv_17520_smartds.parquet" if measured else local_path
    d = pd.read_parquet(path)
    limit = d["s_kva"].quantile(B1_QUANTILE)

    def strata(g):
        hour = pd.to_datetime(g["timestamp"]).dt.hour.to_numpy()
        ghi_q = pd.qcut(g["Ghi"].rank(method="first"), 4, labels=False).to_numpy()
        return hour * 4 + ghi_q

    out = _cells(
        d,
        ["feeder", "ev_pen", "pv_pen"],
        "p_kw",
        "q_kvar",
        "v_worst_pu",
        lambda _g: limit,
        V_LOWER_PU,
        strata_of=strata,
    )
    out.insert(0, "record", "b1_measured" if measured else "b1_synth")
    out.attrs["s_limit"] = limit
    return out


def _us_record() -> Path:
    """The North American record, preferring the full year over the shorter runs.

    The 90-day record covers a single season, which is a poor basis for a claim
    about dependence that the chapter then reads across the year.
    """
    for name in (
        "pole_transformer_pqv_17520_year.parquet",
        "pole_transformer_pqv_4320_d121_loaded.parquet",
        "pole_transformer_pqv_336_quick_d212.parquet",
    ):
        if (ml_energy_repo / name).exists():
            return ml_energy_repo / name
    raise FileNotFoundError(f"no US record under {ml_energy_repo}")


def b2() -> pd.DataFrame:
    """North American pole transformers, one row per transformer and scenario."""
    d = pd.read_parquet(_us_record())
    out = _cells(
        d,
        ["trafo", "ev_pen", "pv_pen"],
        "p_kw",
        "q_kvar",
        "v_lv_min_pu",
        lambda g: B2_UTILISATION * g["kva_rating"].iloc[0],
        V_LOWER_PU,
        # No irradiance is recorded here, so the stratification is on time of day
        # alone, derived from the half-hourly step index.
        strata_of=lambda g: (g["step"].to_numpy() % 48) // 2,
    )
    out.insert(0, "record", "b2")
    return out


def summarise(out: pd.DataFrame) -> str:
    lines = []
    for record, g in out.groupby("record"):
        act = g[g["thermal_joint"] > 0]
        lines.append(f"{record}: {len(g)} cells, {len(act)} with thermal events")
        near = g[g.r_pq.abs() < 0.3]
        if len(near):
            lines.append(
                f"  cells with |Pearson| < 0.3: {len(near)}, "
                f"median distance correlation there {near.dcor_pq.median():.2f}"
            )
        lines.append(
            f"  corr(P,Q)   {g.r_pq.min():+.2f} to {g.r_pq.max():+.2f}"
            f"   corr(|P|,Q) {g.r_absp_q.min():+.2f} to {g.r_absp_q.max():+.2f}"
            f"   corr(P,V) {g.r_pv.min():+.2f} to {g.r_pv.max():+.2f}"
        )
        cond = act.dropna(subset=["cond_err_pct"])
        if len(cond):
            lines.append(
                f"  conditional (within strata): median {cond.cond_err_pct.median():+.0f}%"
                f"  range {cond.cond_err_pct.min():+.0f}% to {cond.cond_err_pct.max():+.0f}%"
            )
        boot = act.dropna(subset=["boot_err"])
        if len(boot):
            w = boot.loc[boot.thermal_err_pct.idxmin()]
            lines.append(f"  worst cell bootstrap: {w.boot_err:+.0f}% [{w.boot_lo:+.0f}%, {w.boot_hi:+.0f}%]")
        if len(act):
            e = act.thermal_err_pct
            lines.append(
                f"  thermal error from independence: median {e.median():+.0f}%"
                f"  range {e.min():+.0f}% to {e.max():+.0f}%"
            )
            u = g[g["union_joint"] > 0].union_err_pct
            lines.append(
                f"  union error from independence  : median {u.median():+.0f}%"
                f"  range {u.min():+.0f}% to {u.max():+.0f}%"
            )
    return "\n".join(lines)


def reported_values(out: pd.DataFrame) -> dict[str, float]:
    """The numbers the chapter quotes, so prose and analysis cannot drift apart.

    Every value here appears verbatim in Section 6.3. ``check_chapter.py``
    compares them against the LaTeX source, which is what caught two silent
    disagreements while the section was being written.
    """
    values: dict[str, float] = {}
    for record, label in (("b1_measured", "eu"), ("b2", "us")):
        g = out[out["record"] == record]
        act = g[g["thermal_joint"] > 0]
        near = g[g["r_pq"].abs() < 0.3]
        values[f"{label}_median_err"] = round(act["thermal_err_pct"].median())
        values[f"{label}_worst_err"] = round(act["thermal_err_pct"].min())
        values[f"{label}_cond_median"] = round(act["cond_err_pct"].median())
        values[f"{label}_near_zero_cells"] = len(near)
        values[f"{label}_near_zero_dcor"] = round(near["dcor_pq"].median(), 2)
        values[f"{label}_corr_min"] = round(g["r_pq"].min(), 2)
        values[f"{label}_corr_max"] = round(g["r_pq"].max(), 2)
    worst = out[(out["record"] == "b1_measured") & (out["thermal_joint"] > 0)]
    w = worst.loc[worst["thermal_err_pct"].idxmin()]
    values["eu_boot_err"] = round(w["boot_err"])
    values["eu_boot_lo"] = round(w["boot_lo"])
    values["eu_boot_hi"] = round(w["boot_hi"])
    values["eu_worst_joint_pct"] = round(100 * w["thermal_joint"], 1)
    values["eu_worst_indep_pct"] = round(100 * w["thermal_indep"], 1)
    return values


def main() -> None:
    out = pd.concat([b1(measured=True), b1(measured=False), b2()], ignore_index=True)
    out.to_csv(script_dir / "dependence_cost.csv", index=False)
    (script_dir / "dependence_cost_reported.json").write_text(json.dumps(reported_values(out), indent=2) + "\n")
    print(summarise(out))
    log.info("wrote %s", script_dir / "dependence_cost.csv")
    log.info("wrote %s", script_dir / "dependence_cost_reported.json")


if __name__ == "__main__":
    main()
