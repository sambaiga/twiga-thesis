r"""Chapter 6: what the recorded networks look like, before any model is fitted.

These are data figures, in the sense of the dataset section of Chapter 2: they
characterise the records rather than report a result. Their purpose is to show
that the coupling the chapter goes on to model is visible in the measurements,
so the case for a joint representation is made from the data and not only from
the argument of Section 6.1.

The joint panels are the measured counterpart of the drawn illustration in
Section 6.1: the same axes, the same apparent-power arc, and the same question
of how much probability mass lies beyond it, but with clouds produced by
time-series power flow rather than by construction.

Run:  uv run python -c "from chapters.ch7_multitarget.figures.data import render_all; render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twiga_thesis.paths import external_data_dir, thesis_data_dir
from twiga_thesis.plotting import style as S, tokens as T

ml_energy_repo = external_data_dir() / "experiments/lv-multitarget/data"
EU = ml_energy_repo / "substation_pqv_17520_smartds.parquet"
US_YEAR = ml_energy_repo / "pole_transformer_pqv_17520_year.parquet"
US_LOADED = ml_energy_repo / "pole_transformer_pqv_4320_d121_loaded.parquet"
US_QUICK = ml_energy_repo / "pole_transformer_pqv_336_quick_d212.parquet"

# The functions above (_eu(), eu_joint_*, hourly_p/q/v) read the multi-feeder
# 'weak'/'stiff' record and are named "eu" in this module for historical
# reasons, but Chapter-2.tex's own captions correctly call that record
# DSS-UK (two-feeder, Low Carbon London demand, Q(U) droop -- see
# Chapter-2.tex's dataset table). DSS_EU_TRUE below is the actual DSS-EU
# record this session rebuilt: the four-feeder Hybrid European MV-LV network
# (Koirala et al. 2020), one monitored busbar, no per-feeder dimension.
DSS_EU_TRUE = thesis_data_dir() / "DSS-EU.parquet"


def _dss_eu_true() -> pd.DataFrame:
    d = pd.read_parquet(DSS_EU_TRUE)
    d["hour"] = pd.to_datetime(d["timestamp"]).dt.hour + pd.to_datetime(d["timestamp"]).dt.minute / 60
    return d


def _dss_eu_true_hourly(column: str, label: str) -> plt.Figure:
    """Median and 10th to 90th percentile band by time of day, on the real DSS-EU record."""
    d = _dss_eu_true()
    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.86))
    cases = (
        (0.0, 0.0, "no DER", T.TEXT_MUTED, "--"),
        (0.6, 0.6, r"60\,\% EV and PV", T.PRIMARY, "-"),
    )
    for ev, pv, name, colour, dash in cases:
        sub = d[(d["ev_pen"] == ev) & (d["pv_pen"] == pv)]
        if sub.empty:
            continue
        g = sub.groupby("hour")[column]
        med, lo, hi = g.median(), g.quantile(0.10), g.quantile(0.90)
        ax.fill_between(med.index, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(med.index, med.to_numpy(), dash, color=colour, lw=1.4, label=name)
    ax.set_xlim(0, 23.5)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(label)
    if column == "p_kw":
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return fig


def dss_eu_hourly_p() -> plt.Figure:
    """Active power against time of day, real DSS-EU (four-feeder busbar)."""
    return _dss_eu_true_hourly("p_kw", r"$P$ (kW)")


def dss_eu_hourly_q() -> plt.Figure:
    """Reactive power against time of day, real DSS-EU (four-feeder busbar)."""
    return _dss_eu_true_hourly("q_kvar", r"$Q$ (kvar)")


def dss_eu_hourly_v() -> plt.Figure:
    """Busbar voltage against time of day, real DSS-EU (four-feeder busbar)."""
    return _dss_eu_true_hourly("v_lv_pu", r"$V$ (pu)")


# The European model rates its secondary transformers at 800 kVA, which the
# recorded feeders never approach, so the line drawn on those panels is the
# screening level a planner would use rather than the nameplate.
EU_SCREEN_Q = 0.995


def _eu() -> pd.DataFrame:
    d = pd.read_parquet(EU)
    d["hour"] = pd.to_datetime(d["timestamp"]).dt.hour + pd.to_datetime(d["timestamp"]).dt.minute / 60
    return d


def _us() -> pd.DataFrame:
    """The North American record, preferring the full year over the shorter runs."""
    for path in (US_YEAR, US_LOADED, US_QUICK):
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(f"no US record under {ml_energy_repo}")


def _us_hourly_frame() -> pd.DataFrame:
    """The US record on one pole transformer, with an hour-of-day column.

    A feeder carries hundreds of pole transformers and only a handful bind, so
    an average over all of them would describe no asset a planner acts on. The
    most heavily loaded is taken instead, which is the monitoring point the case
    study uses. The record is indexed by simulation step rather than timestamp,
    so the hour is recovered from the 30-minute step.
    """
    d = _us()
    peak = d.groupby("trafo").apply(lambda g: (g["s_kva"] / g["kva_rating"]).max(), include_groups=False)
    d = d[d["trafo"] == peak.idxmax()].copy()
    d["hour"] = (d["step"] % 48) / 2.0
    return d


def _joint_panel(p, q, limit, title, note) -> plt.Figure:
    """One (P, Q) cloud against its apparent-power limit, with the margins shown."""
    fig = plt.figure(figsize=S.panel(0.31, aspect=1.05))
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 6), wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    hi_p = max(np.abs(p).max(), limit) * 1.08
    lo_p = min(p.min(), 0.0) * 1.08
    hi_q = max(q.max(), limit) * 1.08

    theta = np.linspace(0, np.pi, 400)
    ax.plot(limit * np.cos(theta), limit * np.sin(theta), color=T.DANGER, lw=1.4, zorder=4)

    # Rasterized: thousands of points per panel as vector glyphs bloats the
    # PDF and makes it slow to open/scroll; axes, text, and the limit arc
    # stay vector since only these two point-cloud artists set the flag.
    over = np.hypot(p, q) > limit
    ax.plot(p[~over], q[~over], "o", ms=1.1, color=T.PRIMARY, alpha=0.18, mew=0, zorder=2, rasterized=True)
    if over.any():
        ax.plot(p[over], q[over], "o", ms=1.6, color=T.DANGER, alpha=0.55, mew=0, zorder=3, rasterized=True)

    ax.set_xlim(lo_p, hi_p)
    ax.set_ylim(0, hi_q)
    ax.set_xlabel(r"$P$ (kW)")
    ax.set_ylabel(r"$Q$ (kvar)")
    ax.text(
        0.97,
        0.96,
        note,
        ha="right",
        transform=ax.transAxes,
        fontsize=8,
        color=T.DANGER,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    axt.hist(p, bins=70, color=T.PRIMARY_TINT, lw=0)
    axr.hist(q, bins=70, orientation="horizontal", color=T.PRIMARY_TINT, lw=0)
    for a in (axt, axr):
        a.axis("off")
    axt.set_title(title, fontsize=9, pad=4)
    return fig


def _eu_joint(ev: float, pv: float, title: str) -> plt.Figure:
    d = _eu()
    limit = d["s_kva"].quantile(EU_SCREEN_Q)
    g = d[(d["feeder"] == "weak") & (d["ev_pen"] == ev) & (d["pv_pen"] == pv)]
    p, q = g["p_kw"].to_numpy(), g["q_kvar"].to_numpy()
    return _joint_panel(p, q, limit, title, rf"$\rho={np.corrcoef(p, q)[0, 1]:+.2f}$")


def _us_joint(ev: float, pv: float, title: str) -> plt.Figure:
    """One transformer held fixed across scenarios, so the panels compare like with like."""
    d = _us()
    trafo = d.loc[(d["s_kva"] / d["kva_rating"]).idxmax(), "trafo"]
    g = d[(d["trafo"] == trafo) & (d["ev_pen"] == ev) & (d["pv_pen"] == pv)]
    p, q = g["p_kw"].to_numpy(), g["q_kvar"].to_numpy()
    return _joint_panel(p, q, float(g["kva_rating"].iloc[0]), title, rf"$\rho={np.corrcoef(p, q)[0, 1]:+.2f}$")


def us_joint_evpv() -> plt.Figure:
    """The only North American scenario that exceeds the transformer rating.

    On the most loaded pole transformer, 1 of the 16 penetration scenarios puts
    apparent power beyond the nameplate, and it is the combined one: 60 per cent
    charging with 40 per cent generation. Charging alone reaches 0.82 of the
    rating and generation alone 0.37, so a panel drawn at either would show an
    arc no point crosses and would not illustrate what the figure is for.
    """
    return _us_joint(0.6, 0.4, r"US, 60\,\% EV and 40\,\% PV")


def eu_joint_nodter() -> plt.Figure:
    """European feeder without distributed energy resources."""
    return _eu_joint(0.0, 0.0, "EU, no DER")


def eu_joint_ev() -> plt.Figure:
    """The same feeder under charging load alone, which binds by demand."""
    return _eu_joint(0.6, 0.0, r"EU, 60\,\% EV")


def eu_joint_pv() -> plt.Figure:
    """The same feeder under generation alone, which binds by export."""
    return _eu_joint(0.0, 0.6, r"EU, 60\,\% PV")


def us_joint_nodter() -> plt.Figure:
    """North American pole transformer without distributed energy resources."""
    return _us_joint(0.0, 0.0, "US, no DER")


def us_joint_ev() -> plt.Figure:
    """The same transformer under charging load alone."""
    return _us_joint(0.6, 0.0, r"US, 60\,\% EV")


def us_joint_pv() -> plt.Figure:
    """The same transformer under generation alone."""
    return _us_joint(0.0, 0.6, r"US, 60\,\% PV")


def _hourly(column: str, label: str) -> plt.Figure:
    """Median and interquartile band by time of day, without and with DER."""
    d = _eu()
    d = d[d["feeder"] == "weak"]
    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.86))
    cases = (
        (0.0, 0.0, "no DER", T.TEXT_MUTED, "--"),
        (0.6, 0.6, r"60\,\% EV and PV", T.PRIMARY, "-"),
    )
    for ev, pv, name, colour, dash in cases:
        g = d[(d["ev_pen"] == ev) & (d["pv_pen"] == pv)].groupby("hour")[column]
        med, lo, hi = g.median(), g.quantile(0.10), g.quantile(0.90)
        ax.fill_between(med.index, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(med.index, med.to_numpy(), dash, color=colour, lw=1.4, label=name)
    ax.set_xlim(0, 23.5)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(label)
    if column == "p_kw":
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return fig


def _hourly_us(column: str, label: str) -> plt.Figure:
    """Median and 10th to 90th percentile band by time of day, on the US record."""
    d = _us_hourly_frame()
    fig, ax = plt.subplots(figsize=S.panel(0.31, aspect=0.86))
    cases = (
        (0.0, 0.0, "no DER", T.TEXT_MUTED, "--"),
        (0.6, 0.6, r"60\,\% EV and PV", T.PRIMARY, "-"),
    )
    for ev, pv, name, colour, dash in cases:
        sub = d[(d["ev_pen"] == ev) & (d["pv_pen"] == pv)]
        if sub.empty:
            continue
        g = sub.groupby("hour")[column]
        med, lo, hi = g.median(), g.quantile(0.10), g.quantile(0.90)
        ax.fill_between(med.index, lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot(med.index, med.to_numpy(), dash, color=colour, lw=1.4, label=name)
    ax.set_xlim(0, 23.5)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(label)
    if column == "p_kw":
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return fig


def hourly_us_p() -> plt.Figure:
    """Active power against time of day, North American pole transformer."""
    return _hourly_us("p_kw", r"$P$ (kW)")


def hourly_us_q() -> plt.Figure:
    """Reactive power against time of day, North American pole transformer."""
    return _hourly_us("q_kvar", r"$Q$ (kvar)")


def hourly_us_v() -> plt.Figure:
    """Worst customer voltage against time of day, North American pole transformer."""
    return _hourly_us("v_lv_min_pu", r"$V$ (pu)")


def hourly_p() -> plt.Figure:
    """Active power against time of day."""
    return _hourly("p_kw", r"$P$ (kW)")


def hourly_q() -> plt.Figure:
    """Reactive power against time of day."""
    return _hourly("q_kvar", r"$Q$ (kvar)")


def hourly_v() -> plt.Figure:
    """Worst customer voltage against time of day."""
    return _hourly("v_worst_pu", r"$V$ (pu)")


FIGURES = {
    "fig_ch6_joint_eu_nodter.pdf": eu_joint_nodter,
    "fig_ch6_joint_eu_ev.pdf": eu_joint_ev,
    "fig_ch6_joint_eu_pv.pdf": eu_joint_pv,
    "fig_ch6_joint_us_nodter.pdf": us_joint_nodter,
    "fig_ch6_joint_us_ev.pdf": us_joint_ev,
    "fig_ch6_joint_us_pv.pdf": us_joint_pv,
    "fig_ch6_joint_us_evpv.pdf": us_joint_evpv,
    "fig_ch6_hourly_p.pdf": hourly_p,
    "fig_ch6_hourly_q.pdf": hourly_q,
    "fig_ch6_hourly_v.pdf": hourly_v,
    "fig_ch6_hourly_us_p.pdf": hourly_us_p,
    "fig_ch6_hourly_us_q.pdf": hourly_us_q,
    "fig_ch6_hourly_us_v.pdf": hourly_us_v,
    "fig_ch6_hourly_eu_p.pdf": dss_eu_hourly_p,
    "fig_ch6_hourly_eu_q.pdf": dss_eu_hourly_q,
    "fig_ch6_hourly_eu_v.pdf": dss_eu_hourly_v,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render every data figure to ``out`` and to the Chapter 6 image folder."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    thesis = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-6"
    for name, fn in FIGURES.items():
        fig = fn()
        # 300 dpi for the rasterized point-cloud panels (fig_ch6_joint_*): print-
        # crisp without the multi-MB file a vector-per-point PDF would produce.
        fig.savefig(out / name, dpi=300)
        if thesis.exists():
            fig.savefig(thesis / name, dpi=300)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
