"""Figures that explain the decision events from the records alone, before any forecast is applied.

Two figures for Chapter 6's operational decision experiment:

* ``event_rates_grid``: how often each event occurs across the EV x PV grid,
  per network (rows: sustained overload above the pre-DER headroom, above
  1.25 times it, and customer under-voltage; columns: DSS-UK weak and stiff,
  DSS-US rank 8). It shows which scenarios make each event rare, common or
  certain, and so where a forecast can add value at all.
* ``event_anatomy_<scenario>``: what one scenario's events look like: when in
  the day apparent power crosses the overload tiers, on which days, why the
  thermal event needs P and Q jointly (it is a circle in the P-Q plane), and,
  where customer voltages are recorded, how the busbar stays inside the
  statutory band while the feeder's lowest customer drops out of it.

Every number comes from the test months (June to December), the period the
decision experiment scores.

Usage::

    uv run python experiment/ch7_multitarget/experiment_3_decision/plot_decision_events.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(script_dir), str(script_dir.parent / "experiment_1_benchmark"), str(script_dir.parents[1])]

import chapters.ch7_multitarget.experiment_3_decision.decision_scoring as ds  # noqa: E402
import chapters.ch7_multitarget.experiment_3_decision.fixed_decision_sweep as sweep  # noqa: E402
from twiga_thesis.plotting import style as S, tokens as T  # noqa: E402

TEST_MONTH_START = 6
NETWORK_LABEL = {"DSS-UK:weak": "DSS-UK weak", "DSS-UK:stiff": "DSS-UK stiff", "DSS-US:8": "DSS-US rank 8"}
RATE_CMAP = LinearSegmentedColormap.from_list("rate", [T.WHITE, T.PRIMARY])
LOADING_CMAP = LinearSegmentedColormap.from_list("loading", [T.PRIMARY, T.WHITE, T.DANGER])


def pct() -> str:
    """A percent sign that survives both renderers: escaped only when TeX typesets the figure."""
    return r"\%" if matplotlib.rcParams["text.usetex"] else "%"


def test_months(record: pd.DataFrame) -> pd.DataFrame:
    """The record's rows in the scored test months."""
    return record[pd.to_datetime(record["timestamp"], utc=True).dt.month >= TEST_MONTH_START]


def daily(values: np.ndarray, horizon: int = 48) -> np.ndarray:
    """Whole days of a series, ``(K, horizon)``."""
    return values[: (len(values) // horizon) * horizon].reshape(-1, horizon)


def scenario_rates(record_key: str, limits: ds.DecisionLimits, ev: float, pv: float) -> dict[str, float]:
    """Share of test-month days with each event, from the recorded values."""
    frame = test_months(sweep.scenario_record(f"{record_key}:{ev:g}:{pv:g}"))
    apparent = daily(np.hypot(frame["p_kw"], frame["q_kvar"]).to_numpy())
    rates = {
        f"{k:g}x": float(ds._has_run(apparent > k * limits.rating, limits.min_duration).mean())
        for k in limits.severities
    }
    if "v_cust_min_pu" in frame.columns:
        low, high = limits.voltage_band
        v = daily(frame["v_cust_min_pu"].to_numpy())
        rates["voltage"] = float(((v < low) | (v > high)).any(axis=1).mean())
    return rates


def plot_rates_grid(limits: dict[str, ds.DecisionLimits], out: Path) -> None:
    """Heatmaps of the share of days with each event over EV x PV, one column per network."""
    rows = [
        ("1x", r"Overload, $1\times$ headroom"),
        ("1.25x", r"Overload, $1.25\times$ headroom"),
        ("voltage", "Customer under-voltage"),
    ]
    networks = list(limits)
    levels = sweep.LEVELS
    fig, axes = plt.subplots(len(rows), len(networks), figsize=S.panel(1.0, aspect=0.95), squeeze=False)
    grids = {
        net: {(ev, pv): scenario_rates(net, limits[net], ev, pv) for ev in levels for pv in levels} for net in networks
    }
    for i, (key, title) in enumerate(rows):
        for j, net in enumerate(networks):
            ax = axes[i, j]
            grid = (
                np.array([[grids[net][(ev, pv)][key] for pv in levels] for ev in levels])
                if key in grids[net][(0.0, 0.0)]
                else np.zeros((len(levels), len(levels)))
            )
            if not grid.any():
                message = "not recorded" if key not in grids[net][(0.0, 0.0)] else "never breached"
                ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, color=T.TEXT_MUTED)
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(NETWORK_LABEL[net], fontsize=9)
                continue
            ax.imshow(grid, cmap=RATE_CMAP, vmin=0.0, vmax=1.0, origin="lower")
            for (r, c), value in np.ndenumerate(grid):
                ink = T.TEXT_LIGHT if value > 0.6 else T.TEXT_DARK
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=6, color=ink)
            ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels], fontsize=7)
            ax.set_yticks(range(len(levels)), [f"{v:g}" for v in levels], fontsize=7)
            ax.tick_params(which="minor", left=False, bottom=False, top=False, right=False)
            ax.tick_params(which="major", length=0)
            if i == len(rows) - 1:
                ax.set_xlabel("PV penetration", fontsize=8)
            if j == 0:
                ax.set_ylabel("EV penetration", fontsize=8)
                ax.text(-0.42, 0.5, title, rotation=90, ha="center", va="center", fontsize=9, transform=ax.transAxes)
            if i == 0:
                ax.set_title(NETWORK_LABEL[net], fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"event_rates_grid.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_anatomy(record_key: str, ev: float, pv: float, limits: ds.DecisionLimits, out: Path) -> None:
    """Four views of one scenario's events from its recorded values."""
    frame = test_months(sweep.scenario_record(f"{record_key}:{ev:g}:{pv:g}"))
    apparent = daily(np.hypot(frame["p_kw"], frame["q_kvar"]).to_numpy())
    hours = np.arange(apparent.shape[1]) / 2.0
    days = pd.to_datetime(frame["timestamp"], utc=True).iloc[:: apparent.shape[1]].iloc[: apparent.shape[0]]
    has_voltage = "v_cust_min_pu" in frame.columns
    fig, grid = plt.subplots(2, 2, figsize=S.panel(1.0, aspect=0.75))
    axes = [grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]]
    tiers = [(k, T.TEXT_DARK if k == 1.0 else T.DANGER) for k in limits.severities]

    # (a) When in the day: median and 10-90% band of apparent power by hour, against the tiers.
    ax = axes[0]
    lo, mid, hi = np.quantile(apparent, [0.1, 0.5, 0.9], axis=0)
    ax.fill_between(hours, lo, hi, color=T.PRIMARY_TINT, lw=0, label=f"10-90{pct()} of days")
    ax.plot(hours, mid, color=T.PRIMARY, lw=1.5, label="median day")
    for k, colour in tiers:
        ax.axhline(k * limits.rating, color=colour, lw=0.8, ls="--", label=rf"${k:g}\times$ headroom")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Apparent power (kVA)")
    top = max(float(hi.max()), max(k for k, _ in tiers) * limits.rating)
    ax.set_ylim(top=top * 1.45)
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)
    ax.set_xticks([0, 6, 12, 18, 24])

    # (b) Which days: loading relative to headroom, day by half-hour, diverging at 1.
    ax = axes[1]
    ratio = apparent / limits.rating
    image = ax.imshow(
        ratio.T,
        aspect="auto",
        cmap=LOADING_CMAP,
        norm=TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=max(2.0, float(ratio.max()))),
        origin="lower",
        extent=(0, ratio.shape[0], 0, 24),
    )
    month_starts = np.flatnonzero(days.dt.day.to_numpy() == 1)
    ax.set_xticks(month_starts + 0.5, [d.strftime("%b") for d in days.iloc[month_starts]])
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_ylabel("Hour of day")
    fig.colorbar(image, ax=ax, label="Loading / headroom", fraction=0.046, pad=0.04)

    # (c) Why it is joint: the overload tiers are circles in the P-Q plane.
    ax = axes[2]
    ax.scatter(frame["p_kw"], frame["q_kvar"], s=1, color=T.TEXT_MUTED, alpha=0.25, rasterized=True)
    angle = np.linspace(-np.pi, np.pi, 300)
    for k, colour in tiers:
        radius = k * limits.rating
        ax.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            color=colour,
            lw=0.9,
            ls="--",
            label=rf"${k:g}\times$ headroom",
        )
    ax.set_xlim(*S.pad_limits(frame["p_kw"].to_numpy(), np.array([-1.3, 1.3]) * limits.rating))
    ax.set_ylim(*S.pad_limits(frame["q_kvar"].to_numpy(), np.array([0.0])))
    ax.set_xlabel("Active power $P$ (kW)")
    ax.set_ylabel("Reactive power $Q$ (kvar)")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    # (d) Where compliance is judged: busbar versus lowest customer voltage, by hour;
    # without customer voltages, loading against the transformer's nameplate instead.
    ax = axes[3]
    if not has_voltage and "kva_rating" in frame.columns:
        nameplate = float(frame["kva_rating"].iloc[0])
        utilisation = apparent / nameplate
        u_lo, u_mid, u_hi = np.quantile(utilisation, [0.1, 0.5, 0.9], axis=0)
        ax.fill_between(hours, u_lo, u_hi, color=T.PRIMARY_TINT, lw=0)
        ax.plot(hours, u_mid, color=T.PRIMARY, lw=1.5, label="loading, median day")
        ax.plot(hours, utilisation.max(axis=0), color=T.PRIMARY, lw=0.8, ls=":", label="loading, highest at each hour")
        ax.axhline(1.0, color=T.TEXT_DARK, lw=0.8, label=f"nameplate ({nameplate:g} kVA)")
        ax.axhline(limits.rating / nameplate, color=T.TEXT_DARK, lw=0.8, ls="--", label=r"$1\times$ headroom")
        ax.set_xlabel("Hour of day")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylabel("Loading / nameplate")
        ax.set_ylim(0.0, 1.1)
        ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    if has_voltage:
        low, _ = limits.voltage_band
        for column, label, colour in (("v_lv_pu", "busbar", T.PRIMARY), ("v_cust_min_pu", "lowest customer", T.DANGER)):
            v = daily(frame[column].to_numpy())
            v_lo, v_mid = np.quantile(v, [0.02, 0.5], axis=0)
            ax.fill_between(hours, v_lo, v_mid, color=colour, alpha=0.18, lw=0)
            ax.plot(hours, v_mid, color=colour, lw=1.5, label=label)
            ax.plot(hours, v_lo, color=colour, lw=0.8, ls=":")
        ax.axhline(low, color=T.TEXT_DARK, lw=0.8, ls="--", label="statutory minimum")
        ax.set_xlabel("Hour of day")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylabel("Voltage (pu)")
        # Median (solid) and 2nd percentile (dotted) per series; the caption says so.
        ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    for ax in axes:
        S.despine(ax)
    fig.tight_layout()
    slug = f"{record_key}_{ev:g}_{pv:g}".replace(":", "_").replace(".", "p").lower()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"event_anatomy_{slug}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw the event-rate grid and the anatomy of the requested scenarios."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--anatomy",
        nargs="*",
        default=["DSS-UK:weak:0.4:0.4", "DSS-US:8:0.6:0"],
        help="scenarios to draw in detail",
    )
    ap.add_argument("--output", type=Path, default=sweep.DEFAULT_OUTPUT_ROOT / "event_figures")
    ap.add_argument("--no-latex", action="store_true")
    args = ap.parse_args()
    S.configure(latex=not args.no_latex)
    args.output.mkdir(parents=True, exist_ok=True)
    limits = {record: sweep.decision_limits(record) for record in sweep.RECORDS}
    plot_rates_grid(limits, args.output)
    for key in args.anatomy:
        record_key, ev, pv = key.rsplit(":", 2)
        plot_anatomy(record_key, float(ev), float(pv), limits[record_key], args.output)
    print(f"written under {args.output}")


if __name__ == "__main__":
    main()
