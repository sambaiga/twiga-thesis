r"""Chapter 3: evidence figures for the rank-fusion consensus method.

Two figures back the prose added to Section 3.2 (Influence of Exogenous
Variables): the ACF-derived lookback/lag split that `_characterised_periods`
in `twiga/utils/data_loading.py` performs, and the per-metric ranker
disagreement on DSS-EU's reactive-power target that motivates rank fusion in
the first place -- the ANOVA F-score alone ranks Temperature above Ghi there,
disagreeing with every other constituent metric. Both figures are built from
the same real dataset files and the same
`twiga.core.data.selection.select_top_features` / `characterisation` code
path the thesis text names, not from illustrative or synthetic numbers.

Run:  uv run python -c "import sys; sys.path.insert(0,'experiment');
      from thesis_figures import ch3_features as F; F.render_all()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from twiga.core.data.characterisation import CharacterisationConfig, SignalCharacteriser
from twiga.core.data.selection import select_top_features
from twiga.core.stats.seasonality import get_acf_values
from twiga.utils.data_loading import _read_dataset_parquet, normalise_timestamp_column

from twiga_thesis.plotting import style as S, tokens as T

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"

METRIC_LABELS = {
    "abs_corr": "p-corr",
    "f_score": "F-score",
    "mi_score": "MI",
    "xi_score": r"$\xi$-corr",
    "pps_score": "pps",
    "rf_importance": "RF",
}


def _mlvs_pt_netload() -> tuple[np.ndarray, str]:
    df = _read_dataset_parquet(DATA / "MLVS-PT.parquet")
    df = normalise_timestamp_column(df, "timestamp")
    return df["NetLoad(kW)"].dropna().to_numpy(), "NetLoad(kW)"


def acf_lookback_split() -> plt.Figure:
    """ACF of MLVS-PT net load, marking the lookback pick and retained lag candidates.

    Reproduces exactly what `_characterised_periods` computes: the shortest
    detected seasonal peak becomes the lookback recommendation, held back
    from the longer peaks kept as explicit lag-feature candidates.
    """
    series, _ = _mlvs_pt_netload()
    n_samples = 48
    config = CharacterisationConfig(n_samples_per_day=n_samples)
    characteriser = SignalCharacteriser(config)
    arr = characteriser._validate_series(series, "NetLoad(kW)")  # noqa: SLF001
    temporal = characteriser._analyse_temporal_structure(arr)  # noqa: SLF001

    maxlag = 400
    acf_values, _ = get_acf_values(arr, maxlag=maxlag)
    lags = np.arange(1, maxlag + 1)

    seasonal = sorted(set(temporal.seasonal_periods))
    lookback = seasonal[0] if seasonal else None
    lag_candidates = list(seasonal[1:]) if len(seasonal) > 1 else []

    fig, ax = plt.subplots(figsize=S.panel(1.0, aspect=0.42))
    ax.stem(lags, acf_values, linefmt=T.TEXT_MUTED, markerfmt=" ", basefmt=" ")
    for lag in lag_candidates:
        if lag <= maxlag:
            ax.axvline(lag, color=T.DANGER, lw=1.1, ls="--", zorder=3)
    if lookback is not None and lookback <= maxlag:
        ax.axvline(lookback, color=T.PRIMARY, lw=1.4, zorder=3)

    ax.set_xlabel("Lag (samples, 48 per day)")
    ax.set_ylabel("ACF")
    ax.set_xlim(0, maxlag)
    handles = [
        plt.Line2D([0], [0], color=T.PRIMARY, lw=1.4, label=f"lookback = {lookback}"),
        plt.Line2D([0], [0], color=T.DANGER, lw=1.1, ls="--", label="retained lag candidates"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def dss_eu_metric_disagreement() -> plt.Figure:
    """Per-metric rank of Ghi vs Temperature for reactive power on DSS-EU.

    The ANOVA F-score alone ranks Temperature above Ghi; every other
    constituent metric ranks Ghi first. PPS is excluded from the plot
    (uninformative, zero for both candidates on this target), leaving the
    five metrics that carry a real ranking.
    """
    df = _read_dataset_parquet(DATA / "DSS-EU.parquet")
    df = df[(df.feeder == "default") & (df.ev_pen == 0.4) & (df.pv_pen == 0.4)]
    df = normalise_timestamp_column(df.reset_index(drop=True), "timestamp")
    target = "q_kvar"
    candidates = ["Ghi", "Temperature"]
    df = df[["timestamp", target] + candidates].dropna()

    _, scores = select_top_features(
        data=df, features=candidates, target=target, top_k=len(candidates), return_scores=True
    )
    scores = scores.set_index("feature")

    metrics = [m for m in METRIC_LABELS if m != "pps_score"]
    ghi_rank = [1 if scores.loc["Ghi", m] > scores.loc["Temperature", m] else 2 for m in metrics]
    temperature_rank = [1 if scores.loc["Temperature", m] > scores.loc["Ghi", m] else 2 for m in metrics]

    fig, ax = plt.subplots(figsize=S.panel(0.75, aspect=0.62))
    x = np.arange(len(metrics))
    ax.plot(x, ghi_rank, "o-", color=T.PRIMARY, ms=6, lw=1.3, label="Ghi")
    ax.plot(x, temperature_rank, "s-", color=T.DANGER, ms=6, lw=1.3, label="Temperature")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["ranks 1st", "ranks 2nd"])
    ax.set_ylim(2.4, 0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig


FIGURES = {
    "fig_ch3_acf_lookback_split.pdf": acf_lookback_split,
    "fig_ch3_dss_eu_metric_disagreement.pdf": dss_eu_metric_disagreement,
}


def render_all(out: Path | None = None, latex: bool = True) -> None:
    """Render both figures to `out` and to the Chapter 3 image folder."""
    S.configure(latex=latex)
    here = Path(__file__).resolve().parent
    out = out or here / "output"
    out.mkdir(exist_ok=True)
    thesis = here.parents[1] / "Papers" / "PhD thesis -Final-Revision" / "Images" / "Chapter-3"
    for name, fn in FIGURES.items():
        fig = fn()
        fig.savefig(out / name)
        if thesis.exists():
            fig.savefig(thesis / name)
        plt.close(fig)
        print(f"  {name}")


if __name__ == "__main__":
    render_all()
