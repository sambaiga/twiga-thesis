"""Showing the difference a joint predictive distribution makes.

Marginal fan charts cannot display Chapter 6's claim. The joint and independent
heads have near-identical marginals by construction, so a panel of P against
time looks the same for both; the difference lives in the dependence *between*
targets, which a per-target plot has no axis for.

Three views that do show it:

1. **Bivariate sample clouds.** Samples drawn from both heads, plotted pairwise
   in the (P,Q), (P,V) and (Q,V) planes with the observation marked. The
   independent cloud is axis-aligned by construction; the joint cloud tilts
   along the physical relationship. Marginal histograms are drawn on the axes of
   each panel to make the point explicit: the two models agree on every margin
   and disagree only on the joint.

2. **Correlation matrices.** The empirical correlation of the targets, the
   correlation the joint head learns, and the identity the independent head
   implies. This is the mechanism behind view 1.

3. **Energy score.** The multivariate generalisation of CRPS, and the proper
   score that can actually separate these two models. Marginal CRPS cannot: it
   sees each target alone, so it is blind to the term under test.

Sampling is paired. Both heads consume the same standard normal draws, so the
clouds differ only through the Cholesky factor and not through sampling noise.

    uv run python experiment/ch7_multitarget/joint_vs_independent.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from twiga.core.metrics.parametric import energy_score  # noqa: E402
from twiga.core.utils import get_logger  # noqa: E402
from twiga.models.nn.mlpfmvnormal_model import MLPFMVNormalConfig, MLPFMVNormalModel  # noqa: E402
from twiga.models.nn.mlpfnormal_model import MLPFNormalConfig, MLPFNormalModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chapters.ch7_multitarget.shared.forecast_examples import _cfg, build_windows  # noqa: E402
from twiga_thesis.plotting.style import configure  # noqa: E402
from twiga_thesis.plotting.tokens import HIGHLIGHT, PALETTE, PRIMARY, TEXT_DARK as TEXT  # noqa: E402

# Diverging map built from the thesis palette rather than matplotlib's RdBu:
# teal for negative correlation, white at zero, orange for positive. These are
# PALETTE[0] and PALETTE[2], the two colours carried through every figure in the
# thesis, so the correlation plot reads as part of the same set.
CORR_CMAP = LinearSegmentedColormap.from_list("thesis_diverging", [PALETTE[0], "#ffffff", PALETTE[2]])

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
OUT = script_dir.parent / "joint_vs_independent"

JOINT_C = PRIMARY
INDEP_C = PALETTE[5]
OBS_C = HIGHLIGHT
SYM = {"p_kw": "$P$ (kW)", "q_kvar": "$Q$ (kvar)", "v_lv_pu": "$V$ (pu)"}


def fit(name, cls, cfg, d, root):
    model = cls(model_config=cfg)
    model.data_pipeline = None
    model.resume_training = False
    model.checkpoints_path = root / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["x_tr"], d["y_tr"], d["x_val"], d["y_val"])
    model.model.eval()
    with torch.no_grad():
        return model.model.forecast(torch.from_numpy(d["x_te"]))


def ellipse(ax, pts, colour, label):
    """Draw the 90% Gaussian-equivalent ellipse of a 2-D point cloud."""
    mu = pts.mean(0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0, 2 * np.pi, 200)
    # 90% of a bivariate normal lies inside sqrt(chi2_2(0.9)) = sqrt(4.605).
    circle = np.c_[np.cos(theta), np.sin(theta)] * np.sqrt(4.605 * vals)
    pts_e = circle @ vecs.T + mu
    ax.plot(pts_e[:, 0], pts_e[:, 1], color=colour, lw=1.6, label=label, zorder=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="DSS-EU:default:0.4:0.4")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--lead", type=int, default=24, help="lead-time step to show the clouds at")
    ap.add_argument("--eval-days", type=int, default=60, help="test days sampled for the energy score")
    ap.add_argument("--es-samples", type=int, default=200, help="ensemble size for the energy score; it is O(S^2)")
    ap.add_argument("--replot", action="store_true", help="redraw from cached forecasts")
    ap.add_argument("--no-latex", action="store_true", help="skip LaTeX typesetting")
    args = ap.parse_args()

    try:
        configure(latex=not args.no_latex)
    except Exception as exc:  # noqa: BLE001
        log.warning("thesis style unavailable (%s)", exc)
    OUT.mkdir(parents=True, exist_ok=True)

    d = build_windows(args.scenario, args.lookback, args.horizon)
    targets = d["targets"]
    mu_s, sd_s = d["mu"], d["sd"]
    truth = d["y_te"] * sd_s + mu_s

    cache = OUT / "heads.npz"
    if args.replot and cache.exists():
        b = np.load(cache)
        loc, scale, tril = b["loc"], b["scale"], b["tril"]
        loc_i, scale_i = b["loc_i"], b["scale_i"]
        log.info("loaded cached forecasts from %s", cache)
    else:
        root = Path(tempfile.mkdtemp(prefix="jvi_"))
        kw = {"epochs": args.epochs, "seed": args.seed, "horizon": args.horizon, "lookback": args.lookback}
        log.info("  training joint ...")
        oj = fit("joint", MLPFMVNormalModel, _cfg(MLPFMVNormalConfig, d, **kw), d, root)
        log.info("  training independent ...")
        oi = fit("independent", MLPFNormalModel, _cfg(MLPFNormalConfig, d, **kw), d, root)
        loc, scale, tril = (np.asarray(oj[k]) for k in ("loc", "scale", "scale_tril"))
        loc_i, scale_i = np.asarray(oi["loc"]), np.asarray(oi["scale"])
        np.savez_compressed(cache, loc=loc, scale=scale, tril=tril, loc_i=loc_i, scale_i=scale_i)
        log.info("cached forecasts -> %s", cache)

    n_days, horizon, n_t = loc.shape
    rng = np.random.default_rng(args.seed)
    log.info("test set: %d days x %d steps x %d targets", n_days, horizon, n_t)

    # Sample a subset of days rather than all of them. Drawing S samples for
    # every day and lead time allocates S x days x horizon x targets floats,
    # which at the defaults is about 15 GB and is killed before it finishes.
    # The energy score is an average over observations, so a random subset
    # estimates it to well within the spread between the two models.
    n_eval = min(args.eval_days, n_days)
    days = rng.choice(n_days, size=n_eval, replace=False)
    log.info("sampling %d draws on %d of %d test days", args.samples, n_eval, n_days)

    # Paired draws: both heads see the same standard normals, so the clouds
    # differ only through the Cholesky factor and not through sampling noise.
    z = rng.standard_normal((args.samples, n_eval, horizon, n_t))
    # Broadcast the Cholesky factor implicitly over the sample axis. Expanding
    # it with broadcast_to first materialises a (S, days, H, D, D) array, 1.7 GB
    # at the defaults, for no benefit.
    s_joint = loc[days][None] + np.einsum("htij,shtj->shti", tril[days], z)
    s_indep = loc_i[days][None] + z * scale_i[days][None]
    s_joint = s_joint * sd_s + mu_s
    s_indep = s_indep * sd_s + mu_s
    truth = truth[days]

    # ── energy score: the proper multivariate score ───────────────────────────
    # The energy score is O(S^2 * d) in the ensemble size: the second term is an
    # expectation over pairs of samples. Flattening every day and lead time into
    # one d-dimensional vector makes d = days * H * D and the pairwise term
    # 4000^2 x 17280, which is what killed the process. Score each (day, lead
    # time) triple separately with a smaller ensemble and average instead: that
    # is the multivariate score over the three coupled targets, which is the
    # quantity under test, and the cross-time dependence is not modelled by
    # either head anyway.
    n_es = min(args.es_samples, args.samples)
    print(f"\n=== energy score over (P, Q, V), {n_es} draws (lower is better) ===", flush=True)
    es = {}
    for name, s in (("joint", s_joint), ("independent", s_indep)):
        vals = [float(energy_score(truth[d_i, h], s[:n_es, d_i, h])) for d_i in range(n_eval) for h in range(horizon)]
        es[name] = float(np.mean(vals))
        print(f"  {name:12s} {es[name]:.5f}", flush=True)
    better = "joint" if es["joint"] < es["independent"] else "independent"
    delta = 100 * (es["independent"] - es["joint"]) / es["independent"]
    print(f"  -> {better} better; joint improves on independent by {delta:+.2f}%")

    # ── correlation matrices ──────────────────────────────────────────────────
    emp = np.corrcoef(truth.reshape(-1, n_t).T)
    learned = np.corrcoef(s_joint.reshape(args.samples, -1, n_t).reshape(-1, n_t).T)
    _plot_corr(emp, learned, np.eye(n_t), targets)

    # ── bivariate clouds ──────────────────────────────────────────────────────
    day = int(np.abs(truth[:, args.lead, :] - truth[:, args.lead, :].mean(0)).sum(1).argmin())
    _plot_clouds(s_joint[:, day, args.lead], s_indep[:, day, args.lead], truth[day, args.lead], targets, es, delta)
    log.info("figures %s", OUT)


def _plot_corr(emp, learned, ident, targets):
    """Observed, learned and implied correlation, on the thesis palette."""
    # constrained_layout, because a colorbar attached to a list of axes is not
    # something tight_layout can place: it warned and drew the bar over the
    # third matrix.
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), layout="constrained")
    labels = [SYM[t_].split()[0] for t_ in targets]
    for ax, m, t in zip(
        axes,
        (emp, learned, ident),
        ("observed", "joint head (learned)", "independent head (implied)"),
        strict=True,
    ):
        im = ax.imshow(m, vmin=-1, vmax=1, cmap=CORR_CMAP)
        ax.set_xticks(range(len(targets)), labels)
        ax.set_yticks(range(len(targets)), labels)
        for i in range(len(targets)):
            for j in range(len(targets)):
                # Dark text on the pale centre of the map, white out at the ends,
                # so every cell stays legible whatever its value.
                shade = "white" if abs(m[i, j]) > 0.55 else TEXT
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9, color=shade)
        ax.set_title(t, fontsize=10)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="correlation", ticks=[-1, -0.5, 0, 0.5, 1])
    fig.savefig(OUT / "correlation_matrices.pdf")
    fig.savefig(OUT / "correlation_matrices.png", dpi=150, facecolor="white")
    plt.close(fig)


def _plot_clouds(sj, si, obs, targets, es, delta):
    """Pairwise sample clouds with marginal histograms on each axis."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig = plt.figure(figsize=(11.5, 4.4), layout="constrained")
    # One gridspec for the whole figure. Building a fresh 2x3 grid inside the
    # loop stacked three overlapping layouts on top of each other, which is what
    # produced the tight_layout warnings and the titles landing on the marginal
    # histograms.
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 4))
    for k, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, k])
        axt = fig.add_subplot(gs[0, k], sharex=ax)

        ax.scatter(si[:, a], si[:, b], s=3, c=INDEP_C, alpha=0.18, lw=0, zorder=1)
        ax.scatter(sj[:, a], sj[:, b], s=3, c=JOINT_C, alpha=0.18, lw=0, zorder=2)
        ellipse(ax, si[:, [a, b]], INDEP_C, "independent 90%")
        ellipse(ax, sj[:, [a, b]], JOINT_C, "joint 90%")
        ax.scatter([obs[a]], [obs[b]], marker="*", s=180, c=OBS_C, ec="k", lw=0.4, zorder=6, label="observed")

        ax.set_xlabel(SYM[targets[a]])
        ax.set_ylabel(SYM[targets[b]])
        rj = np.corrcoef(sj[:, a], sj[:, b])[0, 1]
        ri = np.corrcoef(si[:, a], si[:, b])[0, 1]
        # Title on the marginal strip, so it cannot collide with the histograms.
        axt.set_title(f"joint $\\rho$ {rj:+.2f}   independent $\\rho$ {ri:+.2f}", fontsize=9)

        # The marginals are the point: they coincide, so no per-target plot can
        # tell these two models apart.
        bins = np.linspace(min(si[:, a].min(), sj[:, a].min()), max(si[:, a].max(), sj[:, a].max()), 40)
        axt.hist(si[:, a], bins=bins, color=INDEP_C, alpha=0.55, density=True, lw=0)
        axt.hist(sj[:, a], bins=bins, color=JOINT_C, alpha=0.55, density=True, lw=0)
        axt.axis("off")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if k == 0:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(
        f"Same marginals, different joint. Energy score: joint {es['joint']:.3f} vs "
        f"independent {es['independent']:.3f} ({delta:+.1f}\\%)",
        fontsize=11,
    )
    fig.savefig(OUT / "sample_clouds.pdf")
    fig.savefig(OUT / "sample_clouds.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
