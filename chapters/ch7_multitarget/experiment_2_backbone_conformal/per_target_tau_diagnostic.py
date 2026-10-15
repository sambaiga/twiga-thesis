"""Does FPQR place its quantile grid where the theory says it should?

Chapter 5 motivates a learned grid with a quadrature argument: expanding the
quantile function about each interval midpoint and minimising the total
approximation error subject to the widths summing to one gives

    dtau*_n  proportional to  |Q''(tau_n)|^(-1/3)

Chapter 5 then reports an honest negative: the learned grid adapts to lead time
but not to the day being forecast. That was tested on an axis where the physics
gives no strong prior that grids *should* differ. Across targets the prior is
strong and measurable. At a low-voltage substation the three targets have
visibly different distributional shapes, so the equation above predicts three
different optimal grids.

This module therefore does not ask the descriptive question "do the learned
grids differ", which any two randomly initialised heads would answer yes to. It
asks the directional one: **does the learned grid track the grid the theory
prescribes, and does a per-target grid track it better than one shared grid?**
That has a clean null, and the null is informative.

Design notes, following the Chapter 5 diagnostic protocol:

* **Seeds.** Every configuration is repeated, and dispersion across seeds is
  reported alongside the effect. A single run cannot separate signal from
  initialisation, and the fold-level spread on this data is large.
* **Collapse is checked before the result is read.** ``tau_hats`` is clamped to
  [0.01, 0.99], and a slot pinned at a bound carries no information about where
  the model wants its levels. The fraction of pinned levels is reported first
  and gates the correlations: above a threshold the run is declared
  uninterpretable rather than reported, because a collapsed proposal can produce
  any correlation at all for reasons that have nothing to do with adaptivity.
* **The grid is averaged over instances and lead times** before it is compared.
  The theory grid is derived from the unconditional marginal of each target, so
  the like-for-like learned quantity is the unconditional grid. Whether the grid
  also adapts *within* those axes is the separate Chapter 5 question and is not
  what this module tests.
* **A shared-grid control** is trained under identical conditions, seed for
  seed, so the comparison isolates the grid rather than the head or the data.
  The shared model has one grid, which is scored against all three targets: that
  is precisely the constraint per-target grids are meant to relax.

A note on what the rho test can carry. It is computed over ``n_quantiles``
points, so the number of slots *is* its sample size. At the default of 9 the
test has roughly 31% power against the largest effect seen so far, which makes a
null there uninformative rather than negative; about 26 slots are needed for 80%
power. Slots, not seeds, are the lever: repeated seeds produce near-identical
grids and therefore identical rank correlations. The widths are also a cumsum of
a softmax, so neighbouring slots are correlated and the nominal p-values are
anti-conservative. The paired comparison between the two arms is the more robust
claim, since it needs no assumption about the slots being independent.

    uv run python experiment/ch7_multitarget/per_target_tau_diagnostic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist
import torch
from twiga.core.utils import get_logger, release_torch_memory
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig, MLPFFPQRModel
from twiga.utils.data_loading import load_dataset

from twiga_thesis.paths import thesis_data_dir

log = get_logger(__name__)
script_dir = Path(__file__).resolve().parent
TAU_CLAMP = (0.01, 0.99)  # matches the proposal's clamp
CLAMP_TOL = 1e-4


def build_windows(scenario: str, lookback: int, horizon: int):
    """Supervised windows plus the standardised targets used for the theory grid."""
    data, targets, _cal, exog, _lags, _lat, _lon, _lookback = load_dataset(scenario, data_path=str(thesis_data_dir()))
    y = data[targets].to_numpy(dtype=np.float64)
    x_exog = data[exog].to_numpy(dtype=np.float64)

    ts = pd.to_datetime(data["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    cal = np.c_[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ]

    # Standardise on training rows only. Using whole-series statistics would leak
    # test-period mean and variance into the scaler, and the theory grid below is
    # derived from these same standardised targets, so the leak would reach the
    # reference the learned grid is scored against.
    n_windows = len(data) - lookback - horizon
    fit_rows = int(0.8 * n_windows) + lookback
    yz = (y - y[:fit_rows].mean(0)) / y[:fit_rows].std(0)
    ez = (x_exog - x_exog[:fit_rows].mean(0)) / x_exog[:fit_rows].std(0)

    past = np.c_[yz, ez, cal]
    future = np.c_[ez, cal]
    n = n_windows
    idx = np.arange(n)
    xp = np.stack([past[i : i + lookback] for i in idx])
    xf = np.stack([future[i + lookback : i + lookback + horizon] for i in idx])
    yy = np.stack([yz[i + lookback : i + lookback + horizon] for i in idx])
    pad = np.zeros((n, horizon, xp.shape[2] - xf.shape[2]))
    x = np.concatenate([xp, np.concatenate([pad, xf], axis=2)], axis=1).astype(np.float32)

    cut = int(0.8 * n)
    # The theory grid uses training rows only, for the same reason as the scaler.
    return x[:cut], yy[:cut].astype(np.float32), x[cut:], yy[cut:].astype(np.float32), yz[:fit_rows], targets


def theory_widths(y_train: np.ndarray, n_slots: int) -> np.ndarray:
    """Interval widths prescribed by the Chapter 5 quadrature result.

    Estimates the empirical quantile function per target, differentiates it
    twice, and applies ``dtau* ~ |Q''|^(-1/3)`` normalised to sum to one.

    Args:
        y_train: Standardised training targets, ``(T, D)``.
        n_slots: Number of quantile slots ``N``.

    Returns:
        Optimal widths of shape ``(N, D)``.
    """
    edges = np.linspace(0.0, 1.0, n_slots + 1)
    mid = (edges[:-1] + edges[1:]) / 2.0
    fine = np.linspace(0.02, 0.98, 400)
    out = np.zeros((n_slots, y_train.shape[1]))
    for d in range(y_train.shape[1]):
        q = np.quantile(y_train[:, d], fine)
        d2 = np.gradient(np.gradient(q, fine), fine)
        w = np.abs(d2) ** (-1.0 / 3.0)
        w = np.interp(mid, fine, w)
        out[:, d] = w / w.sum()
    return out


def train(
    x_tr,
    y_tr,
    x_te,
    y_te,
    *,
    per_target: bool,
    seed: int,
    epochs: int,
    lookback: int,
    horizon: int,
    ckpt_root: Path,
    entropy_weight: float = 1.0,
    n_quantiles: int = 9,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = MLPFFPQRConfig(
        num_target_feature=y_tr.shape[2],
        num_past_features=x_tr.shape[2] - y_tr.shape[2],
        num_calendar_features=0,
        num_known_future_features=0,
        num_forecast_period_features=0,
        forecast_horizon=horizon,
        lookback_window_size=lookback,
        latent_size=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        max_epochs=epochs,
        batch_size=64,
        seed=seed,
        per_target_tau=per_target,
        entropy_weight=entropy_weight,
        n_quantiles=n_quantiles,
        num_cosines=0,
        rope_type="tau_pos",
        qvn_type="monotone",
    )
    model = MLPFFPQRModel(model_config=cfg)
    model.data_pipeline = None  # no scaling: windows are pre-standardised above
    # Each run gets its own checkpoint directory and never resumes. Sharing a
    # directory would let one configuration restore another's weights, which for
    # an ablation is the difference between a comparison and a confound.
    model.resume_training = False
    model.checkpoints_path = ckpt_root / f"{'per_target' if per_target else 'shared'}_seed{seed}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_tr, y_tr, x_te, y_te)
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.from_numpy(x_te))
    return model, np.asarray(out["quantile_levels"])


def _spearman_p(rho: float, n: int) -> float:
    """Two-tailed p-value for a Spearman correlation, via the t approximation."""
    r = float(np.clip(rho, -0.999999, 0.999999))
    return float(2 * t_dist.sf(abs(r * np.sqrt((n - 2) / (1 - r**2))), n - 2))


def clamp_fraction(tau: np.ndarray) -> float:
    """Fraction of levels sitting on a clamp bound, i.e. evidence of collapse."""
    lo, hi = TAU_CLAMP
    return float(((tau <= lo + CLAMP_TOL) | (tau >= hi - CLAMP_TOL)).mean())


def learned_widths(tau: np.ndarray) -> np.ndarray:
    """Interval widths implied by the learned levels, ``(N, D)``.

    ``tau`` is ``(B, N, H)`` for a shared grid or ``(B, N, H, D)`` per target.
    Widths are differenced along the slot axis and averaged over instances and
    lead times, which is the grid the theory comparison is about.
    """
    if tau.ndim == 3:
        tau = tau[..., None]
    grid = tau.mean(axis=(0, 2))  # (N, D)
    w = np.diff(grid, axis=0, prepend=0.0)
    return w / w.sum(axis=0, keepdims=True)


def run(args) -> pd.DataFrame:
    ckpt_root = Path(tempfile.mkdtemp(prefix="fpqr_diag_"))
    x_tr, y_tr, x_te, y_te, y_raw, targets = build_windows(args.scenario, args.lookback, args.horizon)
    log.info("scenario %s | train %s test %s | targets %s", args.scenario, x_tr.shape, x_te.shape, targets)

    rows = []
    for entropy_weight in args.entropy_weight:
        for seed_i in range(args.seeds):
            seed = args.seed0 + seed_i
            for per_target in (False, True):
                model, tau = train(
                    x_tr,
                    y_tr,
                    x_te,
                    y_te,
                    per_target=per_target,
                    seed=seed,
                    epochs=args.epochs,
                    lookback=args.lookback,
                    horizon=args.horizon,
                    ckpt_root=ckpt_root / f"ew{entropy_weight}_n{args.n_quantiles}",
                    entropy_weight=entropy_weight,
                    n_quantiles=args.n_quantiles,
                )
                n_slots = tau.shape[1]
                theory = theory_widths(y_raw, n_slots)
                learned = learned_widths(tau)
                collapsed = clamp_fraction(tau)

                offsets = None
                if per_target:
                    offsets = model.model.model.head.quantile_proposal.target_ctx.offset.weight.detach().numpy()

                for d, name in enumerate(targets):
                    lw = learned[:, 0] if learned.shape[1] == 1 else learned[:, d]
                    rows.append(
                        {
                            "entropy_weight": entropy_weight,
                            "seed": seed,
                            "grid": "per_target" if per_target else "shared",
                            "target": name,
                            "rho_vs_theory": spearmanr(lw, theory[:, d]).statistic,
                            # Distance from the uniform grid separates the two
                            # failure modes. Near zero means the proposal has
                            # learned nothing a fixed grid does not already give,
                            # which is what a large entropy weight forces; a large
                            # value with a high clamp count means it has collapsed
                            # instead. FPQR earns its place only between the two,
                            # and only with rho positive there.
                            "tv_from_uniform": float(np.abs(lw - 1.0 / len(lw)).sum() / 2.0),
                            "clamped_frac": collapsed,
                            "offset_max": float(np.abs(offsets).max()) if offsets is not None else 0.0,
                        }
                    )

                # The grid itself is printed, not just the clamp fraction. Levels
                # concentrating in BOTH tails is what the |Q''|^(-1/3) rule
                # predicts for a heavy-tailed target and is a correct result;
                # levels crushed against ONE bound is degenerate. A clamp count
                # alone cannot tell the two apart, so the grid has to be visible.
                # tau.mean over batch and horizon leaves (N,) for a shared grid
                # and (N, D) per target. Flattening the latter interleaves slots
                # and targets into something that is not a grid at all, so each
                # target's grid is printed on its own line.
                mean_grid = tau.mean(axis=(0, 2))
                arm = "per_target" if per_target else "shared    "
                head = f"  ew={entropy_weight:<5g} seed {seed} {arm} clamped={collapsed:5.1%}"
                if mean_grid.ndim == 1:
                    print(f"{head}  grid[{' '.join(f'{v:.3f}' for v in mean_grid)}]", flush=True)
                else:
                    print(head, flush=True)
                    for d, tname in enumerate(targets):
                        print(f"      {tname:>8}: [{' '.join(f'{v:.3f}' for v in mean_grid[:, d])}]", flush=True)

                # Each (entropy_weight, seed, per_target) combination builds and
                # trains a fresh Lightning model/trainer; without this, CUDA-cached
                # memory and unreferenced objects accumulate across the whole sweep.
                release_torch_memory()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="DSS-EU:default:0.4:0.4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=20260904)
    ap.add_argument(
        "--entropy-weight",
        type=float,
        nargs="+",
        default=[1.0],
        help="proposal entropy regulariser weight(s); pass several to sweep",
    )
    ap.add_argument(
        "--n-quantiles",
        type=int,
        default=9,
        help=(
            "Number of quantile slots. This also sets the sample size of the rho "
            "test: at 9 slots the test has only 31% power against the largest "
            "effect observed so far, so a null there is uninformative. 80% power "
            "needs about 26 slots."
        ),
    )
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=48)
    args = ap.parse_args()

    df = run(args)
    df.attrs["n_slots"] = args.n_quantiles
    df.to_csv(script_dir / "per_target_tau_diagnostic.csv", index=False)

    print("\n── collapse check ──────────────────────────────────────────────")
    coll = df.groupby(["entropy_weight", "grid"]).clamped_frac.agg(["mean", "max"])
    print(coll.round(4).to_string())
    # Gate per entropy weight, not globally: a setting that collapses says
    # nothing about a setting that does not, and one collapsed weight in a sweep
    # must not invalidate the rest of the table.
    collapsed_at = sorted({w for (w, _), v in coll["mean"].items() if v > 0.2})
    clean_at = sorted({w for (w, _), v in coll["mean"].items() if v <= 0.05})
    if collapsed_at:
        print(f"\nCOLLAPSED at entropy_weight={collapsed_at}: a large share of levels")
        print("sit on the clamp bounds, so no conclusion about adaptivity is")
        print("supported at those weights, whatever their correlations say.")
    if clean_at:
        print(f"Interpretable at entropy_weight={clean_at}.")

    print("\n── is the grid doing anything a fixed grid would not? ──────────")
    print(df.groupby(["entropy_weight", "grid"]).tv_from_uniform.agg(["mean", "std"]).round(4).to_string())
    print("Total-variation distance from the uniform grid. Near zero means the")
    print("proposal reduces to fixed-grid quantile regression.")

    print("\n── does the learned grid track the theory-optimal grid? ────────")
    piv = df.pivot_table(
        index=["entropy_weight", "target"], columns="grid", values="rho_vs_theory", aggfunc=["mean", "std"]
    )
    print(piv.round(3).to_string())
    print("\nSpearman rho between learned and theory-optimal interval widths.")
    print("The test is whether per_target exceeds shared: that is the claim")
    print("per-target grids exist to make. Equal values are a clean null.")

    # rho is computed over n_slots points, which is a small sample. Reporting it
    # without the significance threshold invites reading noise as a trend, so
    # the threshold is printed next to the numbers rather than left implicit.
    n_slots = int(df.attrs.get("n_slots", 9))
    print(f"(rho computed over {n_slots} slots)")
    crit = next(r for r in np.linspace(0.0, 1.0, 10001) if _spearman_p(r, n_slots) < 0.05)
    print(f"\nWith n={n_slots} slots, |rho| must exceed {crit:.3f} to reach p<0.05.")
    print("Treat that threshold as optimistic. The widths are a cumsum of a")
    print("softmax, so neighbouring slots are correlated by construction and the")
    print("effective sample size is below the nominal one, which makes the")
    print("p-values anti-conservative. Read rho as an effect size; the claim")
    print("that carries the least statistical baggage is the paired one, whether")
    print("per_target and shared differ at all, since both arms see identical")
    print("data and differ in exactly one thing.")
    largest = df.rho_vs_theory.abs().max()
    if largest < crit:
        print(f"The largest |rho| observed is {largest:.3f}, so no correlation here")
        print("is distinguishable from chance and none should be read as a trend.")
    print(f"\nwritten to {script_dir / 'per_target_tau_diagnostic.csv'}")


if __name__ == "__main__":
    main()
