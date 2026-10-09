"""Does the lead-time-only adaptivity of the learned grid survive perturbation?

The single-run diagnostic in `learned_tau.py` found that MLPQF's proposal grid
varies mainly with lead time and only weakly with the day being forecast. Three
explanations were consistent with that, and this script separates them:

    seeds       is the finding stable at all, or an artefact of one run?
    n_quantiles is N = 9 too coarse for instance-level placement to pay?
    entropy     is the entropy regulariser suppressing adaptation?
    proposal    is a single linear map too weak to express a conditional grid?

Each run appends a row to `s564_tau_matrix.csv`, so partial results are usable
if the sweep is interrupted.

Run:  uv run python experiment/chapter-5-fpqr/tau_matrix.py
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chapters.ch5_fpqr.learned_tau import SPLITS, build, capture, find_head  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT = script_dir / "s564_tau_matrix.csv"


def conditions() -> list[dict]:
    """Baseline at three seeds, then one perturbation at a time."""
    c = [{"arm": "baseline", "seed": s} for s in (0, 1, 2)]
    c += [{"arm": f"N={n}", "seed": 0, "n_quantiles": n} for n in (5, 21)]
    c += [{"arm": "no entropy", "seed": 0, "entropy_weight": 0.0}]
    c += [{"arm": "horizon proposal", "seed": 0, "proposal_type": "horizon"}]
    return c


def _detrend(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.vstack([np.ones_like(x), x]).T
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def diagnostics(tau: np.ndarray, ghi_mean: np.ndarray) -> dict:
    """The three numbers Addition 9 rests on, plus grid entropy."""
    n_windows, n_slots, horizon = tau.shape
    t = np.arange(n_windows)
    sd_steps = float(tau.std(axis=2).mean())
    sd_windows = float(tau.std(axis=0).mean())
    prof = tau.mean(axis=0)
    mid = tau[:, :, 20:32].mean(axis=2)
    raw = max(abs(float(np.corrcoef(mid[:, k], ghi_mean)[0, 1])) for k in range(n_slots))
    par = max(abs(float(np.corrcoef(_detrend(mid[:, k], t), _detrend(ghi_mean, t))[0, 1])) for k in range(n_slots))
    edges = np.concatenate([np.zeros((n_windows, 1, horizon)), tau, np.ones((n_windows, 1, horizon))], axis=1)
    w = np.diff(edges, axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    ent = float((-(w * np.log(w + 1e-12)).sum(axis=1)).mean())
    return {
        "n_windows": n_windows,
        "n_slots": n_slots,
        "across_day_range": float((prof.max(axis=1) - prof.min(axis=1)).mean()),
        "sd_steps": sd_steps,
        "sd_windows": sd_windows,
        "dispersion_ratio": sd_steps / sd_windows,
        "raw_corr_ghi": raw,
        "detrended_corr_ghi": par,
        "grid_entropy": ent,
        "max_entropy": float(np.log(n_slots + 1)),
    }


if __name__ == "__main__":
    rows: list[dict] = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    done = {(r["dataset"], r["arm"], r["seed"]) for r in rows}

    for cond in conditions():
        for ds, (a, b, c, d) in SPLITS.items():
            key = (ds, cond["arm"], cond["seed"])
            if key in done:
                print(f"skip {key}", flush=True)
                continue
            print(f"\n===== {ds} | {cond['arm']} | seed {cond['seed']} =====", flush=True)
            torch.manual_seed(cond["seed"])
            kw = {k: v for k, v in cond.items() if k not in ("arm", "seed")}
            cfg, forecaster, train_df, test_df = build(ds, a, b, c, d, **kw)
            forecaster.fit(train_df)
            obj = forecaster.fitted_models if hasattr(forecaster, "fitted_models") else forecaster
            head = find_head(obj)
            tau = capture(head, forecaster, test_df)

            raw = pd.read_parquet(Path(cfg.data_path) / f"{cfg.dataset_name}.parquet")
            ghi = raw["Ghi"].to_numpy()
            first = int(test_df.index[0]) + cfg.input_window_size
            H = tau.shape[2]
            gm = np.array(
                [
                    (lambda seg: float(seg[seg > 1e-6].mean()) if (seg > 1e-6).any() else 0.0)(
                        ghi[first + w * H : first + w * H + H]
                    )
                    for w in range(tau.shape[0])
                ]
            )
            row = {"dataset": ds, **cond, **diagnostics(tau, gm)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                    if k not in ("n_windows", "max_entropy")
                },
                flush=True,
            )
    print(f"\nwrote {OUT}")
