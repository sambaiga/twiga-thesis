"""Quick bake-off across the multi-target heads Chapter 6 proposes.

Six variants on one scenario, one split, one seed. This is a scoping run, not
the case study: it establishes which comparisons separate and which do not, so
the backtested experiment can be designed around the ones that do.

    independent-normal   marginal Gaussian, the Chapter 6 baseline
    independent-laplace  marginal Laplace, the head Section 6.2 selects
    joint-normal         MultivariateNormalDistribution
    joint-laplace        MultivariateLaplaceDistribution, Section 6.4
    fpqr-shared          one learned quantile grid across targets
    fpqr-per-target      a grid conditioned on the target, Section 6.5

The quantile heads are additionally coupled post hoc with the Gaussian copula of
Section 6.5, fitted on validation PIT values, which is what lets them be scored
against the joint criterion at all.

Metrics are the three tiers the chapter argues for: marginal (CRPS), joint
(energy and variogram scores), and operational (constraint exceedance against
realised frequency). The variogram score is included because the energy score is
known to have limited power against misspecified dependence.

This script scores; it does not draw. Fan charts on the best and worst day come
from forecast_examples.py, which ranks days on error pooled over every model so
that all variants are judged on the same day.

    uv run python experiment/ch7_multitarget/multitarget_bakeoff.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
import torch
from twiga.core.metrics.parametric import energy_score, variogram_score
from twiga.core.utils import get_logger, release_torch_memory

log = get_logger(__name__)
from twiga.distributions.nn.copula_sampling import (
    coupled_samples,
    fit_gaussian_copula,
)
from twiga.models.nn.mlpffpqr_model import MLPFFPQRConfig, MLPFFPQRModel
from twiga.models.nn.mlpflaplace_model import MLPFLaplaceConfig, MLPFLaplaceModel
from twiga.models.nn.mlpfmvlaplace_model import MLPFMVLaplaceConfig, MLPFMVLaplaceModel
from twiga.models.nn.mlpfmvnormal_model import MLPFMVNormalConfig, MLPFMVNormalModel
from twiga.models.nn.mlpfnormal_model import MLPFNormalConfig, MLPFNormalModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from forecast_examples import _cfg, build_windows, sample_parametric  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT = script_dir.parent / "bakeoff"


def family_of(name: str) -> str:
    """Marginal family and scale convention for a variant, keyed by its name."""
    if "laplace" not in name:
        return "normal"
    return "laplace-sd" if name.startswith("joint") else "laplace-dispersion"


VARIANTS = {
    "independent-normal": (MLPFNormalModel, MLPFNormalConfig, {}),
    "independent-laplace": (MLPFLaplaceModel, MLPFLaplaceConfig, {}),
    "joint-normal": (MLPFMVNormalModel, MLPFMVNormalConfig, {}),
    "joint-laplace": (MLPFMVLaplaceModel, MLPFMVLaplaceConfig, {}),
    "fpqr-shared": (MLPFFPQRModel, MLPFFPQRConfig, {"per_target_tau": False}),
    "fpqr-per-target": (MLPFFPQRModel, MLPFFPQRConfig, {"per_target_tau": True}),
}


def fit_forecast(name, cls, cfg_cls, extra, d, root, kw):
    """Train one variant and return its forecast dict on the test split."""
    model = cls(model_config=_cfg(cfg_cls, d, **kw, **extra))
    model.data_pipeline = None
    model.resume_training = False
    model.checkpoints_path = root / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(d["x_tr"], d["y_tr"], d["x_val"], d["y_val"])
    model.model.eval()
    with torch.no_grad():
        out = model.model.forecast(torch.from_numpy(d["x_te"]))
        val = model.model.forecast(torch.from_numpy(d["x_val"]))
    return {k: np.asarray(v) for k, v in out.items() if isinstance(v, torch.Tensor | np.ndarray)}, {
        k: np.asarray(v) for k, v in val.items() if isinstance(v, torch.Tensor | np.ndarray)
    }


def crps_ensemble(obs: np.ndarray, ens: np.ndarray) -> float:
    """CRPS estimated from an ensemble, per target, averaged."""
    term1 = np.abs(ens - obs[None]).mean(0)
    term2 = np.abs(ens[:, None] - ens[None, :]).mean((0, 1)) / 2.0
    return float(np.mean(term1 - term2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="DSS-EU:default:0.4:0.4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--eval-days", type=int, default=40)
    ap.add_argument("--rated-q", type=float, default=0.98, help="rating as a quantile of recorded S")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = build_windows(args.scenario, args.lookback, args.horizon)
    mu_s, sd_s = d["mu"], d["sd"]
    kw = {"epochs": args.epochs, "seed": args.seed, "horizon": args.horizon, "lookback": args.lookback}
    rng = np.random.default_rng(args.seed)
    root = Path(tempfile.mkdtemp(prefix="bakeoff_"))

    forecasts = {}
    for name, (cls, cfg_cls, extra) in VARIANTS.items():
        log.info("  training %s ...", name)
        forecasts[name] = fit_forecast(name, cls, cfg_cls, extra, d, root, kw)
        # Each variant builds and trains a fresh Lightning model/trainer; without
        # this, CUDA-cached memory and unreferenced objects accumulate across
        # all six variants in this loop.
        release_torch_memory()

    truth = d["y_te"] * sd_s + mu_s
    y_val = d["y_val"] * sd_s + mu_s
    n_days = truth.shape[0]
    days = rng.choice(n_days, size=min(args.eval_days, n_days), replace=False)
    # The rating is set from the record rather than assumed, so the exceedance
    # tier is a real screening level for this feeder.
    s_all = np.hypot(truth[..., 0], truth[..., 1])
    rated = float(np.quantile(s_all, args.rated_q))

    rows = {}
    for name in VARIANTS:
        out, val = forecasts[name]
        if "quantiles" in out:
            # Fitted on the validation split, so the correlation never sees the
            # rows the coupled model is scored on.
            corr = fit_gaussian_copula((y_val - mu_s) / sd_s, val["quantiles"], val["quantile_levels"])
            variants = {name: None, f"{name}+copula": corr}
        else:
            joint = "scale_tril" in out
            variants = {name: joint}
        for label, spec in variants.items():
            if "quantiles" in out:
                s = coupled_samples(
                    out["quantiles"], out["quantile_levels"], args.samples, spec, random_state=args.seed
                )
            else:
                # The sampler is shared with forecast_examples so that the scores
                # and the panels describe the same ensemble. It also carries the
                # dispersion conversion: the marginal Laplace head reports the
                # Laplace b while the joint heads report a standard deviation, and
                # sampling both at face value drew the marginal ensemble a factor
                # of sqrt(2) too narrow, which inflated its CRPS.
                marginal = {k: v for k, v in out.items() if k != "scale_tril"}
                s = sample_parametric(out if spec else marginal, family_of(name), args.samples, rng)
            # Both branches predict in standardised units; scoring happens in
            # physical ones, so the rescaling belongs to both equally.
            s = s * sd_s + mu_s
            s = s[:, days]
            t = truth[days]
            crps = np.mean([crps_ensemble(t[i, h], s[:, i, h]) for i in range(len(days)) for h in range(t.shape[1])])
            es = np.mean(
                [float(energy_score(t[i, h], s[:, i, h])) for i in range(len(days)) for h in range(t.shape[1])]
            )
            vs = np.mean([variogram_score(t[i, h], s[:, i, h]) for i in range(len(days)) for h in range(t.shape[1])])
            s_hat = np.hypot(s[..., 0], s[..., 1])
            pred_rate = float((s_hat > rated).mean())
            obs_rate = float((np.hypot(t[..., 0], t[..., 1]) > rated).mean())
            rows[label] = {
                "crps": float(crps),
                "energy": float(es),
                "variogram": float(vs),
                "exceed_pred": pred_rate,
                "exceed_obs": obs_rate,
                "exceed_err_pct": float(100 * (pred_rate - obs_rate) / obs_rate) if obs_rate else float("nan"),
            }
            print(
                f"  {label:26s} CRPS {crps:.4f}  ES {es:.4f}  VS {vs:.5f}  "
                f"exceed {100 * pred_rate:.2f}% vs {100 * obs_rate:.2f}%",
                flush=True,
            )

    (OUT / "bakeoff.json").write_text(json.dumps({"rated_kva": rated, "results": rows}, indent=2) + "\n")
    log.info("wrote %s", OUT / "bakeoff.json")
    print("forecast panels: run forecast_examples.py, which selects the day on")
    print("error pooled over models rather than on one variant's own error")


if __name__ == "__main__":
    main()
