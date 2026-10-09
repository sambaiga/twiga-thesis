"""Measure how much cross-target dependence survives conditioning on the backbone.

Chapter 6 originally contrasted an "independent" model against a "joint" one.
That contrast does not hold at the architecture level: every MLPF head sits on
one backbone trained on all three targets at once, so the conditional means are
already coupled through the shared representation. What separates the variants
is the likelihood, not whether they are multivariate.

This script measures the quantity that decides whether the likelihood still has
work left to do. If conditioning on the backbone absorbed the dependence, the
residuals would be close to uncorrelated and a joint density would be modelling
noise. If it does not, the residual dependence is exactly what a joint
likelihood has to carry, and the chapter's methods sections are motivated.

Three quantities are reported, all on the held-out split:

    raw         correlation between the targets themselves
    residual    correlation between the targets and their predicted means
                removed, so the backbone has been conditioned on
    asserted    the correlation the joint Gaussian head emits from its own
                Cholesky factor, which is what the likelihood claims

The comparison that matters is asserted against residual: the head is right
about a pair when the two agree. Per-day spread is reported alongside, because
a pair whose sign changes day to day is not usefully summarised by its mean.

    uv run python experiment/ch7_multitarget/residual_dependence.py
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
from twiga.core.utils import get_logger
from twiga.models.nn.mlpfmvnormal_model import MLPFMVNormalConfig, MLPFMVNormalModel
from twiga.models.nn.mlpfnormal_model import MLPFNormalConfig, MLPFNormalModel

log = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from chapters.ch7_multitarget.shared.forecast_examples import _cfg, build_windows  # noqa: E402

script_dir = Path(__file__).resolve().parent
OUT = script_dir / "residual_dependence.json"

# Fixed pair order so the JSON keys, the CSV rows and the chapter table agree.
PAIRS: tuple[tuple[str, int, int], ...] = (("pq", 0, 1), ("pv", 0, 2), ("qv", 1, 2))


def pair_correlations(values: np.ndarray) -> dict[str, float]:
    """Correlation of each target pair, pooled over windows and lead times.

    Args:
        values: array of shape ``(windows, horizon, targets)``.

    Returns:
        Mapping from pair name to Pearson correlation.
    """
    flat = values.reshape(-1, values.shape[-1])
    matrix = np.corrcoef(flat, rowvar=False)
    return {name: float(matrix[i, j]) for name, i, j in PAIRS}


def daily_correlations(values: np.ndarray) -> dict[str, np.ndarray]:
    """Correlation of each target pair within each forecast day.

    A pair whose correlation is stable across days can be summarised by one
    number; a pair that changes sign cannot, and the spread is the evidence for
    saying so.

    Args:
        values: array of shape ``(windows, horizon, targets)``.

    Returns:
        Mapping from pair name to an array of per-window correlations, with
        windows too degenerate to correlate dropped.
    """
    out: dict[str, list[float]] = {name: [] for name, _, _ in PAIRS}
    for window in values:
        for name, i, j in PAIRS:
            a, b = window[:, i], window[:, j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            out[name].append(float(np.corrcoef(a, b)[0, 1]))
    return {name: np.asarray(vals) for name, vals in out.items()}


def asserted_correlations(scale_tril: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Correlations implied by the joint head's Cholesky factor.

    Args:
        scale_tril: array of shape ``(windows, horizon, targets, targets)``.

    Returns:
        Tuple of mean correlation per pair, and per-window mean correlation per
        pair, both derived from ``L L^T``.
    """
    cov = np.einsum("whij,whkj->whik", scale_tril, scale_tril)
    sd = np.sqrt(np.einsum("whii->whi", cov))
    corr = cov / (sd[..., :, None] * sd[..., None, :])
    mean = {name: float(corr[..., i, j].mean()) for name, i, j in PAIRS}
    per_day = {name: corr[..., i, j].mean(axis=1) for name, i, j in PAIRS}
    return mean, per_day


def fit_head(name, model_cls, config_cls, windows, root, settings):
    """Train one head and return its forecast on the held-out split."""
    model = model_cls(model_config=_cfg(config_cls, windows, **settings))
    model.data_pipeline = None
    model.resume_training = False
    model.checkpoints_path = root / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(windows["x_tr"], windows["y_tr"], windows["x_val"], windows["y_val"])
    model.model.eval()
    with torch.no_grad():
        out = model.model.forecast(torch.from_numpy(windows["x_te"]))
    return {k: np.asarray(v) for k, v in out.items() if isinstance(v, torch.Tensor | np.ndarray)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="DSS-EU:default:0.4:0.4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=48)
    args = parser.parse_args()

    windows = build_windows(args.scenario, args.lookback, args.horizon)
    settings = {
        "epochs": args.epochs,
        "seed": args.seed,
        "horizon": args.horizon,
        "lookback": args.lookback,
    }
    root = Path(tempfile.mkdtemp(prefix="residual_dep_"))
    truth = windows["y_te"]

    log.info("  training independent-normal ...")
    independent = fit_head("independent", MLPFNormalModel, MLPFNormalConfig, windows, root, settings)
    log.info("  training joint-normal ...")
    joint = fit_head("joint", MLPFMVNormalModel, MLPFMVNormalConfig, windows, root, settings)

    raw = pair_correlations(truth)
    residual_independent = pair_correlations(truth - independent["loc"])
    residual_joint = pair_correlations(truth - joint["loc"])
    asserted, asserted_daily = asserted_correlations(joint["scale_tril"])
    residual_daily = daily_correlations(truth - joint["loc"])

    # The headline is the mean absolute correlation over the three pairs: it
    # answers "how much dependence is there" without a pair-by-pair reading, and
    # is comparable before and after conditioning.
    raw_mean = float(np.mean([abs(v) for v in raw.values()]))
    residual_mean = float(np.mean([abs(v) for v in residual_joint.values()]))

    report = {
        "scenario": args.scenario,
        "epochs": args.epochs,
        "seed": args.seed,
        "raw_mean_abs": round(raw_mean, 3),
        "residual_mean_abs": round(residual_mean, 3),
        "raw": {k: round(v, 3) for k, v in raw.items()},
        "residual_independent": {k: round(v, 3) for k, v in residual_independent.items()},
        "residual_joint": {k: round(v, 3) for k, v in residual_joint.items()},
        "asserted": {k: round(v, 3) for k, v in asserted.items()},
    }
    for name, _, _ in PAIRS:
        daily = asserted_daily[name]
        report[f"asserted_{name}_min"] = round(float(daily.min()), 3)
        report[f"asserted_{name}_max"] = round(float(daily.max()), 3)
        report[f"residual_{name}_daily_sd"] = round(float(residual_daily[name].std()), 3)

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n  mean |corr|  raw {raw_mean:.3f}  residual {residual_mean:.3f}")
    print(f"  {'pair':6s} {'raw':>8s} {'residual':>10s} {'asserted':>10s} {'day min':>9s} {'day max':>9s}")
    for name, _, _ in PAIRS:
        print(
            f"  {name:6s} {raw[name]:8.3f} {residual_joint[name]:10.3f} {asserted[name]:10.3f} "
            f"{asserted_daily[name].min():9.3f} {asserted_daily[name].max():9.3f}"
        )
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
