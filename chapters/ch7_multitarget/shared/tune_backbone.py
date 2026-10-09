"""Tune the shared MLPF backbone once, so the likelihood comparison is fair.

Chapter 6 compares likelihoods placed on one backbone. If each head were tuned
separately, a difference between them could be a difference in how well each was
tuned, and the comparison would answer the wrong question. So the backbone is
tuned once here and every condition then runs on the result.

The search is scored on validation CRPS rather than on the training objective.
The objectives are not comparable across heads, a Gaussian negative
log-likelihood against a pinball loss, whereas CRPS is proper, is defined for
both, and is in the units of the target. Tuning is done with the marginal
Gaussian head because it is the cheapest to fit and carries no dependence
structure of its own, so it cannot bias the result toward the joint likelihoods
that are the subject of the comparison.

Search is random rather than grid: with six hyperparameters a grid coarse enough
to be affordable resolves each one badly, and random search covers the
individually important dimensions better at the same budget.

    uv run python experiment/ch7_multitarget/tune_backbone.py --trials 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
from scipy.stats import norm
import torch
from twiga.core.utils import get_logger, release_torch_memory
from twiga.models.nn.mlpfnormal_model import MLPFNormalConfig, MLPFNormalModel

log = get_logger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_examples import GRID_LEVELS, build_backtest_folds

script_dir = Path(__file__).resolve().parent
OUT = script_dir / "backbone_params.json"  # rebound per record in main()

# Re-centred after a first search put five of six parameters on a boundary: the
# optimiser wanted more latent capacity, less hidden width, more dropout, a
# faster learning rate and more weight decay than any value offered. Three layers
# is dropped rather than widened, being settled by that search at a median
# validation CRPS of 0.540 against 0.482 for one and two layers, and the freed
# budget pays for the extra dropout and learning-rate levels.
#
# Widened again after a full 40-epoch run with early stopping disabled showed
# every trial's true optimum sitting at epoch 0-2: validation NLL more than
# doubled by epoch 40 while training loss fell by 60%, so every prior trial was
# stopped correctly but none of them had escaped immediate overfitting. The
# strongest weight decay on offer was 1e-3; the run that exposed this used
# 1e-5. Both ends are raised so the search can actually reach a regime where
# validation improves past the first few epochs.
SPACE = {
    "latent_size": [64, 128, 256, 512],
    "hidden_dim": [32, 64, 128, 256],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "lr": [1e-3, 2e-3, 3e-3, 5e-3],
    "weight_decay": [1e-4, 1e-3, 1e-2, 1e-1],
}


def sample_config(rng):
    """One draw from the search space."""
    return {name: values[rng.integers(len(values))] for name, values in SPACE.items()}


def validation_crps(params, windows, epochs, seed, root, horizon, lookback, patience):
    """Mean CRPS on the validation split, averaged over targets.

    Returns:
        CRPS in the units of each target, averaged after scaling each target by
        its own spread so active power in kW does not swamp voltage in pu.
    """
    config = MLPFNormalConfig(
        num_target_feature=windows["y_tr"].shape[2],
        num_past_features=windows["x_tr"].shape[2] - windows["y_tr"].shape[2],
        num_calendar_features=0,
        num_known_future_features=0,
        num_forecast_period_features=0,
        forecast_horizon=horizon,
        lookback_window_size=lookback,
        latent_size=int(params["latent_size"]),
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        optimizer_params={"lr": float(params["lr"]), "weight_decay": float(params["weight_decay"])},
        max_epochs=epochs,
        batch_size=64,
        seed=seed,
        early_stop_patience=patience,
    )
    model = MLPFNormalModel(model_config=config)
    model.data_pipeline = None
    model.resume_training = False
    model.checkpoints_path = root / "trial"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(windows["x_tr"], windows["y_tr"], windows["x_val"], windows["y_val"])
    model.model.eval()
    with torch.no_grad():
        out = model.model.forecast(torch.from_numpy(windows["x_val"]))

    loc, scale = np.asarray(out["loc"]), np.asarray(out["scale"])
    quantiles = loc[:, None] + norm.ppf(GRID_LEVELS)[None, :, None, None] * scale[:, None]
    truth = windows["y_val"]
    # CRPS from the quantile grid, the same estimator the study reports, scaled
    # per target so the three contribute comparably to one number.
    spread = truth.reshape(-1, truth.shape[2]).std(0)
    per_target = []
    for i in range(truth.shape[2]):
        diff = np.abs(quantiles[:, :, :, i] - truth[:, None, :, i])
        per_target.append(float(diff.mean()) / float(spread[i]))
    return float(np.mean(per_target))


def tune(
    scenario: str = "DSS-EU:default:0.4:0.4",
    trials: int = 16,
    epochs: int = 30,
    seed: int = 20260906,
    train_months: int = 4,
    lookback: int = 48,
    horizon: int = 48,
    train_stride: int = 8,
    patience: int = 15,
) -> dict:
    """Random-search the shared MLPF backbone on one scenario, write and return the result.

    The importable core of this script's own CLI (below) -- pulled out so the
    other hand-rolled Chapter 6 scripts (``copula_variants.py``,
    ``operational_decision.py``, ``penetration_sweep.py``, none of which use
    :class:`~twiga.experiment.ExperimentEngine` and so have no
    ``run_backbone_hpo`` of their own) can tune the same shared backbone
    in-process via a ``--run-hpo`` flag, rather than requiring this script to
    be run separately first. See the module docstring for why the search is
    scored on validation CRPS with a marginal Gaussian head, and why it is
    random rather than grid.

    Returns:
        The same payload written to disk (``best`` params, ``best_score``,
        every trial in ``all``), so a caller can use the result immediately
        without a round trip through the file it also writes.
    """
    # Tuning uses the first backtest fold only. Tuning on the whole record would
    # select settings using months that later folds hold out, so every fold after
    # the first would be scored by a backbone that had already seen its test
    # period. Only this fold's training and validation rows are touched.
    folds = build_backtest_folds(
        scenario,
        lookback,
        horizon,
        train_months=train_months,
        test_months=1,
        train_stride=train_stride,
    )
    if not folds:
        raise SystemExit("no folds; the record is shorter than the training span")
    windows = folds[0]
    log.info("scenario %s", scenario)
    log.info("tuning on fold %s only (train %s)", windows["fold"], windows["train_span"])
    log.info(
        "split    train %d  val %d  test %d windows",
        windows["x_tr"].shape[0],
        windows["x_val"].shape[0],
        windows["x_te"].shape[0],
    )
    rng = np.random.default_rng(seed)
    root = Path(tempfile.mkdtemp(prefix="tune_"))

    results = []
    for trial in range(trials):
        params = sample_config(rng)
        score = validation_crps(params, windows, epochs, seed, root / str(trial), horizon, lookback, patience)
        results.append({"trial": trial, "score": score, **{k: float(v) for k, v in params.items()}})
        print(f"  trial {trial:2d}  CRPS {score:.5f}  {params}", flush=True)
        # Each trial builds and trains a fresh Lightning model/trainer; without
        # this, CUDA-cached memory and unreferenced objects accumulate across
        # all `trials` iterations (16+ by default).
        release_torch_memory()

    results.sort(key=lambda r: r["score"])
    best = results[0]
    payload = {
        "scenario": scenario,
        "tuned_on_fold": windows["fold"],
        "trials": trials,
        "tuning_epochs": epochs,
        "train_stride": train_stride,
        "patience": patience,
        "objective": "validation CRPS, per-target scaled, marginal Gaussian head",
        "best": {k: best[k] for k in SPACE},
        "best_score": best["score"],
        "all": results,
    }
    out = script_dir / f"backbone_params_{scenario.split(':')[0].lower()}_{scenario.split(':')[1]}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    OUT.write_text(json.dumps(payload, indent=2) + "\n")  # the record most recently tuned
    log.info("best CRPS %.5f with %s", best["score"], payload["best"])
    log.info("wrote %s and %s", out, OUT)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="DSS-EU:default:0.4:0.4")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30, help="epochs per trial; the final study trains longer")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=8,
        help=(
            "spacing between training window starts. A full 40-epoch run at stride 1 showed "
            "validation NLL more than double while training loss fell 60%%, with every trial's "
            "true optimum at epoch 0-2: consecutive windows overlap in 95 of 96 steps at stride "
            "1, so 'more windows' was mostly the same few sequences repeated with a 1-step "
            "shift, trivial to memorise. A wider stride trades window count for independence."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="early-stopping patience in epochs; raised from the 10-epoch default to give the "
        "wider stride room to show whether validation genuinely keeps improving past epoch 2",
    )
    args = parser.parse_args()
    tune(
        scenario=args.scenario,
        trials=args.trials,
        epochs=args.epochs,
        seed=args.seed,
        train_months=args.train_months,
        lookback=args.lookback,
        horizon=args.horizon,
        train_stride=args.train_stride,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
