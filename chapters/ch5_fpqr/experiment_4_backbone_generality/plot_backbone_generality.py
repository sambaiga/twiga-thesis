"""Per-metric benchmark plots for Chapter 5's backbone-generality comparison.

Compares mlpqf/qr/bnn/laplace on MLPF and N-HiTS. Aggregates
``results/ch5_fpqr_full.csv`` to one row per fold before plotting
(matching the fold-then-mean convention used elsewhere, e.g.
``experiment/ch6_conformal/design_ablation_significance.py``), so the
``errorbar='sd'`` bars reflect fold-to-fold spread rather than the raw
per-timestamp sample size.

Output: figures/{nrmse,crps,winkler_score,cwe_stratified}_backbone_generality.pdf
"""

from pathlib import Path

import arviz as az
from matplotlib import pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (registers the "science" style)
import seaborn as sns

az.style.use(["science", "arviz-doc"])

script_dir = Path(__file__).parent
OUT_DIR = script_dir / "figures"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    "mlpqf": "MLPQF",
    "qr": "QR",
    "bnn": "BNN",
    "laplace": "Laplace",
    "nhits_mlpqf": "MLPQF",
    "nhits_qr": "QR",
    "nhits_bnn": "BNN",
    "nhits_laplace": "Laplace",
}
BACKBONE_NAMES = {
    "benchmark": "MLPF",
    "benchmark_nhits": "N-HiTS",
}
MODEL_ORDER = ["MLPQF", "QR", "BNN", "Laplace"]

METRICS = {
    "nrmse": "NRMSE",
    "crps": "CRPS (kW)",
    "winkler_score": "Winkler score",
    "cwe_stratified": "CWE (stratified)",
}


def load_fold_level() -> pd.DataFrame:
    df = pd.read_csv(script_dir / "results" / "ch5_fpqr_full.csv")
    sub = df[df["group"].isin(["benchmark", "benchmark_nhits"])].copy()
    fold = sub.groupby(["dataset", "group", "condition", "Folds"], as_index=False)[list(METRICS)].mean()
    fold["Model"] = fold["condition"].map(MODEL_NAMES)
    fold["Backbone"] = fold["group"].map(BACKBONE_NAMES)
    fold["Dataset"] = fold["dataset"]
    return fold


def plot_metric(data: pd.DataFrame, metric: str, xlabel: str) -> sns.axisgrid.FacetGrid:
    g = sns.catplot(
        data=data,
        y="Model",
        x=metric,
        hue="Backbone",
        col="Dataset",
        order=MODEL_ORDER,
        palette={"MLPF": "C0", "N-HiTS": "C1"},
        sharex=False,
        margin_titles=True,
        height=2.5,
        aspect=1.25,
        legend=True,
        kind="bar",
        errorbar="sd",
    )
    g.set_axis_labels(xlabel, "")
    g.set_titles(col_template="{col_name}")
    sns.move_legend(g, "lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.15), frameon=False, title=None)
    g.figure.tight_layout()
    return g


def main() -> None:
    fold = load_fold_level()
    for metric, xlabel in METRICS.items():
        g = plot_metric(fold, metric, xlabel)
        out = OUT_DIR / f"{metric}_backbone_generality.pdf"
        g.savefig(out, dpi=480, bbox_inches="tight")
        plt.close(g.figure)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
