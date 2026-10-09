"""One-off generator for the README's banner image.

Not part of the reproducibility code; run manually whenever it needs regenerating:

    uv run python scripts/generate_readme_assets.py

The README's other image, assets/roadmap.png, is copied directly from the real
Chapter 1 figure (chapters/ch1_introduction/figures/roadmap.py's output), not
generated here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from twiga_thesis.plotting.tokens import PALETTE

OUT = Path(__file__).resolve().parent.parent / "assets"
SERIF = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}


def make_banner() -> None:
    plt.rcParams.update(SERIF)
    fig, ax = plt.subplots(figsize=(12, 3), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    n = len(PALETTE)
    band_h = 0.22
    for i, color in enumerate(PALETTE):
        ax.add_patch(plt.Rectangle((i * 12 / n, 0), 12 / n, band_h, color=color, linewidth=0))

    ax.text(6, 1.95, "twiga-thesis", fontsize=50, weight="bold", color=PALETTE[0], ha="center", va="center")
    ax.text(
        6,
        1.1,
        "Efficient and Robust Machine Learning for Net-Load Forecasting\nin Low-Voltage Distribution Systems",
        fontsize=17,
        color=PALETTE[6],
        ha="center",
        va="center",
        linespacing=1.6,
    )
    fig.savefig(OUT / "banner.png", bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_banner()
    print(f"wrote banner.png to {OUT}")


if __name__ == "__main__":
    main()
