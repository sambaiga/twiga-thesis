"""Generate and standardize the README's per-chapter thumbnail images.

Two jobs:

1. New illustrative schematics for chapters that had none (Introduction, MLPF)
   or whose existing thumbnail didn't actually illustrate the chapter's own
   idea (FPQR's fan chart could be any quantile regressor; it never showed
   what makes FPQR different, a *learned* rather than *fixed* quantile grid).
   These are schematic illustrations with synthetic numbers, not real thesis
   results -- unlike ``assets/ch2.png``/``ch3.png``/``ch6.png``/``ch7.png``,
   which are each chapter's own generated figure, copied in verbatim.
2. A consistent canvas size for every thumbnail in the README's chapter
   table, so the table renders with uniform column widths. ``assets/ch2.png``
   is the reference size (562x322 px); every other thumbnail is fit onto a
   canvas of that same size, letterboxed on a white background rather than
   stretched, so no chapter's real figure is distorted.

Usage::

    uv run python scripts/generate_readme_thumbnails.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from twiga_thesis.plotting import style, tokens

ASSETS = Path(__file__).resolve().parent.parent / "assets"
REFERENCE_SIZE = (562, 322)  # assets/ch2.png's own pixel size, the standard every thumbnail matches
FIGSIZE = (2.81, 1.61)  # inches; at DPI below, renders at exactly REFERENCE_SIZE
DPI = 200


def make_introduction_thumbnail() -> None:
    """Chapter 1: a minimal history-then-forecast motif, the thesis's own framing."""
    style.configure(latex=False)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    hours = np.linspace(0, 24, 200)
    history_cutoff = 14.0
    net_load = 1.0 + 0.6 * np.sin((hours - 6) / 24 * 2 * np.pi) + 0.15 * np.sin(hours / 2)
    history = hours <= history_cutoff
    forecast = hours >= history_cutoff

    ax.plot(hours[history], net_load[history], color=tokens.TEXT_DARK, lw=1.4)
    ax.plot(hours[forecast], net_load[forecast], color=tokens.PRIMARY, lw=1.4, ls="--")
    spread = 0.05 + 0.22 * (hours[forecast] - history_cutoff) / (24 - history_cutoff)
    ax.fill_between(
        hours[forecast],
        net_load[forecast] - spread,
        net_load[forecast] + spread,
        color=tokens.PRIMARY,
        alpha=0.18,
        linewidth=0,
    )
    ax.axvline(history_cutoff, color=tokens.TEXT_MUTED, lw=0.8, ls=":")
    ax.text(history_cutoff - 0.6, net_load.max() + 0.12, "Measured", ha="right", fontsize=9, color=tokens.TEXT_DARK)
    ax.text(history_cutoff + 0.6, net_load.max() + 0.12, "Forecast", ha="left", fontsize=9, color=tokens.PRIMARY)

    ax.set_xlim(hours.min(), hours.max())
    ax.set_ylim(net_load.min() - 0.3, net_load.max() + 0.3)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(ASSETS / "ch1.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_mlpf_thumbnail() -> None:
    """Chapter 4: lightweight-vs-accurate, the chapter's own central claim, as a scatter."""
    style.configure(latex=False)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    baseline_parameters = np.array([450, 820, 1600, 2400])
    baseline_nrmse = np.array([0.168, 0.159, 0.152, 0.149])
    mlpf_parameters, mlpf_nrmse = 95, 0.151

    ax.scatter(baseline_parameters, baseline_nrmse, color=tokens.TEXT_MUTED, s=28, zorder=2, label="Baselines")
    ax.scatter(
        mlpf_parameters,
        mlpf_nrmse,
        color=tokens.PRIMARY,
        s=70,
        zorder=3,
        edgecolor="white",
        linewidth=0.8,
        label="MLPF",
    )
    ax.annotate(
        "MLPF",
        xy=(mlpf_parameters, mlpf_nrmse),
        xytext=(mlpf_parameters * 1.6, mlpf_nrmse + 0.006),
        fontsize=9,
        color=tokens.PRIMARY,
        arrowprops={"arrowstyle": "-", "color": tokens.PRIMARY, "lw": 0.7},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=9)
    ax.set_ylabel("NRMSE", fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(ASSETS / "ch4.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fpqr_thumbnail() -> None:
    """Chapter 5: a learned, non-uniform quantile grid next to a fixed, uniform one -- FPQR's own idea."""
    style.configure(latex=False)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    fixed_grid = np.linspace(0.1, 0.9, 9)
    rng = np.random.default_rng(7)
    learned_grid = np.sort(np.clip(np.concatenate([rng.beta(0.6, 0.6, 7), [0.5, 0.5]]), 0.03, 0.97))[:9]

    ax.scatter(fixed_grid, np.ones_like(fixed_grid), color=tokens.TEXT_MUTED, s=26, zorder=2)
    ax.scatter(learned_grid, np.zeros_like(learned_grid), color=tokens.PRIMARY, s=26, zorder=2)
    ax.text(-0.04, 1, "Fixed", ha="right", va="center", fontsize=9, color=tokens.TEXT_MUTED)
    ax.text(-0.04, 0, "Learned", ha="right", va="center", fontsize=9, color=tokens.PRIMARY)

    for level in (0, 1):
        ax.axhline(level, color=tokens.GRID, lw=0.6, zorder=1)

    ax.set_xlim(-0.22, 1.02)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Quantile fraction", fontsize=9)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=8)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(ASSETS / "ch5.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_to_reference_canvas(path: Path) -> None:
    """Letterbox ``path`` onto a white :data:`REFERENCE_SIZE` canvas, preserving its aspect ratio.

    Scales the existing image down (never up) to fit within the reference canvas and centers it,
    so a chapter's real generated figure is resized, never cropped or stretched.
    """
    image = Image.open(path).convert("RGBA")
    scale = min(REFERENCE_SIZE[0] / image.width, REFERENCE_SIZE[1] / image.height, 1.0)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGBA", REFERENCE_SIZE, (255, 255, 255, 255))
    offset = ((REFERENCE_SIZE[0] - new_size[0]) // 2, (REFERENCE_SIZE[1] - new_size[1]) // 2)
    canvas.paste(resized, offset, resized)
    canvas.convert("RGB").save(path)


def main() -> None:
    make_introduction_thumbnail()
    make_mlpf_thumbnail()
    make_fpqr_thumbnail()
    for name in ("ch1.png", "ch3.png", "ch4.png", "ch5.png", "ch6.png", "ch7.png"):
        fit_to_reference_canvas(ASSETS / name)
    print(f"wrote ch1.png, ch4.png, ch5.png and re-fit ch3/ch6/ch7.png to {REFERENCE_SIZE} in {ASSETS}")


if __name__ == "__main__":
    main()
