"""Hand-drawn concept diagrams for the book, using matplotlib directly.

Unlike everything else in :mod:`ark.plot`, these are not data visualisations:
each function draws a fixed, illustrative diagram for one specific concept
(matplotlib's Figure/Axes model, NumPy broadcasting, ...). They are matplotlib
code on purpose, not Mermaid or a static image asset, so they render as a real
image everywhere a tutorial notebook is opened, including plain Jupyter and
VS Code, not just when processed by Quarto.
"""

from __future__ import annotations

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

from .icons import ICONS, icon_font
from .tokens import (
    AI_ACCENT,
    DANGER,
    ENERGY_ACCENT,
    INFO,
    INFO_TINT,
    PRIMARY,
    SUCCESS,
    TEXT_MUTED,
    WARNING,
    WARNING_TINT,
)


def figure_axes_diagram() -> plt.Figure:
    """Draw a labelled diagram of one Figure containing two Axes.

    Illustrates the distinction Part 5 opens with: a Figure is the whole
    canvas, an Axes is one plot inside it, and a single Figure can hold
    more than one. The two panels are real Axes with real plotted data,
    annotated rather than drawn as abstract boxes.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    for ax, title in zip(axes, ["Axes object 1", "Axes object 2"], strict=True):
        x = np.linspace(0, 10, 60)
        ax.plot(x, np.sin(x) + rng.normal(0, 0.08, 60), color=PRIMARY, linewidth=1.4)
        ax.set_xlabel("x label")
        ax.set_ylabel("y label")
        ax.set_title(title, fontsize=10, color=INFO, fontweight="bold")

    fig.subplots_adjust(top=0.72, bottom=0.18, left=0.12, right=0.97, wspace=0.35)

    # Draw the Figure's own boundary explicitly - this is the one line in
    # the whole diagram that has no equivalent in the rendered chart itself.
    # Its top edge sits below the label text so the two never overlap.
    boundary = plt.Rectangle(
        (0.01, 0.01),
        0.98,
        0.85,
        transform=fig.transFigure,
        fill=False,
        edgecolor=WARNING,
        linewidth=2,
        linestyle="dashed",
        clip_on=False,
    )
    fig.add_artist(boundary)
    fig.text(
        0.5,
        0.93,
        "One Figure (the dashed boundary - the whole canvas, what fig.savefig() writes)",
        ha="center",
        fontsize=10,
        color=WARNING,
        fontweight="bold",
    )
    return fig


def broadcasting_diagram() -> plt.Figure:
    """Draw a grid diagram of a (3,) array broadcasting against a (5, 3) matrix.

    The single row at the top represents the smaller array; the dashed
    arrows show it being virtually repeated for every row of the larger
    array below, which is the mental model Part 4's broadcasting rule
    depends on.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    n_rows, n_cols = 5, 3
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # The (5, 3) matrix: a grid of cells, one row per sample.
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((c, -r), 0.92, 0.92, facecolor=INFO_TINT, edgecolor=INFO, linewidth=1))
    ax.text(n_cols / 2, -n_rows - 0.3, "shape (5, 3)", ha="center", color=INFO, fontweight="bold")

    # The (3,) array: one row, drawn above the matrix, with dashed arrows
    # showing it "stretching" down to every row it gets broadcast against.
    for c in range(n_cols):
        ax.add_patch(plt.Rectangle((c, 1.6), 0.92, 0.92, facecolor=WARNING_TINT, edgecolor=WARNING, linewidth=1))
        ax.annotate(
            "",
            xy=(c + 0.46, -n_rows + 1.1),
            xytext=(c + 0.46, 1.55),
            arrowprops={"arrowstyle": "->", "color": WARNING, "linestyle": "dashed", "alpha": 0.6},
        )
    ax.text(n_cols / 2, 2.75, "shape (3,)", ha="center", color=WARNING, fontweight="bold")
    ax.text(
        n_cols / 2,
        -n_rows / 2 + 0.5,
        "broadcast\n(no data copied)",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_MUTED,
        style="italic",
    )

    ax.set_xlim(-0.3, n_cols + 0.3)
    ax.set_ylim(-n_rows - 0.8, 3.1)
    ax.set_title("X - feature_mean: (5, 3) and (3,)", fontsize=11, color=PRIMARY, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def merge_join_types_diagram() -> plt.Figure:
    """Draw the four `pd.merge` join types as overlapping-circle diagrams.

    Each panel shows the same two circles, "left" keys and "right" keys,
    with the kept region shaded: only the overlap for inner, all of left
    plus the overlap for left, all of right plus the overlap for right, and
    everything for outer. The circles never change, only what gets shaded
    does, which is the point: a join type is a choice about which rows to
    keep, not a different way of matching them.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    joins = [
        ("inner", "intersection only", False, True, False),
        ("left", "all of left, matched right", True, True, False),
        ("right", "all of right, matched left", False, True, True),
        ("outer", "everything, matched or not", True, True, True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

    for ax, (name, subtitle, keep_left_only, keep_overlap, keep_right_only) in zip(axes, joins, strict=True):
        left = Circle((-0.45, 0), 1.0, facecolor="none", edgecolor=INFO, linewidth=1.6)
        right = Circle((0.45, 0), 1.0, facecolor="none", edgecolor=WARNING, linewidth=1.6)

        if keep_left_only:
            ax.add_patch(Circle((-0.45, 0), 1.0, facecolor=INFO, alpha=0.25, edgecolor="none"))
        if keep_right_only:
            ax.add_patch(Circle((0.45, 0), 1.0, facecolor=WARNING, alpha=0.25, edgecolor="none"))
        if keep_overlap:
            overlap = Circle((0, 0), 0.55, facecolor=SUCCESS, alpha=0.45, edgecolor="none")
            ax.add_patch(overlap)

        ax.add_patch(left)
        ax.add_patch(right)
        ax.text(-0.95, 1.15, "left", color=INFO, fontsize=9, fontweight="bold", ha="center")
        ax.text(0.95, 1.15, "right", color=WARNING, fontsize=9, fontweight="bold", ha="center")
        ax.set_title(f'how="{name}"', fontsize=11, color=PRIMARY, fontweight="bold")
        ax.text(0, -1.4, subtitle, ha="center", fontsize=8.5, color=TEXT_MUTED, style="italic")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.8, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("pd.merge(left, right, on=key, how=...) - shaded rows are kept", fontsize=11, color=PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def groupby_split_apply_combine_diagram() -> plt.Figure:
    """Draw groupby as three stages: split, apply, combine.

    A small table of rows is split into groups by key, the same function
    is applied independently to each group, and the per-group results are
    combined back into a single table, one row per group. This is the
    mental model behind every `df.groupby(...)` call, regardless of which
    aggregation function ends up inside `.apply()`.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    group_colors = [INFO, WARNING, SUCCESS]
    group_labels = ["program = CS", "program = DS", "program = IT"]

    # Stage 1: split - one small stack of rows per group.
    split_x = 0.5
    for i, color in enumerate(group_colors):
        y0 = 2.6 - i * 1.3
        for r in range(3):
            ax.add_patch(
                plt.Rectangle((split_x, y0 - r * 0.32), 1.1, 0.26, facecolor=color, alpha=0.25, edgecolor=color)
            )
        ax.text(split_x + 0.55, y0 + 0.5, group_labels[i], ha="center", fontsize=8.5, color=color, fontweight="bold")
    ax.text(split_x + 0.55, 4.1, "split", ha="center", fontsize=11, color=PRIMARY, fontweight="bold")

    # Stage 2: apply - the same function applied to each group independently.
    apply_x = 3.3
    for i, color in enumerate(group_colors):
        y0 = 2.6 - i * 1.3
        ax.add_patch(plt.Rectangle((apply_x, y0 - 0.55), 1.1, 0.75, facecolor="none", edgecolor=color, linewidth=1.6))
        ax.text(apply_x + 0.55, y0 - 0.17, ".mean()", ha="center", fontsize=8.5, color=color)
        ax.annotate(
            "",
            xy=(apply_x - 0.1, y0 - 0.17),
            xytext=(split_x + 1.2, y0 - 0.17),
            arrowprops={"arrowstyle": "->", "color": TEXT_MUTED, "alpha": 0.7},
        )
    ax.text(apply_x + 0.55, 4.1, "apply", ha="center", fontsize=11, color=PRIMARY, fontweight="bold")

    # Stage 3: combine - one result row per group, back in a single table.
    combine_x = 6.0
    combine_values = ["0.61", "0.58", "0.74"]
    for i, color in enumerate(group_colors):
        y0 = 2.6 - i * 1.3
        ax.add_patch(plt.Rectangle((combine_x, y0 - 0.55), 1.6, 0.34, facecolor=color, alpha=0.3, edgecolor=color))
        ax.text(
            combine_x + 0.8,
            y0 - 0.38,
            f"{group_labels[i].split(' = ')[1]}: {combine_values[i]}",
            ha="center",
            fontsize=8,
            color=PRIMARY,
        )
        ax.annotate(
            "",
            xy=(combine_x - 0.1, y0 - 0.38),
            xytext=(apply_x + 1.2, y0 - 0.17),
            arrowprops={"arrowstyle": "->", "color": TEXT_MUTED, "alpha": 0.7},
        )
    ax.text(combine_x + 0.8, 4.1, "combine", ha="center", fontsize=11, color=PRIMARY, fontweight="bold")

    ax.set_xlim(0, 8)
    ax.set_ylim(-1.3, 4.5)
    ax.axis("off")
    ax.set_title(
        'df.groupby("program")["final_score"].mean() - one function, run once per group',
        fontsize=10.5,
        color=PRIMARY,
    )
    fig.tight_layout()
    return fig


def _curved_flow_arrow(
    ax: plt.Axes,
    xy_start: tuple[float, float],
    xy_end: tuple[float, float],
    color: str,
    rad: float = 0.15,
    linewidth: float = 1.6,
) -> None:
    """A single curved, single-headed arrow, the shared idiom every panel below uses for power flow."""
    ax.add_patch(
        FancyArrowPatch(
            xy_start,
            xy_end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=linewidth,
            shrinkA=6,
            shrinkB=6,
        )
    )


def _grid_tier_zone(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, label: str) -> None:
    """A soft, dashed rounded-rectangle zone with its label sitting above it, not fighting the icons inside."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            edgecolor=TEXT_MUTED,
            facecolor="none",
            alpha=0.55,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height + 0.1,
        label,
        fontsize=8,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="bottom",
        alpha=0.85,
    )


def _grid_panel(ax: plt.Axes, title: str, decentralized: bool) -> None:
    """One side of the centralized/decentralized comparison; see `centralized_vs_decentralized_grid_diagram`."""
    # Deliberately not a repeated template: the two panels use their own
    # hand-picked coordinates, box sizes, and arrow curvatures rather than
    # one geometry shared by both and recolored, the thing that made the
    # first version of this diagram read as auto-generated despite the
    # curved arrows. Every number below was chosen for this specific panel.
    if decentralized:
        plant_xy = (0.53, 5.62)
        sub_xy = (0.49, 3.7)
        house_xys = [(0.11, 1.1), (0.52, 0.86), (0.88, 1.16)]
        zones = [(0.0, 4.9, 1.0, 1.02), (0.01, 2.7, 0.99, 1.5), (0.03, 0.02, 0.95, 2.08)]
        node_r, sub_r = 0.175, 0.235
        icon_sizes = (26, 17, 12)
    else:
        plant_xy = (0.47, 5.5)
        sub_xy = (0.45, 3.62)
        house_xys = [(0.16, 1.0), (0.48, 0.83), (0.83, 1.04)]
        zones = [(0.03, 4.82, 0.94, 1.08), (0.0, 2.62, 1.0, 1.58), (0.01, 0.08, 0.98, 2.0)]
        node_r, sub_r = 0.165, 0.245
        icon_sizes = (25, 18, 12)

    _grid_tier_zone(ax, zones[0][:2], zones[0][2], zones[0][3], "HV transmission")
    _grid_tier_zone(ax, zones[1][:2], zones[1][2], zones[1][3], "MV substation")
    _grid_tier_zone(ax, zones[2][:2], zones[2][2], zones[2][3], "LV feeder")

    # Hand-picked per panel too: a real illustrator wouldn't re-derive one
    # arrow curvature from a loop index for six different arrows, they'd
    # nudge each one until it looked right. These numbers are that nudging.
    if decentralized:
        plant_sub_rad = 0.13
        down_rads = [-0.19, -0.04, 0.09]
        up_rads = [-0.06, 0.07, 0.19]
    else:
        plant_sub_rad = 0.06
        feed_rads = [-0.16, 0.01, 0.13]

    ax.text(
        plant_xy[0],
        plant_xy[1],
        ICONS["building-fill"],
        fontproperties=icon_font(icon_sizes[0]),
        color=PRIMARY,
        ha="center",
        va="center",
    )
    ax.text(plant_xy[0], plant_xy[1] - 0.38, "power plant", fontsize=7.5, color=TEXT_MUTED, ha="center")

    ax.add_patch(Circle(sub_xy, sub_r, facecolor="white", edgecolor=PRIMARY, linewidth=1.4))
    ax.text(
        sub_xy[0],
        sub_xy[1],
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(icon_sizes[1]),
        color=PRIMARY,
        ha="center",
        va="center",
    )
    ax.text(sub_xy[0], sub_xy[1] - 0.42, "substation", fontsize=7.5, color=TEXT_MUTED, ha="center")

    _curved_flow_arrow(
        ax, (plant_xy[0], plant_xy[1] - 0.3), (sub_xy[0], sub_xy[1] + 0.3), TEXT_MUTED, rad=plant_sub_rad
    )

    for i, house_xy in enumerate(house_xys):
        if decentralized:
            # Two distinct, offset arrows rather than one double-headed
            # arrow: at this scale a "<->" arrowhead reads as one
            # direction, not two. Grey supplies down, amber exports back up.
            _curved_flow_arrow(
                ax,
                (sub_xy[0] - 0.05, sub_xy[1] - 0.28),
                (house_xy[0] - 0.05, house_xy[1] + 0.24),
                TEXT_MUTED,
                rad=down_rads[i],
                linewidth=1.2,
            )
            _curved_flow_arrow(
                ax,
                (house_xy[0] + 0.05, house_xy[1] + 0.24),
                (sub_xy[0] + 0.05, sub_xy[1] - 0.28),
                WARNING,
                rad=up_rads[i],
                linewidth=1.4,
            )
        else:
            _curved_flow_arrow(
                ax,
                (sub_xy[0], sub_xy[1] - 0.28),
                (house_xy[0], house_xy[1] + 0.24),
                TEXT_MUTED,
                rad=feed_rads[i],
                linewidth=1.3,
            )

        # The third house on the decentralized side reads DANGER (a voltage
        # violation), not because that's inevitable, but because Chapter 1
        # of Part 4 spends real effort establishing that DER growth can push
        # a feeder past its limits; a diagram that shows only green would
        # undersell the point the surrounding chapter is actually making.
        node_color = DANGER if (decentralized and i == 2) else SUCCESS
        ax.add_patch(Circle(house_xy, node_r, facecolor=node_color, edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(
            house_xy[0],
            house_xy[1],
            ICONS["house-fill"],
            fontproperties=icon_font(icon_sizes[2]),
            color="white",
            ha="center",
            va="center",
            zorder=4,
        )
        if decentralized and i != 1:
            badge_icon = "sun-fill" if i == 0 else "ev-front-fill"
            badge_color = WARNING if i == 0 else PRIMARY
            badge_xy = (house_xy[0] + 0.16, house_xy[1] + 0.15)
            ax.add_patch(Circle(badge_xy, 0.105, facecolor=badge_color, edgecolor="white", linewidth=0.8, zorder=5))
            ax.text(
                badge_xy[0],
                badge_xy[1],
                ICONS[badge_icon],
                fontproperties=icon_font(9),
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )
        if i == 1:
            # One labeled example, not one badge per house: the point is
            # "this is where a smart meter sits," a concept, not a claim
            # that every house in this schematic has one. Same icon the
            # book's own cover page already uses for "Meter Signal," so
            # the two pages read as the same visual language.
            meter_xy = (house_xy[0] + 0.19, house_xy[1] - 0.16)
            ax.add_patch(Circle(meter_xy, 0.105, facecolor=INFO, edgecolor="white", linewidth=0.8, zorder=5))
            ax.text(
                meter_xy[0],
                meter_xy[1],
                ICONS["speedometer2"],
                fontproperties=icon_font(9),
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )
            ax.text(
                meter_xy[0] + 0.14,
                meter_xy[1],
                "smart meter",
                fontsize=6.5,
                color=INFO,
                ha="left",
                va="center",
                style="italic",
            )

    caption = "power flows both ways" if decentralized else "power flows one way only"
    caption_color = WARNING if decentralized else TEXT_MUTED
    ax.text(
        0.5,
        -0.2,
        caption,
        fontsize=8.5,
        color=caption_color,
        ha="center",
        style="italic",
        fontweight="bold" if decentralized else "normal",
    )

    ax.set_title(title, fontsize=11.5, color=PRIMARY, fontweight="bold", loc="center", pad=8)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.42, 6.25)
    ax.set_aspect("equal")
    ax.axis("off")


def centralized_vs_decentralized_grid_diagram() -> plt.Figure:
    """Draw the centralized-vs-decentralized grid comparison Part 4's first chapter opens with.

    Left panel: the traditional grid, one-way power flow from a plant
    through a substation down to passive customers. Right panel: the same
    chain, but with rooftop solar and EV charging at the low-voltage tier,
    where power now flows both ways, and one customer is deliberately
    marked as a voltage violation, the exact strain the surrounding chapter
    spends its first few sections establishing.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4))
    _grid_panel(axes[0], "Centralized grid", decentralized=False)
    _grid_panel(axes[1], "Decentralized, DER-rich grid", decentralized=True)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _failure_mode_panel(
    ax: plt.Axes,
    title: str,
    badge_icon: str,
    badge_color: str,
    arrow_up: bool,
    arrow_color: str,
    gauge_label: str,
    gauge_color: str,
) -> None:
    """One panel of `pv_vs_ev_failure_mode_diagram`: a substation, one house, and one flow arrow."""
    sub_xy = (0.5, 1.42)
    house_xy = (0.5, 0.32)

    ax.add_patch(Circle(sub_xy, 0.26, facecolor="white", edgecolor=PRIMARY, linewidth=1.6, zorder=2))
    ax.text(
        sub_xy[0],
        sub_xy[1],
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(19),
        color=PRIMARY,
        ha="center",
        va="center",
        zorder=3,
    )

    ax.add_patch(Circle(house_xy, 0.22, facecolor=SUCCESS, edgecolor="white", linewidth=1.6, zorder=2))
    ax.text(
        house_xy[0],
        house_xy[1],
        ICONS["house-fill"],
        fontproperties=icon_font(15),
        color="white",
        ha="center",
        va="center",
        zorder=3,
    )
    badge_xy = (house_xy[0] + 0.22, house_xy[1] + 0.17)
    ax.add_patch(Circle(badge_xy, 0.135, facecolor=badge_color, edgecolor="white", linewidth=1.0, zorder=4))
    ax.text(
        badge_xy[0],
        badge_xy[1],
        ICONS[badge_icon],
        fontproperties=icon_font(11),
        color="white",
        ha="center",
        va="center",
        zorder=5,
    )

    start, end = (house_xy, sub_xy) if arrow_up else (sub_xy, house_xy)
    _curved_flow_arrow(
        ax,
        (start[0] + 0.07, start[1] + (0.28 if arrow_up else -0.28)),
        (end[0] + 0.07, end[1] + (-0.28 if arrow_up else 0.28)),
        arrow_color,
        rad=0.2 if arrow_up else -0.2,
        linewidth=2.4,
    )

    gauge_xy = (0.98, 1.42)
    ax.add_patch(Circle(gauge_xy, 0.155, facecolor="white", edgecolor=gauge_color, linewidth=2.6, zorder=2))
    ax.text(
        gauge_xy[0],
        gauge_xy[1],
        ICONS["speedometer2"],
        fontproperties=icon_font(11),
        color=gauge_color,
        ha="center",
        va="center",
        zorder=3,
    )
    ax.text(
        gauge_xy[0],
        gauge_xy[1] - 0.24,
        gauge_label,
        fontsize=7.5,
        color=gauge_color,
        ha="center",
        va="top",
        fontweight="bold",
    )

    ax.set_title(title, fontsize=13, color=PRIMARY, fontweight="bold", loc="center", pad=10)
    ax.set_xlim(0.02, 1.24)
    ax.set_ylim(-0.02, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")


def pv_vs_ev_failure_mode_diagram() -> plt.Figure:
    """Draw the PV-vs-EV contrast Chapter 2 builds its second half around.

    Same feeder, same house, two different real problems: PV exports
    power at midday and pushes voltage toward its statutory ceiling
    first; EV charging imports power at the evening peak and pushes the
    transformer toward its thermal rating first, the opposite direction,
    the opposite time of day, and a different binding constraint, exactly
    the finding Chapter 2's own hosting-capacity sweeps land on.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    _failure_mode_panel(
        axes[0],
        "PV, midday",
        badge_icon="sun-fill",
        badge_color=WARNING,
        arrow_up=True,
        arrow_color=WARNING,
        gauge_label="voltage\nbinds first",
        gauge_color=DANGER,
    )
    _failure_mode_panel(
        axes[1],
        "EV, evening",
        badge_icon="ev-front-fill",
        badge_color=INFO,
        arrow_up=False,
        arrow_color=INFO,
        gauge_label="thermal\nbinds first",
        gauge_color=DANGER,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def voltage_correlation_vs_magnitude_diagram() -> plt.Figure:
    """Draw the "correlation, not magnitude" contrast Chapter 3's phase-identification method rests on.

    Three synthetic customers on a feeder: two share phase A but sit at
    different distances from the transformer, so their absolute voltage
    levels differ; the third sits on phase B, but happens to land at an
    absolute level between the other two. Raw magnitude (left) groups by
    proximity to the transformer, pairing the wrong two customers.
    Correlation (right) reveals which voltage traces actually move
    together, correctly pairing the two customers who share phase A, the
    same distinction the surrounding Key Concept box states in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    hours = np.linspace(0, 24, 96)
    phase_a_fluctuation = 0.02 * np.sin(2 * np.pi * (hours - 6) / 24) + 0.01 * np.sin(2 * np.pi * hours / 6)
    phase_b_fluctuation = 0.02 * np.sin(2 * np.pi * (hours - 14) / 24) + 0.008 * np.sin(2 * np.pi * (hours + 2) / 5)

    near_transformer = 1.02 + phase_a_fluctuation
    far_on_phase_a = 0.97 + 1.3 * phase_a_fluctuation
    on_phase_b = 1.00 + phase_b_fluctuation

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    ax = axes[0]
    ax.plot(hours, near_transformer, color=WARNING, linewidth=2.2, label="Customer A1 (phase A, near)")
    ax.plot(hours, far_on_phase_a, color=INFO, linewidth=2.2, label="Customer A2 (phase A, far)")
    ax.plot(hours, on_phase_b, color=DANGER, linewidth=2.2, linestyle="--", label="Customer B1 (phase B)")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("Cluster on raw magnitude", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Wrong pair: A1 and B1 sit closest in level",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=DANGER,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(hours, near_transformer - near_transformer.mean(), color=WARNING, linewidth=2.2, label="Customer A1")
    ax.plot(hours, far_on_phase_a - far_on_phase_a.mean(), color=INFO, linewidth=2.2, label="Customer A2")
    ax.plot(
        hours,
        on_phase_b - on_phase_b.mean(),
        color=DANGER,
        linewidth=2.2,
        linestyle="--",
        label="Customer B1",
    )
    ax.set_ylabel("Voltage fluctuation (centered)")
    ax.set_title("Cluster on correlation", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Right pair: A1 and A2 move together",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def _household_curve(peak: float, *, hours: np.ndarray) -> np.ndarray:
    """A synthetic evening-peaking household load curve, scaled to a given peak (kW)."""
    shape = 0.15 + 0.85 * np.exp(-0.5 * ((hours - 19) / 2.3) ** 2)
    return shape * peak


def shape_vs_magnitude_diagram() -> plt.Figure:
    """Draw the "shape, not magnitude" contrast Chapter 4's clustering feature choice rests on.

    Two households share the same evening-peaking rhythm but draw very
    different absolute power: a small apartment (peak 1.0 kW) and a large
    house (peak 4.5 kW). Clustering on raw magnitude (left) separates them
    by size; clustering on peak-normalized shape (right), the choice this
    chapter actually makes, groups them together because their rhythm
    matches, the same distinction the surrounding Key Concept box states
    in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    hours = np.linspace(0, 24, 96)
    small = _household_curve(1.0, hours=hours)
    large = _household_curve(4.5, hours=hours)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    ax = axes[0]
    ax.plot(hours, small, color=WARNING, linewidth=2.2, label="Household A (peak 1.0 kW)")
    ax.plot(hours, large, color=INFO, linewidth=2.2, label="Household B (peak 4.5 kW)")
    ax.set_ylabel("Demand (kW)")
    ax.set_title("Cluster on raw magnitude", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Different clusters: far apart in kW",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=DANGER,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(hours, small / small.max(), color=WARNING, linewidth=2.2, label="Household A")
    ax.plot(hours, large / large.max(), color=INFO, linewidth=2.2, linestyle="--", label="Household B")
    ax.set_ylabel("Normalized demand")
    ax.set_title("Cluster on peak-normalized shape", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.24,
        "Same cluster: same evening rhythm",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def cluster_confidence_set_diagram() -> plt.Figure:
    """Draw the geometry behind a conformal cluster-membership confidence set.

    Three archetype centroids in embedding space, each surrounded by a
    circle of radius equal to the calibrated conformal threshold. A
    customer landing inside exactly one circle gets a confident,
    single-archetype set (Customer 1). One landing inside two overlapping
    circles gets an honest two-archetype set instead of a forced single
    guess (Customer 2). One landing outside every circle gets an empty
    set, itself a signal that this customer's own behavior does not
    confidently resemble any archetype (Customer 3). The same geometric
    idea Chapter 3 uses for phase centroids and Chapter 4 uses for
    archetype centroids.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    centroids = {
        "Archetype 1": ((-1.4, 0.6), INFO, "left"),
        "Archetype 2": ((1.4, 0.6), WARNING, "right"),
        "Archetype 3": ((0.0, -1.7), SUCCESS, "below"),
    }
    threshold_radius = 1.05

    for name, (xy, color, side) in centroids.items():
        ax.add_patch(Circle(xy, threshold_radius, facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.8))
        ax.scatter(*xy, s=70, color=color, zorder=3)
        if side == "left":
            label_xy, ha = (xy[0] - threshold_radius - 0.1, xy[1]), "right"
        elif side == "right":
            label_xy, ha = (xy[0] + threshold_radius + 0.1, xy[1]), "left"
        else:
            label_xy, ha = (xy[0], xy[1] - threshold_radius - 0.25), "center"
        ax.text(*label_xy, name, ha=ha, va="center", fontsize=9.5, color=color, fontweight="bold")

    customers = [
        ("Customer 1", "confident: {1}", (-1.4, 1.35), (0.0, 0.32), "center", TEXT_MUTED),
        ("Customer 2", "ambiguous: {1, 2}", (0.0, 0.6), (0.0, 0.32), "center", TEXT_MUTED),
        ("Customer 3", "no match: {}", (2.5, -0.6), (0.0, 0.32), "center", DANGER),
    ]
    for name, status, xy, text_offset, ha, color in customers:
        ax.scatter(*xy, s=90, marker="x", color=color, linewidth=2.4, zorder=4)
        ax.text(
            xy[0] + text_offset[0],
            xy[1] + text_offset[1],
            f"{name}\n{status}",
            fontsize=8,
            color=color,
            ha=ha,
            va="bottom",
        )

    ax.set_xlim(-3.0, 3.2)
    ax.set_ylim(-3.2, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Each circle's radius is the calibrated threshold\ninside one, inside two, or inside none",
        fontsize=10,
        color=PRIMARY,
        fontweight="bold",
    )

    fig.tight_layout()
    plt.close(fig)
    return fig


def _lever_checklist_panel(
    ax: plt.Axes,
    title: str,
    badge_icon: str,
    badge_color: str,
    available: dict[str, bool],
) -> None:
    """One panel of `lever_applicability_diagram`: a house, a badge, and a checklist of levers."""
    house_xy = (0.28, 0.5)
    ax.add_patch(Circle(house_xy, 0.2, facecolor=PRIMARY, edgecolor="white", linewidth=1.6, zorder=2))
    ax.text(
        house_xy[0],
        house_xy[1],
        ICONS["house-fill"],
        fontproperties=icon_font(15),
        color="white",
        ha="center",
        va="center",
        zorder=3,
    )
    badge_xy = (house_xy[0] + 0.16, house_xy[1] + 0.15)
    ax.add_patch(Circle(badge_xy, 0.1, facecolor=badge_color, edgecolor="white", linewidth=1.0, zorder=4))
    ax.text(
        badge_xy[0],
        badge_xy[1],
        ICONS[badge_icon],
        fontproperties=icon_font(8),
        color="white",
        ha="center",
        va="center",
        zorder=5,
    )

    list_x = 0.58
    top_y = 0.86
    row_height = 0.19
    for row, (lever, is_available) in enumerate(available.items()):
        y = top_y - row * row_height
        mark_color = SUCCESS if is_available else DANGER
        mark = "✓" if is_available else "✕"
        ax.text(list_x, y, mark, color=mark_color, fontsize=13, fontweight="bold", ha="center", va="center")
        label_color = PRIMARY if is_available else TEXT_MUTED
        ax.text(list_x + 0.08, y, lever, color=label_color, fontsize=10, ha="left", va="center")

    ax.set_title(title, fontsize=12, color=PRIMARY, fontweight="bold", pad=10)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")


def lever_applicability_diagram() -> plt.Figure:
    """Draw which mitigation levers can even act on a voltage violation versus a thermal one.

    Volt-Watt and Volt-VAr are `InvControl` elements: they only govern a
    `PVSystem` element's own real and reactive power output as local
    voltage moves. A customer whose violation comes from EV charging
    pushing the transformer's own thermal loading has no PVSystem for
    either lever to control at all, not a weaker option, an inapplicable
    one. Storage is the only lever that reaches both violation types,
    the same physical constraint the recommendation section's own case-
    based retrieval is scoped around.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    _lever_checklist_panel(
        axes[0],
        "Voltage violation (PV export)",
        badge_icon="sun-fill",
        badge_color=WARNING,
        available={"Volt-Watt": True, "Volt-VAr": True, "Storage": True, "No action": True},
    )
    _lever_checklist_panel(
        axes[1],
        "Thermal violation (EV charging)",
        badge_icon="ev-front-fill",
        badge_color=INFO,
        available={"Volt-Watt": False, "Volt-VAr": False, "Storage": True, "No action": True},
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def _feeder_line_panel(ax: plt.Axes, title: str, subtitle: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Draw a transformer and two customers at different distances along one feeder line.

    Returns the (near, far) customer node positions so the caller can add
    a panel-specific annotation on top of the shared schematic.
    """
    transformer_xy = (0.3, 0.5)
    near_xy = (1.6, 0.5)
    far_xy = (3.4, 0.5)

    ax.plot([transformer_xy[0], far_xy[0]], [0.5, 0.5], color=TEXT_MUTED, linewidth=2.0, zorder=1)

    ax.add_patch(Circle(transformer_xy, 0.22, facecolor=PRIMARY, edgecolor="white", linewidth=1.2, zorder=3))
    ax.text(
        *transformer_xy,
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(15),
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )

    for xy, label in [(near_xy, "Customer near"), (far_xy, "Customer far")]:
        ax.add_patch(Circle(xy, 0.19, facecolor=SUCCESS, edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(
            *xy, ICONS["house-fill"], fontproperties=icon_font(12), color="white", ha="center", va="center", zorder=4
        )
        ax.text(xy[0], xy[1] - 0.42, label, fontsize=8, color=TEXT_MUTED, ha="center")
        ax.text(xy[0], xy[1] + 0.32, "same real shape", fontsize=7.5, color=TEXT_MUTED, ha="center", style="italic")

    ax.set_title(title, fontsize=11.5, color=PRIMARY, fontweight="bold", pad=14)
    ax.text(2.0, -0.35, subtitle, fontsize=8.5, color=TEXT_MUTED, ha="center", style="italic", wrap=True)
    ax.set_xlim(-0.3, 4.0)
    ax.set_ylim(-0.6, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    return near_xy, far_xy


def voltage_vs_thermal_position_diagram() -> plt.Figure:
    """Draw why bus position matters for voltage risk but barely matters for thermal risk.

    Two customers share the exact same real smart-meter shape, only their
    distance from the transformer differs. Voltage rise depends on how
    much line impedance sits between a customer and the source, on top of
    how much they draw or export, something a smart-meter reading alone
    says nothing about, so the far customer sees a real, different
    voltage outcome (left). Thermal loading at the transformer depends
    only on how much power is flowing at its own peak moment, exactly
    what a smart-meter reading already captures, so position adds nothing
    once shape is known (right). The physical reason retrieval-from-shape
    works for one risk and struggles for the other, the same distinction
    the surrounding Key Concept box states in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    near_xy, far_xy = _feeder_line_panel(
        axes[0],
        "Voltage: depends on where you are",
        "Impedance between customer and source adds up with distance",
    )
    ax = axes[0]
    pu_y = near_xy[1] + 0.95
    ax.text(near_xy[0], pu_y, "1.01 pu", fontsize=9, color=SUCCESS, fontweight="bold", ha="center")
    ax.text(far_xy[0], pu_y, "0.96 pu", fontsize=9, color=DANGER, fontweight="bold", ha="center")
    ax.annotate(
        "",
        xy=(far_xy[0], pu_y - 0.14),
        xytext=(near_xy[0], pu_y - 0.14),
        arrowprops={"arrowstyle": "-|>", "color": WARNING, "linewidth": 1.6},
    )
    ax.text(2.5, pu_y + 0.28, "voltage drifts with distance", fontsize=7.5, color=WARNING, ha="center", style="italic")

    near_xy, far_xy = _feeder_line_panel(
        axes[1],
        "Thermal: depends on what you draw",
        "Transformer loading sums instantaneous power, position is invisible to it",
    )
    ax = axes[1]
    arrow_y = near_xy[1] + 0.85
    for xy in (near_xy, far_xy):
        _curved_flow_arrow(ax, (xy[0], arrow_y - 0.1), (0.3, arrow_y), INFO, rad=0.2)
    ax.text(2.0, arrow_y + 0.28, "same contribution either way", fontsize=7.5, color=INFO, ha="center", style="italic")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _customer_row(ax: plt.Axes, y: float, n: int = 5) -> list[tuple[float, float]]:
    """Five evenly-spaced house icons along one row, returns their positions."""
    xs = np.linspace(0.6, 4.2, n)
    positions = [(x, y) for x in xs]
    for xy in positions:
        ax.add_patch(Circle(xy, 0.22, facecolor=SUCCESS, edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(
            *xy, ICONS["house-fill"], fontproperties=icon_font(13), color="white", ha="center", va="center", zorder=4
        )
    return positions


def per_customer_vs_network_view_diagram() -> plt.Figure:
    """Draw why per-customer detection cannot see a coincident event that a network view catches immediately.

    Five real customers, the same real moment. Left: each customer's own
    self-baseline detector only ever sees its own history, and a
    regionally sunny day already sits inside that history, so nothing
    looks wrong. Right: the same five customers share one real
    transformer, and their real exports arrive at the same real moment;
    summed, they cross a real physical limit no single customer's own
    meter could ever reveal, the structural blind spot Section 2's own
    Key Concept box states in words.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))

    ax = axes[0]
    positions = _customer_row(ax, 1.1)
    for xy in positions:
        ax.text(xy[0], xy[1] - 0.42, "own history", fontsize=7, color=TEXT_MUTED, ha="center", style="italic")
        ax.text(xy[0], xy[1] + 0.4, "✓", fontsize=13, color=SUCCESS, fontweight="bold", ha="center")
    ax.set_title(
        "Per-customer view: five isolated self-baselines",
        fontsize=10.8,
        color=PRIMARY,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        2.4,
        -0.15,
        "A regionally sunny day already sits inside each customer's own real history",
        fontsize=8.2,
        color=TEXT_MUTED,
        ha="center",
        style="italic",
        wrap=True,
    )
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    transformer_xy = (2.4, 0.2)
    positions = _customer_row(ax, 1.5)
    ax.add_patch(Circle(transformer_xy, 0.24, facecolor=DANGER, edgecolor="white", linewidth=1.4, zorder=3))
    ax.text(
        *transformer_xy,
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(16),
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )
    for xy in positions:
        ax.text(xy[0], xy[1] + 0.4, ICONS["sun-fill"], fontproperties=icon_font(11), color=WARNING, ha="center")
        _curved_flow_arrow(ax, (xy[0], xy[1] - 0.22), (transformer_xy[0], transformer_xy[1] + 0.24), DANGER, rad=0.12)
    ax.text(
        transformer_xy[0],
        transformer_xy[1] - 0.5,
        "real overload",
        fontsize=8.5,
        color=DANGER,
        fontweight="bold",
        ha="center",
    )
    ax.set_title(
        "Network view: the same real moment, summed",
        fontsize=10.8,
        color=PRIMARY,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        2.4,
        -0.9,
        "Five individually unremarkable exports, one real, physical overload",
        fontsize=8.2,
        color=TEXT_MUTED,
        ha="center",
        style="italic",
        wrap=True,
    )
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-1.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _persistent_signal(hours: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """A clean, strongly repeating daily cycle plus a little real noise."""
    daily = 0.5 + 0.45 * np.sin((hours - 6) / 24 * 2 * np.pi)
    return daily + rng.normal(0, 0.03, size=hours.shape)


def _volatile_signal(hours: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """The same average level and daily cycle, buried under much larger real noise."""
    daily = 0.5 + 0.15 * np.sin((hours - 6) / 24 * 2 * np.pi)
    return daily + rng.normal(0, 0.22, size=hours.shape)


def forecastability_diagram() -> plt.Figure:
    """Draw the "some signals are just harder to forecast" contrast this chapter opens with.

    Two signals share the same average level and the same underlying daily
    cycle. The left one repeats cleanly, day after day, real noise but not
    enough to hide the pattern: low entropy, a Hurst exponent above 0.5, the
    persistent case a model can learn from directly. The right one carries
    the same cycle under noise large enough to swamp it: high entropy, a
    Hurst exponent near the random-walk boundary, the volatile case no
    amount of modelling sophistication fixes on its own. This is the real
    distinction every metric in this chapter measures from a different
    angle, not a specific customer's own reading.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(7)
    hours = np.linspace(0, 24 * 3, 300)  # three real days

    persistent = _persistent_signal(hours, rng=rng)
    volatile = _volatile_signal(hours, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)

    ax = axes[0]
    ax.plot(hours, persistent, color=SUCCESS, linewidth=1.6)
    ax.set_title("Persistent: forecastable", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.26,
        "Low entropy, Hurst > 0.5\nyesterday predicts today",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(hours, volatile, color=DANGER, linewidth=1.6)
    ax.set_title("Volatile: hard to forecast", fontsize=12, color=PRIMARY, fontweight="bold")
    ax.text(
        0.5,
        -0.26,
        "High entropy, Hurst near 0.5\nthe same cycle, buried in noise",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=DANGER,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("Hour (three real days)")
        ax.set_xlim(hours.min(), hours.max())
        ax.set_xticks([0, 24, 48, 72])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Load (normalized)")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    color: str,
    filled: bool = True,
) -> None:
    """A single labeled rounded box, the shared unit every panel below is built from."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6,
            edgecolor=color,
            facecolor=color if filled else "none",
            alpha=1.0 if filled else 0.9,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        fontsize=8.3,
        color="white" if filled else color,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _flow_arrow(ax: plt.Axes, xy_start: tuple[float, float], xy_end: tuple[float, float], color: str) -> None:
    """A single straight downward flow arrow, the shared idiom both panels below use."""
    ax.add_patch(
        FancyArrowPatch(
            xy_start,
            xy_end,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            linewidth=1.5,
            shrinkA=2,
            shrinkB=2,
        )
    )


def known_vs_new_customer_diagram() -> plt.Figure:
    """Draw the two different mechanisms this chapter's real result calls for.

    Left: a customer the system already has real history for. Their own
    distance to each archetype centroid is just another feature, fed into
    the same single shared model as the lag features, a legitimate
    personalization signal because the model has already trained on this
    exact customer. Right: a brand-new customer with no history in any
    trained model yet. The same distance features, computed from whatever
    partial history exists, first pass through Chapter 5's own calibrated
    retrieval-confidence gate, reused directly, not reimplemented, before
    deciding whether to trust a distance-weighted blend of the already-
    trained archetype specialists or fall back to the safe global model.
    Two different answers to "how should clustering and forecasting
    combine," chosen by whether a customer is known or new, not by
    preference.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))

    # ---- left panel: known customer ----
    ax = axes[0]
    ax.set_title("Known customer", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (0.0, 3.6), 1.7, 0.6, "Lag features", color=INFO)
    _flow_box(ax, (2.1, 3.6), 1.9, 0.6, "Own archetype\ndistance", color=SUCCESS)
    _flow_box(ax, (0.9, 2.0), 2.2, 0.7, "One shared model", color=PRIMARY)
    _flow_arrow(ax, (0.85, 3.6), (1.6, 2.7), color=INFO)
    _flow_arrow(ax, (3.05, 3.6), (2.4, 2.7), color=SUCCESS)
    _flow_arrow(ax, (2.0, 2.0), (2.0, 1.0), color=PRIMARY)
    ax.text(
        2.0,
        0.55,
        "Personalized forecast",
        fontsize=9.5,
        color=SUCCESS,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        2.0,
        4.5,
        "A static distance signature\ndoubles as a personalization signal",
        fontsize=8.2,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="bottom",
    )

    # ---- right panel: new customer ----
    ax = axes[1]
    ax.set_title("New customer", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (0.9, 4.3), 2.2, 0.6, "Partial-history distance", color=WARNING)
    _flow_arrow(ax, (2.0, 4.3), (2.0, 3.5), color=WARNING)
    ax.scatter([2.0], [3.15], s=1400, marker="D", facecolor="white", edgecolor=PRIMARY, linewidth=1.6, zorder=3)
    ax.text(2.0, 3.15, "Confident\nmatch?", fontsize=7.3, color=PRIMARY, fontweight="bold", ha="center", va="center")

    _flow_arrow(ax, (2.5, 2.9), (3.3, 2.15), color=SUCCESS)
    ax.text(3.45, 2.55, "yes", fontsize=8, color=SUCCESS, fontweight="bold", ha="left", va="center")
    _flow_box(ax, (2.75, 1.55), 2.1, 0.65, "Blend of archetype\nspecialists", color=SUCCESS)

    _flow_arrow(ax, (1.5, 2.9), (0.7, 2.15), color=DANGER)
    ax.text(0.55, 2.55, "no", fontsize=8, color=DANGER, fontweight="bold", ha="right", va="center")
    _flow_box(ax, (-0.35, 1.55), 2.1, 0.65, "Global model\n(safe fallback)", color=TEXT_MUTED)

    _flow_arrow(ax, (0.7, 1.55), (1.8, 0.75), color=TEXT_MUTED)
    _flow_arrow(ax, (3.8, 1.55), (2.2, 0.75), color=SUCCESS)
    ax.text(
        2.0,
        0.4,
        "Forecast, never worse than the floor",
        fontsize=8.6,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="center",
    )

    for ax in axes:
        ax.set_xlim(-0.8, 5.0)
        ax.set_ylim(0.0, 5.1)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def archetype_distance_and_conformal_diagram() -> plt.Figure:
    """Draw the two mechanics behind this chapter's real result.

    Distance-to-centroid weighting, and the split-conformal calibration
    that gates it. Left: a customer's own shape sits some distance from each of 5 archetype
    centroids. Those distances become a softmax over negative distance, the
    same mixture-of-experts idea Jacobs, Jordan, Nowlan and Hinton introduced
    in 1991, closer centroids get more weight, not a hard, winner-take-all
    assignment. Right: before trusting any of this for a new customer, a
    calibrated distance threshold, tau, decides whether a real match exists
    at all: tau is the (1-alpha) quantile of nearest-neighbor distances on a
    held-back calibration split, the split-conformal guarantee Lei, G'Sell,
    Rinaldo, Tibshirani and Wasserman formalized in 2018. A customer inside
    tau is trusted; one beyond it safely falls back to the blind model.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))

    # ---- left panel: distance-to-centroid weighting ----
    ax = axes[0]
    ax.set_title("Distance becomes weight", fontsize=12, color=PRIMARY, fontweight="bold")
    customer_xy = (0.0, 0.0)
    rng = np.random.default_rng(3)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.3
    radii = np.array([1.0, 2.6, 1.8, 3.4, 2.2])
    weights = np.exp(-radii / 1.3)
    weights = weights / weights.sum()

    for angle, radius, weight in zip(angles, radii, weights, strict=True):
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        ax.plot([customer_xy[0], cx], [customer_xy[1], cy], color=TEXT_MUTED, linewidth=1.0 + 5 * weight, zorder=1)
        ax.add_patch(
            Circle(
                (cx, cy),
                0.42,
                facecolor=SUCCESS,
                edgecolor="white",
                linewidth=1.2,
                zorder=2,
                alpha=0.55 + 0.45 * weight,
            )
        )
        ax.text(
            cx, cy, f"{weight:.0%}", fontsize=8, color="white", fontweight="bold", ha="center", va="center", zorder=3
        )

    ax.add_patch(Circle(customer_xy, 0.32, facecolor=PRIMARY, edgecolor="white", linewidth=1.4, zorder=4))
    ax.text(
        customer_xy[0],
        customer_xy[1] - 0.85,
        "New customer's\npartial-history shape",
        fontsize=8.2,
        color=PRIMARY,
        ha="center",
        va="top",
        fontweight="bold",
    )
    ax.text(
        0, 3.9, r"$w_i = e^{-d_i/T} \,/\, \sum_j e^{-d_j/T}$", fontsize=10.5, color=PRIMARY, ha="center", va="center"
    )
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- right panel: split-conformal calibration ----
    ax = axes[1]
    ax.set_title('A calibrated "do we trust this" gate', fontsize=12, color=PRIMARY, fontweight="bold")
    calib_scores = rng.gamma(2.2, 0.35, size=400)
    tau = np.quantile(calib_scores, 0.9)
    counts, bin_edges = np.histogram(calib_scores, bins=28)
    bin_width = bin_edges[1] - bin_edges[0]
    for left, count in zip(bin_edges[:-1], counts, strict=True):
        trusted_bin = left + bin_width / 2 <= tau
        ax.bar(
            left,
            count,
            width=bin_width * 0.95,
            align="edge",
            color=SUCCESS if trusted_bin else DANGER,
            alpha=0.75,
            linewidth=0,
        )

    ax.axvline(tau, color=PRIMARY, linestyle="--", linewidth=1.6)
    ax.set_ylim(0, counts.max() * 1.35)
    ax.text(
        tau, counts.max() * 1.18, r"$\tau$ = 90th percentile", fontsize=9, color=PRIMARY, ha="center", fontweight="bold"
    )
    ax.annotate(
        "Trusted:\nblend the specialists",
        xy=(tau * 0.45, counts.max() * 0.55),
        fontsize=8.4,
        color=SUCCESS,
        ha="center",
        fontweight="bold",
    )
    ax.annotate(
        "Flagged:\nfall back to blind",
        xy=(tau * 1.5, counts.max() * 0.55),
        fontsize=8.4,
        color=DANGER,
        ha="center",
        fontweight="bold",
    )
    ax.set_xlabel("Distance to nearest labeled neighbor")
    ax.set_ylabel("Calibration customers")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def three_uq_paradigms_diagram() -> plt.Figure:
    """Draw the three probabilistic paradigms this chapter checks, side by side.

    Left: a parametric density (a real Gaussian curve), the only paradigm
    that can price a risk-of-exceedance question directly, the shaded tail
    is literally the probability a real load exceeds some threshold. Middle:
    a quantile forecast, several quantile levels fit directly to noisy real
    data with no assumed shape, the P95 line answering a capacity-sizing
    question on its own. Right: a conformal interval, a fixed-width band
    around a central forecast with a real, finite-sample coverage guarantee,
    the only one of the three that holds regardless of whether the
    underlying model's own assumptions are correct.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # ---- left: parametric density ----
    ax = axes[0]
    ax.set_title("Parametric", fontsize=11.5, color=PRIMARY, fontweight="bold")
    x = np.linspace(-4, 4, 400)
    y = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    threshold = 1.4
    ax.plot(x, y, color=PRIMARY, linewidth=1.8)
    ax.fill_between(x[x >= threshold], y[x >= threshold], color=DANGER, alpha=0.55)
    ax.axvline(threshold, color=DANGER, linestyle="--", linewidth=1.2)
    ax.text(threshold + 0.15, y.max() * 0.55, "P(load > X)", fontsize=9, color=DANGER, fontweight="bold")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, y.max() * 1.15)
    ax.axis("off")

    # ---- middle: quantile forecast ----
    ax = axes[1]
    ax.set_title("Quantile", fontsize=11.5, color=PRIMARY, fontweight="bold")
    t = np.linspace(0, 10, 60)
    noise_scale = 0.3 + 0.5 * np.abs(np.sin(t / 3))
    points = 1.5 + 0.4 * t + rng.normal(0, 1, size=t.shape) * noise_scale
    ax.scatter(t, points, s=10, color=TEXT_MUTED, alpha=0.6, zorder=1)
    p50 = 1.5 + 0.4 * t
    p95 = p50 + 1.8 * noise_scale
    p05 = p50 - 1.8 * noise_scale
    ax.plot(t, p50, color=PRIMARY, linewidth=1.6, zorder=2)
    ax.plot(t, p95, color=WARNING, linewidth=1.6, linestyle="--", zorder=2)
    ax.plot(t, p05, color=WARNING, linewidth=1.6, linestyle="--", zorder=2)
    ax.text(t[-1], p95[-1], "  P95", fontsize=9, color=WARNING, fontweight="bold", va="center")
    ax.text(t[-1], p50[-1], "  P50", fontsize=9, color=PRIMARY, fontweight="bold", va="center")
    ax.text(t[-1], p05[-1], "  P5", fontsize=9, color=WARNING, fontweight="bold", va="center")
    ax.set_xlim(0, 12.5)
    ax.axis("off")

    # ---- right: conformal guarantee band ----
    ax = axes[2]
    ax.set_title("Conformal", fontsize=11.5, color=PRIMARY, fontweight="bold")
    t2 = np.linspace(0, 10, 60)
    center = 2.0 + 0.25 * t2 + 0.6 * np.sin(t2 / 1.6)
    band = 1.3
    ax.fill_between(t2, center - band, center + band, color=SUCCESS, alpha=0.28, zorder=1)
    ax.plot(t2, center, color=SUCCESS, linewidth=1.8, zorder=2)
    covered = rng.uniform(-1, 1, size=12) * band * 0.8
    idx = np.linspace(3, 56, 12).astype(int)
    ax.scatter(t2[idx], center[idx] + covered, s=16, color=PRIMARY, zorder=3)
    ax.text(
        5,
        center.min() - band - 0.35,
        r"$\geq 90\%$ coverage, guaranteed",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
        ha="center",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(center.min() - band - 0.9, center.max() + band + 0.5)
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def ground_truth_vs_discovered_clusters_diagram() -> plt.Figure:
    """Draw what ARI and NMI actually measure.

    Agreement between two real partitions of the same objects, corrected
    for chance. Left: the same 15 points colored by a real, known category (three
    building types, say). Right: the same 15 points at the same positions,
    colored by what an unsupervised clustering pass actually discovered
    from shape alone, with a handful of real disagreements, a point whose
    true category and discovered cluster do not match, marked with an
    outline. Both indices summarize the whole grid of matches and
    mismatches into one number: Hubert and Arabie's Adjusted Rand Index
    counts agreeing and disagreeing pairs of points, corrected for the
    agreement chance alone would produce; Strehl and Ghosh's Normalized
    Mutual Information asks how much knowing one partition reduces
    uncertainty about the other. Neither reaches its maximum just because
    two partitions have the same number of groups, only if the actual
    grouping of objects agrees.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(5)
    true_colors = [SUCCESS, WARNING, INFO]
    # three explicit 2x3 grids, well separated, no overlap between or within groups
    group_origins = [(-1.3, 0.9), (1.1, 0.9), (-0.1, -1.1)]
    positions, true_labels = [], []
    for g, (ox, oy) in enumerate(group_origins):
        for r in range(2):
            for c in range(3):
                positions.append((ox + c * 0.55, oy - r * 0.55))
                true_labels.append(g)
    positions = np.array(positions)
    true_labels = np.array(true_labels)

    # a discovered clustering that mostly, but not perfectly, agrees
    discovered_labels = true_labels.copy()
    swap_idx = rng.choice(len(true_labels), size=3, replace=False)
    for idx in swap_idx:
        discovered_labels[idx] = (discovered_labels[idx] + 1) % 3

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    ax = axes[0]
    ax.set_title("Ground truth", fontsize=12, color=PRIMARY, fontweight="bold")
    for i, (x, y) in enumerate(positions):
        ax.add_patch(
            Circle((x, y), 0.14, facecolor=true_colors[true_labels[i]], edgecolor="white", linewidth=1.2, zorder=2)
        )

    ax = axes[1]
    ax.set_title("Discovered by clustering", fontsize=12, color=PRIMARY, fontweight="bold")
    for i, (x, y) in enumerate(positions):
        mismatched = discovered_labels[i] != true_labels[i]
        ax.add_patch(
            Circle(
                (x, y),
                0.14,
                facecolor=true_colors[discovered_labels[i]],
                edgecolor=DANGER if mismatched else "white",
                linewidth=2.4 if mismatched else 1.2,
                zorder=2,
            )
        )
    mismatch_idx = np.flatnonzero(discovered_labels != true_labels)
    for rank, i in enumerate(mismatch_idx):
        x, y = positions[i]
        label_y = 1.75 - rank * 0.32
        ax.annotate(
            "disagreement",
            xy=(x, y),
            xytext=(2.05, label_y),
            fontsize=8,
            color=DANGER,
            va="center",
            arrowprops={"arrowstyle": "-", "color": DANGER, "linewidth": 0.8},
        )

    for ax in axes:
        ax.set_xlim(-1.8, 2.4)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[1].set_xlim(-1.8, 3.2)

    fig.tight_layout()
    plt.close(fig)
    return fig


def cross_population_retrieval_trust_diagram() -> plt.Figure:
    """Draw why a calibrated retrieval radius does not travel for free.

    Left: AusNet's own labeled customers in shape-embedding space, with the
    split-conformal radius tau drawn around them, calibrated so that tau
    covers 90% of how far a held-back AusNet customer's own nearest real
    neighbor ever sits. Every AusNet point at that scale sits inside its
    own calibration, by construction. Right: the same tau circle, unmoved,
    with real NREL ResStock buildings added. A real US home whose own
    shape happens to resemble an AusNet household still falls inside tau
    and gets a trustworthy retrieval. A real US home whose shape reflects
    a genuinely different climate or appliance mix, electric resistance
    heating in a cold state, say, falls outside it, and the same
    conformal mechanism that flagged AusNet's own unusual customers
    flags this one too, honestly, rather than silently extrapolating.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(7)
    tau_radius = 1.0
    ausnet_pts = rng.normal(scale=0.42, size=(40, 2))

    nrel_inside = rng.normal(scale=0.55, size=(10, 2))
    nrel_inside = nrel_inside / np.maximum(np.linalg.norm(nrel_inside, axis=1, keepdims=True) / 0.85, 1.0)
    angles = rng.uniform(0, 2 * np.pi, 9)
    radii = rng.uniform(1.15, 1.75, 9)
    nrel_outside = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))

    ax = axes[0]
    ax.set_title("Calibrated on AusNet alone", fontsize=11.5, color=PRIMARY, fontweight="bold")
    ax.add_patch(Circle((0, 0), tau_radius, facecolor=SUCCESS, alpha=0.08, edgecolor=SUCCESS, linewidth=1.6))
    ax.scatter(ausnet_pts[:, 0], ausnet_pts[:, 1], s=34, color=INFO, edgecolor="white", linewidth=0.6, zorder=3)
    ax.text(0, -1.32, r"$\tau$ = 90% of AusNet's own held-back distances", fontsize=8.5, color=SUCCESS, ha="center")

    ax = axes[1]
    ax.set_title("Same $\\tau$, real NREL buildings added", fontsize=11.5, color=PRIMARY, fontweight="bold")
    ax.add_patch(Circle((0, 0), tau_radius, facecolor=SUCCESS, alpha=0.08, edgecolor=SUCCESS, linewidth=1.6))
    ax.scatter(
        ausnet_pts[:, 0], ausnet_pts[:, 1], s=28, color=INFO, alpha=0.5, edgecolor="white", linewidth=0.5, zorder=2
    )
    ax.scatter(
        nrel_inside[:, 0],
        nrel_inside[:, 1],
        s=42,
        color=SUCCESS,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="NREL, trusted",
    )
    ax.scatter(
        nrel_outside[:, 0],
        nrel_outside[:, 1],
        s=42,
        color=DANGER,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="NREL, flagged",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    for ax in axes:
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def diversity_weighted_ranking_diagram() -> plt.Figure:
    """Draw why a redundant metric should not count as much as an independent one.

    Each metric is drawn as an arrow from the origin, its direction set by
    how it ranks a fixed set of models: four pointwise-error metrics, MAE,
    RMSE, WMAPE, SMAPE, tend to agree on which model is better, so their
    arrows cluster at a similar angle. Corr asks a different question,
    whether a model tracks the real signal's shape, and disagrees with the
    error metrics often enough that its arrow points somewhere else
    entirely. The bars on the right show the consequence: a metric whose
    own arrow sits close to several others gets down-weighted in the final
    ranking, exactly what this book's own diversity-weighted Borda count
    does, since a nearly unanimous cluster of agreeing metrics should not
    outvote the one metric actually saying something new.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    error_color = WARNING
    corr_color = SUCCESS

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1.1, 0.9]})

    ax = axes[0]
    ax.set_title("Where each metric points", fontsize=11.5, color=PRIMARY, fontweight="bold")
    error_angles_deg = [24, 34, 14, 44]
    error_labels = ["MAE", "RMSE", "WMAPE", "SMAPE"]
    for angle_deg, label in zip(error_angles_deg, error_labels, strict=True):
        angle = np.deg2rad(angle_deg)
        ax.annotate(
            "",
            xy=(np.cos(angle), np.sin(angle)),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": error_color, "linewidth": 2.0},
        )
        ax.text(
            np.cos(angle) * 1.12, np.sin(angle) * 1.12, label, fontsize=9, color=error_color, ha="left", va="center"
        )

    corr_angle = np.deg2rad(150)
    ax.annotate(
        "",
        xy=(np.cos(corr_angle), np.sin(corr_angle)),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": corr_color, "linewidth": 2.4},
    )
    ax.text(
        np.cos(corr_angle) * 1.15,
        np.sin(corr_angle) * 1.15,
        "Corr",
        fontsize=9.5,
        color=corr_color,
        fontweight="bold",
        ha="right",
        va="center",
    )
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=TEXT_MUTED, linewidth=0.8, linestyle="dashed"))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.3, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Weight in the final ranking", fontsize=11.5, color=PRIMARY, fontweight="bold")
    metrics = ["MAE", "RMSE", "WMAPE", "SMAPE", "Corr"]
    weights = [0.164, 0.164, 0.164, 0.18, 0.328]
    colors = [error_color] * 4 + [corr_color]
    y_pos = np.arange(len(metrics))[::-1]
    ax.barh(y_pos, weights, color=colors, height=0.55)
    for y, w in zip(y_pos, weights, strict=True):
        ax.text(w + 0.01, y, f"{w:.3f}", va="center", fontsize=9, color=TEXT_MUTED)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=9.5)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("diversity weight", fontsize=9, color=TEXT_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def multi_task_shared_encoder_diagram() -> plt.Figure:
    """Draw one shared encoder feeding a state head and a power head.

    The single coupling this chapter's whole architecture rests on: one
    representation of the input window, computed once, drives both a
    per-appliance state classifier and a per-appliance multi-quantile power
    regressor, rather than two unrelated models trained as if they had
    nothing to do with each other.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.4))

    _flow_box(ax, (0.7, 0.2), 2.6, 0.6, "Input window\n(100 readings)", color=TEXT_MUTED)
    _flow_arrow(ax, (2.0, 0.8), (2.0, 1.55), color=PRIMARY)

    _flow_box(ax, (0.4, 1.55), 3.2, 0.7, "Shared encoder\n(CNN1D / U-Net)", color=PRIMARY)

    _flow_arrow(ax, (1.55, 2.25), (0.5, 3.1), color=SUCCESS)
    _flow_arrow(ax, (2.45, 2.25), (3.5, 3.1), color=WARNING)

    _flow_box(ax, (-0.6, 3.1), 2.2, 0.65, "State head\nsoftmax, 2-way", color=SUCCESS)
    _flow_box(ax, (2.5, 3.1), 2.2, 0.65, "Power head\n5 quantiles", color=WARNING)

    _flow_arrow(ax, (0.5, 3.75), (0.5, 4.35), color=SUCCESS)
    _flow_arrow(ax, (3.6, 3.75), (3.6, 4.35), color=WARNING)

    ax.text(0.5, 4.55, "on / off", fontsize=9.5, color=SUCCESS, fontweight="bold", ha="center", va="bottom")
    ax.text(3.6, 4.55, "power band", fontsize=9.5, color=WARNING, fontweight="bold", ha="center", va="bottom")

    ax.text(
        2.0,
        4.9,
        "One representation, two tasks",
        fontsize=11.5,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    ax.set_xlim(-1.1, 5.1)
    ax.set_ylim(-0.1, 5.3)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _icon_circle(ax: plt.Axes, center: tuple[float, float], radius: float, color: str) -> None:
    """A plain circle outline, the shared frame every small icon glyph below sits inside."""
    ax.add_patch(Circle(center, radius, facecolor="none", edgecolor=color, linewidth=1.8))


def nilm_frequency_split_diagram() -> plt.Figure:
    """Draw one meter signal splitting into the book's two acquisition regimes.

    A single real meter reading can be sampled two very different ways.
    Sampled in the kHz range and up, it supports load identification:
    telling appliances apart by their own transient signature (Chapters
    2-3). Sampled at the rate a real smart meter reports, it supports
    energy estimation instead: how much power, not which appliance
    exactly (Chapter 4). Same signal, two different jobs, depending only
    on how fast it gets read.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    _icon_circle(ax, (0.75, 2.1), 0.62, INFO)
    step_x = [0.45, 0.45, 0.62, 0.62, 0.85, 0.85, 1.05]
    step_y = [1.85, 2.25, 2.25, 2.45, 2.45, 1.95, 1.95]
    ax.plot(step_x, step_y, color=INFO, linewidth=1.8, solid_joinstyle="round")
    ax.text(0.75, 1.25, "Meter Signal", ha="center", fontsize=9.5, color=PRIMARY, fontweight="bold")

    _curved_flow_arrow(ax, (1.35, 2.3), (2.75, 3.15), color=AI_ACCENT, rad=0.25)
    _curved_flow_arrow(ax, (1.35, 1.9), (2.75, 1.05), color=SUCCESS, rad=-0.25)

    _flow_box(ax, (2.8, 2.75), 3.0, 0.85, "HIGH FREQUENCY\nkHz and up\n(Chapters 2-3)", color=AI_ACCENT)
    _flow_box(ax, (2.8, 0.6), 3.0, 0.85, "LOW FREQUENCY\nsmart-meter rate\n(Chapter 4)", color=SUCCESS)

    _flow_arrow(ax, (5.85, 3.18), (6.55, 3.18), color=AI_ACCENT)
    _flow_arrow(ax, (5.85, 1.03), (6.55, 1.03), color=SUCCESS)

    _icon_circle(ax, (7.1, 3.18), 0.55, AI_ACCENT)
    ax.plot(
        [6.85, 6.98, 7.1, 7.22, 7.35],
        [3.0, 3.35, 3.05, 3.35, 3.0],
        color=AI_ACCENT,
        linewidth=1.6,
        solid_joinstyle="round",
    )
    ax.text(7.1, 2.45, "Load\nIdentification", ha="center", va="top", fontsize=9, color=PRIMARY, fontweight="bold")

    _icon_circle(ax, (7.1, 1.03), 0.55, SUCCESS)
    ax.plot(
        [6.85, 6.98, 7.1, 7.22, 7.35],
        [0.9, 0.95, 1.1, 1.05, 1.3],
        color=SUCCESS,
        linewidth=1.6,
        solid_joinstyle="round",
    )
    ax.text(7.1, 0.35, "Energy\nEstimation", ha="center", va="top", fontsize=9, color=PRIMARY, fontweight="bold")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig


def nilm_transient_steady_state_diagram() -> plt.Figure:
    """Draw a switch-on event's current waveform splitting into transient and steady state.

    The first fraction of a second after a switch-on is transient: a fast,
    decaying oscillation whose shape, duration, and harmonic content are
    what make one appliance's fingerprint different from another's. Once
    that dies out, the signal settles into the smooth, repeating cycle of
    steady state, active power, reactive power, current, the same three
    numbers every appliance eventually settles at.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    t_transient = np.linspace(0, 1, 300)
    envelope = np.exp(-3.2 * t_transient)
    transient = envelope * np.sin(2 * np.pi * 9 * t_transient)

    t_steady = np.linspace(1, 2.4, 300)
    steady = 0.18 * np.sin(2 * np.pi * 4 * (t_steady - 1))

    ax.axvspan(0, 1, color=AI_ACCENT, alpha=0.12)
    ax.axvspan(1, 2.4, color=INFO, alpha=0.10)
    ax.axvline(1, color=TEXT_MUTED, linestyle="dashed", linewidth=1.3)

    ax.plot(t_transient, transient, color=PRIMARY, linewidth=1.8)
    ax.plot(t_steady, steady, color=PRIMARY, linewidth=1.8)

    ax.text(0.5, 1.15, "TRANSIENT", ha="center", fontsize=12, color=AI_ACCENT, fontweight="bold")
    ax.text(0.5, 1.0, "shape, duration, harmonics", ha="center", fontsize=9.5, color=AI_ACCENT)
    ax.text(1.7, 1.15, "STEADY STATE", ha="center", fontsize=12, color=INFO, fontweight="bold")
    ax.text(1.7, 1.0, "active power, reactive power, current", ha="center", fontsize=9.5, color=INFO)
    ax.text(1.2, -1.35, "switch-on settles", ha="center", fontsize=9.5, color=TEXT_MUTED, style="italic")

    ax.set_xlim(0, 2.4)
    ax.set_ylim(-1.5, 1.3)
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig


def _pill_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, label: str, color: str) -> None:
    """A rounded, unfilled pill, the shared idiom for a small result readout in the panels below."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            linewidth=1.6,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, fontsize=8.5, color=PRIMARY, ha="center", va="center")


def nilm_single_vs_multi_appliance_diagram() -> plt.Figure:
    """Draw single-appliance learning against multi-appliance, multi-label learning.

    Top: single-appliance learning trains and runs one model per
    appliance, each blind to what the others are doing, a separate pass
    through the same aggregate signal for every appliance. Bottom:
    multi-appliance, multi-label learning trains one shared model that
    reads every appliance's state off the same signal in a single pass,
    the mechanism Chapter 3 builds on directly.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2))

    appliances = ["Fridge", "Kettle", "Heater"]
    states = ["ON", "OFF", "ON"]
    row_ys = [3.0, 1.8, 0.6]

    # ---- top panel: single-appliance ----
    ax = axes[0]
    ax.set_title("SINGLE-APPLIANCE", fontsize=13, color=PRIMARY, fontweight="bold", pad=14)
    ax.text(
        4.0,
        4.15,
        "a separate model, and a separate pass, per appliance",
        ha="center",
        fontsize=9.5,
        color=TEXT_MUTED,
        style="italic",
    )
    _flow_box(ax, (0.0, 1.75), 1.7, 0.9, "Aggregate\nsignal", color=INFO)
    for i, (appl, state, y) in enumerate(zip(appliances, states, row_ys, strict=True)):
        _curved_flow_arrow(ax, (1.75, 2.2), (2.7, y + 0.35), color=TEXT_MUTED, rad=0.18 * (1 - i))
        _flow_box(ax, (2.75, y), 2.0, 0.7, f"{appl} model", color=TEXT_MUTED, filled=False)
        _flow_arrow(ax, (4.8, y + 0.35), (5.3, y + 0.35), color=TEXT_MUTED)
        _pill_box(ax, (5.35, y), 1.7, 0.7, f"{appl}: {state}", color=TEXT_MUTED)
    ax.text(
        7.6, 1.95, "three\nseparate\npasses", ha="left", va="center", fontsize=8.5, color=TEXT_MUTED, style="italic"
    )
    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(0.2, 4.5)
    ax.axis("off")

    # ---- bottom panel: multi-appliance ----
    ax = axes[1]
    ax.set_title("MULTI-APPLIANCE (MULTI-LABEL)", fontsize=13, color=AI_ACCENT, fontweight="bold", pad=14)
    ax.text(
        4.0,
        4.15,
        "one shared model, one pass, every appliance at once",
        ha="center",
        fontsize=9.5,
        color=AI_ACCENT,
        style="italic",
    )
    _flow_box(ax, (0.0, 1.75), 1.7, 0.9, "Aggregate\nsignal", color=INFO)
    _flow_arrow(ax, (1.75, 2.2), (2.7, 2.2), color=AI_ACCENT)
    _flow_box(ax, (2.75, 1.75), 2.3, 0.9, "Multi-label\nmodel", color=AI_ACCENT)
    for i, (appl, state, y) in enumerate(zip(appliances, states, row_ys, strict=True)):
        _curved_flow_arrow(ax, (5.1, 2.2), (5.85, y + 0.35), color=AI_ACCENT, rad=0.18 * (1 - i))
        _pill_box(ax, (5.9, y), 1.7, 0.7, f"{appl}: {state}", color=AI_ACCENT)
    ax.text(
        7.9,
        1.95,
        "one\npass",
        ha="left",
        va="center",
        fontsize=8.5,
        color=AI_ACCENT,
        style="italic",
        fontweight="bold",
    )
    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(0.2, 4.5)
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def nilm_activation_extraction_diagram() -> plt.Figure:
    """Draw one state transition splitting a waveform into a before-window and an after-window.

    A fixed number of AC cycles immediately before a detected state
    transition, and the same fixed number immediately after, aligned at
    the voltage zero-crossing so the subtraction lines up in phase, not
    just in time. Subtracting the two isolates whatever changed: the
    appliance that just switched.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    t_before = np.linspace(-8, 0, 260)
    before = 0.55 * np.sin(2 * np.pi * 1.0 * t_before)

    t_after = np.linspace(0, 8, 260)
    after = 0.55 * np.sin(2 * np.pi * 1.0 * t_after) + 0.5 * np.sin(2 * np.pi * 1.6 * t_after + 0.6)

    ax.axvspan(-8, 0, color=TEXT_MUTED, alpha=0.10)
    ax.axvspan(0, 8, color=AI_ACCENT, alpha=0.12)
    ax.axvline(0, color=PRIMARY, linestyle="dashed", linewidth=1.6)

    ax.plot(t_before, before, color=TEXT_MUTED, linewidth=1.8)
    ax.plot(t_after, after, color=AI_ACCENT, linewidth=1.8)

    ax.text(0, 1.55, "ONE EVENT, TWO WINDOWS", ha="center", fontsize=13, color=PRIMARY, fontweight="bold")
    ax.text(0, 1.28, "state transition", ha="center", fontsize=9.5, color=TEXT_MUTED)
    ax.text(-4, -1.15, "Ns cycles before", ha="center", fontsize=10, color=TEXT_MUTED, fontweight="bold")
    ax.text(4, -1.15, "Ns cycles after", ha="center", fontsize=10, color=AI_ACCENT, fontweight="bold")

    ax.axhline(-1.55, color="#E5E7EB", linewidth=1.0, xmin=0.02, xmax=0.98)
    ax.text(
        0,
        -1.85,
        "activation current = after \u2212 before, aligned at the voltage zero-crossing",
        ha="center",
        fontsize=9.5,
        color=PRIMARY,
    )

    ax.set_xlim(-8, 8)
    ax.set_ylim(-2.1, 1.75)
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig


def _icon_box(ax: plt.Axes, center: tuple[float, float], size: float, color: str, *, is_circle: bool = False) -> None:
    """An empty icon frame (box or circle) at `center`; the caller draws its own glyph on top."""
    if is_circle:
        ax.add_patch(Circle(center, size / 2, facecolor="none", edgecolor=color, linewidth=1.8))
    else:
        ax.add_patch(
            FancyBboxPatch(
                (center[0] - size / 2, center[1] - size / 2),
                size,
                size,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.8,
                edgecolor=color,
                facecolor="none",
            )
        )


def nilm_wrg_pipeline_diagram() -> plt.Figure:
    """Draw the activation-to-label pipeline: window, distance matrix, WRG image, CNN, label.

    An activation window becomes a pairwise distance matrix, which
    becomes a weighted recurrence graph image, which a small CNN
    classifies directly, the same three-layer network Chapter 2 reuses
    for the plain voltage-current (V-I) image and the amplitude-weighted
    variant (AWRG) alike. Five stages, one flow, no step skipped.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(7.6, 3.4))

    xs = [0.7, 2.6, 4.5, 6.4, 8.1]
    y = 2.0
    size = 1.15

    _icon_box(ax, (xs[0], y), size, AI_ACCENT)
    t = np.linspace(0, 1, 40)
    wave = 0.32 * np.sin(2 * np.pi * 2.2 * t) * np.exp(-1.4 * t)
    ax.plot(xs[0] - 0.4 + t * 0.8, y + wave, color=AI_ACCENT, linewidth=1.4)

    _icon_box(ax, (xs[1], y), size, TEXT_MUTED)
    rng = np.random.default_rng(3)
    grid_vals = rng.uniform(0.15, 0.55, (4, 4))
    for r in range(4):
        for c in range(4):
            ax.add_patch(
                plt.Rectangle(
                    (xs[1] - 0.44 + c * 0.22, y + 0.44 - r * 0.22 - 0.2),
                    0.2,
                    0.2,
                    facecolor=TEXT_MUTED,
                    alpha=grid_vals[r, c],
                    edgecolor="none",
                )
            )

    _icon_box(ax, (xs[2], y), size, AI_ACCENT)
    grid_vals2 = rng.uniform(0.25, 0.85, (3, 3))
    for r in range(3):
        for c in range(3):
            ax.add_patch(
                plt.Rectangle(
                    (xs[2] - 0.33 + c * 0.22, y + 0.33 - r * 0.22 - 0.2),
                    0.2,
                    0.2,
                    facecolor=AI_ACCENT,
                    alpha=grid_vals2[r, c],
                    edgecolor="none",
                )
            )

    _icon_box(ax, (xs[3], y), size, TEXT_MUTED)
    for dx, w in zip([-0.3, 0.02, 0.28], [0.2, 0.14, 0.1], strict=True):
        ax.add_patch(
            plt.Rectangle((xs[3] + dx, y - 0.38), w, 0.76, facecolor="none", edgecolor=TEXT_MUTED, linewidth=1.4)
        )

    _icon_box(ax, (xs[4], y), size, AI_ACCENT, is_circle=True)
    ax.plot(
        [xs[4] - 0.28, xs[4] - 0.06, xs[4] + 0.32],
        [y - 0.02, y - 0.25, y + 0.28],
        color=AI_ACCENT,
        linewidth=2.2,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    labels = [
        ("Activation\nwindow", AI_ACCENT),
        ("Distance\nmatrix D", PRIMARY),
        ("Weighted\nrecurrence graph", AI_ACCENT),
        ("CNN\nclassifier", PRIMARY),
        ("Appliance", PRIMARY),
    ]
    for x, (label, color) in zip(xs, labels, strict=True):
        ax.text(x, y - 0.95, label, ha="center", va="top", fontsize=9, color=color, fontweight="bold")

    for x0, x1 in zip(xs[:-1], xs[1:], strict=True):
        ax.plot([x0 + size / 2, x1 - size / 2], [y, y], color="#CBD5E1", linewidth=1.4, zorder=0)

    ax.text(
        (xs[0] + xs[-1]) / 2,
        3.15,
        "FROM ONE ACTIVATION TO ONE LABEL",
        ha="center",
        fontsize=13,
        color=PRIMARY,
        fontweight="bold",
    )

    ax.axhline(0.35, color="#E5E7EB", linewidth=1.0, xmin=0.02, xmax=0.98)
    ax.text(
        (xs[0] + xs[-1]) / 2,
        0.0,
        "the same three-layer CNN classifies WRG, V-I, and AWRG images alike",
        ha="center",
        fontsize=9.5,
        color=PRIMARY,
    )

    ax.set_xlim(-0.2, 8.9)
    ax.set_ylim(-0.4, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig


def nilm_softmax_vs_sigmoid_diagram() -> plt.Figure:
    """Draw sigmoid-plus-threshold against the chapter's own two-logit softmax.

    Top: the usual approach, one logit through a sigmoid, giving a
    probability that only becomes a decision once an external 0.5 cutoff,
    fixed after training, is bolted on. Bottom: this chapter's approach,
    an off-logit and an on-logit through a two-way softmax, where the
    decision boundary is learned as part of training rather than assumed
    afterward.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6))

    # ---- top panel: sigmoid + threshold ----
    ax = axes[0]
    ax.set_title("SIGMOID + THRESHOLD", fontsize=13, color=PRIMARY, fontweight="bold", pad=14)
    ax.text(
        4.4,
        2.15,
        "the usual approach: one logit, a cutoff bolted on after training",
        ha="center",
        fontsize=9.5,
        color=TEXT_MUTED,
        style="italic",
    )
    _flow_box(ax, (0.1, 0.85), 1.4, 0.9, "logit", color=PRIMARY, filled=False)
    _flow_arrow(ax, (1.5, 1.3), (2.0, 1.3), color=TEXT_MUTED)
    _icon_circle(ax, (2.55, 1.3), 0.55, INFO)
    ax.text(2.55, 1.3, r"$\sigma$", ha="center", va="center", fontsize=17, color=INFO)
    _flow_arrow(ax, (3.1, 1.3), (3.6, 1.3), color=TEXT_MUTED)
    _pill_box(ax, (3.65, 0.95), 1.9, 0.7, "P(on) = 0.62", color=PRIMARY)
    ax.axvline(6.05, color=TEXT_MUTED, linestyle="dashed", linewidth=1.4, ymin=0.2, ymax=0.85)
    ax.text(6.05, 0.05, "0.5 cutoff, fixed", ha="center", fontsize=8.5, color=TEXT_MUTED, style="italic")
    _flow_arrow(ax, (5.55, 1.3), (6.35, 1.3), color=TEXT_MUTED)
    _pill_box(ax, (6.4, 0.95), 2.0, 0.7, "Decision: ON", color=PRIMARY)
    ax.set_xlim(-0.2, 8.8)
    ax.set_ylim(0.0, 2.5)
    ax.axis("off")

    # ---- bottom panel: two-logit softmax ----
    ax = axes[1]
    ax.set_title("TWO-LOGIT SOFTMAX", fontsize=13, color=AI_ACCENT, fontweight="bold", pad=14)
    ax.text(
        4.4,
        2.15,
        "this chapter's approach: the cutoff is learned, not bolted on",
        ha="center",
        fontsize=9.5,
        color=AI_ACCENT,
        style="italic",
    )
    _pill_box(ax, (0.0, 1.55), 1.5, 0.55, "off logit", color=AI_ACCENT)
    _pill_box(ax, (0.0, 0.75), 1.5, 0.55, "on logit", color=AI_ACCENT)
    _curved_flow_arrow(ax, (1.5, 1.83), (2.0, 1.55), color=AI_ACCENT, rad=-0.2)
    _curved_flow_arrow(ax, (1.5, 1.03), (2.0, 1.3), color=AI_ACCENT, rad=0.2)
    _flow_box(ax, (2.05, 0.95), 1.6, 0.9, "softmax\n(2-way)", color=AI_ACCENT)
    _flow_arrow(ax, (3.65, 1.4), (4.15, 1.4), color=AI_ACCENT)
    _pill_box(ax, (4.2, 1.05), 2.55, 0.7, "P(off)=0.35, P(on)=0.65", color=AI_ACCENT)
    _flow_arrow(ax, (6.75, 1.4), (7.25, 1.4), color=AI_ACCENT)
    _pill_box(ax, (7.3, 1.05), 2.0, 0.7, "Decision: ON", color=AI_ACCENT)
    ax.set_xlim(-0.2, 9.6)
    ax.set_ylim(0.4, 2.5)
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _tau_ticks(ax: plt.Axes, y: float, taus: list[float], color: str, *, x0: float = 0.0, width: float = 4.0) -> None:
    """A row of tick marks on a [0, 1] axis at the given tau levels; the shared idiom both panels below use."""
    ax.plot([x0, x0 + width], [y, y], color=TEXT_MUTED, linewidth=1.2, zorder=1)
    for tau in taus:
        ax.plot([x0 + tau * width], [y], marker="|", markersize=14, markeredgewidth=2.2, color=color, zorder=2)
    ax.text(x0, y - 0.28, "0", fontsize=8, color=TEXT_MUTED, ha="center")
    ax.text(x0 + width, y - 0.28, "1", fontsize=8, color=TEXT_MUTED, ha="center")


def fpqr_vs_standard_qr_diagram() -> plt.Figure:
    """Draw standard (fixed-tau) quantile regression against FPQR's learned-tau version.

    Left: the common approach, a single Quantile Value Network (QVN) maps a
    context vector to quantile values at a handful of levels fixed by the
    user in advance, evenly spaced, chosen by convention rather than learned.
    Right: FPQR adds a Fraction Proposal Network (FPN) that learns where to
    place those levels from the same context vector, unevenly spaced,
    concentrating wherever the conditional distribution's own shape calls
    for finer resolution, before the QVN maps the learned levels to values.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.4))

    # ---- left panel: standard QR, fixed tau ----
    ax = axes[0]
    ax.set_title("Standard QR", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (1.0, 3.3), 2.0, 0.65, "Context $\\phi$", color=TEXT_MUTED)
    _flow_arrow(ax, (2.0, 3.3), (2.0, 2.55), color=INFO)
    _flow_box(ax, (0.7, 1.85), 2.6, 0.7, "QVN", color=INFO)
    _flow_arrow(ax, (2.0, 1.85), (2.0, 1.15), color=INFO)
    _tau_ticks(ax, 0.85, [0.05, 0.25, 0.5, 0.75, 0.95], color=TEXT_MUTED)
    ax.text(2.0, 0.35, "fixed by the user", fontsize=8.6, color=TEXT_MUTED, style="italic", ha="center")
    ax.text(
        2.0,
        4.35,
        "one network,\nfixed quantile levels",
        fontsize=8.8,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="bottom",
    )

    # ---- right panel: FPQR, learned tau ----
    ax = axes[1]
    ax.set_title("FPQR", fontsize=12, color=WARNING, fontweight="bold")
    _flow_box(ax, (1.0, 3.3), 2.0, 0.65, "Context $\\phi$", color=TEXT_MUTED)
    _flow_arrow(ax, (1.55, 3.3), (0.9, 2.55), color=WARNING)
    _flow_arrow(ax, (2.45, 3.3), (2.9, 2.55), color=INFO)
    _flow_box(ax, (-0.2, 1.85), 2.2, 0.7, "FPN", color=WARNING)
    _flow_box(ax, (2.0, 1.85), 2.0, 0.7, "QVN", color=INFO)
    _flow_arrow(ax, (0.9, 1.85), (2.4, 2.4), color=WARNING)
    _flow_arrow(ax, (3.0, 1.85), (3.0, 1.15), color=INFO)
    _tau_ticks(ax, 0.85, [0.04, 0.09, 0.22, 0.55, 0.97], color=WARNING)
    ax.text(2.9, 0.35, "learned per window", fontsize=8.6, color=WARNING, style="italic", ha="center")
    ax.text(
        2.9,
        4.35,
        "two networks,\nlearned quantile levels",
        fontsize=8.8,
        color=WARNING,
        style="italic",
        ha="center",
        va="bottom",
    )

    for ax in axes:
        ax.set_xlim(-0.6, 5.2)
        ax.set_ylim(0.0, 4.6)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def trained_specialist_vs_zero_shot_diagram() -> plt.Figure:
    """Draw the two ways a forecaster can come to know a population.

    Left: the trained specialist every earlier chapter in this part builds,
    a per-population pipeline, collect history, engineer lag and distance
    features, fit a model, that only knows the population it was fit on;
    a new population (NREL, say) means repeating every step. Right: a
    foundation model, pretrained once on a large general time-series corpus
    before this book ever saw the data, that consumes a customer's own raw
    context window directly and forecasts with no fitting step at all, on
    AusNet or on NREL alike.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.4))

    # ---- left panel: trained specialist, one pipeline per population ----
    ax = axes[0]
    ax.set_title("Trained specialist", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (0.9, 4.4), 2.2, 0.6, "Population history", color=INFO)
    _flow_arrow(ax, (2.0, 4.4), (2.0, 3.7), color=INFO)
    _flow_box(ax, (0.9, 3.05), 2.2, 0.6, "Lag + distance\nfeatures", color=INFO)
    _flow_arrow(ax, (2.0, 3.05), (2.0, 2.35), color=INFO)
    _flow_box(ax, (0.9, 1.7), 2.2, 0.6, "Fit one model", color=PRIMARY)
    _flow_arrow(ax, (2.0, 1.7), (2.0, 1.0), color=PRIMARY)
    ax.text(
        2.0,
        0.6,
        "Forecast, this\npopulation only",
        fontsize=8.8,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.add_patch(
        FancyArrowPatch(
            (3.3, 4.7),
            (3.3, 1.9),
            connectionstyle="arc3,rad=0.5",
            arrowstyle="-|>",
            mutation_scale=13,
            color=WARNING,
            linewidth=1.5,
        )
    )
    ax.text(
        3.85,
        3.3,
        "new\npopulation:\nrepeat",
        fontsize=7.6,
        color=WARNING,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # ---- right panel: zero-shot foundation model, one model for any population ----
    ax = axes[1]
    ax.set_title("Zero-shot foundation model", fontsize=12, color=PRIMARY, fontweight="bold")
    _flow_box(ax, (0.9, 4.4), 2.2, 0.6, "Pretrained once,\ngeneral corpus", color=TEXT_MUTED, filled=False)
    _flow_arrow(ax, (2.0, 4.4), (2.0, 3.7), color=TEXT_MUTED)
    _flow_box(ax, (0.9, 3.05), 2.2, 0.6, "Frozen model", color=SUCCESS)
    _flow_arrow(ax, (0.5, 1.7), (1.75, 2.95), color=SUCCESS)
    _flow_box(ax, (-0.6, 1.1), 2.2, 0.6, "AusNet customer's\nown context window", color=SUCCESS, filled=False)
    _flow_arrow(ax, (3.5, 1.7), (2.25, 2.95), color=SUCCESS)
    _flow_box(ax, (2.4, 1.1), 2.2, 0.6, "NREL customer's\nown context window", color=SUCCESS, filled=False)
    _flow_arrow(ax, (2.0, 3.05), (2.0, 0.4), color=SUCCESS)
    ax.text(
        2.0,
        0.05,
        "Forecast, either population,\nno fitting step",
        fontsize=8.8,
        color=SUCCESS,
        fontweight="bold",
        ha="center",
        va="center",
    )

    for ax in axes:
        ax.set_xlim(-1.0, 5.0)
        ax.set_ylim(-0.3, 5.2)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def _net_injection_panel(
    ax: plt.Axes,
    title: str,
    demand_height: float,
    net_color: str,
    voltage_label: str,
) -> None:
    """One panel of `net_injection_offset_diagram`: one customer bus, PV export against local demand."""
    bus_xy = (0.5, 0.05)
    ax.add_patch(Circle(bus_xy, 0.16, facecolor=PRIMARY, edgecolor="white", linewidth=1.6, zorder=3))
    ax.text(
        bus_xy[0],
        bus_xy[1],
        ICONS["house-fill"],
        fontproperties=icon_font(13),
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )

    export_top = 0.05 + 0.95
    ax.add_patch(
        FancyArrowPatch(
            (0.25, bus_xy[1] + 0.18),
            (0.25, export_top),
            arrowstyle="-|>",
            mutation_scale=16,
            color=WARNING,
            linewidth=2.2,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(0.12, export_top - 0.05, "PV export", fontsize=7.8, color=WARNING, fontweight="bold", ha="right")

    ax.add_patch(
        FancyArrowPatch(
            (0.75, bus_xy[1] + 0.18 + demand_height),
            (0.75, bus_xy[1] + 0.18),
            arrowstyle="-|>",
            mutation_scale=16,
            color=INFO,
            linewidth=2.2,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        0.88,
        bus_xy[1] + 0.18 + demand_height + 0.03,
        "local\ndemand",
        fontsize=7.8,
        color=INFO,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    net_top = 0.05 + 0.95 - demand_height
    ax.add_patch(
        FancyArrowPatch(
            (0.5, bus_xy[1] + 0.18),
            (0.5, max(net_top, bus_xy[1] + 0.24)),
            arrowstyle="-|>",
            mutation_scale=20,
            color=net_color,
            linewidth=3.2,
            shrinkA=0,
            shrinkB=0,
        )
    )

    ax.text(
        0.5,
        1.14,
        f"net injection\n{voltage_label}",
        fontsize=8.4,
        color=net_color,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.set_title(title, fontsize=12, color=PRIMARY, fontweight="bold", pad=22)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.15, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def net_injection_offset_diagram() -> plt.Figure:
    """Draw why a customer's own local demand offsets PV export at the point of common coupling.

    Left: a low-demand customer, most of a fixed PV export flows onto the
    network unopposed, a large net injection and a large voltage rise.
    Right: the same fixed PV export, but a customer with high local demand
    consumes most of it before it ever reaches the network, a small net
    injection and a small voltage rise. The mechanism behind this chapter's
    own real finding, that a forecast's upper-demand quantile is not a
    uniformly worse case once real PV penetration is high enough.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))

    _net_injection_panel(
        axes[0], "Low local demand", demand_height=0.15, net_color=DANGER, voltage_label="large, $\\Delta V$ high"
    )
    _net_injection_panel(
        axes[1], "High local demand", demand_height=0.65, net_color=SUCCESS, voltage_label="small, $\\Delta V$ low"
    )

    fig.tight_layout()
    plt.close(fig)
    return fig


def _dsse_feeder_panel(ax: plt.Axes, title: str, customer_measured: list[bool]) -> None:
    """One panel of `pseudo_measurement_observability_diagram`: one substation, a row of customer buses."""
    sub_xy = (0.5, 1.5)
    ax.add_patch(Circle(sub_xy, 0.22, facecolor=PRIMARY, edgecolor="white", linewidth=1.6, zorder=3))
    ax.text(
        sub_xy[0],
        sub_xy[1],
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(16),
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )

    n = len(customer_measured)
    xs = np.linspace(0.3, 0.3 + 0.85 * (n - 1), n)
    for x, measured in zip(xs, customer_measured, strict=True):
        color = SUCCESS if measured else TEXT_MUTED
        ax.plot([sub_xy[0], x], [sub_xy[1] - 0.22, 0.32], color=color, linewidth=1.3, alpha=0.8, zorder=1)
        ax.add_patch(Circle((x, 0.15), 0.155, facecolor=color, edgecolor="white", linewidth=1.3, zorder=2))
        ax.text(
            x,
            0.15,
            ICONS["house-fill"],
            fontproperties=icon_font(10),
            color="white",
            ha="center",
            va="center",
            zorder=3,
        )
        if measured:
            badge_xy = (x + 0.13, 0.15 + 0.17)
            ax.add_patch(Circle(badge_xy, 0.09, facecolor=SUCCESS, edgecolor="white", linewidth=0.8, zorder=4))
            ax.text(
                badge_xy[0],
                badge_xy[1],
                ICONS["speedometer2"],
                fontproperties=icon_font(6.5),
                color="white",
                ha="center",
                va="center",
                zorder=5,
            )
        else:
            ax.text(x, 0.15, "?", fontsize=9, color="white", fontweight="bold", ha="center", va="center", zorder=5)

    ax.set_title(title, fontsize=12, color=PRIMARY, fontweight="bold", pad=10)
    ax.set_xlim(-0.05, xs[-1] + 0.35)
    ax.set_ylim(-0.15, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")


def clustering_paradigm_landscape_diagram() -> plt.Figure:
    """Draw where this chapter's five clustering methods sit relative to each other.

    Two axes, neither one a strict contrast: the horizontal axis is how a
    load curve gets represented before anything is clustered, a hand-
    engineered shape feature on the left, a learned embedding on the
    right; the vertical axis is the structure-discovery mechanism applied
    to that representation, a flat partition at the bottom, a
    hierarchical or two-stage grouping in the middle, a spectral or
    manifold view at the top. Shape-KMeans, this book's own established
    baseline, sits at the hand-engineered, partition corner. CROCS moves
    up the mechanism axis without changing representation, a two-stage
    grouping over the same kind of engineered daily-behavior features.
    Tucker decomposition and Chronos-2 embeddings both move right, a
    learned multi-way factorization and a learned foundation-model
    embedding respectively, each still clustered by a flat partition.
    Diffusion maps moves up instead, a nonlinear manifold view applied on
    top of whichever representation feeds it. No method in this chapter
    occupies more than one point in this space alone; the real question
    later sections ask is whether different points in this space agree
    on what counts as real structure.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    methods = [
        ("Shape k-means", 0.12, 0.14, PRIMARY),
        ("CROCS\n(RLS + WSMD)", 0.24, 0.56, INFO),
        ("Tucker\ndecomposition", 0.62, 0.20, SUCCESS),
        ("Chronos-2\nembeddings", 0.90, 0.26, AI_ACCENT),
        ("Diffusion maps", 0.66, 0.90, WARNING),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 6.0))

    ax.add_patch(FancyArrowPatch((0, 0), (1.05, 0), arrowstyle="-|>", mutation_scale=14, color=TEXT_MUTED, lw=1.2))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1.05), arrowstyle="-|>", mutation_scale=14, color=TEXT_MUTED, lw=1.2))

    ax.text(0.0, -0.08, "Hand-engineered\nfeature", fontsize=9, color=TEXT_MUTED, ha="left", va="top")
    ax.text(1.05, -0.08, "Learned\nembedding", fontsize=9, color=TEXT_MUTED, ha="right", va="top")
    ax.text(
        -0.06,
        0.0,
        "Partition",
        fontsize=9,
        color=TEXT_MUTED,
        ha="right",
        va="bottom",
        rotation=90,
    )
    ax.text(
        -0.06,
        1.05,
        "Spectral / manifold",
        fontsize=9,
        color=TEXT_MUTED,
        ha="right",
        va="top",
        rotation=90,
    )
    ax.text(
        -0.06, 0.55, "Hierarchical /\ntwo-stage", fontsize=8.5, color=TEXT_MUTED, ha="right", va="center", rotation=90
    )

    for label, x, y, color in methods:
        ax.add_patch(Circle((x, y), 0.028, facecolor=color, edgecolor="white", linewidth=1.2, zorder=3))
        label_dy = 0.11 if y < 0.8 else -0.11
        va = "bottom" if y < 0.8 else "top"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x, y + label_dy),
            fontsize=9,
            color=color,
            fontweight="bold",
            ha="center",
            va=va,
        )

    ax.set_title("Where this chapter's five methods sit", fontsize=12.5, color=PRIMARY, fontweight="bold", pad=14)
    ax.set_xlim(-0.32, 1.18)
    ax.set_ylim(-0.22, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def pseudo_measurement_observability_diagram() -> plt.Figure:
    """Draw why smart-meter readings can turn an unobservable feeder into an observable one.

    Left: a substation with its own real telemetry (voltage and power), and
    every customer bus downstream genuinely unknown, no measurement of any
    kind, the real starting point for most real low-voltage feeders below
    the substation. Right: the same feeder, the same one real substation
    measurement, but every customer's own smart-meter reading now treated
    as a real pseudo-measurement, turning an unobservable network into one
    a real weighted-least-squares state estimator can actually solve.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    _dsse_feeder_panel(axes[0], "Substation telemetry only", [False, False, False, False, False])
    _dsse_feeder_panel(axes[1], "Smart-meter pseudo-measurements", [True, True, True, True, True])

    fig.tight_layout()
    plt.close(fig)
    return fig


def clustering_objective_families_diagram() -> plt.Figure:
    """Draw what each of the four clustering objective families actually sees.

    Four small panels, the same kind of question, four different answers.
    Partition: k-means finds three round blobs by distance to a centroid,
    marked with a star, and cannot represent a shape a centroid does not
    describe well. Hierarchical: the same points merge pairwise into a
    tree, a dendrogram, read at whatever height gives the number of
    groups wanted, not fixed in advance. Density-connectivity: DBSCAN
    finds two interleaved crescents no centroid could separate, plus a
    few real noise points it refuses to force into either one. Spectral:
    a similarity graph gets cut into two groups by removing the fewest,
    lightest edges, a global graph property no local distance rule sees.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))

    # Partition: three round blobs, centroid marked
    ax = axes[0, 0]
    centers = [(-1.1, 0.7), (1.1, 0.6), (0.0, -1.1)]
    colors = [PRIMARY, INFO, SUCCESS]
    for (cx, cy), color in zip(centers, colors, strict=True):
        pts = rng.normal(loc=(cx, cy), scale=0.24, size=(22, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color=color, alpha=0.75, edgecolor="none")
        ax.scatter([cx], [cy], marker="*", s=180, color=color, edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_title("Partition (k-means)", fontsize=10.5, color=PRIMARY, fontweight="bold")

    # Hierarchical: a small dendrogram
    ax = axes[0, 1]
    leaf_x = [0, 1, 2, 3, 4]
    leaf_y = 0.0
    for x in leaf_x:
        ax.plot([x, x], [leaf_y, 0.35], color=TEXT_MUTED, linewidth=1.3)
    merges = [((0, 1), 0.35, 0.6), ((3, 4), 0.35, 0.75), ((0.5, 2), 0.6, 1.15), ((1.25, 3.5), 1.15, 1.6)]
    for (a, b), y0, y1 in merges:
        ax.plot([a, a], [y0, y1], color=WARNING, linewidth=1.6)
        ax.plot([b, b], [y0, y1], color=WARNING, linewidth=1.6)
        ax.plot([a, b], [y1, y1], color=WARNING, linewidth=1.6)
    ax.axhline(0.9, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax.text(4.15, 1.0, "cut height", fontsize=7.5, color=TEXT_MUTED, va="bottom")
    ax.set_xlim(-0.5, 5.3)
    ax.set_ylim(-0.1, 1.85)
    ax.set_title("Hierarchical (linkage)", fontsize=10.5, color=PRIMARY, fontweight="bold")

    # Density-connectivity: two interleaved crescents plus noise
    ax = axes[1, 0]
    theta1 = rng.uniform(0.1, np.pi - 0.1, 60)
    moon1 = np.column_stack([np.cos(theta1), np.sin(theta1)]) + rng.normal(scale=0.04, size=(60, 2))
    theta2 = rng.uniform(0.1, np.pi - 0.1, 60)
    moon2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5]) + rng.normal(scale=0.04, size=(60, 2))
    noise = rng.uniform([-0.8, -1.0], [1.8, 1.3], size=(6, 2))
    ax.scatter(moon1[:, 0], moon1[:, 1], s=14, color=PRIMARY, alpha=0.75, edgecolor="none")
    ax.scatter(moon2[:, 0], moon2[:, 1], s=14, color=INFO, alpha=0.75, edgecolor="none")
    ax.scatter(noise[:, 0], noise[:, 1], s=30, color=DANGER, linewidth=1.3, marker="x")
    ax.set_title("Density-connectivity (DBSCAN)", fontsize=10.5, color=PRIMARY, fontweight="bold")

    # Spectral: a small graph cut into two groups
    ax = axes[1, 1]
    left_nodes = [(-1.1, 0.5), (-1.4, -0.2), (-0.6, -0.5)]
    right_nodes = [(0.9, 0.4), (1.3, -0.2), (0.7, -0.6)]
    within_edges = [(0, 1), (1, 2), (0, 2)]
    for a, b in within_edges:
        ax.plot(*zip(left_nodes[a], left_nodes[b], strict=True), color=PRIMARY, linewidth=1.3, zorder=1)
        ax.plot(*zip(right_nodes[a], right_nodes[b], strict=True), color=INFO, linewidth=1.3, zorder=1)
    cut_edges = [(left_nodes[2], right_nodes[2]), (left_nodes[0], right_nodes[0])]
    for p1, p2 in cut_edges:
        ax.plot(*zip(p1, p2, strict=True), color=DANGER, linewidth=1.2, linestyle=":", zorder=1)
    for x, y in left_nodes:
        ax.add_patch(Circle((x, y), 0.09, facecolor=PRIMARY, edgecolor="white", linewidth=0.8, zorder=3))
    for x, y in right_nodes:
        ax.add_patch(Circle((x, y), 0.09, facecolor=INFO, edgecolor="white", linewidth=0.8, zorder=3))
    ax.text(0, 0.95, "cut here", fontsize=7.5, color=DANGER, ha="center")
    ax.set_title("Graph / spectral (Ncut)", fontsize=10.5, color=PRIMARY, fontweight="bold")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def choosing_k_diagram() -> plt.Figure:
    """Draw why an elbow curve and a silhouette curve do not read the same way.

    Left: distortion (within-cluster distance) against K, decreasing
    monotonically by construction, with a real elbow near K=3 that is
    faint enough to miss entirely. Right: the silhouette score for the
    same clustering runs, peaking sharply and unambiguously at the same
    K=3, the more reliable of the two signals for exactly the reason the
    left panel is easy to misread.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    k = np.arange(2, 9)
    distortion = 8.0 / k + 0.35 * np.log(k)
    silhouette = np.array([0.42, 0.71, 0.55, 0.46, 0.41, 0.37, 0.33])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    ax = axes[0]
    ax.plot(k, distortion, marker="o", color=PRIMARY, linewidth=1.8, markersize=5)
    ax.axvline(3, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax.annotate(
        "elbow here,\neasy to miss",
        xy=(3, distortion[1]),
        xytext=(5.0, distortion[1] + 0.9),
        fontsize=8.5,
        color=WARNING,
        arrowprops={"arrowstyle": "-", "color": WARNING, "linewidth": 0.9},
    )
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("distortion", fontsize=9)
    ax.set_title("Distortion vs K", fontsize=11, color=PRIMARY, fontweight="bold")

    ax = axes[1]
    ax.plot(k, silhouette, marker="o", color=SUCCESS, linewidth=1.8, markersize=5)
    ax.axvline(3, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax.annotate(
        "clear, unambiguous\npeak",
        xy=(3, silhouette[1]),
        xytext=(5.2, silhouette[1] - 0.08),
        fontsize=8.5,
        color=SUCCESS,
        arrowprops={"arrowstyle": "-", "color": SUCCESS, "linewidth": 0.9},
    )
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("silhouette score", fontsize=9)
    ax.set_title("Silhouette vs K", fontsize=11, color=PRIMARY, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def dtw_vs_euclidean_diagram() -> plt.Figure:
    """Draw why a shifted-but-similar routine needs more than Euclidean distance.

    Left: two evening-routine-shaped curves, one a few time steps later
    than the other, compared point by point. The straight vertical
    matching lines connect a peak on one curve to a trough on the other,
    exactly the false mismatch Euclidean distance reports. Right: the
    same two curves compared by dynamic time warping, whose matching
    lines bend to align each real peak and trough regardless of the
    time shift, reporting the low distance the shared shape deserves.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    t = np.linspace(0, 10, 60)
    curve_a = np.exp(-((t - 5.0) ** 2) / 3.0) + 0.15 * np.exp(-((t - 1.5) ** 2) / 1.0)
    curve_b = np.exp(-((t - 6.2) ** 2) / 3.0) + 0.15 * np.exp(-((t - 2.7) ** 2) / 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    ax = axes[0]
    ax.plot(t, curve_a, color=PRIMARY, linewidth=1.8, label="Customer A")
    ax.plot(t, curve_b, color=INFO, linewidth=1.8, label="Customer B")
    for i in range(0, 60, 6):
        ax.plot([t[i], t[i]], [curve_a[i], curve_b[i]], color=DANGER, linewidth=0.8, alpha=0.8, zorder=1)
    ax.set_title("Euclidean: point by point", fontsize=11, color=PRIMARY, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(t, curve_a, color=PRIMARY, linewidth=1.8)
    ax.plot(t, curve_b, color=INFO, linewidth=1.8)
    shift_steps = 7
    for i in range(0, 60, 6):
        j = min(max(i + shift_steps, 0), 59)
        ax.plot([t[i], t[j]], [curve_a[i], curve_b[j]], color=SUCCESS, linewidth=0.8, alpha=0.8, zorder=1)
    ax.set_title("DTW: warped alignment", fontsize=11, color=PRIMARY, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("time of day", fontsize=9)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def must_link_cannot_link_diagram() -> plt.Figure:
    """Draw how background knowledge overrides a distance function alone.

    A single population where distance alone would draw the cluster
    boundary right through two customers known, from real network
    topology, to share one physical feeder. A must-link constraint
    forces them together regardless of their own distance; a
    cannot-link constraint, drawn between two nearby points known to sit
    on different feeders, forbids the opposite mistake.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(11)
    group_a = rng.normal(loc=(-0.5, 0.25), scale=0.3, size=(14, 2))
    group_b = rng.normal(loc=(0.5, -0.25), scale=0.3, size=(14, 2))

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(group_a[:, 0], group_a[:, 1], s=34, color=PRIMARY, alpha=0.8, edgecolor="none", label="Feeder A")
    ax.scatter(group_b[:, 0], group_b[:, 1], s=34, color=INFO, alpha=0.8, edgecolor="none", label="Feeder B")

    # must-link: the two farthest-apart points within the same real feeder,
    # exactly the pair a distance function alone would be tempted to split.
    dist_within_a = np.linalg.norm(group_a[:, None, :] - group_a[None, :, :], axis=-1)
    i_a, j_a = np.unravel_index(np.argmax(dist_within_a), dist_within_a.shape)
    must_link = (group_a[i_a], group_a[j_a])
    ax.plot(*zip(must_link[0], must_link[1], strict=True), color=SUCCESS, linewidth=2.2, zorder=4)
    mid_must = (must_link[0] + must_link[1]) / 2
    ax.annotate(
        "must-link\n(same real feeder)",
        xy=tuple(mid_must),
        xytext=(mid_must[0] - 0.1, mid_must[1] + 0.75),
        fontsize=8.5,
        color=SUCCESS,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": SUCCESS, "linewidth": 0.9},
    )

    # cannot-link: the closest cross-feeder pair, exactly what a distance
    # function alone would be tempted to merge.
    dist_cross = np.linalg.norm(group_a[:, None, :] - group_b[None, :, :], axis=-1)
    i_b, j_b = np.unravel_index(np.argmin(dist_cross), dist_cross.shape)
    cannot_link = (group_a[i_b], group_b[j_b])
    ax.plot(*zip(cannot_link[0], cannot_link[1], strict=True), color=DANGER, linewidth=2.0, linestyle="--", zorder=4)
    mid_cannot = (cannot_link[0] + cannot_link[1]) / 2
    ax.annotate(
        "cannot-link\n(different real feeders)",
        xy=tuple(mid_cannot),
        xytext=(mid_cannot[0] + 0.15, mid_cannot[1] - 0.7),
        fontsize=8.5,
        color=DANGER,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": DANGER, "linewidth": 0.9},
    )

    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def trust_gated_composite_score_diagram() -> plt.Figure:
    """Draw why gating separation by trust must be multiplicative, not additive.

    Left: two illustrative candidates plotted on separation (x) vs. trust
    (y). Candidate A sits far right, near-perfect separation, but low on
    the y-axis; candidate B sits centrally, good but not exceptional
    separation, high trust. The shaded band below a veto threshold is
    Roy's own outranking concept: any candidate whose trust falls in that
    band is vetoed regardless of how far right it sits
    [@roy1990outranking]. Right: the same two candidates scored two ways.
    A flat additive average lets A's huge separation edge partially
    compensate for its trust deficit, enough here to win outright, a
    compensatory combination in Gilbride and Allenby's own sense
    [@gilbrideallenby2004screening]. The multiplicative gate this book
    uses instead, separation times trust, crushes A's score toward zero
    and correctly prefers B: no amount of separation can buy back a
    candidate trust has already vetoed.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    candidate_a = {"label": "Candidate A", "separation": 0.99, "trust": 0.02, "color": WARNING}
    candidate_b = {"label": "Candidate B", "separation": 0.45, "trust": 0.50, "color": INFO}
    veto_threshold = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={"width_ratios": [1.05, 0.95]})

    ax = axes[0]
    ax.set_title("Separation vs. trust", fontsize=11.5, color=PRIMARY, fontweight="bold")
    ax.axhspan(0, veto_threshold, color=DANGER, alpha=0.08, zorder=0)
    ax.axhline(veto_threshold, color=DANGER, linewidth=1.0, linestyle="dashed", alpha=0.6)
    ax.text(0.02, veto_threshold + 0.02, "veto threshold (Roy 1990)", fontsize=7.8, color=DANGER, va="bottom")

    for cand in (candidate_a, candidate_b):
        ax.scatter(
            [cand["separation"]],
            [cand["trust"]],
            s=180,
            color=cand["color"],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        ax.annotate(
            cand["label"],
            xy=(cand["separation"], cand["trust"]),
            xytext=(cand["separation"] - 0.02, cand["trust"] + 0.06),
            fontsize=9.5,
            color=cand["color"],
            fontweight="bold",
            ha="right",
        )

    ax.set_xlabel("Separation ensemble", fontsize=9.5, color=TEXT_MUTED)
    ax.set_ylabel("Trust factor", fontsize=9.5, color=TEXT_MUTED)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.08)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.set_title("Resulting composite score", fontsize=11.5, color=PRIMARY, fontweight="bold")
    rules = ["Additive\naverage", "Multiplicative\ngate (this book)"]
    scores_a = [
        (candidate_a["separation"] + candidate_a["trust"]) / 2,
        candidate_a["separation"] * candidate_a["trust"],
    ]
    scores_b = [
        (candidate_b["separation"] + candidate_b["trust"]) / 2,
        candidate_b["separation"] * candidate_b["trust"],
    ]
    x = np.arange(len(rules))
    width = 0.32
    ax.bar(x - width / 2, scores_a, width, color=candidate_a["color"], label=candidate_a["label"])
    ax.bar(x + width / 2, scores_b, width, color=candidate_b["color"], label=candidate_b["label"])
    for xi, (sa, sb) in enumerate(zip(scores_a, scores_b, strict=True)):
        ax.text(xi - width / 2, sa + 0.02, f"{sa:.2f}", ha="center", fontsize=8.5, color=candidate_a["color"])
        ax.text(xi + width / 2, sb + 0.02, f"{sb:.2f}", ha="center", fontsize=8.5, color=candidate_b["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=9.5)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Composite score", fontsize=9.5, color=TEXT_MUTED)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def same_blind_spot_diagram() -> plt.Figure:
    """Draw why silhouette, Davies-Bouldin, and Calinski-Harabasz all miss the same failure.

    Two scatters, deliberately unlabelled on the axes since neither is a
    real feature space, just illustrative points. Left: a real two-way
    split, evenly sized, well separated, silhouette 0.78. Right: the
    MAC000037 pattern this chapter opens with, one dominant group plus two
    isolated points, silhouette 0.99, the higher score of the two despite
    representing almost no one. Both numbers come from formulas built from
    the same two ingredients, within-cluster compactness and between-
    cluster separation; population share, the one thing that would tell
    these two scatters apart, appears in neither formula.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(3)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))

    ax = axes[0]
    ax.set_title("A real split", fontsize=11.5, color=PRIMARY, fontweight="bold")
    group_a = rng.normal(loc=(-1.4, 0.0), scale=0.35, size=(40, 2))
    group_b = rng.normal(loc=(1.4, 0.0), scale=0.35, size=(40, 2))
    ax.scatter(*group_a.T, s=26, color=SUCCESS, alpha=0.85, edgecolor="none")
    ax.scatter(*group_b.T, s=26, color=INFO, alpha=0.85, edgecolor="none")
    ax.text(0, -1.7, "silhouette 0.78 · 50% / 50%", fontsize=9.5, color=TEXT_MUTED, ha="center")

    ax = axes[1]
    ax.set_title("`MAC000037`'s split", fontsize=11.5, color=PRIMARY, fontweight="bold")
    main_group = rng.normal(loc=(0.0, 0.0), scale=0.5, size=(78, 2))
    ax.scatter(*main_group.T, s=22, color=INFO, alpha=0.75, edgecolor="none")
    outliers = np.array([[2.7, 1.8], [2.9, -2.0]])
    ax.scatter(*outliers.T, s=90, color=DANGER, edgecolor="white", linewidth=1.2, zorder=3)
    ax.text(0, -3.0, "silhouette 0.99 · 98.7% / 1.3%", fontsize=9.5, color=TEXT_MUTED, ha="center")

    for ax in axes:
        ax.set_xlim(-3.0, 3.6)
        ax.set_ylim(-3.6, 3.0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.text(
        0.5,
        0.03,
        "Same formulas, same blind spot: the higher score belongs to the fake split",
        fontsize=10.5,
        color=TEXT_MUTED,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    plt.close(fig)
    return fig


def separation_index_blind_spot_diagram() -> plt.Figure:
    """Draw why separation-based indices cannot tell a real split from an isolated outlier.

    The general-purpose version of :func:`same_blind_spot_diagram`, generic
    rather than tied to this book's own `MAC000037` finding, since it
    introduces the concept before any later chapter demonstrates it on
    real data. Left: a real, evenly sized two-way split. Right: one
    dominant group plus two isolated points, an illustrative population
    share, not a real book number. Both scatters are deliberately
    unlabelled on their axes; neither is a real feature space.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))

    ax = axes[0]
    ax.set_title("A real split", fontsize=11.5, color=PRIMARY, fontweight="bold")
    group_a = rng.normal(loc=(-1.4, 0.0), scale=0.35, size=(40, 2))
    group_b = rng.normal(loc=(1.4, 0.0), scale=0.35, size=(40, 2))
    ax.scatter(*group_a.T, s=26, color=SUCCESS, alpha=0.85, edgecolor="none")
    ax.scatter(*group_b.T, s=26, color=INFO, alpha=0.85, edgecolor="none")
    ax.text(0, -1.7, "well separated · 50% / 50%", fontsize=9.5, color=TEXT_MUTED, ha="center")

    ax = axes[1]
    ax.set_title("An isolated outlier", fontsize=11.5, color=PRIMARY, fontweight="bold")
    main_group = rng.normal(loc=(0.0, 0.0), scale=0.5, size=(78, 2))
    ax.scatter(*main_group.T, s=22, color=INFO, alpha=0.75, edgecolor="none")
    outliers = np.array([[2.7, 1.8], [2.9, -2.0]])
    ax.scatter(*outliers.T, s=90, color=DANGER, edgecolor="white", linewidth=1.2, zorder=3)
    ax.text(0, -3.0, "also well separated · 98.7% / 1.3%", fontsize=9.5, color=TEXT_MUTED, ha="center")

    for ax in axes:
        ax.set_xlim(-3.0, 3.6)
        ax.set_ylim(-3.6, 3.0)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.text(
        0.5,
        0.03,
        "Same formulas, same blind spot: neither reads population share",
        fontsize=10.5,
        color=TEXT_MUTED,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    plt.close(fig)
    return fig


def _two_cluster_scatter(
    ax: plt.Axes,
    rng: np.random.Generator,
    loc1: tuple[float, float],
    loc2: tuple[float, float],
    n1: int,
    n2: int,
    *,
    scale: float = 0.3,
    s: float = 20,
    alpha: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Two illustrative point clouds, the shared unit every scene in `resampling_validity_diagram` is built from."""
    pts1 = rng.normal(loc=loc1, scale=scale, size=(n1, 2))
    pts2 = rng.normal(loc=loc2, scale=scale, size=(n2, 2))
    ax.scatter(*pts1.T, s=s, color=INFO, alpha=alpha, edgecolor="none", zorder=2)
    ax.scatter(*pts2.T, s=s, color=SUCCESS, alpha=alpha, edgecolor="none", zorder=2)
    return pts1, pts2


def _centroid_star(ax: plt.Axes, pts: np.ndarray, color: str, *, s: float = 220) -> tuple[float, float]:
    """A star marker at a point cloud's own mean, the shared "fitted centroid" unit."""
    center = pts.mean(axis=0)
    ax.scatter(*center, s=s, marker="*", color=color, edgecolor="white", linewidth=1.0, zorder=4)
    return float(center[0]), float(center[1])


_LABEL_BBOX = {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}


def _prediction_strength_panel(ax: plt.Axes, rng: np.random.Generator) -> None:
    """One real illustrative split: centroids from Half A predict Half B's own points."""
    ax.set_title("Prediction strength", fontsize=12, color=PRIMARY, fontweight="bold")

    _two_cluster_scatter(ax, rng, (0.6, 5.6), (2.5, 5.5), 14, 10, scale=0.28)
    ax.text(1.55, 6.2, "Full data", fontsize=9.5, color=PRIMARY, fontweight="bold", ha="center")
    ax.text(1.55, 4.95, ICONS["scissors"], fontproperties=icon_font(20), color=PRIMARY, ha="center", va="center")
    ax.text(
        2.35,
        4.95,
        "split in two",
        fontsize=7.3,
        color=TEXT_MUTED,
        ha="left",
        va="center",
        style="italic",
        bbox=_LABEL_BBOX,
    )
    _curved_flow_arrow(ax, (1.05, 4.65), (-0.55, 3.55), PRIMARY, rad=-0.2, linewidth=1.3)
    _curved_flow_arrow(ax, (2.05, 4.65), (3.6, 3.55), PRIMARY, rad=0.2, linewidth=1.3)

    a1, a2 = _two_cluster_scatter(ax, rng, (-1.15, 3.05), (0.15, 2.95), 7, 5, scale=0.24, s=18)
    ax.text(-0.5, 3.65, "Half A", fontsize=9, color=INFO, fontweight="bold", ha="center")
    star_a1 = _centroid_star(ax, a1, INFO)
    star_a2 = _centroid_star(ax, a2, SUCCESS)
    ax.text(-0.5, 2.2, "centroids from A", fontsize=7, color=TEXT_MUTED, ha="center", style="italic")

    _two_cluster_scatter(ax, rng, (3.25, 3.05), (4.55, 2.95), 7, 5, scale=0.24, s=18)
    ax.text(3.9, 3.65, "Half B", fontsize=9, color=INFO, fontweight="bold", ha="center")
    _curved_flow_arrow(ax, star_a1, (3.25, 3.05), PRIMARY, rad=-0.22, linewidth=1.4)
    _curved_flow_arrow(ax, star_a2, (4.55, 2.95), PRIMARY, rad=0.22, linewidth=1.4)
    ax.text(
        1.7,
        2.75,
        "predict nearest\ncentroid",
        fontsize=7.3,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=_LABEL_BBOX,
    )

    ax.text(
        3.9, 1.75, ICONS["check-circle-fill"], fontproperties=icon_font(16), color=SUCCESS, ha="center", va="center"
    )
    ax.text(3.9, 1.15, "B's own pairs agree\n→ high prediction strength", fontsize=7, color=SUCCESS, ha="center")

    ax.set_xlim(-2.1, 5.5)
    ax.set_ylim(0.85, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _cluster_wise_stability_panel(ax: plt.Axes, rng: np.random.Generator) -> None:
    """One real illustrative bootstrap: a fragile cluster C either survives resampling or does not."""
    ax.set_title("Cluster-wise stability", fontsize=12, color=PRIMARY, fontweight="bold")

    _two_cluster_scatter(ax, rng, (0.7, 5.6), (2.6, 5.5), 14, 10, scale=0.28)
    fragile_home = np.array([4.55, 5.35])
    ax.scatter(*fragile_home, s=55, color=DANGER, edgecolor="white", linewidth=1.0, zorder=3)
    ax.add_patch(Circle(fragile_home, 0.32, facecolor="none", edgecolor=DANGER, linewidth=1.1, linestyle="dashed"))
    ax.text(4.55, 5.82, "cluster $C$", fontsize=7.5, color=DANGER, fontweight="bold", ha="center", style="italic")
    ax.text(1.65, 6.2, "Full data", fontsize=9.5, color=PRIMARY, fontweight="bold", ha="center")

    ax.text(0.75, 4.4, ICONS["dice-5-fill"], fontproperties=icon_font(20), color=WARNING, ha="center", va="center")
    ax.text(
        2.55,
        4.4,
        "bootstrap resample\n(with replacement)",
        fontsize=7.3,
        color=TEXT_MUTED,
        ha="left",
        va="center",
        style="italic",
        bbox=_LABEL_BBOX,
    )
    _flow_arrow(ax, (1.65, 5.05), (1.65, 3.75), color=WARNING)

    r1, r2 = _two_cluster_scatter(ax, rng, (0.7, 3.05), (2.6, 2.95), 14, 10, scale=0.28, s=18)
    faded_home = np.array([4.5, 3.15])
    ax.scatter(*faded_home, s=40, color=DANGER, alpha=0.35, edgecolor="none", zorder=2)
    ax.text(4.5, 3.5, "C's household:\npresent this time?", fontsize=6.6, color=DANGER, ha="center", style="italic")

    star_r1 = _centroid_star(ax, r1, INFO, s=170)
    ax.scatter([star_r1[0] + 0.34], [star_r1[1] - 0.08], s=55, marker="x", color=DANGER, linewidth=2.0, zorder=4)
    _centroid_star(ax, r2, SUCCESS, s=170)
    ax.text(
        1.65,
        1.85,
        "re-clustered: C's lone\nmember absorbed nearby",
        fontsize=6.6,
        color=TEXT_MUTED,
        ha="center",
        style="italic",
    )

    left_c, right_c = (0.95, 0.55), (1.95, 0.55)
    ax.add_patch(Circle(left_c, 0.42, facecolor="none", edgecolor=DANGER, linewidth=1.4))
    ax.add_patch(Circle(right_c, 0.42, facecolor="none", edgecolor=AI_ACCENT, linewidth=1.4))
    ax.add_patch(Circle((1.45, 0.55), 0.22, facecolor=AI_ACCENT, alpha=0.4, edgecolor="none"))
    ax.text(left_c[0] - 1.02, 0.55, "cluster $C$\n(original)", fontsize=6.4, color=DANGER, ha="center", va="center")
    ax.text(right_c[0] + 1.02, 0.55, "best match\n(resample)", fontsize=6.4, color=AI_ACCENT, ha="center", va="center")
    ax.text(
        1.45, -0.05, "Jaccard overlap = stability(C)", fontsize=7.3, color=AI_ACCENT, fontweight="bold", ha="center"
    )

    ax.set_xlim(-1.8, 5.9)
    ax.set_ylim(-0.35, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _stability_gauge_panel(ax: plt.Axes) -> None:
    """The two mechanisms' own real verdicts on this book's own data, not illustrative numbers."""
    ax.set_title(
        "Reading the result: the same idea, two different verdicts",
        fontsize=11.5,
        color=PRIMARY,
        fontweight="bold",
        pad=12,
    )

    y_ps = 1.55
    ax.add_patch(plt.Rectangle((0, y_ps - 0.13), 1.0, 0.26, facecolor=TEXT_MUTED, alpha=0.1, edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.8, y_ps - 0.13), 0.2, 0.26, facecolor=SUCCESS, alpha=0.2, edgecolor="none"))
    ax.text(-0.02, y_ps, "Prediction strength", fontsize=9.5, color=PRIMARY, fontweight="bold", ha="right", va="center")
    ax.text(
        0.9,
        y_ps - 0.32,
        "Tibshirani & Walther's\n0.8-0.9 cutoff",
        fontsize=6.5,
        color=SUCCESS,
        ha="center",
        style="italic",
    )
    for x, label, color, dy in [(1.0, "London\n1.000", SUCCESS, 0.4), (0.749, "AusNet\n0.749", WARNING, 0.4)]:
        ax.scatter([x], [y_ps], s=120, marker="D", color=color, edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(x, y_ps + dy, label, fontsize=7.3, color=color, fontweight="bold", ha="center", va="center")

    y_st = 0.25
    bands = [(0.0, 0.5, DANGER, "dissolved"), (0.5, 0.25, WARNING, "patterned"), (0.75, 0.25, SUCCESS, "stable")]
    for start, width, color, label in bands:
        ax.add_patch(plt.Rectangle((start, y_st - 0.13), width, 0.26, facecolor=color, alpha=0.18, edgecolor="none"))
        ax.text(start + width / 2, y_st - 0.32, label, fontsize=6.8, color=color, ha="center", style="italic")
    ax.text(
        -0.02, y_st, "Cluster-wise stability", fontsize=9.5, color=PRIMARY, fontweight="bold", ha="right", va="center"
    )
    for x, label, color, dy in [
        (1.0, "settled archetypes\n1.000", SUCCESS, 0.4),
        (0.63, "`MAC000037`'s cluster\n0.63", WARNING, 0.4),
    ]:
        ax.scatter([x], [y_st], s=120, marker="D", color=color, edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(x, y_st + dy, label, fontsize=7.3, color=color, fontweight="bold", ha="center", va="center")

    ax.set_xlim(-0.62, 1.14)
    ax.set_ylim(-0.35, 2.15)
    ax.axis("off")


def resampling_validity_diagram() -> plt.Figure:
    """Draw the two resampling mechanisms this chapter checks a clustering against, plus their real verdicts.

    Top-left: Tibshirani and Walther's prediction strength, drawn as an
    actual illustrative split rather than a labelled box: two synthetic
    clusters get cut into Half A and Half B, centroids fit on A predict
    B's own points, and B's own pairs either agree or they do not.
    Top-right: Hennig's cluster-wise stability, the same underlying idea
    run the other way: a fragile, near-singleton cluster C sits inside
    the full data, survives (or does not) a bootstrap resample, and gets
    scored by how much of itself it recovers. Bottom: the two mechanisms'
    own real verdicts on this book's own data, prediction strength
    comfortably above Tibshirani and Walther's 0.8-0.9 cutoff for London
    and just under it for AusNet, cluster-wise stability landing at 1.000
    for a genuine archetype and at 0.63, inside Hennig's own "patterned,
    not stable" band, for `MAC000037`'s isolated cluster.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(11.4, 7.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.3], hspace=0.12, wspace=0.15)

    _prediction_strength_panel(fig.add_subplot(gs[0, 0]), rng)
    _cluster_wise_stability_panel(fig.add_subplot(gs[0, 1]), rng)
    _stability_gauge_panel(fig.add_subplot(gs[1, :]))

    plt.close(fig)
    return fig


def resampling_mechanisms_diagram() -> plt.Figure:
    """Draw the two resampling mechanisms on their own, with no real-data verdict attached.

    The general-purpose, illustrative version of the top row of
    :func:`resampling_validity_diagram`, reusing the same two panel
    functions unchanged. This one carries no book-specific numbers, since
    it introduces the two mechanisms themselves, before any later chapter
    applies them to this book's own real data.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(11)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    _prediction_strength_panel(axes[0], rng)
    _cluster_wise_stability_panel(axes[1], rng)

    fig.tight_layout()
    plt.close(fig)
    return fig


def component_selection_diagram() -> plt.Figure:
    """Draw why a fixed variance threshold and profile likelihood can disagree on how many components to keep.

    One illustrative eigenvalue spectrum, one dominant direction followed
    by a long, smoothly decaying tail, the shape this book's own real
    load-shape spectra share. The 90%-cumulative-variance rule and Zhu and
    Ghodsi's profile-likelihood change point are marked at different
    points along that same curve, since the tail decays too smoothly for
    either method to agree on where it stops mattering by construction.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    rng = np.random.default_rng(4)
    p = 14
    eigenvalues = 9.0 * np.exp(-0.55 * np.arange(p)) + 0.15
    eigenvalues += rng.normal(0, 0.03, size=p)
    eigenvalues = np.sort(eigenvalues)[::-1]
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    variance_q = int(np.searchsorted(cumulative, 0.90) + 1)
    profile_q = 2  # illustrative: profile likelihood finds the one genuine change point early

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    components = np.arange(1, p + 1)
    ax.plot(components, eigenvalues, color=PRIMARY, linewidth=1.8, zorder=2)
    ax.scatter(components, eigenvalues, color=PRIMARY, s=32, zorder=3)

    ax.axvline(profile_q + 0.5, color=SUCCESS, linewidth=1.6, linestyle="--", zorder=1)
    ax.text(
        profile_q + 0.65,
        eigenvalues[0] * 0.92,
        f"profile likelihood\nstops at q={profile_q}",
        fontsize=9,
        color=SUCCESS,
        fontweight="bold",
        va="top",
    )
    ax.axvline(variance_q + 0.5, color=WARNING, linewidth=1.6, linestyle="--", zorder=1)
    ax.text(
        variance_q + 0.65,
        eigenvalues[0] * 0.55,
        f"90% variance rule\nstops at q={variance_q}",
        fontsize=9,
        color=WARNING,
        fontweight="bold",
        va="top",
    )

    ax.set_xlabel("Component index", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Eigenvalue", fontsize=10, color=TEXT_MUTED)
    ax.set_xticks(components)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)

    fig.tight_layout()
    plt.close(fig)
    return fig


def customer_feeder_network_aggregation_diagram() -> plt.Figure:
    """Draw the same clustering recipe applied at three real levels of aggregation.

    Bottom: nine real customers, three real archetypes (Chapter 3's own
    colors), each a peak-normalized daily shape. Middle: those customers
    group under two real feeders, a residential-heavy one and a
    commercial-heavy one, the aggregate shape each feeder's own customers
    sum to. Top: every feeder groups under one real network. The
    mechanism never changes, shape-based features, a validity-index scan,
    a resampling trust gate; only the unit being clustered does, one
    customer, one feeder, one network at a time.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell.
    """
    fig, ax = plt.subplots(figsize=(7.4, 6.6))

    customer_groups = [(0, 3, SUCCESS), (3, 6, INFO), (6, 9, WARNING)]
    xs = np.linspace(0.5, 8.5, 9)
    y_customer = 0.6
    customer_xy: list[tuple[float, float]] = []
    for x, color in zip(xs, [c for start, end, c in customer_groups for _ in range(end - start)], strict=True):
        ax.add_patch(Circle((x, y_customer), 0.32, facecolor=color, edgecolor="white", linewidth=1.3, zorder=3))
        ax.text(
            x, y_customer, ICONS["house-fill"], fontproperties=icon_font(13), color="white", ha="center", va="center"
        )
        customer_xy.append((x, y_customer))
    ax.text(4.5, y_customer - 0.7, "Customers: 3 archetypes (Chapter 3)", fontsize=8.6, color=TEXT_MUTED, ha="center")

    y_feeder = 3.1
    feeder_xy = [(2.5, y_feeder), (6.5, y_feeder)]
    feeder_labels = ["Feeder A\nresidential-heavy", "Feeder B\ncommercial-heavy"]
    feeder_colors = [SUCCESS, WARNING]
    for (fx, fy), label, color in zip(feeder_xy, feeder_labels, feeder_colors, strict=True):
        ax.add_patch(Circle((fx, fy), 0.5, facecolor=color, edgecolor="white", linewidth=1.6, zorder=3))
        ax.text(
            fx,
            fy,
            ICONS["lightning-charge-fill"],
            fontproperties=icon_font(18),
            color="white",
            ha="center",
            va="center",
        )
        ax.text(fx, fy - 0.85, label, fontsize=8.6, color=color, fontweight="bold", ha="center", va="top")
    for (cx, cy), (fx, fy) in zip(customer_xy, [feeder_xy[0]] * 5 + [feeder_xy[1]] * 4, strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (cx, cy + 0.32), (fx, fy - 0.5), arrowstyle="-", color=TEXT_MUTED, linewidth=0.8, alpha=0.55, zorder=1
            )
        )
    ax.text(6.9, y_feeder + 1.15, "Feeders: 2 archetypes (this chapter)", fontsize=8.6, color=TEXT_MUTED, ha="left")

    y_network = 5.1
    ax.add_patch(Circle((4.5, y_network), 0.62, facecolor=PRIMARY, edgecolor="white", linewidth=1.8, zorder=3))
    ax.text(
        4.5,
        y_network,
        ICONS["hdd-network-fill"],
        fontproperties=icon_font(20),
        color="white",
        ha="center",
        va="center",
    )
    ax.text(4.5, y_network + 0.85, "Network: real AusNet MV-LV topology", fontsize=8.6, color=TEXT_MUTED, ha="center")
    for fx, fy in feeder_xy:
        ax.add_patch(
            FancyArrowPatch(
                (fx, fy + 0.5),
                (4.5, y_network - 0.62),
                arrowstyle="-|>",
                mutation_scale=13,
                color=PRIMARY,
                linewidth=1.6,
                shrinkA=1,
                shrinkB=1,
                zorder=2,
            )
        )

    ax.text(
        4.5,
        6.55,
        "One recipe, three units of aggregation",
        fontsize=12.5,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
    )

    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(-0.3, 6.9)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def coe_aies_theory_of_change_diagram() -> plt.Figure:
    """Draw the Theory of Change behind the CoE-AIES concept note, VLIR-UOS style.

    Follows the VLIR-UOS Theory of Change template's own four named
    elements, not a generic Input-Activity-Output-Outcome chain: one
    root problem statement (Context), branching into exactly two Change
    Domains capped at two by that template's own convention, not three
    pillars plus a fourth capacity line, each Domain grouping the
    Pathways of Change (outputs and activities merged, the template's
    own term for that stage) that produce it, both converging on one
    Dream, the long-term SDG-tagged impact every Domain is meant to add
    up to. Assumptions and Context sit in their own labeled strip along
    the bottom, the conditions that have to hold for the whole chain to
    work, grounded in this book's own concept note (the TANESCO data-
    sharing dependency, the PI-structure note, the AMI-coverage gap),
    not invented placeholders. Built entirely from this file's own
    established idioms, `_icon_circle` for an unfilled badge frame,
    `ICONS`/`icon_font` for the glyph inside it, `_curved_flow_arrow`
    for the connective tissue, and `_grid_tier_zone` for each Domain's
    dashed grouping box, the same restrained, mostly-white, color-only-
    where-it-means-something style `resampling_validity_diagram` uses.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell or
        save to an SVG asset.
    """
    fig, ax = plt.subplots(figsize=(15.4, 8.6))

    # Shared vertical anchor: the midpoint between the two Change Domains'
    # own centers, so the Context and Dream nodes sit symmetrically between
    # the branch each domain takes, not accidentally offset toward one.
    mid_y = 5.08

    # ---- Context: the one root problem, branching into both Domains ----
    ax.text(1.1, 7.35, "Context", fontsize=10.5, color=DANGER, fontweight="bold", ha="center")
    _icon_circle(ax, (1.1, mid_y), 0.52, DANGER)
    ax.text(1.1, mid_y, ICONS["x-circle-fill"], fontproperties=icon_font(20), color=DANGER, ha="center", va="center")
    ax.text(
        1.1,
        mid_y - 0.8,
        "Too little visibility into\nthe grid, too little skill\nto act on it",
        fontsize=8.4,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="top",
    )

    _curved_flow_arrow(ax, (1.62, mid_y + 0.25), (2.95, 6.33), color=TEXT_MUTED, rad=0.15)
    _curved_flow_arrow(ax, (1.62, mid_y - 0.2), (2.95, 3.84), color=TEXT_MUTED, rad=-0.15)

    # ---- Pathways of Change, grouped into the two capped Change Domains ----
    # Domain 1 rows sit at 6.75/5.9 (spacing 0.85), its zone padded 0.3
    # above/below the icon edges; Domain 2 mirrors the same spacing and
    # padding lower down, with a 0.4 gap between the two zones so neither
    # dashed border crosses the other zone's text or the other zone itself.
    _grid_tier_zone(ax, (3.15, 5.28), 4.5, 2.09, "Change Domain 1 — grid & energy-access institutions")
    domain1_specs = [
        (6.75, ICONS["lightning-charge-fill"], INFO, "Fewer blind decisions\non the low-voltage grid"),
        (5.9, ICONS["sun-fill"], ENERGY_ACCENT, "Cheaper reliability for\noff-grid communities"),
    ]
    icon_x = 3.65
    for y, glyph, color, label in domain1_specs:
        _icon_circle(ax, (icon_x, y), 0.32, color)
        ax.text(icon_x, y, glyph, fontproperties=icon_font(13), color=color, ha="center", va="center")
        ax.text(icon_x + 0.5, y, label, fontsize=7.6, color=PRIMARY, ha="left", va="center")
    ax.text(
        8.05,
        6.33,
        "→ utilities and mini-grid operators\nplan instead of guess",
        fontsize=7.3,
        color=PRIMARY,
        style="italic",
        ha="left",
        va="center",
    )

    _grid_tier_zone(ax, (3.15, 2.79), 4.5, 2.09, "Change Domain 2 — local industrial & innovation capacity")
    domain2_specs = [
        (4.26, ICONS["building-fill"], WARNING, "Idle factory flexibility\nbecomes grid resilience"),
        (3.41, ICONS["arrow-down-up"], AI_ACCENT, "Capability that\noutlasts the grant"),
    ]
    for y, glyph, color, label in domain2_specs:
        _icon_circle(ax, (icon_x, y), 0.32, color)
        ax.text(icon_x, y, glyph, fontproperties=icon_font(13), color=color, ha="center", va="center")
        ax.text(icon_x + 0.5, y, label, fontsize=7.6, color=PRIMARY, ha="left", va="center")
    ax.text(
        8.05,
        3.84,
        "→ idle capacity becomes counted,\ntrusted flexibility",
        fontsize=7.3,
        color=PRIMARY,
        style="italic",
        ha="left",
        va="center",
    )

    _curved_flow_arrow(ax, (10.05, 6.33), (11.15, mid_y + 0.25), color=TEXT_MUTED, rad=-0.15)
    _curved_flow_arrow(ax, (10.05, 3.84), (11.15, mid_y - 0.2), color=TEXT_MUTED, rad=0.15)

    # ---- The Dream: the one SDG-tagged impact both Domains add up to ----
    ax.text(11.9, 7.35, "The Dream", fontsize=10.5, color=SUCCESS, fontweight="bold", ha="center")
    _icon_circle(ax, (11.9, mid_y), 0.55, SUCCESS)
    ax.text(
        11.9,
        mid_y,
        ICONS["check-circle-fill"],
        fontproperties=icon_font(21),
        color=SUCCESS,
        ha="center",
        va="center",
    )
    ax.text(
        11.9,
        mid_y - 0.9,
        "A self-sustaining, locally-led\ndigital capability for East\nAfrica's energy transition",
        fontsize=8.2,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="top",
    )
    ax.text(
        11.9,
        mid_y - 2.15,
        "SDG 7 — Affordable & Clean Energy\nSDG 9 — Industry, Innovation & Infrastructure",
        fontsize=7,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="top",
    )
    ax.text(
        11.9,
        mid_y - 2.95,
        "Cross-cutting: less diesel burned,\nless renewable energy curtailed",
        fontsize=7,
        color=ENERGY_ACCENT,
        style="italic",
        ha="center",
        va="top",
    )

    # ---- Assumptions & Context: the conditions the whole chain leans on ----
    _grid_tier_zone(ax, (0.6, 0.55), 13.8, 1.15, "Assumptions & context")
    assumption_specs = [
        (2.6, "TANESCO feeder-level\ndata-sharing agreement signed"),
        (7.5, "A full-time UDSM\nfaculty PI is secured"),
        (12.4, "AMI / smart-meter interval\ncoverage keeps expanding"),
    ]
    for x, label in assumption_specs:
        ax.text(x, 1.125, label, fontsize=7.4, color=TEXT_MUTED, style="italic", ha="center", va="center")

    ax.set_xlim(0.0, 14.4)
    ax.set_ylim(0.3, 7.7)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def salama_nafaka_theory_of_change_diagram() -> plt.Figure:
    """Draw the Theory of Change behind the Salama Nafaka AI concept note, VLIR-UOS style.

    Reuses the exact skeleton `coe_aies_theory_of_change_diagram` established
    (one root Context node branching into two capped Change Domains, each
    grouping its own Pathways of Change, converging on one SDG-tagged Dream,
    with an explicit Assumptions & Context strip), not a fresh layout
    invented per project. Only the content changes: Context is the concept
    note's own single unmeasured moment ("is this maize dry enough to store,
    right now?"); Domain 1 is farmer decision-making (the risk model plus the
    Swahili SMS/voice advisory that acts on it); Domain 2 is the market and
    storage systems around that decision (the storage-partner network plus
    the premium off-take linkage); the Dream is the concept note's own
    Phase-3 scale language (licensed across the maize belt, replicated via
    Smart Africa), tagged to SDG 2 and SDG 3; and the three assumptions are
    pulled directly from the note's own SWOT and Phase-1 dependencies (the
    lab partner, the AMCOS/supplier MOUs, a representative season), not
    invented placeholders.

    Returns:
        The matplotlib Figure, ready to display in a notebook cell or
        save to an SVG asset.
    """
    fig, ax = plt.subplots(figsize=(15.4, 8.6))

    mid_y = 5.08

    # ---- Context: the one unmeasured moment, branching into both Domains ----
    ax.text(1.1, 7.35, "Context", fontsize=10.5, color=DANGER, fontweight="bold", ha="center")
    _icon_circle(ax, (1.1, mid_y), 0.52, DANGER)
    ax.text(1.1, mid_y, ICONS["x-circle-fill"], fontproperties=icon_font(20), color=DANGER, ha="center", va="center")
    ax.text(
        1.1,
        mid_y - 0.8,
        "Is this maize dry enough\nto store, right now?\nNo one can answer that",
        fontsize=8.4,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="top",
    )

    _curved_flow_arrow(ax, (1.62, mid_y + 0.25), (2.95, 6.33), color=TEXT_MUTED, rad=0.15)
    _curved_flow_arrow(ax, (1.62, mid_y - 0.2), (2.95, 3.84), color=TEXT_MUTED, rad=-0.15)

    # ---- Pathways of Change, grouped into the two capped Change Domains ----
    _grid_tier_zone(ax, (3.15, 5.28), 4.5, 2.09, "Change Domain 1 — farmer decision-making")
    domain1_specs = [
        (6.75, ICONS["speedometer2"], INFO, "Weekly ward-level\nstorage-risk score"),
        (5.9, ICONS["record-circle-fill"], SUCCESS, "Swahili voice & SMS\nadvisory at harvest"),
    ]
    icon_x = 3.65
    for y, glyph, color, label in domain1_specs:
        _icon_circle(ax, (icon_x, y), 0.32, color)
        ax.text(icon_x, y, glyph, fontproperties=icon_font(13), color=color, ha="center", va="center")
        ax.text(icon_x + 0.5, y, label, fontsize=7.6, color=PRIMARY, ha="left", va="center")
    ax.text(
        8.05,
        6.33,
        "→ farmers know when grain is\nsafe to store, not guessing",
        fontsize=7.3,
        color=PRIMARY,
        style="italic",
        ha="left",
        va="center",
    )

    _grid_tier_zone(ax, (3.15, 2.79), 4.5, 2.09, "Change Domain 2 — market & storage systems")
    domain2_specs = [
        (4.26, ICONS["building-fill"], WARNING, "Safe-storage partner\nnetwork (bags, AMCOS)"),
        (3.41, ICONS["arrow-down-up"], AI_ACCENT, "Premium off-take\nfor verified safe grain"),
    ]
    for y, glyph, color, label in domain2_specs:
        _icon_circle(ax, (icon_x, y), 0.32, color)
        ax.text(icon_x, y, glyph, fontproperties=icon_font(13), color=color, ha="center", va="center")
        ax.text(icon_x + 0.5, y, label, fontsize=7.6, color=PRIMARY, ha="left", va="center")
    ax.text(
        8.05,
        3.84,
        "→ safe grain reaches a buyer\nwho pays for it, not a rejection",
        fontsize=7.3,
        color=PRIMARY,
        style="italic",
        ha="left",
        va="center",
    )

    _curved_flow_arrow(ax, (10.05, 6.33), (11.15, mid_y + 0.25), color=TEXT_MUTED, rad=-0.15)
    _curved_flow_arrow(ax, (10.05, 3.84), (11.15, mid_y - 0.2), color=TEXT_MUTED, rad=0.15)

    # ---- The Dream: the one SDG-tagged impact both Domains add up to ----
    ax.text(11.9, 7.35, "The Dream", fontsize=10.5, color=SUCCESS, fontweight="bold", ha="center")
    _icon_circle(ax, (11.9, mid_y), 0.55, SUCCESS)
    ax.text(
        11.9,
        mid_y,
        ICONS["check-circle-fill"],
        fontproperties=icon_font(21),
        color=SUCCESS,
        ha="center",
        va="center",
    )
    ax.text(
        11.9,
        mid_y - 0.9,
        "Safe storage licensed across\nTanzania's maize belt, replicated\nto other AU states via Smart Africa",
        fontsize=8.2,
        color=PRIMARY,
        fontweight="bold",
        ha="center",
        va="top",
    )
    ax.text(
        11.9,
        mid_y - 2.15,
        "SDG 2 — Zero Hunger\nSDG 3 — Good Health & Well-being",
        fontsize=7,
        color=TEXT_MUTED,
        style="italic",
        ha="center",
        va="top",
    )
    ax.text(
        11.9,
        mid_y - 2.95,
        "Cross-cutting: climate-resilient\ndecision-making, not a generic fix",
        fontsize=7,
        color=SUCCESS,
        style="italic",
        ha="center",
        va="top",
    )

    # ---- Assumptions & Context: the conditions the whole chain leans on ----
    _grid_tier_zone(ax, (0.6, 0.55), 13.8, 1.15, "Assumptions & context")
    assumption_specs = [
        (2.6, "Food-safety lab partner signed\nearly for ground-truth testing"),
        (7.5, "AMCOS, miller, and hermetic-bag\nsupplier MOUs signed"),
        (12.4, "A representative harvest season,\nnot unusually dry or wet"),
    ]
    for x, label in assumption_specs:
        ax.text(x, 1.125, label, fontsize=7.4, color=TEXT_MUTED, style="italic", ha="center", va="center")

    ax.set_xlim(0.0, 14.4)
    ax.set_ylim(0.3, 7.7)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig
