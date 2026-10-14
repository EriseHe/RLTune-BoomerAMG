"""Pure plotting of recorded relaxation actions for the official online study."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt

DISPLAY_METHODS = ("bandit_lstdq",)
LABELS = {"bandit_lstdq": "LinUCB–LSTDQ"}


def _action_matrix(
    rows: Sequence[Dict[str, Any]],
    max_cycles: int,
    *,
    native_default_weight: float | None = None,
) -> np.ndarray:
    matrix = np.full((int(max_cycles), len(rows)), np.nan, dtype=float)
    for instance, row in enumerate(rows):
        outcome = row["outcome"]
        actions = np.asarray(outcome.get("cycle_actions", []), dtype=float)
        if (
            actions.size == 0
            and native_default_weight is not None
            and (outcome.get("native_status") is not None)
        ):
            actions = np.full(
                int(outcome.get("iterations", 0)),
                float(native_default_weight),
                dtype=float,
            )
        matrix[: min(actions.size, max_cycles), instance] = actions[:max_cycles]
    return matrix


def plot_action_trajectory_grid(
    rows_by_seed_method: Dict[tuple[int, str], list[Dict[str, Any]]],
    controller_seeds: Sequence[int],
    max_cycles: int,
    output: Path,
    *,
    action_min: float,
    action_max: float,
    methods: Sequence[str] = DISPLAY_METHODS,
    method_labels: Dict[str, str] | None = None,
    column_labels: Dict[int, str] | None = None,
    x_label: str = "Training instance",
    title: str = "Per-instance, per-cycle action trajectories (white: solve already terminated)",
    panel_width: float = 5.4,
    panel_height: float = 2.2,
    native_default_weight: float | None = None,
    activation_cases: Dict[str, int] | None = None,
) -> None:
    seeds = tuple((int(seed) for seed in controller_seeds))
    plotted_methods = tuple(methods)
    labels = LABELS if method_labels is None else method_labels
    (fig, axes) = plt.subplots(
        len(plotted_methods),
        len(seeds),
        figsize=(
            float(panel_width) * len(seeds),
            float(panel_height) * len(plotted_methods),
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    colormap = plt.colormaps["viridis"].copy()
    colormap.set_bad("white")
    image = None
    for row_index, method in enumerate(plotted_methods):
        for column_index, controller_seed in enumerate(seeds):
            axis = axes[row_index, column_index]
            rows = rows_by_seed_method[controller_seed, method]
            matrix = _action_matrix(
                rows, max_cycles, native_default_weight=native_default_weight
            )
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                vmin=float(action_min),
                vmax=float(action_max),
                cmap=colormap,
                extent=(0.5, len(rows) + 0.5, -0.5, max_cycles - 0.5),
            )
            axis.grid(False)
            activation_case = int((activation_cases or {}).get(method, 0))
            if activation_case > 0:
                boundary = float(activation_case) + 0.5
                axis.axvline(boundary, color="#d95f02", linestyle="--", linewidth=1.3)
                axis.text(
                    boundary,
                    0.97,
                    f"RL starts at {activation_case + 1:,}",
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                    color="#8c2d04",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 1.0,
                    },
                )
            if row_index == 0:
                axis.set_title(
                    f"Controller seed {controller_seed}"
                    if column_labels is None
                    else column_labels[int(controller_seed)]
                )
            if column_index == 0:
                axis.set_ylabel(f"{labels.get(method, method)}\nAMG cycle")
            if row_index == len(plotted_methods) - 1:
                axis.set_xlabel(x_label)
    if image is not None:
        fig.colorbar(
            image, ax=axes.ravel().tolist(), label="Relaxation weight w", shrink=0.92
        )
    fig.suptitle(title)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
