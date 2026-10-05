"""Regenerate Run 03 figures with publication layout fixes.

The completed experiment's plotting source is hash-frozen. This separate
presentation revision reuses that generator without changing its source,
observations, estimators, policy selection, or illustrative seed selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Annotation

from experiments.archive.paper_development import plot_05_policy_refresh as original
from experiments.archive.paper_development.module05_figures.style import Exporter, SEED_COLORS


MARKERS = ("o", "s", "^", "D", "P", "X")


def seed_legend(fig, seeds, *, distinct_markers=False):
    handles = [
        Line2D([], [], linestyle="none", color=color,
               marker=MARKERS[i] if distinct_markers else "o", markersize=4,
               label=f"Seed {seed}")
        for i, (seed, color) in enumerate(zip(seeds, SEED_COLORS))
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, -.075),
               ncol=len(seeds), frameon=False, columnspacing=1.3)


class PublicationExporter(Exporter):
    def __init__(self, directory, dpi=600):
        super().__init__(directory, dpi)
        self.seeds = json.loads((Path(directory).parent / "summary.json").read_text())["seeds"]

    def save(self, fig, name, title, caption, **kwargs):
        if name == "04_rl_seed_atlas":
            axes = [ax for ax in fig.axes if ax.images]
            last_cycles = []
            for ax in axes:
                recorded = np.isfinite(np.ma.asarray(ax.images[0].get_array()).filled(np.nan))
                rows = np.flatnonzero(recorded.any(axis=1))
                last_cycles.append(int(rows[-1] + 1) if rows.size else 0)
            width = max(30, 5 * ((max(last_cycles) + 4) // 5))
            for ax in axes:
                ax.set_ylim(.5, width + .5)
                ax.set_yticks([1, *range(5, width + 1, 5)])
            caption += " The shared cycle limit includes every recorded RL cycle across all six checkpoints."
        elif name == "06_rl_advantage":
            seed_legend(fig, self.seeds)
        elif name == "07_run02_vs_run03":
            for ax in fig.axes:
                for annotation in list(ax.texts):
                    if isinstance(annotation, Annotation):
                        annotation.remove()
                for collection, marker in zip(ax.collections, MARKERS):
                    style = MarkerStyle(marker)
                    collection.set_paths([style.get_path().transformed(style.get_transform())])
                    collection.set_sizes([30])
                    collection.set_edgecolors("white")
                    collection.set_linewidths(.35)
            seed_legend(fig, self.seeds, distinct_markers=True)
            caption += " Checkpoints are identified by both color and marker shape in the common legend."
        super().save(fig, name, title, caption, **kwargs)


def generate(output):
    output = Path(output)
    previous_exporter = original.Exporter
    original.Exporter = PublicationExporter
    try:
        provenance = original.generate(output)
    finally:
        original.Exporter = previous_exporter
    destination = output / "analysis/paper_figures_best_seed"
    provenance["layout_revision"] = {
        "script": "experiments/archive/paper_development/polish_05_policy_refresh_figures.py",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "changes": [
            "RL seed atlas uses a common limit containing all recorded RL cycles",
            "Direct advantage plot includes the seed color legend",
            "Run comparison uses distinct markers and a legend instead of overlapping seed labels",
        ],
        "numerical_data_and_estimators_unchanged": True,
        "frozen_experiment_sources_unchanged": True,
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(__file__, destination / "reproduction/experiments/paper_final" / Path(__file__).name)
    with zipfile.ZipFile(destination / "figure_pack.zip", "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != "figure_pack.zip":
                archive.write(path, path.relative_to(destination))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = generate(parser.parse_args().output)
    print(json.dumps({"figure_count": result["figure_count"],
                      "best_seed_selection": result["best_seed_selection"],
                      "layout_revision": result["layout_revision"]}, indent=2))
