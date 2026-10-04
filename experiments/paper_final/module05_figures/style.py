"""Consistent two-column manuscript styling and figure exports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


COLORS = {"reference": "#777777", "fixed": "#454f5b", "oracle": "#9b609a",
          "periodic": "#d58b19", "prefix": "#00856c", "rl": "#2166ac",
          "native": "#8b6948", "chebyshev": "#ac503d"}
SEED_COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#555555"]
BLANK = "#efefef"


def apply_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8., "axes.labelsize": 8.,
        "axes.titlesize": 8.5, "figure.titlesize": 10., "xtick.labelsize": 7., "ytick.labelsize": 7.,
        "legend.fontsize": 7., "axes.linewidth": .6, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": .6, "ytick.major.width": .6, "grid.linewidth": .4,
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": True, "image.interpolation": "nearest"})


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=7)


def clean(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, alpha=.2)


class Exporter:
    def __init__(self, directory, dpi=600):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.entries = []
        self.atlas = PdfPages(self.root / "main_figure_atlas.pdf", metadata={
            "Title": "Module 05: matched-hierarchy policy evaluation",
            "Subject": "Scientific figures; all six checkpoints; diffusion 60 cubed"})

    def save(self, fig, name, title, caption, *, category="main", takeaway="", inputs=None):
        stem = self.root / name
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(stem.with_suffix("." + ext), dpi=self.dpi, bbox_inches="tight", pad_inches=.06)
        preview = stem.with_name(stem.name + "_preview.png")
        fig.savefig(preview, dpi=170, bbox_inches="tight", pad_inches=.06)
        if category == "main":
            self.atlas.savefig(fig, dpi=self.dpi, bbox_inches="tight", pad_inches=.06)
        self.entries.append({"name": name, "title": title, "caption": caption,
                             "takeaway": takeaway, "category": category, "inputs": inputs or {},
                             "files": {**{ext: str(stem.with_suffix("."+ext).relative_to(self.root)) for ext in ("pdf", "svg", "png")},
                                       "preview": str(preview.relative_to(self.root))}})
        plt.close(fig)

    def close(self):
        self.atlas.close()
        (self.root / "figure_catalog.json").write_text(json.dumps(self.entries, indent=2) + "\n")
