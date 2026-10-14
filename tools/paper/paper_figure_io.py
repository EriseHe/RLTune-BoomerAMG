"""Shared output formatting for current paper figures."""
import matplotlib.pyplot as plt


def save(fig, paper, name):
    for ax in fig.axes:
        ax.grid(False, which="both")
    for extension, dpi in (("svg", 600), ("png", 180)):
        path = paper / "figures" / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
