"""
Plot cumulative runtime for policies run on the same persisted instance stream.

Inputs
------
One or more policy CSV logs produced by `run_amg_policy_on_stream.py`.

Outputs
-------
Saved under the run directory:
  - cumulative_runtime.png
  - summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

# Avoid Matplotlib writing to user home (may be non-writable in some environments).
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_bandits_for_setup"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_policy_csv(path: Path) -> np.ndarray:
    rt: List[float] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rt.append(float(row["runtime_sec"]))
    return np.asarray(rt, dtype=float)


def _label_from_path(p: Path) -> str:
    name = p.stem
    if name.startswith("policy_"):
        name = name[len("policy_") :]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="directory containing policy_*.csv logs")
    ap.add_argument("--logs", type=str, nargs="*", default=[], help="explicit list of policy CSV logs")
    ap.add_argument("--title", type=str, default="BoomerAMG Setup: timing individually")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.logs:
        logs = [Path(p).resolve() for p in args.logs]
    else:
        logs = sorted(run_dir.glob("policy_*.csv"))
    if not logs:
        raise ValueError(f"no policy logs found in {run_dir}")

    # Remove stale plots from older versions of this script.
    for stale in ("cumulative_wu.png", "wu_vs_time_scatter.png"):
        try:
            (run_dir / stale).unlink()
        except FileNotFoundError:
            pass

    series: Dict[str, np.ndarray] = {}
    T = None
    for p in logs:
        label = _label_from_path(p)
        rt = _read_policy_csv(p)
        if T is None:
            T = int(rt.size)
        elif int(rt.size) != int(T):
            raise ValueError(f"log {p} has T={rt.size}, expected T={T}")
        series[label] = rt

    assert T is not None
    t_axis = np.arange(1, T + 1)

    # --- cumulative runtime plot ---
    plt.figure(figsize=(9, 5))
    for label, rt in series.items():
        cum = np.cumsum(rt)
        plt.plot(cum, T - t_axis + 1, linewidth=2, label=label)
    plt.xlabel("cumulative runtime (seconds)", fontsize=13)
    plt.ylabel("instances remaining", fontsize=13)
    plt.title(args.title, fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()
    rt_plot = run_dir / "cumulative_runtime.png"
    plt.savefig(rt_plot, dpi=256)
    plt.close()

    summary = {
        "run_dir": str(run_dir),
        "T": int(T),
        "logs": [str(p) for p in logs],
        "totals": {k: {"total_runtime_sec": float(np.sum(rt))} for k, rt in series.items()},
        "plots": {
            "cumulative_runtime": str(rt_plot),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("=== Plots written ===")
    print(f"run_dir: {run_dir}")
    print(f"{rt_plot.name}")


if __name__ == "__main__":
    main()
