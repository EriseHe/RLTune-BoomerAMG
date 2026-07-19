"""
Generate and persist a deterministic BoomerAMG instance stream (60x60x1, 27pt).

Purpose
-------
When comparing *wall time* across policies, timing multiple configs back-to-back
on the same instance can introduce systematic bias (warm caches, allocator
state, "first call" effects). A cleaner approach is:
  1) generate the instance stream once,
  2) run each policy separately on the exact same stream,
  3) compare cumulative runtime curves.

This script implements (1): it writes a CSV with all per-round randomization
variables needed to reconstruct each linear system and RHS.

The distribution matches `scripts/learning_amg_linucb.py`:
  s1,s2,s3 ~ Uniform(0.5, 2.0)
  c_diag   ~ Uniform(0.0, 5.0)
  rhs_seed ~ Uniform integer in [0, 2^63-1]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Reduce noisy OpenMPI TCP binding warnings on macOS environments.
os.environ.setdefault("OMPI_MCA_btl", "self,sm")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import _project_paths  # noqa: E402,F401

from problems.amg import A1_BASE, A2_BASE, A3_BASE, stencil_27_laplace
from utils.paths import SETUP_RESULTS_ROOT


def sample_instance(rng: np.random.Generator, *, nz: int) -> Dict[str, Any]:
    mkw, _context, meta = stencil_27_laplace(rng=rng, nz=nz, stencil=27)
    return {
        "s1": float(meta["s1"]),
        "s2": float(meta["s2"]),
        "s3": float(meta["s3"]),
        "c_diag": float(meta["c_diag"]),
        "a0": float(mkw["a0"]),
        "a1": float(mkw["a1"]),
        "a2": float(mkw["a2"]),
        "a3": float(mkw["a3"]),
        "rhs_seed": int(meta["rhs_seed"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000, help="number of instances to generate")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the stream")
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--ny", type=int, default=60)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--stencil", type=int, default=27, choices=[27], help="stencil type (currently only 27)")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="output CSV path (default: results/setup/timing_individual/T{T}_seed{seed}/instances.csv)",
    )
    args = ap.parse_args()

    T = int(args.T)
    seed = int(args.seed)
    nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
    stencil = int(args.stencil)
    if stencil != 27:
        raise ValueError("only 27pt stencil is supported by this stream generator")

    out_dir = SETUP_RESULTS_ROOT / "timing_individual" / f"T{T}_seed{seed}"
    out_csv = Path(args.out) if args.out else (out_dir / "instances.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    meta = {
        "T": T,
        "seed": seed,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "stencil": stencil,
        "A1_BASE": A1_BASE,
        "A2_BASE": A2_BASE,
        "A3_BASE": A3_BASE,
        "dist": {
            "s1": "Uniform(0.5,2.0)",
            "s2": "Uniform(0.5,2.0)",
            "s3": "Uniform(0.5,2.0)",
            "c_diag": "Uniform(0.0,5.0)",
            "rhs_seed": "Uniform integer [0,2^63-1]",
        },
    }
    (out_csv.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    fieldnames = ["t", "s1", "s2", "s3", "c_diag", "a0", "a1", "a2", "a3", "rhs_seed"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in range(1, T + 1):
            row = sample_instance(rng, nz=nz)
            row["t"] = t
            w.writerow(row)

    print("=== Generated instance stream ===")
    print(f"T={T}, seed={seed}")
    print(f"grid: {nx}x{ny}x{nz}, stencil={stencil}")
    print(f"csv:  {out_csv}")
    print(f"meta: {out_csv.parent / 'meta.json'}")


if __name__ == "__main__":
    main()
