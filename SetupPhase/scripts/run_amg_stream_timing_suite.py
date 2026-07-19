"""
End-to-end helper for "timing individually" comparisons on the AMG instance stream.

This script:
  1) generates `instances.csv` (deterministic stream)
  2) runs one or more policies *separately* on that stream
  3) plots cumulative runtime and WU overlays

All outputs are written under:
  results/setup/timing_individual/<run_id>/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Reduce noisy OpenMPI TCP binding warnings on macOS environments.
os.environ.setdefault("OMPI_MCA_btl", "self,sm")


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260209)
    ap.add_argument("--loss", type=str, default="runtime", choices=["wu", "runtime"], help="bandit update loss")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--baseline-set", type=str, default="minimal", choices=["minimal", "full"])
    ap.add_argument("--include-tsallis", action="store_true", help="also run TsallisINF_AMG (strong_threshold only)")
    ap.add_argument("--tag", type=str, default="", help="optional suffix for the run directory name")
    args = ap.parse_args()

    T = int(args.T)
    seed = int(args.seed)

    base_dir = Path(__file__).resolve().parent.parent
    repo_root = base_dir.parent
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = repo_root / "results" / "setup" / "timing_individual" / f"T{T}_seed{seed}_loss{args.loss}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    instances_csv = run_dir / "instances.csv"

    gen = base_dir / "scripts" / "generate_amg_instance_stream.py"
    run_policy = base_dir / "scripts" / "run_amg_policy_on_stream.py"
    plot = base_dir / "scripts" / "plot_amg_stream_timing.py"

    # 1) Generate stream (always overwrite instances.csv for this run_dir)
    _run(
        [
            sys.executable,
            "-u",
            str(gen),
            "--T",
            str(T),
            "--seed",
            str(seed),
            "--out",
            str(instances_csv),
        ],
        cwd=base_dir,
    )

    # 2) Run policies separately
    baselines = [
        ("default", {"strong_threshold": 0.25, "coarsen_type": 10, "interp_type": 6}),
    ]
    if args.baseline_set == "full":
        baselines.extend(
            [
                ("th_0p10", {"strong_threshold": 0.10, "coarsen_type": 10, "interp_type": 6}),
                ("th_0p40", {"strong_threshold": 0.40, "coarsen_type": 10, "interp_type": 6}),
                ("coarsen_8", {"strong_threshold": 0.25, "coarsen_type": 8, "interp_type": 6}),
                ("interp_8", {"strong_threshold": 0.25, "coarsen_type": 10, "interp_type": 8}),
                ("coarsen_8_interp_8", {"strong_threshold": 0.25, "coarsen_type": 8, "interp_type": 8}),
            ]
        )

    for name, p in baselines:
        _run(
            [
                sys.executable,
                "-u",
                str(run_policy),
                "--policy",
                "fixed",
                "--name",
                name,
                "--instances",
                str(instances_csv),
                "--out",
                str(run_dir / f"policy_{name}.csv"),
                "--strong-threshold",
                str(p["strong_threshold"]),
                "--coarsen-type",
                str(p["coarsen_type"]),
                "--interp-type",
                str(p["interp_type"]),
            ],
            cwd=base_dir,
        )

    _run(
        [
            sys.executable,
            "-u",
            str(run_policy),
            "--policy",
            "linucb",
            "--name",
            "linucb",
            "--instances",
            str(instances_csv),
            "--out",
            str(run_dir / "policy_linucb.csv"),
            "--seed",
            str(seed + 1),
            "--alpha",
            str(float(args.alpha)),
            "--l2",
            str(float(args.l2)),
            "--loss",
            str(args.loss),
        ],
        cwd=base_dir,
    )

    if bool(args.include_tsallis):
        _run(
            [
                sys.executable,
                "-u",
                str(run_policy),
                "--policy",
                "tsallis",
                "--name",
                "tsallis",
                "--instances",
                str(instances_csv),
                "--out",
                str(run_dir / "policy_tsallis.csv"),
                "--seed",
                str(seed + 2),
                "--loss",
                str(args.loss),
            ],
            cwd=base_dir,
        )

    # 3) Plot overlays
    _run(
        [
            sys.executable,
            "-u",
            str(plot),
            "--run-dir",
            str(run_dir),
            "--title",
            f"BoomerAMG Setup: timing individually (T={T}, seed={seed}, loss={args.loss})",
        ],
        cwd=base_dir,
    )

    print("\n=== Suite complete ===")
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()
