"""
Thin wrapper for the canonical LinUCB AMG experiment.

This keeps the historical entrypoint name while reusing the shared
runner path in scripts/learning_amg_linucb.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learning_amg_linucb import main as run_linucb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260209)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--ma", type=int, default=25)
    ap.add_argument("--fail-penalty", type=float, default=1e6)
    ap.add_argument(
        "--loss",
        type=str,
        default="wu",
        choices=["wu", "runtime"],
        help="bandit update loss",
    )
    ap.add_argument(
        "--tune",
        type=str,
        default="th_coarsen_interp",
        choices=["th_coarsen_interp", "th_mxrs_tr"],
        help="3-parameter set to tune",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / "results" / "setup" / "linucb_amg" / f"T{args.T}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "T": int(args.T),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "l2": float(args.l2),
        "ma": int(args.ma),
        "fail_penalty": float(args.fail_penalty),
        "loss": str(args.loss),
        "tune": str(args.tune),
        "note": "Wrapper delegates to scripts/learning_amg_linucb.py",
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, sort_keys=True))

    run_linucb(
        [
            "--T",
            str(args.T),
            "--trials",
            "1",
            "--seed",
            str(args.seed),
            "--alpha",
            str(args.alpha),
            "--l2",
            str(args.l2),
            "--ma",
            str(args.ma),
            "--fail-penalty",
            str(args.fail_penalty),
            "--loss",
            str(args.loss),
            "--tune",
            str(args.tune),
            "--out-dir",
            str(run_dir),
        ]
    )


if __name__ == "__main__":
    main()
