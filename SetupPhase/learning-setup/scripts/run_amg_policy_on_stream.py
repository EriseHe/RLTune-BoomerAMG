"""
Run one policy on a persisted BoomerAMG instance stream and log WU + wall time.

This is intended to support "timing individually" comparisons:
  - generate an instance stream once (generate_amg_instance_stream.py)
  - run each policy separately on that same stream
  - compare cumulative runtime curves without within-round timing-order bias
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Reduce noisy OpenMPI TCP binding warnings on macOS environments.
os.environ.setdefault("OMPI_MCA_btl", "self,sm")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learners.LinUCB_AMG import LinUCB_AMG
from learners.TsallisINF_AMG import TsallisINF_AMG
from solver import solve
from utils.problem_amg import CONTEXT_DIM, build_context, build_matrix_kwargs
from utils.setup_amg import build_actions_th_coarsen_interp


def _load_meta(instances_csv: Path) -> Dict[str, Any]:
    meta_path = instances_csv.parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"nx": 60, "ny": 60, "nz": 1, "stencil": 27}


def _iter_instances(instances_csv: Path) -> Iterable[Dict[str, Any]]:
    with open(instances_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def _mkw_from_row(meta: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    return build_matrix_kwargs(
        s1=float(row["s1"]),
        s2=float(row["s2"]),
        s3=float(row["s3"]),
        c_diag=float(row["c_diag"]),
        rhs_seed=int(row["rhs_seed"]),
        nx=int(meta.get("nx", 60)),
        ny=int(meta.get("ny", 60)),
        nz=int(meta.get("nz", 1)),
        stencil=int(meta.get("stencil", 27)),
    )


def _context_from_row(meta: Dict[str, Any], row: Dict[str, Any]) -> np.ndarray:
    return build_context(
        s1=float(row["s1"]),
        s2=float(row["s2"]),
        s3=float(row["s3"]),
        c_diag=float(row["c_diag"]),
        nx=int(meta.get("nx", 60)),
        ny=int(meta.get("ny", 60)),
        nz=int(meta.get("nz", 1)),
    )


def _safe_solve(params: Dict[str, Any], mkw: Dict[str, Any], *, fail_penalty: float) -> Tuple[float, float]:
    start = time.perf_counter_ns()
    try:
        wu = float(solve(params=params, **mkw).work_units)
        elapsed_sec = (time.perf_counter_ns() - start) / 1e9
        return wu, elapsed_sec
    except Exception:
        elapsed_sec = (time.perf_counter_ns() - start) / 1e9
        return float(fail_penalty), elapsed_sec


def _parse_json_dict(s: str) -> Dict[str, Any]:
    out = json.loads(s)
    if not isinstance(out, dict):
        raise ValueError("--params-json must decode to a JSON object")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str, required=True, help="path to instances.csv")
    ap.add_argument("--out", type=str, default="", help="output CSV path (default: alongside instances.csv)")
    ap.add_argument("--name", type=str, default="", help="policy name used for default output filenames")
    ap.add_argument("--policy", type=str, required=True, choices=["fixed", "linucb", "tsallis"])
    ap.add_argument("--seed", type=int, default=0, help="policy RNG seed (tie-breaking etc.)")
    ap.add_argument("--warmup", type=int, default=1, help="number of untimed warmup solves (uses default params)")
    ap.add_argument("--fail-penalty", type=float, default=1e6)

    # Fixed policy
    ap.add_argument("--params-json", type=str, default="", help="JSON dict of solver params (overrides flag params)")
    ap.add_argument("--strong-threshold", type=float, default=0.25)
    ap.add_argument("--coarsen-type", type=int, default=10)
    ap.add_argument("--interp-type", type=int, default=6)

    # LinUCB policy
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--loss", type=str, default="runtime", choices=["wu", "runtime"], help="bandit update loss")
    ap.add_argument("--fixed-extras-json", type=str, default="", help="JSON dict of constant params applied to all actions")

    # Tsallis policy (strong_threshold only; other knobs come from --params-json/flags)
    ap.add_argument("--tsallis-arms", type=int, default=19)
    ap.add_argument("--tsallis-low", type=float, default=0.05)
    ap.add_argument("--tsallis-high", type=float, default=0.95)

    args = ap.parse_args()

    instances_csv = Path(args.instances).resolve()
    meta = _load_meta(instances_csv)

    out_csv = Path(args.out).resolve() if args.out else (instances_csv.parent / f"policy_{args.policy}.csv")
    if args.name:
        out_csv = Path(args.out).resolve() if args.out else (instances_csv.parent / f"policy_{args.name}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fixed_extras: Dict[str, Any] = _parse_json_dict(args.fixed_extras_json) if args.fixed_extras_json else {}

    fixed_params: Dict[str, Any]
    if args.params_json:
        fixed_params = _parse_json_dict(args.params_json)
    else:
        fixed_params = {
            "strong_threshold": float(args.strong_threshold),
            "coarsen_type": int(args.coarsen_type),
            "interp_type": int(args.interp_type),
        }
    fixed_params = dict(fixed_extras, **fixed_params)

    # Preload first instance for warmup and to know T.
    rows = list(_iter_instances(instances_csv))
    if not rows:
        raise ValueError(f"no instances found in {instances_csv}")

    # Warmup solve(s) to reduce first-call effects. Do not update bandits here.
    if int(args.warmup) > 0:
        mkw0 = _mkw_from_row(meta, rows[0])
        for _ in range(int(args.warmup)):
            _ = solve(params=fixed_params, **mkw0)

    # Policy init
    linucb: Optional[LinUCB_AMG] = None
    tsallis: Optional[TsallisINF_AMG] = None

    # LinUCB actions fixed to match scripts/learning_amg_linucb.py by default.
    thresholds = [0.10, 0.25, 0.40, 0.50, 0.70]
    coarsen_types = [10, 8, 6]
    interp_types = [6, 8, 0]

    if args.policy == "linucb":
        actions = build_actions_th_coarsen_interp(
            thresholds,
            coarsen_types,
            interp_types,
            fixed_params=fixed_extras,
        )
        linucb = LinUCB_AMG(
            actions,
            context_dim=CONTEXT_DIM,
            alpha=float(args.alpha),
            l2_reg=float(args.l2),
            seed=int(args.seed),
        )
    elif args.policy == "tsallis":
        # TsallisINF_AMG uses NumPy's global RNG for sampling; seed it for reproducibility.
        np.random.seed(int(args.seed))
        grid = np.linspace(float(args.tsallis_low), float(args.tsallis_high), int(args.tsallis_arms))
        tsallis = TsallisINF_AMG(grid, len(rows))

    # Run
    fieldnames = [
        "t",
        "s1",
        "s2",
        "s3",
        "c_diag",
        "rhs_seed",
        "a0",
        "a1",
        "a2",
        "a3",
        "chosen_strong_threshold",
        "chosen_coarsen_type",
        "chosen_interp_type",
        "wu",
        "runtime_sec",
        "loss_used",
        "pred_mean",
        "pred_uncert",
    ]

    total_wu = 0.0
    total_rt = 0.0

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for row in rows:
            t = int(row["t"])
            mkw = _mkw_from_row(meta, row)
            context = _context_from_row(meta, row)

            if args.policy == "fixed":
                params = fixed_params
                pred_mean = float("nan")
                pred_unc = float("nan")
            elif args.policy == "linucb":
                assert linucb is not None
                params = linucb.predict(context)
                step = linucb.history[-1]
                pred_mean = float(step.pred_mean)
                pred_unc = float(step.pred_uncert)
            else:
                assert tsallis is not None
                th = float(tsallis.predict())
                params = dict(fixed_params)
                params["strong_threshold"] = th
                pred_mean = float("nan")
                pred_unc = float("nan")

            wu, rt = _safe_solve(params, mkw, fail_penalty=float(args.fail_penalty))
            total_wu += wu
            total_rt += rt

            if args.policy == "linucb":
                assert linucb is not None
                # If using runtime as loss, use milliseconds to keep LinUCB's
                # exploration term (alpha * uncertainty) on a comparable scale.
                loss_used = (rt * 1e3) if args.loss == "runtime" else wu
                linucb.update(loss_used)
            elif args.policy == "tsallis":
                assert tsallis is not None
                # TsallisINF_AMG expects loss in the same convention as existing code (loss >= 1).
                # For runtime, shift by +1 to preserve positivity and keep (loss-1) meaningful.
                loss_used = (1.0 + rt * 1e3) if args.loss == "runtime" else wu
                tsallis.update(loss_used)
            else:
                loss_used = float("nan")

            w.writerow(
                {
                    "t": t,
                    "s1": float(row["s1"]),
                    "s2": float(row["s2"]),
                    "s3": float(row["s3"]),
                    "c_diag": float(row["c_diag"]),
                    "rhs_seed": int(row["rhs_seed"]),
                    "a0": float(row["a0"]),
                    "a1": float(row["a1"]),
                    "a2": float(row["a2"]),
                    "a3": float(row["a3"]),
                    "chosen_strong_threshold": float(params.get("strong_threshold", float("nan"))),
                    "chosen_coarsen_type": int(params.get("coarsen_type", -1)),
                    "chosen_interp_type": int(params.get("interp_type", -1)),
                    "wu": float(wu),
                    "runtime_sec": float(rt),
                    "loss_used": float(loss_used),
                    "pred_mean": float(pred_mean),
                    "pred_uncert": float(pred_unc),
                }
            )

    summary = {
        "policy": str(args.policy),
        "name": str(args.name) if args.name else str(args.policy),
        "instances": str(instances_csv),
        "out_csv": str(out_csv),
        "T": int(len(rows)),
        "total_wu": float(total_wu),
        "total_runtime_sec": float(total_rt),
        "loss": str(args.loss) if args.policy in ("linucb", "tsallis") else "",
        "linucb": {"alpha": float(args.alpha), "l2": float(args.l2)} if args.policy == "linucb" else {},
        "fixed_params": fixed_params if args.policy in ("fixed", "tsallis") else {},
    }
    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=== Policy run complete ===")
    print(f"policy: {args.policy} ({summary['name']})")
    print(f"T={len(rows)}, seed={args.seed}")
    print(f"total runtime (sec): {total_rt:.6f}")
    print(f"total WU:            {total_wu:.3f}")
    print(f"log:     {out_csv}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
