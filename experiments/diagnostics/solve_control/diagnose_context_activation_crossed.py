"""Small frozen-policy x recorded-setup diagnostic for the 4D branches.

This reuses benchmark matrices to explain observed mechanisms. It is not an
independent benchmark, a rerun of either training trajectory, or evidence about
the causal effect of changing only the activation time.
"""
from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from composable_joint_4k import build_composable_solve_runtime
from hypre.bindings.config import configure_smoother_profile
from joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from online_td_experiment_common import _json_ready, _write_json
from run_lstdq_v3_stability import _run_case
from setup_aware_compare_common import augment_setup_params, solve_no_rl_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    raw = json.loads((args.run / "experiment_config.json").read_text())
    runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
    configure_smoother_profile(raw["solve"]["smoother_profile"])
    methods = ("context4d_fixed", "context4d_dynamic")
    rows = {}
    for method in methods:
        with (args.run / "trajectories" / f"{method}.jsonl").open() as f:
            rows[method] = [json.loads(line) for line in f]
    assert all(a["mkw"] == b["mkw"] for a,b in zip(*rows.values()))
    args.output.mkdir(parents=True)
    native_args = SimpleNamespace(tol=1e-6, max_cycles=50)
    rng = np.random.default_rng(91342026)
    records = []
    # 25 evenly spaced problems per checkpoint; decided before any crossed solve.
    selections = {2000: list(range(2000, 3000, 40)), 5000: list(range(4000, 5000, 40))}
    _write_json(args.output / "protocol.json", {
        "source": str(args.run.resolve()), "case_indices_zero_based": selections,
        "learn": False, "explore": False, "native_solves": 500,
        "scoring": ["lcb_beta2", "mean_beta0", "default_weight1"],
        "smoother_profile": raw["solve"]["smoother_profile"],
        "interpretation": __doc__,
    })
    with (args.output / "trajectories.jsonl").open("w") as handle:
        for checkpoint, indices in selections.items():
            policies = {}
            snapshots = {}
            for scoring in ("lcb_beta2", "mean_beta0"):
                bundles = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles
                for method in methods:
                    bundle = bundles[method]
                    bundle.load(args.run / "checkpoints" / f"{method}_episode_{checkpoint}.npz")
                    if scoring == "mean_beta0":
                        bundle.controller.spec = replace(bundle.controller.spec, uncertainty_beta=0.)
                    name = f"{method}/{scoring}"
                    policies[name] = bundle
                    c = bundle.controller
                    snapshots[name] = {"a": c.a_matrix.copy(), "b": c.b.copy(), "theta": c.theta.copy(),
                                       "steps": c.steps, "rng": json.dumps(c.rng.bit_generator.state)}
            for index in indices:
                cells = [(setup, policy) for setup in methods for policy in (*policies, "default_weight1")]
                for cell_index in rng.permutation(len(cells)):
                    setup, policy = cells[cell_index]
                    row = rows[setup][index]
                    if policy == "default_weight1":
                        outcome = solve_no_rl_case(params=dict(row["params"]), mkw=dict(row["mkw"]),
                                                  solver_tol=1e-6, solver_max_iter=50,
                                                  augment_params=augment_setup_params)
                    else:
                        outcome = _run_case(bundle=policies[policy], setup_row=row,
                                            args=native_args, learn=False, explore=False)
                    item = {"checkpoint": checkpoint, "case": index+1, "setup_source": setup,
                            "policy": policy, "params": row["params"], "outcome": outcome}
                    records.append(item)
                    handle.write(json.dumps(_json_ready(item))+"\n")
                handle.flush()
                print(json.dumps({"completed_solves": len(records), "total": 500,
                                  "checkpoint": checkpoint, "problem": index+1}), flush=True)
            for name, bundle in policies.items():
                c, snap = bundle.controller, snapshots[name]
                assert np.array_equal(c.a_matrix, snap["a"])
                assert np.array_equal(c.b, snap["b"])
                assert np.array_equal(c.theta, snap["theta"])
                assert c.steps == snap["steps"] and json.dumps(c.rng.bit_generator.state) == snap["rng"]
    groups = defaultdict(list)
    for r in records:
        groups[(r["checkpoint"], r["setup_source"], r["policy"])].append(r["outcome"])
    summaries = []
    for (checkpoint, setup, policy), os in groups.items():
        summaries.append({"checkpoint": checkpoint, "setup_source": setup, "policy": policy,
                          "cases": len(os), "mean_cycles": float(np.mean([o["iterations"] for o in os])),
                          "mean_setup_ms": float(np.mean([o["setup_runtime"] for o in os])*1000),
                          "mean_native_solve_ms": float(np.mean([o.get("native_solve_runtime", o["solve_runtime"]) for o in os])*1000),
                          "mean_end_to_end_ms": float(np.mean([o.get("end_to_end_runtime",
                              o.get("native_runtime", o["runtime"])+o.get("infer_runtime", 0.)) for o in os])*1000),
                          "primary_failures": sum(o.get("primary_status", o.get("attempt_status")) not in (None,"success") for o in os),
                          "failed": sum(bool(o["failed"]) for o in os)})
    _write_json(args.output / "summary.json", {"source": str(args.run.resolve()), "learning_state_unchanged": True,
                                              "cells": summaries})


if __name__ == "__main__":
    main()
