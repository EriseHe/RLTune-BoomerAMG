"""Audit shared-prefix execution and the frozen Module 05 output contract."""
from __future__ import annotations

from experiments.paper_final import run_05_policy as base

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.joint.solve_control.joint_online_common import _validate_recovery_stream

METHODS = ("bandit_default", "bandit_fixed", "bandit_fixed_prior", "bandit_periodic", "bandit_lstdq")


def load_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def verify(output):
    config = base.read(output/"experiment_config.json")
    n = config["stream"]["cases"]
    prefix = config["checkpoint_training"]["shared_prefix_cases"]
    fixed_weights = {m["id"]: m["fixed_weight"] for m in config["methods"] if m["solve"] == "fixed"}
    if tuple(m["id"] for m in config["methods"]) != METHODS or fixed_weights["bandit_fixed_prior"] != 1.60:
        raise AssertionError("Incorrect five-branch configuration")
    rows = {m:load_rows(output/"trajectories"/f"{m}.jsonl") for m in METHODS}
    orders = load_rows(output/"trajectories/method_order.jsonl")
    inputs = base.read(output/"inputs.json")
    if len(orders) != n or len(inputs) != n or any(len(r) != n for r in rows.values()):
        raise AssertionError("Incomplete paired training stream")
    rng = np.random.default_rng(config["seeds"]["method_order"])
    for i in range(n):
        randomized = [METHODS[k] for k in rng.permutation(len(METHODS))]
        expected = [METHODS[0]] if i < prefix else randomized
        if orders[i]["method_order"] != expected or orders[i]["online_index"] != i:
            raise AssertionError("Unexpected native execution order")
        for method in METHODS:
            row = rows[method][i]
            if row["online_index"] != i or row["stream_index"] != i or row["mkw"] != inputs[i]["mkw"]:
                raise AssertionError("Training input alignment failed")
            if i < prefix and base.digest(row) != base.digest(rows[METHODS[0]][i]):
                raise AssertionError("Prefix was not shared exactly")
            o = row["outcome"]
            for key in ("runtime", "setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime", "end_to_end_runtime"):
                if not np.isfinite(o[key]) or o[key] < 0:
                    raise AssertionError("Invalid timing")
            if not o["unrecovered_failure"]:
                r = o["completed_residual_norm"]
                if r is None or not np.isfinite(r) or r > config["solve"]["tolerance"]:
                    raise AssertionError("Completed solution violates the tolerance")
            actions = o.get("cycle_actions", [])
            for attempt in o.get("primary_attempts", []):
                if not 0 <= attempt["cycles"] <= config["solve"]["max_cycles"]:
                    raise AssertionError("Primary attempt exceeded the cycle cap")
            if not 0 <= o["fallback_cycles"] <= config["solve"]["max_cycles"]:
                raise AssertionError("Fallback exceeded the cycle cap")
            if i < prefix or method == "bandit_default":
                if any(w != 1 for w in actions) or o.get("controller_update_committed", False):
                    raise AssertionError("W1 prefix/control used a solve learner")
            elif method in fixed_weights and any(w != fixed_weights[method] for w in actions):
                raise AssertionError("Fixed continuation used the wrong weight")
            elif method == "bandit_periodic" and any(w != (2.85, 1.10)[j%2] for j,w in enumerate(actions)):
                raise AssertionError("Periodic continuation or phase reset is incorrect")
            elif method == "bandit_lstdq" and any(
                not np.any(np.isclose(w, np.linspace(1,3,41), rtol=0, atol=1e-12)) for w in actions):
                raise AssertionError("RL used a weight outside its prescribed grid")
    fork_audits = {}
    for method in METHODS[1:]:
        fork = base.read(output/f"shared_prefix_{method}.json")
        if not fork["valid"] or fork["completed_instances"] != prefix or fork["candidate_schedule_cursor"] != prefix*3:
            raise AssertionError("Invalid shared-prefix state fork")
        summary = fork["controller_summaries_at_fork"]["bandit_lstdq"]
        if summary.get("sample_count", 0) != 0 or summary.get("episode_moment_count", 0) != 0:
            raise AssertionError("LSTDQ was trained during the W1 prefix")
        fork_audits[method] = fork
    recovery = {m:_validate_recovery_stream(r, expect_bandit_transaction=True) for m,r in rows.items()}
    result = base.read(output/"result.json")
    if set(result["bandit_online_steps"].values()) != {n}:
        raise AssertionError("Wrong training budget")
    if result["controllers"]["bandit_lstdq"].get("sample_count", 0) <= 0:
        raise AssertionError("LSTDQ continuation did not learn")
    for method in METHODS:
        with np.load(output/"final_bandit_states"/f"{method}.npz", allow_pickle=False) as state:
            if int(state["candidate_schedule_cursor"]) != n*3:
                raise AssertionError("Final candidate cursor is inconsistent")
            for key in ("A_inv", "b", "failure_A_inv", "failure_b"):
                if not np.all(np.isfinite(state[key])):
                    raise AssertionError("Nonfinite final setup checkpoint")
    frozen = base.read(output/"frozen_artifacts.json")
    functional = config["checkpoint_training"]["functional_check"]
    if frozen["eligible_for_module06"] == functional:
        raise AssertionError("Functional-check checkpoints must not be eligible for Module 06")
    for artifact in frozen["methods"].values():
        for field in ("statistics", "decision_state"):
            if base.file_hash(artifact[field]) != artifact[field+"_sha256"]:
                raise AssertionError("Frozen artifact hash mismatch")
        if not artifact["roundtrip_passed"]:
            raise AssertionError("Final checkpoint was not verified")
    controller = frozen["rl_controller"]
    if base.file_hash(controller["path"]) != controller["sha256"]:
        raise AssertionError("Frozen controller hash mismatch")
    metrics = ("runtime", "setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime", "end_to_end_runtime")
    windows = {"shared_prefix": (0,prefix), "continuation": (prefix,n), "last_1000": (max(prefix,n-1000),n), "logical_full": (0,n)}
    diagnostics = {window:{m:{key:sum(r["outcome"][key] for r in rows[m][start:stop]) for key in metrics}
                          for m in METHODS} for window,(start,stop) in windows.items()}
    physical = rows[METHODS[0]][:prefix] + [r for m in METHODS for r in rows[m][prefix:]]
    audit = {"passed": True, "problems_per_branch": n, "prefix_executed_once": prefix,
             "functional_check": functional,
             "physical_executions": len(physical), "logical_method_problem_exposures": n*len(METHODS),
             "prefix_clones_verified": list(fork_audits), "recovery": recovery,
             "training_costs_are_diagnostics_only": True,
             "actual_inclusive_training_seconds": sum(r["outcome"]["end_to_end_runtime"] for r in physical)}
    base.dump(output/"training_diagnostics.json", diagnostics)
    base.dump(output/"audit.json", audit)
    lines = ["# Module 05 checkpoint preparation", "", f"Verified {prefix} shared W1 problems followed by {n-prefix} problems per continuation.",
             f"Physical method–problem executions: {len(physical)}; logical exposures per branch: {n}.", "",
             "Training cost is diagnostic; faster training is not an acceptance condition.", "",
             "| Branch | Continuation inclusive seconds | Continuation native seconds |",
             "|---|---:|---:|"]
    for m in METHODS:
        d = diagnostics["continuation"][m]
        lines.append(f"| {m} | {d['end_to_end_runtime']:.6f} | {d['runtime']:.6f} |")
    lines += ["", "All five setup snapshots and the final LSTDQ controller are listed in `frozen_artifacts.json`.",
              "The shared prefix appears in each logical branch log, but is charged only once in physical training work.",
              "Future Module 06 evaluates fresh inputs without learning; its own-pair comparison includes setup selection/construction, solve, controller, and recovery.", ""]
    (output/"TRAINING_REPORT.md").write_text("\n".join(lines))
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    print(json.dumps(verify(p.parse_args().output)))
