"""Small development diagnostics, using the existing native and recovery paths.

Run from the repository root with ``python -m experiments.paper_final.run_02_diagnostics``.
The default prepares/validates the protocol; --run executes the prescribed cases.
This is not the stage 3 activation study or the stage 5 final policy evaluation.
"""
from __future__ import annotations

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

import os

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.joint.solve_control.composable_joint_4k import build_composable_solve_runtime
from hypre.bindings import run_with_default_fallback
from hypre.bindings.config import configure_smoother_profile
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.joint_online_common import report_online_outcome
from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json
from problems.streams import (generate_scalar_anisotropic_diffusion_instances,
                             generate_scalar_anisotropic_diffusion_advection_instances)
from experiments.diagnostics.solve_control.run_lstdq_v3_stability import _run_case
from experiments.joint.solve_control.run_paper_final import file_hash, source_state
from setup.space import DEFAULT_SETUP_PARAMS
from experiments.joint.solve_control.setup_aware_compare_common import augment_setup_params, solve_no_rl_case, solve_schedule_case


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "results/paper/paper_test_n60_composite_activation_seed1"
OUTPUT = ROOT / "results/paper_final/02_diagnostics"
NATIVE_ARGS = SimpleNamespace(tol=1e-6, max_cycles=50)
SCHEDULES = {
    "low_high_2": [1., 2.], "high_low_2": [2., 1.],
    "low_high_29": [1., 2.9], "high_low_29": [2.9, 1.],
    "low_low_high": [1., 1., 2.9], "high_low_low": [2.9, 1., 1.],
}


def build_bundle(raw, method, checkpoint=None, beta=2.):
    runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
    bundle = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles[method]
    if checkpoint is not None:
        bundle.load(checkpoint)
    bundle.controller.spec = replace(bundle.controller.spec, uncertainty_beta=beta)
    return bundle


def prepare():
    raw = json.loads((PILOT / "experiment_config.json").read_text())
    with (PILOT / "trajectories/fixed_start.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    counts = Counter(json.dumps(row["params"], sort_keys=True) for row in rows[-1000:])
    chosen, frequency = counts.most_common(1)[0]
    setups = {"reference": dict(DEFAULT_SETUP_PARAMS), "learned": json.loads(chosen)}
    advection = json.loads((Path(__file__).with_name("04_online") / "advection_80_s1.json").read_text())
    advection["seeds"]["controller"] = 9131614
    diffusion_cases = generate_scalar_anisotropic_diffusion_instances(
        count=12, seed=9131611, grid_choices=[(60, 60, 60)], c_min=1, c_max=1000)
    advection_training = generate_scalar_anisotropic_diffusion_advection_instances(
        count=64, seed=9131612, grid_choices=[(80, 80, 80)], c_min=1, c_max=1000,
        advection_min=1, advection_max=1000)
    advection_cases = generate_scalar_anisotropic_diffusion_advection_instances(
        count=12, seed=9131613, grid_choices=[(80, 80, 80)], c_min=1, c_max=1000,
        advection_min=1, advection_max=1000)
    cases = {"diffusion": diffusion_cases, "advection": advection_cases}
    pilot_inputs = {json.dumps(row["mkw"], sort_keys=True) for row in rows}
    if any(json.dumps(mkw, sort_keys=True) in pilot_inputs for mkw, _ in diffusion_cases):
        raise ValueError("Diagnostic input overlaps the pilot training stream")
    weights = {"diffusion": [round(1 + i * .05, 2) for i in range(41)],
               "advection": [1., 1.5, 2., 2.5, 2.9, 3.]}
    checkpoint = PILOT / "checkpoints/fixed_start_final.npz"
    policies = {f"rl_{scoring}": build_bundle(raw, "fixed_start", checkpoint, beta)
                for scoring, beta in (("lcb", 2.), ("mean", 0.))}
    training_bundle = build_bundle(advection, "bandit_lstdq")
    # Reproducible fresh inputs; never replay the pilot's training matrices.
    encoded = {
        "evaluation": {family: [{"mkw": mkw, "context": context.tolist()} for mkw, context in stream]
                       for family, stream in cases.items()},
        "advection_training": [{"mkw": mkw, "context": context.tolist()} for mkw, context in advection_training],
    }
    protocol = {
        "stage": "02_diagnostics", "status": "development_only",
        "setups": setups, "learned_setup_selection": {
            "source": str(PILOT.relative_to(ROOT)), "window": "last_1000_fixed_start",
            "rule": "most frequent parameter tuple, fixed before fresh outcomes",
            "frequency": frequency,
            "advection_use": "transfer of the diffusion setup tuple; not an advection-trained setup learner",
        },
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": file_hash(checkpoint),
        "fresh_input_sha256": hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(),
        "seeds": {"diffusion": 9131611, "advection_training": 9131612,
                  "advection": 9131613, "advection_controller": 9131614, "order": 9131615},
        "cases_per_family": 12, "selection_cases": [1, 2, 3, 4, 5, 6],
        "evaluation_cases": [7, 8, 9, 10, 11, 12],
        "selection_rule": "minimum solve + controller + fallback setup time on cases 1-6; ties by name; exclude only the common primary setup cost",
        "advection_training_cases": 64, "advection_training_setup": "alternate reference and learned",
        "advection_policy_caveat": "fresh short training for functional checks; not a converged or final policy",
        "fixed_weights": weights, "periodic_schedules": SCHEDULES,
        "evaluation_policies": ["rl_lcb", "rl_mean"], "learn": False, "explore": False,
        "smoother_profile": raw["solve"]["smoother_profile"], "tolerance": 1e-6, "cycle_cap": 50,
        "hierarchy_matching": "identical matrix/RHS and setup tuple; hierarchy reconstructed for every policy",
        "recovery": "one fixed-hierarchy primary attempt, then reference AMG fallback; no setup reselection",
        "native_attempts_before_recovery": 64 + sum(12 * 2 * (len(w) + 8) for w in weights.values()),
        "interpretation": "small development screen; no final-paper performance estimate or activation inference",
    }
    return protocol, encoded, setups, cases, weights, policies, training_bundle, advection_training, advection


def run_schedule(mkw, params, pattern):
    schedule = ([(50, pattern[0], 1, 1)] if len(pattern) == 1 else
                [(i + 1, pattern[i % len(pattern)], 1, 1) for i in range(50)])
    recovery = run_with_default_fallback(
        lambda: solve_schedule_case(params=dict(params), mkw=dict(mkw), schedule=schedule,
                                    solve_tol=1e-6, solve_max_cycles=50),
        lambda: solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS), mkw=dict(mkw),
                                solver_tol=1e-6, solver_max_iter=50, augment_params=augment_setup_params),
    )
    outcome = report_online_outcome(recovery.to_result(), bandit_timing={})
    outcome["completed_residual_norm"] = (recovery.fallback.residual_norm if recovery.fallback_used
                                          else recovery.primary.residual_norm)
    return outcome


def audit_outcome(outcome):
    components = [outcome[k] for k in ("setup_runtime", "solve_runtime", "infer_runtime")]
    if not all(np.isfinite(v) and v >= 0 for v in components):
        raise AssertionError("Invalid cost component")
    if not np.isclose(sum(components), outcome["end_to_end_runtime"], rtol=1e-9, atol=1e-12):
        raise AssertionError("Cost accounting mismatch")
    if not outcome["failed"] and not outcome.get("fallback_used", False):
        if not np.isfinite(outcome["residual_norm"]) or outcome["residual_norm"] > 1e-6:
            raise AssertionError("Successful primary solve violates the tolerance")
    if not outcome["failed"] and "completed_residual_norm" in outcome:
        residual = outcome["completed_residual_norm"]
        if not np.isfinite(residual) or residual > 1e-6:
            raise AssertionError("Completed solve violates the tolerance")


def audit_controller(controller):
    eye = np.eye(controller.joint_dim)
    result = {
        "inverse_error_inf": float(np.linalg.norm(controller.a_matrix @ controller.a_inverse - eye, ord=np.inf)),
        "inverse_right_error_inf": float(np.linalg.norm(controller.a_inverse @ controller.a_matrix - eye, ord=np.inf)),
        "theta_batch_error_inf": float(np.linalg.norm(
            controller.theta - np.linalg.solve(controller.a_matrix, controller.b), ord=np.inf)),
        "inverse_is_valid": bool(controller.inverse_is_valid),
        "inverse_rebuild_count": int(controller.inverse_rebuild_count),
    }
    if not result["inverse_is_valid"] or any(
        not np.isfinite(value) or value > 1e-6 for key, value in result.items() if key.endswith("_inf")
    ):
        raise AssertionError(f"Completed-episode numerical audit failed: {result}")
    return result


def audit_frozen_policies(policies, before, rng_before):
    for name, bundle in policies.items():
        after = bundle.controller.snapshot_learning_state()
        for field, value in before[name].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, after[field], err_msg=f"{name}/{field}")
            elif json.dumps(_json_ready(value)) != json.dumps(_json_ready(after[field])):
                raise AssertionError(f"Frozen policy changed: {name}/{field}")
        if rng_before[name] != json.dumps(bundle.controller.rng.bit_generator.state):
            raise AssertionError(f"Frozen policy changed its RNG: {name}")


def summarize(records):
    cells = defaultdict(list)
    for record in records:
        outcome = record["outcome"]
        key = (record["phase"], record["family"], record["setup"], record["policy"])
        cells[key].append(outcome)
    summaries = []
    for (phase, family, setup, policy), outcomes in sorted(cells.items()):
        times = [t for o in outcomes for t in o.get("cycle_times", [])]
        # The terminal cycle time is not a feature of a subsequent action.
        input_times = [t for o in outcomes if o.get("cycle_times")
                       for t in [0., *o["cycle_times"][:-1]]]
        summaries.append({
            "phase": phase, "family": family, "setup": setup, "policy": policy, "cases": len(outcomes),
            "solve_control_sec": sum(o["solve_runtime"] + o["infer_runtime"] for o in outcomes),
            "continuation_sec": sum(o["solve_runtime"] + o["infer_runtime"] + o.get("fallback_setup_runtime", 0)
                                    for o in outcomes),
            "total_sec": sum(o["end_to_end_runtime"] for o in outcomes),
            "setup_sec": sum(o["setup_runtime"] for o in outcomes),
            "controller_sec": sum(o["infer_runtime"] for o in outcomes),
            "fallback_sec": sum(o.get("fallback_setup_runtime", 0) + o.get("fallback_solve_runtime", 0)
                                + o.get("fallback_controller_runtime", 0) for o in outcomes),
            "mean_primary_cycles": float(np.mean([o.get("primary_cycles", o["iterations"]) for o in outcomes])),
            "primary_failures": sum(o["primary_status"] != "success" for o in outcomes),
            "unrecovered_failures": sum(bool(o["failed"]) for o in outcomes),
            "cycle_cap_hits": sum(o.get("primary_cycles", o["iterations"]) >= 50 for o in outcomes),
            "cycle_time_above_10ms_fraction": float(np.mean(np.asarray(times) >= .01)) if times else None,
            "last_cycle_time_inputs_above_10ms_fraction": float(np.mean(np.asarray(input_times) >= .01)) if input_times else None,
        })
    selections = []
    for family in ("diffusion", "advection"):
        for setup in ("reference", "learned"):
            for prefix in ("fixed_", "schedule_"):
                eligible = [c for c in summaries if c["phase"] == "selection" and c["family"] == family
                            and c["setup"] == setup and c["policy"].startswith(prefix)
                            and c["unrecovered_failures"] == 0]
                if eligible:
                    chosen = min(eligible, key=lambda c: (c["continuation_sec"], c["policy"]))
                    heldout = next(c for c in summaries if c["phase"] == "evaluation" and c["family"] == family
                                   and c["setup"] == setup and c["policy"] == chosen["policy"])
                    selections.append({"family": family, "setup": setup, "category": prefix[:-1],
                                       "chosen": chosen["policy"], "selection_sec": chosen["continuation_sec"],
                                       "evaluation": heldout})
    return {"cells": summaries, "selected_baselines": selections}


def resume_advection(output, checkpoint):
    """Append the prespecified frozen advection screen to the retained stage-02 data."""
    checkpoint = checkpoint.resolve()
    prescribed = ROOT / "results/paper_final/03_activation/advection_80_s1/checkpoints/start_1000_final.npz"
    if checkpoint != prescribed:
        raise ValueError("Use the stage-03 start_1000 checkpoint selected before outcomes")
    activation = checkpoint.parent.parent
    completed = json.loads((activation.parent / "summary.json").read_text())
    if not completed["complete"] or completed["completed_replicates"] != 1:
        raise ValueError("Stage 03 is not complete")
    final_audit = json.loads((activation.parent / "final_audit.json").read_text())
    checkpoint_audit = final_audit["checkpoint_numerics"]["start_1000"]
    encoder = Path(str(checkpoint) + ".encoder.json")
    if (not final_audit["prefix_and_cost_audit_valid"]
            or file_hash(checkpoint) != checkpoint_audit["checkpoint_sha256"]
            or file_hash(encoder) != checkpoint_audit["encoder_sha256"]):
        raise ValueError("The audited checkpoint or encoder changed")
    protocol = json.loads((output / "protocol.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    if hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest() != protocol["fresh_input_sha256"]:
        raise ValueError("The prespecified fresh inputs changed")
    if (protocol["periodic_schedules"] != SCHEDULES or protocol["tolerance"] != 1e-6
            or protocol["cycle_cap"] != 50 or protocol["learn"] or protocol["explore"]
            or protocol["selection_cases"] != list(range(1, 7))
            or protocol["evaluation_cases"] != list(range(7, 13))):
        raise ValueError("Unexpected diagnostic protocol")
    raw = json.loads((activation / "experiment_config.json").read_text())
    if raw["solve"]["smoother_profile"] != protocol["smoother_profile"]:
        raise ValueError("Checkpoint and diagnostic smoother profiles differ")
    policies = {f"rl_{scoring}": build_bundle(raw, "start_1000", checkpoint, beta)
                for scoring, beta in (("lcb", 2.), ("mean", 0.))}
    for bundle in policies.values():
        if (bundle.encoder.feature_dim, bundle.controller.joint_dim,
                bundle.encoder.encoding_version, bundle.encoder.problem_context_mode) != (
                34, 306, "space_aware_v2", "canonical_no_c_mean"):
            raise ValueError("Unexpected advection encoder")
    before_numerics = {name: audit_controller(bundle.controller) for name, bundle in policies.items()}

    trajectory = output / "trajectories.jsonl"
    lines = trajectory.read_bytes().splitlines(keepends=True)
    if not lines or not lines[-1].endswith(b"\n"):
        raise ValueError("Incomplete trajectory line; preserve it for inspection")
    records = [json.loads(line) for line in lines]
    keys = [(r["family"], r["phase"], r["case"], r["setup"], r["policy"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate diagnostic cells")
    setups = protocol["setups"]
    rng = np.random.default_rng(protocol["seeds"]["order"])
    plans = {}
    for family in ("diffusion", "advection"):
        names = [*(f"fixed_{w:.2f}" for w in protocol["fixed_weights"][family]),
                 *(f"schedule_{name}" for name in SCHEDULES), *policies]
        cells = [(setup, policy) for setup in setups for policy in names]
        if len(inputs["evaluation"][family]) != 12:
            raise ValueError("Expected the original 12 inputs per family")
        plans[family] = [(family, "selection" if i < 6 else "evaluation", i + 1, *cells[rank])
                         for i in range(12) for rank in rng.permutation(len(cells))]
    # Replaying the 12 diffusion permutations recovers the original advection order.
    prior_keys = plans["diffusion"] + [
        ("advection", "training", i + 1, ("reference", "learned")[i % 2], "rl_train") for i in range(58)
    ]
    plan = plans["advection"]
    if len(plan) != 336 or keys[:len(prior_keys)] != prior_keys:
        raise ValueError("Retained diffusion/short-training records differ from the original execution")
    appended = keys[len(prior_keys):]
    if appended != plan[:len(appended)]:
        raise ValueError("Appended advection records differ from the prespecified order")
    prior_bytes = b"".join(lines[:len(prior_keys)])
    with (activation / "trajectories/setup_only.jsonl").open() as handle:
        training_inputs = {json.dumps(json.loads(line)["mkw"], sort_keys=True) for line in handle}
    if any(json.dumps(row["mkw"], sort_keys=True) in training_inputs for row in inputs["evaluation"]["advection"]):
        raise ValueError("Fresh diagnostic inputs overlap stage-03 training")

    amendment = {
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": file_hash(checkpoint),
        "encoder_sha256": file_hash(encoder), "config_sha256": file_hash(activation / "experiment_config.json"),
        "inputs_sha256": file_hash(output / "inputs.json"), "protocol_sha256": file_hash(output / "protocol.json"),
        "prior_record_count": len(prior_keys), "prior_trajectory_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "additional_primary_attempts": 336, "job_order": plan, "learn": False, "explore": False,
        "model_selection": "start_1000_final selected before stage-03 outcomes; replaces abandoned short training",
        "input_overlap_with_stage_03": False,
    }
    amendment = _json_ready(amendment)
    amendment_path = output / "advection_resume.json"
    if amendment_path.exists():
        existing = json.loads(amendment_path.read_text())
        if any(existing.get(key) != value for key, value in amendment.items()):
            raise ValueError("The recorded advection continuation protocol changed")
    else:
        if appended:
            raise ValueError("Advection outcomes exist without a continuation protocol")
        _write_json(amendment_path, {**amendment, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        _write_json(output / "advection_source_manifest.json", source_state())
    for record in records[len(prior_keys):]:
        if record.get("checkpoint_sha256") != amendment["checkpoint_sha256"]:
            raise ValueError("Existing advection record has a different model")
    for record in records:
        audit_outcome(record["outcome"])
    for key in ("SETUP_RELAX_TYPE", "SETUP_NUM_SWEEPS", "SETUP_CYCLE_TYPE", "SETUP_MAX_LEVELS",
                "AMG_RELAX_TYPE", "AMG_COARSE_RELAX_TYPE", "AMG_CYCLE_TYPE"):
        os.environ.pop(key, None)
    configure_smoother_profile(protocol["smoother_profile"])
    patterns = {f"fixed_{w:.2f}": [w] for w in protocol["fixed_weights"]["advection"]}
    patterns.update({f"schedule_{name}": pattern for name, pattern in SCHEDULES.items()})
    before = {name: bundle.controller.snapshot_learning_state() for name, bundle in policies.items()}
    rng_before = {name: json.dumps(bundle.controller.rng.bit_generator.state) for name, bundle in policies.items()}
    print(json.dumps({"stage": "02_advection", "retained_records": len(prior_keys),
                      "remaining_primary_attempts": len(plan) - len(appended)}), flush=True)
    with trajectory.open("a") as handle:
        for rank, (_family, phase, case, setup, policy) in enumerate(plan[len(appended):], start=len(appended) + 1):
            mkw = inputs["evaluation"]["advection"][case - 1]["mkw"]
            if policy in patterns:
                outcome = run_schedule(mkw, setups[setup], patterns[policy])
            else:
                outcome = _run_case(bundle=policies[policy], setup_row={"mkw": mkw, "params": setups[setup]},
                                    args=NATIVE_ARGS, learn=False, explore=False)
            record = {"phase": phase, "family": "advection", "case": case, "setup": setup,
                      "policy": policy, "mkw": mkw, "params": setups[setup], "outcome": outcome,
                      "checkpoint_sha256": amendment["checkpoint_sha256"]}
            handle.write(json.dumps(_json_ready(record)) + "\n")
            handle.flush()
            records.append(record)
            audit_outcome(outcome)
            if policy in policies and (outcome.get("controller_update_committed", False)
                                       or any(outcome.get("cycle_explored", []))
                                       or any(outcome.get("cycle_epsilons", []))):
                raise AssertionError("Frozen evaluation learned or explored")
            if rank % 28 == 0:
                print(json.dumps({"stage": "02_advection", "completed_cases": case,
                                  "completed_primary_attempts": rank, "total": len(plan)}), flush=True)
    audit_frozen_policies(policies, before, rng_before)
    after_numerics = {name: audit_controller(bundle.controller) for name, bundle in policies.items()}
    with trajectory.open("rb") as handle:
        if handle.read(len(prior_bytes)) != prior_bytes:
            raise AssertionError("Previously completed trajectories changed")
    previous_summary = json.loads((output / "summary.json").read_text())
    report = summarize(records)
    report.update({"complete": True, "development_only": True, "primary_attempts": len(records),
                   "evaluation_primary_attempts": 1512, "advection_training_cases": 58,
                   "advection_comparison_primary_attempts": 336, "diffusion_complete": True,
                   "advection_complete": True, "cost_accounting_verified": True,
                   "preserved_prior_records": len(prior_keys), "duplicate_cells": 0,
                   "frozen_policy_audit": {**previous_summary.get("frozen_policy_audit", {}),
                                           "advection": "All learning-state fields and RNG exactly unchanged"},
                   "boundary_audits": previous_summary["boundary_audits"],
                   "advection_checkpoint": amendment["checkpoint"],
                   "advection_checkpoint_audits": {"before": before_numerics, "after": after_numerics}})
    _write_json(output / "summary.json", report)
    _write_json(output / "advection_complete.json", {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "primary_attempts": 336,
        "trajectory_sha256": file_hash(trajectory), "checkpoint_sha256": file_hash(checkpoint),
        "frozen_state_and_rng_unchanged": True, "prior_records_unchanged": True,
    })
    print(json.dumps({"completed": True, "primary_attempts": len(records), "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-advection", type=Path, metavar="CHECKPOINT",
                        help="Append only the authorized 336 frozen advection comparisons")
    args = parser.parse_args()
    if args.resume_advection is not None:
        with (args.output / ".resume.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            resume_advection(args.output, args.resume_advection)
        return
    protocol, inputs, setups, cases, weights, policies, training, training_cases, advection_raw = prepare()
    if not args.run:
        print(json.dumps(protocol, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ("SETUP_RELAX_TYPE", "SETUP_NUM_SWEEPS", "SETUP_CYCLE_TYPE", "SETUP_MAX_LEVELS",
                "AMG_RELAX_TYPE", "AMG_COARSE_RELAX_TYPE", "AMG_CYCLE_TYPE"):
        os.environ.pop(key, None)
    configure_smoother_profile(protocol["smoother_profile"])
    _write_json(args.output / "protocol.json", protocol)
    _write_json(args.output / "inputs.json", inputs)
    _write_json(args.output / "source_manifest.json", source_state())
    rng = np.random.default_rng(protocol["seeds"]["order"])
    records, boundary_audits = [], []

    def retain(handle, record):
        audit_outcome(record["outcome"])
        records.append(record)
        handle.write(json.dumps(_json_ready(record)) + "\n")
        handle.flush()

    with (args.output / "trajectories.jsonl").open("x") as handle:
        for family in ("diffusion", "advection"):
            if family == "advection":
                for index, (mkw, _context) in enumerate(training_cases):
                    setup = ("reference", "learned")[index % 2]
                    outcome = _run_case(bundle=training, setup_row={"mkw": mkw, "params": setups[setup]},
                                        args=NATIVE_ARGS, learn=True, explore=True)
                    retain(handle, {"phase": "training", "family": family, "case": index + 1,
                                    "setup": setup, "policy": "rl_train", "outcome": outcome})
                    if (index + 1) % 16 == 0:
                        boundary_audits.append({"case": index + 1, **audit_controller(training.controller)})
                        print(json.dumps({"family": family, "training_cases": index + 1}), flush=True)
                checkpoint = args.output / "advection_checkpoint.npz"
                training.save(checkpoint)
                policies = {f"rl_{scoring}": build_bundle(advection_raw, "bandit_lstdq", checkpoint, beta)
                            for scoring, beta in (("lcb", 2.), ("mean", 0.))}
            patterns = {f"fixed_{w:.2f}": [w] for w in weights[family]}
            patterns.update({f"schedule_{name}": pattern for name, pattern in SCHEDULES.items()})
            before = {name: bundle.controller.snapshot_learning_state() for name, bundle in policies.items()}
            rng_before = {name: json.dumps(bundle.controller.rng.bit_generator.state)
                          for name, bundle in policies.items()}
            for index, (mkw, _context) in enumerate(cases[family]):
                cells = [(setup, policy) for setup in setups for policy in (*patterns, *policies)]
                for rank in rng.permutation(len(cells)):
                    setup, policy = cells[rank]
                    if policy in patterns:
                        outcome = run_schedule(mkw, setups[setup], patterns[policy])
                    else:
                        outcome = _run_case(bundle=policies[policy], setup_row={"mkw": mkw, "params": setups[setup]},
                                            args=NATIVE_ARGS, learn=False, explore=False)
                    retain(handle, {"phase": "selection" if index < 6 else "evaluation", "family": family,
                                    "case": index + 1, "setup": setup, "policy": policy, "outcome": outcome})
                print(json.dumps({"family": family, "completed_cases": index + 1,
                                  "total_primary_attempts": len(records)}), flush=True)
            audit_frozen_policies(policies, before, rng_before)
    report = summarize(records)
    report.update({"primary_attempts": len(records), "frozen_policy_state_unchanged": True,
                   "cost_accounting_verified": True, "boundary_audits": boundary_audits})
    _write_json(args.output / "summary.json", report)
    print(json.dumps({"completed": True, "primary_attempts": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
