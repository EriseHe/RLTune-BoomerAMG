"""Frozen, matched-hierarchy Module 05; no learner is trained by this runner.

Use ``prepare``, ``preflight``, then ``run``. The supervisor is resumable and
records development selection before dispatching any test evaluations.
"""
from __future__ import annotations

import os

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"

from experiments.diagnostics.solve_control import _project_paths  # noqa: E402,F401

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI

from experiments.paper_final.run_02_diagnostics import (
    audit_frozen_policies, audit_outcome, build_bundle, run_schedule,
)
from hypre.bindings import RecoveryOutcome, create_env, run_with_default_fallback
from hypre.bindings.config import configure_smoother_profile
from joint_online_common import report_online_outcome
from online_td_experiment_common import _json_ready
from problems.registry import context_for_setup_method
from problems.streams import generate_scalar_anisotropic_diffusion_instances
from run_lstdq_v3_stability import _run_case
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from setup_aware_compare_common import (
    augment_setup_params, build_online_linucb_branch, solve_no_rl_case,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/paper_final/05_policy/20260927_diffusion60_6seeds_100cases"
PROFILE = "l1_jacobi_direct_coarse"
SOURCES = ("bandit_default", "bandit_lstdq")
NATIVE_ARGS = argparse.Namespace(tol=1e-6, max_cycles=50)
SEEDS = {"development": 92705001, "test": 92705002, "preflight": 92705003,
         "candidates": 92705004, "order": 92705005, "repeats": 92705006}


def now():
    return datetime.now(timezone.utc).isoformat()


def ready(value):
    value = _json_ready(value)
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(ready(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def emit(output, message, **fields):
    event = {"at": now(), "message": message, **fields}
    with (output / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(ready(event), sort_keys=True) + "\n")
    print(f"[{event['at']}] {message}" + (" " + json.dumps(ready(fields)) if fields else ""), flush=True)


def seed_directory(seed):
    if seed <= 3:
        batch = "20260925_macmini_module04_3seeds"
    else:
        batch = "20260926_macmini_module04_seeds4to6"
    name = "diffusion_60" if seed == 1 else f"diffusion_60_s{seed}"
    return ROOT / "results/paper_final/04_online" / batch / f"seed_{seed}" / name


def policy_library():
    policies = {}
    for i in range(41):
        w = round(1 + i * .05, 2)
        policies[f"fixed_{w:.2f}"] = {"kind": "fixed", "pattern": [w]}
    grid = [1., 1.5, 2., 2.5, 3.]
    for a in grid:
        for b in grid:
            if a != b:
                policies[f"periodic_{a:.2f}_{b:.2f}"] = {"kind": "periodic", "pattern": [a, b]}
    for a, b in [(1., 3.), (3., 1.)]:
        for prefix in [2, 4, 8]:
            for tail in ([1.5] if prefix == 2 else [1.5, 2.]):
                policies[f"prefix_{a:.2f}_{b:.2f}_n{prefix}_tail{tail:.2f}"] = {
                    "kind": "prefix", "pattern": [a, b] * (prefix // 2) + [tail] * (50 - prefix),
                    "prefix_cycles": prefix, "tail_weight": tail,
                }
    policies["rl_frozen_lcb"] = {"kind": "rl", "learn": False, "explore": False, "beta": 2.}
    policies["native_hypre_default"] = {"kind": "native", "profile": "hypre_default", "params": {}}
    policies["native_chebyshev16"] = {"kind": "native", "profile": PROFILE, "params": {"relax_type": 16}}
    return policies


def load_bundle(output, seed):
    directory = output / "checkpoints" / f"seed_{seed}"
    bundle = build_bundle(read(directory / "experiment_config.json"), "bandit_lstdq",
                          directory / "bandit_lstdq_final.npz")
    # Normalize episode scratch buffers before auditing; this performs no update.
    bundle.controller.start_episode(initial_environment_weight=1.)
    bundle.controller.finish_episode(learned=False)
    return bundle


def frozen_snapshot(bundle):
    return (bundle.controller.snapshot_learning_state(),
            json.dumps(_json_ready(bundle.controller.rng.bit_generator.state)))


def verify_frozen(bundle, before):
    audit_frozen_policies({"rl": bundle}, {"rl": before[0]}, {"rl": before[1]})


def environment_check(output):
    reference = read(ROOT / "results/paper_final/04_online/20260925_macmini_n60/provenance/environment_check.json")
    package_reference = read(ROOT / "results/paper_final/04_online/20260926_macmini_module04_seeds4to6/provenance/packages.json")
    current = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
    native = {p: file_hash(ROOT / p) for p in reference["native_hashes"]}
    result = {"at": now(), "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "thread_environment": {k: os.environ[k] for k in THREAD_KEYS},
              "native_hashes": native, "native_identical": native == reference["native_hashes"],
              "python_identical": sys.version == reference["python"], "packages": current,
              "reference_packages": package_reference,
              "parallelism": "At most three independent processes, one MPI rank and one library thread each; no claimed core affinity"}
    result["mpi_world_size"] = MPI.COMM_WORLD.Get_size()
    result["mpi"] = MPI.Get_library_version().rstrip("\x00\n ")
    result["mpi_identical"] = result["mpi"] == reference["mpi"].rstrip("\x00\n ")
    result["unexpected_setup_overrides"] = {k: os.environ[k] for k in
        ("SETUP_NUM_SWEEPS", "SETUP_CYCLE_TYPE", "SETUP_MAX_LEVELS") if os.environ.get(k, "").strip()}
    # The archive may use either pip's [{name, version}] form or a name map.
    previous = ({r["name"].lower().replace("_", "-"): r["version"] for r in package_reference}
                if isinstance(package_reference, list) else package_reference)
    result["package_differences"] = {k: [v, current.get(k.lower().replace("_", "-"))]
        for k, v in previous.items() if current.get(k.lower().replace("_", "-")) != v}
    dump(output / "environment.json", result)
    if (not result["native_identical"] or not result["python_identical"] or result["package_differences"]
            or not result["mpi_identical"] or result["mpi_world_size"] != 1 or result["unexpected_setup_overrides"]):
        raise RuntimeError("Module 04 environment does not match; see environment.json")
    return result


def checkpoint_payload_hash(path, *, exclude=()):
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as payload:
        for key in sorted(set(payload.files) - set(exclude)):
            a = np.asarray(payload[key])
            h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def prepare(output, *, cases=100, development=100, seeds=tuple(range(1, 7)), seed_offset=0):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        protocol = read(output / "protocol.json")
        if (protocol["test_cases"] != cases or protocol["development_cases"] != development
                or protocol["training_seeds"] != list(seeds) or protocol["seed_offset"] != seed_offset):
            raise ValueError("Existing output has a different protocol; never overwrite it")
        return
    if cases < 1 or development < 1 or not seeds or any(s not in range(1, 7) for s in seeds):
        raise ValueError("Positive case counts and existing training seeds 1..6 are required")
    run_seeds = {k: v + seed_offset for k, v in SEEDS.items()}
    environment_check(output)
    policies = policy_library()
    inputs = {}
    for phase, count in [("development", development), ("test", cases), ("preflight", 4)]:
        inputs[phase] = [{"case_id": i, "input_id": digest(mkw), "mkw": mkw, "context": context.tolist()}
            for i, (mkw, context) in enumerate(generate_scalar_anisotropic_diffusion_instances(
                count=count, seed=run_seeds[phase], grid_choices=[(60, 60, 60)], c_min=1, c_max=1000))]
    all_ids = [r["input_id"] for rows in inputs.values() for r in rows]
    if len(set(all_ids)) != len(all_ids):
        raise AssertionError("Input partitions overlap")
    wanted = set(all_ids)
    old_count = 0
    overlap = []
    for batch in ["20260925_macmini_module04_3seeds", "20260926_macmini_module04_seeds4to6"]:
        for path in (ROOT / "results/paper_final/04_online" / batch).glob("seed_*/*/trajectories/default_setup_default_solve.jsonl"):
            with path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    old_count += 1
                    if digest(record["mkw"]) in wanted:
                        overlap.append(str(path))
    if overlap:
        raise AssertionError(f"New inputs overlap historical training: {overlap}")
    repeat_ids = sorted(random.Random(run_seeds["repeats"]).sample(range(cases), max(1, cases // 10)))
    protocol = {"module": "05", "created_at": now(), "problem": "60^3 scalar anisotropic diffusion only",
        "training_seeds": list(seeds), "test_cases": cases, "development_cases": development,
        "same_inputs_across_training_seeds": True, "sources": list(SOURCES), "seeds": run_seeds,
        "seed_offset": seed_offset, "smoke_only": seed_offset != 0,
        "input_distribution": "Existing Module 04 uniform [1,1000] diffusion coefficients and seeded Gaussian RHS",
        "input_overlap_audit": {"historical_records_checked": old_count, "overlaps": 0},
        "test_repeat_case_ids": repeat_ids, "additional_repeats": 2, "max_workers": 3,
        "profile": PROFILE, "tolerance": 1e-6, "max_cycles": 50,
        "scope": "Frozen component evaluation; no setup or solve learning, no solve-phase bandit",
        "seed_1_note": "Seed 1 was used in earlier development; all Module 05 inputs are fresh, all six checkpoints are included at user request",
        "selection_rule": "Per seed/source: fewest development final failures, then summed solve + controller + recovery setup cost, then policy name",
        "oracle_rule": "Best fixed -- test hindsight: one grid weight per seed/source minimizes summed native solve plus recovery setup over ALL test cases, restricted to policies with no final failure in any prescribed repetition. If none qualifies, report no all-success best fixed and separately report the maximum-coverage choice. Per-problem best fixed minimizes the same cost among successful complete procedures for each case. These are best-observed grid diagnostics, not continuous optima or estimates for unseen problems. Average prescribed repetitions, never select fastest timing.",
        "primary_presentation": "Native solve plus recovery setup first; controller-inclusive continuation and full setup-plus-solve totals also retained",
        "hierarchy_matching": "Each trial creates a fresh matrix/RHS/zero iterate and rebuilds from the same frozen setup tuple; identity hash labels inputs/configuration, not a direct hash of native hierarchy arrays; numerical repeat checks are in preflight.json",
        "recovery": "One primary matched hierarchy, followed on any numerical failure (including native setup failure) by the same reference AMG fallback; the evaluation wrapper supplies the reference fallback for RL setup failures too. No setup reselection or learning. Execution/programming errors abort rather than becoming numerical outcomes.",
        "conventional_references": "Native HYPRE default smoother and native Chebyshev type 16 with library-default polynomial settings; separate practical references, not a tuned polynomial-optimum claim. Their primary comparison includes full setup plus solve cost, including spectral preparation, because that setup work is not common.",
        "test_schedule_rule": "Always both orders of (1,2),(1,3), plus development-selected periodic pair and its reverse, plus development-selected prefix-tail; duplicate policies evaluated once",
        "candidate_selection": "Frozen Module 04 LinUCB statistics and LCB rule, new shared structured512 AOT rows; only the copied AOT cursor is reset to zero; choices saved before solve outcomes",
        "timing": "Three concurrent single-thread workers; randomized serial policy order within each case/source; repetitions cover the same preselected input subset for every policy",
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "policies": policies}
    dump(output / "protocol.json", protocol)
    dump(output / "inputs.json", inputs)
    emit(output, "Fresh development and test inputs saved; no training overlap", cases=cases, development=development, seeds=list(seeds))
    ordered_inputs = [(phase, row) for phase in ["development", "test", "preflight"] for row in inputs[phase]]
    jobs = {phase: [] for phase in inputs}
    selection_audits = []
    for seed in seeds:
        original = seed_directory(seed)
        dest = output / "checkpoints" / f"seed_{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["experiment_config.json", "checkpoints/bandit_lstdq_final.npz", "checkpoints/bandit_lstdq_final.npz.encoder.json"]:
            shutil.copy2(original / name, dest / Path(name).name)
        raw = read(dest / "experiment_config.json")
        space = SetupConfigurationSpace.from_mapping("recommended", raw["setup"]["configuration_spaces"]["recommended"])
        for source in SOURCES:
            original_state = original / "final_bandit_states" / (source + ".npz")
            copied = dest / (source + "_setup_original.npz")
            shutil.copy2(original_state, copied)
            with np.load(copied, allow_pickle=False) as state:
                normalized = {k: state[k].copy() for k in state.files}
            normalized["candidate_schedule_cursor"] = np.asarray(0, dtype=np.int64)
            normalized_path = dest / (source + "_setup_evaluation.npz")
            np.savez_compressed(normalized_path, **normalized)
            if checkpoint_payload_hash(copied, exclude=["candidate_schedule_cursor"]) != checkpoint_payload_hash(normalized_path, exclude=["candidate_schedule_cursor"]):
                raise AssertionError("Setup checkpoint changed beyond its prescribed candidate cursor")
            branch, _ = build_online_linucb_branch(seed=raw["seeds"]["bandit"], learner_kind="linucb",
                tune_dim=7, tune7_variant="categorical", action_space_mode="full_cartesian", solver_tol=1e-6,
                solver_max_iter=50, parameter_resolution=20, configuration_space=space,
                candidate_schedule_dir=output / "candidate_schedule", candidate_schedule_rounds=len(ordered_inputs),
                candidate_schedule_seed=run_seeds["candidates"], candidate_sampling="structured512",
                context_dim=4, context_interaction_indices=(1, 2, 3))
            model = branch.policy.model
            model.load_mutable_state(normalized_path)
            before = checkpoint_payload_hash(normalized_path, exclude=["candidate_schedule_cursor", "rng_state", "metadata"])
            for phase, row in ordered_inputs:
                context = context_for_setup_method(problem_kind="scalar_anisotropic_diffusion", setup_kind="linucb",
                    matrix_kwargs=row["mkw"], stream_context=np.asarray(row["context"]), setup_context="diffusion3d", grid_norm_div=60)
                params, info = branch.policy.select(context=context, parameter_space=branch.parameter_space)
                job = {**row, "seed": seed, "source": source, "params": params, "setup_selection": info,
                       "candidate_row": model._candidate_schedule.cursor - 1}
                job["hierarchy_id"] = digest({"mkw": row["mkw"], "params": params, "profile": PROFILE})
                jobs[phase].append(job)
                branch.policy.cancel_pending()
            after_path = dest / (source + "_setup_after_selection.npz")
            model.save_mutable_state(after_path)
            after = checkpoint_payload_hash(after_path, exclude=["candidate_schedule_cursor", "rng_state", "metadata"])
            if before != after:
                raise AssertionError("Setup learner statistics changed during frozen selection")
            selection_audits.append({"seed": seed, "source": source, "original_sha256": file_hash(copied),
                                     "frozen_statistics_hash": before, "choices": len(ordered_inputs), "unchanged": True})
            del model, branch
        emit(output, "Frozen hierarchy choices prepared", seed=seed)
    for phase, rows in jobs.items():
        random.Random(run_seeds["order"] + list(inputs).index(phase)).shuffle(rows)
        dump(output / f"jobs_{phase}.json", rows)
    dump(output / "setup_freeze_audit.json", selection_audits)
    source_paths = subprocess.check_output(["git", "ls-files", "*.py", "*.c", "*.h"], cwd=ROOT, text=True).splitlines()
    source_paths.append(str(Path(__file__).relative_to(ROOT)))
    source_paths.append("experiments/paper_final/test_05_policy.py")
    source_hashes = {name: file_hash(ROOT / name) for name in sorted(set(source_paths)) if (ROOT / name).is_file()}
    dump(output / "source_manifest.json", source_hashes)
    shutil.copy2(__file__, output / "runner_snapshot.py")
    dump(output / "prepared.json", {"at": now(), "protocol_sha256": file_hash(output / "protocol.json"),
        "inputs_sha256": file_hash(output / "inputs.json"), "jobs_sha256": {p: file_hash(output / f"jobs_{p}.json") for p in inputs},
        "checkpoint_sha256": {str(p.relative_to(output)): file_hash(p) for p in (output / "checkpoints").rglob("*") if p.is_file()}})
    emit(output, "Preparation complete")


def reject_execution_error(outcome):
    attempts = [outcome, *outcome.get("primary_attempts", [])]
    if any(a.get("failure_origin") == "execution" for a in attempts) or outcome.get("fallback_failure_origin") == "execution":
        raise RuntimeError(f"Experiment execution error: {outcome.get('primary_failure_reason', outcome.get('failure_reason'))}")


def run_policy(job, name, spec, bundle):
    started = time.perf_counter()
    configure_smoother_profile(PROFILE)
    if spec["kind"] == "rl":
        outcome = _run_case(bundle=bundle, setup_row=job, args=NATIVE_ARGS, learn=False, explore=False)
        reject_execution_error(outcome)
        if outcome["failed"] and not outcome.get("fallback_used", False):
            # The online learner defers structural setup failure to its setup
            # bandit. Frozen evaluation has no reselection: apply the same
            # reference recovery available to constants and schedules.
            primary_attempt = RecoveryOutcome.from_mapping(outcome).primary
            recovery = run_with_default_fallback(lambda: primary_attempt, lambda: solve_no_rl_case(
                params=dict(DEFAULT_SETUP_PARAMS), mkw=job["mkw"], solver_tol=1e-6,
                solver_max_iter=50, augment_params=augment_setup_params))
            outcome = report_online_outcome(recovery.to_result(), bandit_timing={})
    elif spec["kind"] == "native":
        def primary():
            configure_smoother_profile(spec["profile"])
            try:
                return solve_no_rl_case(params={**job["params"], **spec["params"]}, mkw=job["mkw"],
                    solver_tol=1e-6, solver_max_iter=50, augment_params=augment_setup_params)
            finally:
                configure_smoother_profile(PROFILE)
        recovery = run_with_default_fallback(primary, lambda: solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS),
            mkw=job["mkw"], solver_tol=1e-6, solver_max_iter=50, augment_params=augment_setup_params))
        outcome = report_online_outcome(recovery.to_result(), bandit_timing={})
        outcome["completed_residual_norm"] = (recovery.fallback.residual_norm if recovery.fallback_used else recovery.primary.residual_norm)
    else:
        outcome = run_schedule(job["mkw"], job["params"], spec["pattern"])
    elapsed = time.perf_counter() - started
    reject_execution_error(outcome)
    audit_outcome(outcome)
    # All reported costs include failed attempts/recovery; primary setup is common.
    native_continuation = float(outcome["solve_runtime"]) + float(outcome.get("fallback_setup_runtime", 0))
    controller = float(outcome.get("infer_runtime", 0))
    row = {"seed": job["seed"], "source": job["source"], "case_id": job["case_id"],
           "input_id": job["input_id"], "hierarchy_id": job["hierarchy_id"], "policy": name,
           "kind": spec["kind"], "wall_sec": elapsed, "native_continuation_sec": native_continuation,
           "inclusive_continuation_sec": native_continuation + controller,
           "native_total_sec": float(outcome["setup_runtime"]) + float(outcome["solve_runtime"]),
           "inclusive_total_sec": float(outcome["end_to_end_runtime"]),
           "success": not bool(outcome["failed"]), "outcome": outcome}
    if not all(math.isfinite(row[k]) and row[k] >= 0 for k in ["wall_sec", "native_continuation_sec", "inclusive_continuation_sec", "native_total_sec", "inclusive_total_sec"]):
        raise AssertionError("Invalid timing")
    return row


def preflight(output):
    environment_check(output)
    configure_smoother_profile(PROFILE)
    protocol = read(output / "protocol.json")
    jobs = read(output / "jobs_preflight.json")
    policies = protocol["policies"]
    audit = {"started_at": now(), "checks": [], "passed": False}
    for seed in protocol["training_seeds"]:
        bundle = load_bundle(output, seed)
        before = frozen_snapshot(bundle)
        for source in SOURCES:
            job = next(j for j in jobs if j["seed"] == seed and j["source"] == source and j["case_id"] == 0)
            with create_env(**job["mkw"]) as env:
                env.prepare_rl(params=job["params"])
                stages = env.cycle_relax_types
                if stages != (18, 18, 9):
                    raise AssertionError(f"Wrong primary smoother: {stages}")
            first = run_policy(job, "fixed_1.00", policies["fixed_1.00"], bundle)
            tested = {}
            for name in ["fixed_3.00", "periodic_1.00_3.00", "prefix_1.00_3.00_n4_tail1.50", "rl_frozen_lcb", "native_hypre_default", "native_chebyshev16"]:
                tested[name] = run_policy(job, name, policies[name], bundle)
            native_stages = {}
            for name, expected_stages in [("native_hypre_default", (13, 14, 9)), ("native_chebyshev16", (16, 16, 9))]:
                spec = policies[name]
                configure_smoother_profile(spec["profile"])
                try:
                    with create_env(**job["mkw"]) as env:
                        env.prepare_rl(params={**job["params"], **spec["params"]})
                        native_stages[name] = env.cycle_relax_types
                        if env.cycle_relax_types != expected_stages:
                            raise AssertionError(f"Unexpected native smoother stages: {name}")
                finally:
                    configure_smoother_profile(PROFILE)
            second = run_policy(job, "fixed_1.00", policies["fixed_1.00"], bundle)
            np.testing.assert_allclose(first["outcome"]["cycle_residuals"], second["outcome"]["cycle_residuals"], rtol=1e-12, atol=1e-15)
            if first["success"] != second["success"]:
                raise AssertionError("Repeated reference has a different status")
            verify_frozen(bundle, before)
            audit["checks"].append({"seed": seed, "source": source, "profile": stages,
                "reference_rebuild_repeat_agrees": True, "frozen_controller_unchanged": True,
                "native_reference_stages": native_stages,
                "policies": {name: {"success": row["success"], "recovery": row["outcome"].get("fallback_used", False),
                                     "native_continuation_sec": row["native_continuation_sec"]} for name, row in tested.items()}})
        emit(output, "Native preflight passed", seed=seed)
    audit.update(passed=True, finished_at=now())
    dump(output / "preflight.json", audit)
    return audit


def verify_inputs(output, *, check_source=True):
    saved = read(output / "prepared.json")
    files = {"protocol.json": saved["protocol_sha256"], "inputs.json": saved["inputs_sha256"],
             **{f"jobs_{p}.json": h for p, h in saved["jobs_sha256"].items()},
             **saved["checkpoint_sha256"]}
    for path, expected in files.items():
        if file_hash(output / path) != expected:
            raise RuntimeError(f"Prepared input changed: {path}")
    if check_source:
        for path, expected in read(output / "source_manifest.json").items():
            if file_hash(ROOT / path) != expected:
                raise RuntimeError(f"Experiment source changed after preparation: {path}")


def cell_key(row):
    return (row["seed"], row["source"], row["case_id"], row.get("repeat", 0), row["policy"])


def read_records(path, *, repair_tail=False):
    """Stream durable rows; only a torn final append may be repaired on resume."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r+b" if repair_tail else "rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                return
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not repair_tail or handle.read(1):
                    raise ValueError(f"Corrupt experiment record in {path} at byte {start}")
                path.with_suffix(f".torn.{time.time_ns()}").write_bytes(line)
                handle.seek(start)
                handle.truncate()
                return
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Unterminated record in {path}")
                handle.seek(0, 2)
                handle.write(b"\n")
            yield row


def phase_jobs(output, phase):
    protocol = read(output / "protocol.json")
    if phase != "repetitions":
        return [{**j, "repeat": 0} for j in read(output / f"jobs_{phase}.json")]
    jobs = [{**j, "repeat": repeat} for j in read(output / "jobs_test.json")
            if j["case_id"] in protocol["test_repeat_case_ids"]
            for repeat in range(1, 1 + protocol["additional_repeats"])]
    random.Random(protocol["seeds"]["repeats"]).shuffle(jobs)
    return jobs


def policies_for_job(protocol, selections, job, phase):
    if phase == "development":
        return sorted(name for name, spec in protocol["policies"].items() if spec["kind"] != "rl")
    selected = selections[f"{job['seed']}/{job['source']}"]
    periodic = selected["periodic"]["policy"]
    a, b = protocol["policies"][periodic]["pattern"]
    names = {name for name, spec in protocol["policies"].items() if spec["kind"] in ("fixed", "rl", "native")}
    names.update(["periodic_1.00_2.00", "periodic_2.00_1.00", "periodic_1.00_3.00", "periodic_3.00_1.00",
                  periodic, f"periodic_{b:.2f}_{a:.2f}", selected["prefix"]["policy"]])
    return sorted(names)


def expected_cells(output, phase):
    protocol = read(output / "protocol.json")
    selections = {} if phase == "development" else read(output / "selected_baselines.json")["selections"]
    return {cell_key({**job, "policy": policy}) for job in phase_jobs(output, phase)
            for policy in policies_for_job(protocol, selections, job, phase)}


def worker(output, phase, worker_id, workers):
    directory = output / "raw" / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"worker_{worker_id}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _worker(output, phase, worker_id, workers)


def _worker(output, phase, worker_id, workers):
    if workers not in range(1, 4) or worker_id not in range(workers):
        raise ValueError("At most three single-thread workers")
    verify_inputs(output)
    protocol = read(output / "protocol.json")
    selections = {} if phase == "development" else read(output / "selected_baselines.json")["selections"]
    jobs = phase_jobs(output, phase)[worker_id::workers]
    directory = output / "raw" / phase
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"worker_{worker_id}.jsonl"
    expected = {cell_key({**job, "policy": p}) for job in jobs for p in policies_for_job(protocol, selections, job, phase)}
    done = set()
    wall_done = 0.
    for row in read_records(path, repair_tail=True):
        key = cell_key(row)
        if key in done or key not in expected:
            raise AssertionError("Duplicate or foreign completed trial on resume")
        done.add(key)
        wall_done += row["wall_sec"]
    bundles, frozen = {}, {}
    state = {"pid": os.getpid(), "worker": worker_id, "workers": workers, "phase": phase,
             "total": len(expected), "done": len(done), "status": "running", "started_at": now()}
    progress_path = output / "progress" / f"{phase}_{worker_id}.json"
    with path.open("a", buffering=1) as handle:
        for job in jobs:
            names = policies_for_job(protocol, selections, job, phase)
            random.Random(digest([protocol["seeds"]["order"], phase, job["seed"], job["source"], job["case_id"], job["repeat"]])).shuffle(names)
            seed = job["seed"]
            if seed not in bundles:
                bundles[seed] = load_bundle(output, seed)
                frozen[seed] = frozen_snapshot(bundles[seed])
            for name in names:
                if cell_key({**job, "policy": name}) in done:
                    continue
                state.update(at=now(), seed=seed, source=job["source"], case=job["case_id"] + 1,
                             repeat=job["repeat"], policy=name, done=len(done), wall_sec=wall_done)
                dump(progress_path, state)
                row = run_policy(job, name, protocol["policies"][name], bundles[seed])
                row.update(phase=phase, repeat=job["repeat"], worker=worker_id, at=now())
                handle.write(json.dumps(ready(row), separators=(",", ":"), allow_nan=False) + "\n")
                done.add(cell_key(row))
                wall_done += row["wall_sec"]
            handle.flush()
            os.fsync(handle.fileno())
            verify_frozen(bundles[seed], frozen[seed])
    if done != expected:
        raise AssertionError("Worker completed with missing trials")
    state.update(at=now(), done=len(done), wall_sec=wall_done, status="complete", frozen_audit_passed=True)
    dump(progress_path, state)
    dump(directory / f"worker_{worker_id}.complete.json", state)


COST_FIELDS = ("native_continuation_sec", "inclusive_continuation_sec", "native_total_sec", "inclusive_total_sec", "wall_sec")


def compact_row(row):
    outcome = row["outcome"]
    return {**{k: row[k] for k in ("seed", "source", "case_id", "policy", "kind", "success", *COST_FIELDS)},
            "repeat": row.get("repeat", 0), "recovery": bool(outcome.get("fallback_used", False)),
            "setup_sec": outcome["setup_runtime"], "solve_sec": outcome["solve_runtime"],
            "controller_sec": outcome.get("infer_runtime", 0),
            "recovery_setup_sec": outcome.get("fallback_setup_runtime", 0),
            "primary_cycles": outcome.get("primary_cycles", outcome.get("iterations", 0)),
            "fallback_cycles": outcome.get("fallback_cycles", 0)}


def phase_rows(output, phase):
    seen = set()
    for path in sorted((output / "raw" / phase).glob("worker_*.jsonl")):
        for row in read_records(path):
            key = cell_key(row)
            if key in seen:
                raise AssertionError(f"Duplicate trial in {phase}: {key}")
            seen.add(key)
            yield compact_row(row)
    if seen != expected_cells(output, phase):
        raise AssertionError(f"Incomplete {phase}: {len(seen)} recorded trials")


def totals(rows):
    return {"cases": len(rows), "failures": sum(not r["success"] for r in rows),
            "recoveries": sum(r["recovery"] for r in rows),
            **{k: sum(r[k] for r in rows) for k in (*COST_FIELDS, "setup_sec", "solve_sec", "controller_sec",
                                                  "recovery_setup_sec", "primary_cycles", "fallback_cycles")}}


def choose_development(output):
    grouped = defaultdict(list)
    for row in phase_rows(output, "development"):
        grouped[(row["seed"], row["source"], row["kind"], row["policy"])].append(row)
    selections = defaultdict(dict)
    ranking = defaultdict(list)
    for (seed, source, kind, policy), rows in grouped.items():
        if kind in ("fixed", "periodic", "prefix"):
            ranking[(seed, source, kind)].append({"policy": policy, **totals(rows)})
    for (seed, source, kind), values in ranking.items():
        chosen = min(values, key=lambda r: (r["failures"], r["inclusive_continuation_sec"], r["policy"]))
        selections[f"{seed}/{source}"][kind] = chosen
    result = {"at": now(), "selection_phase": "development", "selections": selections,
              "rule": read(output / "protocol.json")["selection_rule"],
              "raw_sha256": {str(p.relative_to(output)): file_hash(p) for p in sorted((output / "raw/development").glob("*.jsonl"))}}
    path = output / "selected_baselines.json"
    if path.exists():
        previous = read(path)
        if previous["selections"] != ready(result["selections"]) or previous["raw_sha256"] != result["raw_sha256"]:
            raise AssertionError("Development selection changed on resume")
    else:
        dump(path, result)
    return result


def average_repetitions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["source"], row["case_id"], row["policy"])].append(row)
    result = []
    for values in grouped.values():
        first = values[0]
        averaged = {**first, "repeat": "mean", "repetitions": len(values),
                    "success": all(v["success"] for v in values), "recovery": any(v["recovery"] for v in values)}
        for key in (*COST_FIELDS, "setup_sec", "solve_sec", "controller_sec", "recovery_setup_sec", "primary_cycles", "fallback_cycles"):
            averaged[key] = sum(v[key] for v in values) / len(values)
        result.append(averaged)
    return result


def fixed_hindsight(rows):
    """min_w sum_i C_i(w), separately from sum_i min_w C_i(w)."""
    by_policy, by_case = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["kind"] == "fixed":
            by_policy[row["policy"]].append(row)
            by_case[row["case_id"]].append(row)
    candidates = [{"policy": name, **totals(values)} for name, values in by_policy.items()]
    successful = [r for r in candidates if r["failures"] == 0]
    best = min(successful, key=lambda r: (r["native_continuation_sec"], r["policy"])) if successful else None
    coverage_choice = min(candidates, key=lambda r: (r["failures"], r["native_continuation_sec"], r["policy"]))
    individual = []
    uncovered = []
    for case, values in sorted(by_case.items()):
        eligible = [r for r in values if r["success"]]
        if eligible:
            individual.append(min(eligible, key=lambda r: (r["native_continuation_sec"], r["policy"])))
        else:
            uncovered.append(case)
    return {"best_fixed": best, "maximum_coverage_fixed": coverage_choice,
            "per_case": individual, "uncovered_case_ids": uncovered, "fixed_grid": candidates}


def reduction(reference, value):
    return 100 * (1 - value / reference) if reference > 0 else None


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output):
    protocol = read(output / "protocol.json")
    selected = read(output / "selected_baselines.json")["selections"]
    test = list(phase_rows(output, "test"))
    repeated = list(phase_rows(output, "repetitions"))
    blocks = defaultdict(list)
    for row in average_repetitions(test + repeated):
        blocks[(row["seed"], row["source"])].append(row)
    summaries, diagnostics, grids, choices = [], [], [], []
    for (seed, source), rows in sorted(blocks.items()):
        by_policy = defaultdict(list)
        for row in rows:
            by_policy[row["policy"]].append(row)
        oracle = fixed_hindsight(rows)
        best = oracle["best_fixed"]
        reference = totals(by_policy["fixed_1.00"])
        chosen = selected[f"{seed}/{source}"]
        labels = {"Weight 1": "fixed_1.00", "Frozen LSTDQ": "rl_frozen_lcb",
                  "Development fixed": chosen["fixed"]["policy"],
                  "Development periodic": chosen["periodic"]["policy"],
                  "Development prefix-tail": chosen["prefix"]["policy"],
                  "Native HYPRE smoother": "native_hypre_default", "Native Chebyshev": "native_chebyshev16"}
        if best:
            labels["Best fixed - test hindsight"] = best["policy"]
        for label, name in labels.items():
            cost = totals(by_policy[name])
            comparison_cost = "native_total_sec" if protocol["policies"][name]["kind"] == "native" else "native_continuation_sec"
            summaries.append({"seed": seed, "source": source, "method": label, "policy": name, **cost,
                "primary_comparison_cost": comparison_cost,
                "primary_no_overhead_reduction_vs_weight1_pct": reduction(reference[comparison_cost], cost[comparison_cost]),
                "native_total_reduction_vs_weight1_pct": reduction(reference["native_total_sec"], cost["native_total_sec"]),
                "native_reduction_vs_weight1_pct": reduction(reference["native_continuation_sec"], cost["native_continuation_sec"]),
                "inclusive_reduction_vs_weight1_pct": reduction(reference["inclusive_continuation_sec"], cost["inclusive_continuation_sec"]),
                "native_reduction_vs_best_fixed_pct": reduction(best["native_continuation_sec"], cost["native_continuation_sec"]) if best else None})
        eligible_ids = {r["case_id"] for r in oracle["per_case"]}
        paired_ref = totals([r for r in by_policy["fixed_1.00"] if r["case_id"] in eligible_ids])
        individual_cost = totals(oracle["per_case"])
        summaries.append({"seed": seed, "source": source, "method": "Per-problem best fixed - test hindsight",
            "policy": "per_case_oracle", **individual_cost,
            "primary_comparison_cost": "native_continuation_sec",
            "primary_no_overhead_reduction_vs_weight1_pct": reduction(paired_ref["native_continuation_sec"], individual_cost["native_continuation_sec"]),
            "native_total_reduction_vs_weight1_pct": reduction(paired_ref["native_total_sec"], individual_cost["native_total_sec"]),
            "uncovered_cases": len(oracle["uncovered_case_ids"]),
            "native_reduction_vs_weight1_pct": reduction(paired_ref["native_continuation_sec"], individual_cost["native_continuation_sec"]),
            "inclusive_reduction_vs_weight1_pct": reduction(paired_ref["inclusive_continuation_sec"], individual_cost["inclusive_continuation_sec"]),
            "native_reduction_vs_best_fixed_pct": reduction(best["native_continuation_sec"], individual_cost["native_continuation_sec"]) if best else None})
        diagnostics.append({"seed": seed, "source": source, "all_success_best_fixed": best,
            "maximum_coverage_fixed": oracle["maximum_coverage_fixed"], "uncovered_case_ids": oracle["uncovered_case_ids"],
            "repetitions": "Each case has equal weight; prescribed timings are averaged within each case/policy; any repetition failure disqualifies a hindsight choice"})
        grids.extend({"seed": seed, "source": source, **row} for row in oracle["fixed_grid"])
        choices.extend({"seed": seed, "source": source, "case_id": r["case_id"], "policy": r["policy"],
                        "native_continuation_sec": r["native_continuation_sec"]} for r in oracle["per_case"])
    grouped = defaultdict(list)
    for row in summaries:
        grouped[(row["source"], row["method"])].append(row)
    aggregate = []
    for (source, method), rows in sorted(grouped.items()):
        result = {"source": source, "method": method, "training_replicas": len(rows)}
        for metric in ("primary_no_overhead_reduction_vs_weight1_pct", "native_total_reduction_vs_weight1_pct",
                       "native_reduction_vs_weight1_pct", "inclusive_reduction_vs_weight1_pct", "native_reduction_vs_best_fixed_pct"):
            values = [r[metric] for r in rows if r.get(metric) is not None]
            result[metric] = {"values": values, "mean": float(np.mean(values)) if values else None,
                             "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "min": min(values) if values else None, "max": max(values) if values else None}
        aggregate.append(result)
    result = {"at": now(), "protocol": protocol, "per_seed": summaries, "across_seed": aggregate,
              "hindsight_diagnostics": diagnostics,
              "interpretation": "Best-observed finite-grid hindsight; no continuous/global optimality claim. Fresh test inputs are shared across training replicas and hierarchy sources. Across-seed SD is descriptive checkpoint variation, not independent-matrix replication. Costs retain all attempts; coverage shown separately. Concurrent timing is not isolated-core timing. This frozen comparison does not by itself establish that feedback or RL is necessary. Native smoother references use full setup-plus-solve costs as their primary comparison."}
    dump(output / "analysis/summary.json", result)
    write_csv(output / "analysis/per_seed.csv", summaries)
    write_csv(output / "analysis/fixed_grid.csv", grids)
    write_csv(output / "analysis/per_case_fixed_choices.csv", choices)
    return result


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def supervisor(output, workers=3):
    with (output / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _supervisor(output, workers)


def _supervisor(output, workers=3):
    if workers not in range(1, 4):
        raise ValueError("At most three workers")
    verify_inputs(output)
    environment_check(output)
    if not read(output / "preflight.json")["passed"]:
        raise RuntimeError("Native preflight has not passed")
    if (output / "complete.json").exists():
        print("This experiment is already complete.", flush=True)
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())])
    children, logs = [], []
    protocol = read(output / "protocol.json")
    state = {"started_at": now(), "status": "running", "pid": os.getpid(), "workers": workers,
             "awake_pid": awake.pid, "output": str(output), "completed_phases": []}
    dump(output / "status.json", state)
    emit(output, "Module 05 started; Mac sleep inhibited", workers=workers, seeds=protocol["training_seeds"],
         cases=protocol["test_cases"], awake_pid=awake.pid)
    run_start = time.monotonic()
    try:
        for phase in ["development", "test", "repetitions"]:
            if phase == "test":
                choose_development(output)
                emit(output, "Development baselines locked; beginning exhaustive test scan")
            expected = expected_cells(output, phase)
            directory = output / "raw" / phase
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "complete.json"
            if marker.exists():
                saved = read(marker)
                if saved["trials"] != len(expected):
                    raise AssertionError("Completed phase uses a different protocol")
                for name, sha in saved["raw_sha256"].items():
                    if file_hash(output / name) != sha:
                        raise AssertionError(f"Completed data changed: {name}")
                state["completed_phases"].append(phase)
                continue
            state.update(phase=phase, phase_total=len(expected), phase_started_at=now())
            previous_workers = {r["workers"] for p in directory.glob("worker_*.complete.json") for r in [read(p)]}
            for p in (output / "progress").glob(f"{phase}_*.json"):
                previous_workers.add(read(p)["workers"])
            if previous_workers and previous_workers != {workers}:
                raise RuntimeError("Resume must preserve the original worker count")
            children, logs = [], []
            for worker_id in range(workers):
                log = (directory / f"worker_{worker_id}.log").open("a")
                logs.append(log)
                children.append(subprocess.Popen([sys.executable, "-u", "-m", "experiments.paper_final.run_05_policy", "worker",
                    "--output", str(output), "--phase", phase, "--worker-id", str(worker_id), "--workers", str(workers)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            state["worker_pids"] = [p.pid for p in children]
            last_print = 0.
            while True:
                codes = [p.poll() for p in children]
                if any(c not in (None, 0) for c in codes):
                    raise RuntimeError(f"A {phase} worker stopped: exit codes {codes}; inspect raw/{phase}/worker_*.log")
                progress = [read(p) for p in (output / "progress").glob(f"{phase}_*.json")]
                done = sum(r["done"] for r in progress)
                state.update(at=now(), phase_done=done, worker_progress=progress, elapsed_sec=time.monotonic() - run_start)
                dump(output / "status.json", state)
                if time.monotonic() - last_print >= 30:
                    emit(output, f"{phase}: {done:,}/{len(expected):,} evaluations complete")
                    last_print = time.monotonic()
                if all(c == 0 for c in codes):
                    break
                time.sleep(5)
            for log in logs:
                log.close()
            logs = []
            for worker_id in range(workers):
                audit = read(directory / f"worker_{worker_id}.complete.json")
                if not audit["frozen_audit_passed"] or audit["done"] != audit["total"]:
                    raise AssertionError("Worker completion audit failed")
            # Validation also checks the union for exact trial coverage and duplicates.
            count = sum(1 for _ in phase_rows(output, phase))
            dump(marker, {"at": now(), "trials": count,
                "raw_sha256": {str(p.relative_to(output)): file_hash(p) for p in directory.glob("worker_*.jsonl")}})
            state["completed_phases"].append(phase)
            emit(output, f"{phase} complete", evaluations=count)
        state.update(phase="analysis", at=now())
        dump(output / "status.json", state)
        analyze(output)
        verify_inputs(output)
        state.update(status="complete", at=now(), elapsed_sec=time.monotonic() - run_start)
        dump(output / "complete.json", state)
        dump(output / "status.json", state)
        emit(output, "Module 05 complete; full weight scan and summaries saved")
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", at=now(), error=str(exc))
        dump(output / "status.json", state)
        emit(output, "Experiment stopped; completed trials are preserved", error=str(exc))
        raise
    finally:
        stop_children(children)
        for log in logs:
            log.close()
        if awake.poll() is None:
            awake.terminate()
            awake.wait()


def show_status(output, *, watching=False):
    previous = None
    protocol = read(output / "protocol.json")
    while True:
        path = output / "status.json"
        if path.exists():
            state = read(path)
            status = state["status"]
            phase = state.get("phase", "starting")
            done, total = state.get("phase_done", 0), state.get("phase_total", 0)
            marker = (status, phase, done)
            if marker != previous:
                print(f"\nMODULE 05 | 60³ diffusion | {len(protocol['training_seeds'])} frozen training seeds | {protocol['test_cases']} test problems each", flush=True)
                print(f"{datetime.now().strftime('%H:%M:%S')}  {status.upper()}  {phase}: {done:,}/{total:,} evaluations", flush=True)
                for row in sorted(state.get("worker_progress", []), key=lambda r: r["worker"]):
                    source = "Joint hierarchy" if row.get("source") == "bandit_lstdq" else "Setup-only hierarchy"
                    print(f"  Worker {row['worker'] + 1}: seed {row.get('seed', '-')} | {source} | "
                          f"problem {row.get('case', '-')} | {row.get('policy', '-')} | {row['done']:,}/{row['total']:,}", flush=True)
                if status == "running":
                    print("Sleep protection active while the supervisor runs. Closing this viewer does not stop the experiment.", flush=True)
                if state.get("error"):
                    print("Error: " + state["error"], flush=True)
                previous = marker
            if not watching or status in ("complete", "failed", "interrupted"):
                return
        elif not watching:
            print("Prepared; supervisor has not started.")
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "preflight", "run", "worker", "analyze", "status", "watch"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--development", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--seed-offset", type=int, default=0, help="Use disjoint smoke inputs; zero is the formal protocol")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--phase", choices=["development", "test", "repetitions"])
    parser.add_argument("--worker-id", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        prepare(output, cases=args.cases, development=args.development, seeds=tuple(args.seeds), seed_offset=args.seed_offset)
    elif args.command == "preflight":
        verify_inputs(output)
        preflight(output)
    elif args.command == "worker":
        worker(output, args.phase, args.worker_id, args.workers)
    elif args.command == "run":
        supervisor(output, args.workers)
    elif args.command == "analyze":
        analyze(output)
    else:
        show_status(output, watching=args.command == "watch")


if __name__ == "__main__":
    main()
