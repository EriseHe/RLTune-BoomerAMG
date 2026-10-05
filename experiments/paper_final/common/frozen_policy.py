"""Frozen matched-hierarchy execution and its numerical/provenance audits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import time

import numpy as np
from mpi4py import MPI

from experiments.runtime import THREAD_KEYS
from experiments.joint.solve_control.composable_joint_4k import (
    build_composable_solve_runtime,
)
from experiments.joint.solve_control.joint_experiment_config import (
    parse_joint_experiment_config,
    runtime_config_from_spec,
)
from experiments.joint.solve_control.joint_online_common import report_online_outcome
from experiments.joint.solve_control.native_case import run_case as _run_case
from experiments.joint.solve_control.native_evaluation import (
    solve_no_rl_case,
    solve_schedule_case,
)
from hypre.bindings import (
    RecoveryOutcome,
    augment_setup_params,
    run_with_default_fallback,
)
from hypre.bindings.config import configure_smoother_profile
from setup.space import DEFAULT_SETUP_PARAMS
from dataclasses import replace

from .artifacts import (
    ROOT,
    checkpoint_payload_hash,
    dump,
    file_hash,
    now,
    read,
    ready as _json_ready,
)

PROFILE = "l1_jacobi_direct_coarse"
NATIVE_ARGS = argparse.Namespace(tol=1e-6, max_cycles=50)


def build_bundle(raw, method, checkpoint=None, beta=2.0):
    runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
    bundle = build_composable_solve_runtime(
        runtime, specs=runtime.methods
    ).controller_bundles[method]
    if checkpoint is not None:
        bundle.load(checkpoint)
    bundle.controller.spec = replace(bundle.controller.spec, uncertainty_beta=beta)
    return bundle


def run_schedule(mkw, params, pattern):
    schedule = (
        [(50, pattern[0], 1, 1)]
        if len(pattern) == 1
        else [(i + 1, pattern[i % len(pattern)], 1, 1) for i in range(50)]
    )
    recovery = run_with_default_fallback(
        lambda: solve_schedule_case(
            params=dict(params),
            mkw=dict(mkw),
            schedule=schedule,
            solve_tol=1e-6,
            solve_max_cycles=50,
        ),
        lambda: solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=1e-6,
            solver_max_iter=50,
            augment_params=augment_setup_params,
        ),
    )
    outcome = report_online_outcome(recovery.to_result(), bandit_timing={})
    outcome["completed_residual_norm"] = (
        recovery.fallback.residual_norm
        if recovery.fallback_used
        else recovery.primary.residual_norm
    )
    return outcome


def audit_outcome(outcome):
    components = [
        outcome[k] for k in ("setup_runtime", "solve_runtime", "infer_runtime")
    ]
    if not all(np.isfinite(v) and v >= 0 for v in components):
        raise AssertionError("Invalid cost component")
    if not np.isclose(
        sum(components), outcome["end_to_end_runtime"], rtol=1e-9, atol=1e-12
    ):
        raise AssertionError("Cost accounting mismatch")
    if not outcome["failed"] and not outcome.get("fallback_used", False):
        if not np.isfinite(outcome["residual_norm"]) or outcome["residual_norm"] > 1e-6:
            raise AssertionError("Successful primary solve violates the tolerance")
    if not outcome["failed"] and "completed_residual_norm" in outcome:
        residual = outcome["completed_residual_norm"]
        if not np.isfinite(residual) or residual > 1e-6:
            raise AssertionError("Completed solve violates the tolerance")


def audit_frozen_policies(policies, before, rng_before):
    for name, bundle in policies.items():
        after = bundle.controller.snapshot_learning_state()
        for field, value in before[name].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    value, after[field], err_msg=f"{name}/{field}"
                )
            elif json.dumps(_json_ready(value)) != json.dumps(
                _json_ready(after[field])
            ):
                raise AssertionError(f"Frozen policy changed: {name}/{field}")
        if rng_before[name] != json.dumps(bundle.controller.rng.bit_generator.state):
            raise AssertionError(f"Frozen policy changed its RNG: {name}")


def load_bundle(output, seed):
    directory = output / "checkpoints" / f"seed_{seed}"
    bundle = build_bundle(
        read(directory / "experiment_config.json"),
        "bandit_lstdq",
        directory / "bandit_lstdq_final.npz",
    )
    # Normalize episode scratch buffers before auditing; this performs no update.
    bundle.controller.start_episode(initial_environment_weight=1.0)
    bundle.controller.finish_episode(learned=False)
    return bundle


def frozen_snapshot(bundle):
    return (
        bundle.controller.snapshot_learning_state(),
        json.dumps(_json_ready(bundle.controller.rng.bit_generator.state)),
    )


def verify_frozen(bundle, before):
    audit_frozen_policies({"rl": bundle}, {"rl": before[0]}, {"rl": before[1]})


def environment_check(output):
    reference = read(
        ROOT
        / "results/paper_final/04_online/20260925_macmini_n60/provenance/environment_check.json"
    )
    package_reference = read(
        ROOT
        / "results/paper_final/04_online/20260926_macmini_module04_seeds4to6/provenance/packages.json"
    )
    current = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
    }
    native = {p: file_hash(ROOT / p) for p in reference["native_hashes"]}
    result = {
        "at": now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
        "native_hashes": native,
        "native_identical": native == reference["native_hashes"],
        "python_identical": sys.version == reference["python"],
        "packages": current,
        "reference_packages": package_reference,
        "parallelism": "At most three independent processes, one MPI rank and one library thread each; no claimed core affinity",
    }
    result["mpi_world_size"] = MPI.COMM_WORLD.Get_size()
    result["mpi"] = MPI.Get_library_version().rstrip("\x00\n ")
    result["mpi_identical"] = result["mpi"] == reference["mpi"].rstrip("\x00\n ")
    result["unexpected_setup_overrides"] = {
        k: os.environ[k]
        for k in ("SETUP_NUM_SWEEPS", "SETUP_CYCLE_TYPE", "SETUP_MAX_LEVELS")
        if os.environ.get(k, "").strip()
    }
    # The archive may use either pip's [{name, version}] form or a name map.
    previous = (
        {r["name"].lower().replace("_", "-"): r["version"] for r in package_reference}
        if isinstance(package_reference, list)
        else package_reference
    )
    result["package_differences"] = {
        k: [v, current.get(k.lower().replace("_", "-"))]
        for k, v in previous.items()
        if current.get(k.lower().replace("_", "-")) != v
    }
    dump(output / "environment.json", result)
    if (
        not result["native_identical"]
        or not result["python_identical"]
        or result["package_differences"]
        or not result["mpi_identical"]
        or result["mpi_world_size"] != 1
        or result["unexpected_setup_overrides"]
    ):
        raise RuntimeError("Module 04 environment does not match; see environment.json")
    return result


def reject_execution_error(outcome):
    attempts = [outcome, *outcome.get("primary_attempts", [])]
    if (
        any(a.get("failure_origin") == "execution" for a in attempts)
        or outcome.get("fallback_failure_origin") == "execution"
    ):
        raise RuntimeError(
            f"Experiment execution error: {outcome.get('primary_failure_reason', outcome.get('failure_reason'))}"
        )


def run_policy(job, name, spec, bundle):
    started = time.perf_counter()
    configure_smoother_profile(PROFILE)
    if spec["kind"] == "rl":
        outcome = _run_case(
            bundle=bundle, setup_row=job, args=NATIVE_ARGS, learn=False, explore=False
        )
        reject_execution_error(outcome)
        if outcome["failed"] and not outcome.get("fallback_used", False):
            # The online learner defers structural setup failure to its setup
            # bandit. Frozen evaluation has no reselection: apply the same
            # reference recovery available to constants and schedules.
            primary_attempt = RecoveryOutcome.from_mapping(outcome).primary
            recovery = run_with_default_fallback(
                lambda: primary_attempt,
                lambda: solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=job["mkw"],
                    solver_tol=1e-6,
                    solver_max_iter=50,
                    augment_params=augment_setup_params,
                ),
            )
            outcome = report_online_outcome(recovery.to_result(), bandit_timing={})
    elif spec["kind"] == "native":

        def primary():
            configure_smoother_profile(spec["profile"])
            try:
                return solve_no_rl_case(
                    params={**job["params"], **spec["params"]},
                    mkw=job["mkw"],
                    solver_tol=1e-6,
                    solver_max_iter=50,
                    augment_params=augment_setup_params,
                )
            finally:
                configure_smoother_profile(PROFILE)

        recovery = run_with_default_fallback(
            primary,
            lambda: solve_no_rl_case(
                params=dict(DEFAULT_SETUP_PARAMS),
                mkw=job["mkw"],
                solver_tol=1e-6,
                solver_max_iter=50,
                augment_params=augment_setup_params,
            ),
        )
        outcome = report_online_outcome(recovery.to_result(), bandit_timing={})
        outcome["completed_residual_norm"] = (
            recovery.fallback.residual_norm
            if recovery.fallback_used
            else recovery.primary.residual_norm
        )
    else:
        outcome = run_schedule(job["mkw"], job["params"], spec["pattern"])
    elapsed = time.perf_counter() - started
    reject_execution_error(outcome)
    audit_outcome(outcome)
    # All reported costs include failed attempts/recovery; primary setup is common.
    native_continuation = float(outcome["solve_runtime"]) + float(
        outcome.get("fallback_setup_runtime", 0)
    )
    controller = float(outcome.get("infer_runtime", 0))
    row = {
        "seed": job["seed"],
        "source": job["source"],
        "case_id": job["case_id"],
        "input_id": job["input_id"],
        "hierarchy_id": job["hierarchy_id"],
        "policy": name,
        "kind": spec["kind"],
        "wall_sec": elapsed,
        "native_continuation_sec": native_continuation,
        "inclusive_continuation_sec": native_continuation + controller,
        "native_total_sec": float(outcome["setup_runtime"])
        + float(outcome["solve_runtime"]),
        "inclusive_total_sec": float(outcome["end_to_end_runtime"]),
        "success": not bool(outcome["failed"]),
        "outcome": outcome,
    }
    if not all(
        math.isfinite(row[k]) and row[k] >= 0
        for k in [
            "wall_sec",
            "native_continuation_sec",
            "inclusive_continuation_sec",
            "native_total_sec",
            "inclusive_total_sec",
        ]
    ):
        raise AssertionError("Invalid timing")
    return row


def verify_inputs(output, *, check_source=True):
    saved = read(output / "prepared.json")
    files = {
        "protocol.json": saved["protocol_sha256"],
        "inputs.json": saved["inputs_sha256"],
        **{f"jobs_{p}.json": h for p, h in saved["jobs_sha256"].items()},
        **saved["checkpoint_sha256"],
    }
    for path, expected in files.items():
        if file_hash(output / path) != expected:
            raise RuntimeError(f"Prepared input changed: {path}")
    if check_source:
        for path, expected in read(output / "source_manifest.json").items():
            if not (ROOT / path).is_file() or file_hash(ROOT / path) != expected:
                raise RuntimeError(
                    f"Experiment source changed after preparation: {path}"
                )
