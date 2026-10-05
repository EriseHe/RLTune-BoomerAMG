"""Historical Module 05 Run 05: six frozen checkpoints and free pair (2.85,1.10)."""

from __future__ import annotations

from experiments.runtime import (
    configure_single_thread,
    prevent_sleep,
    stop_sleep_prevention,
    single_thread_environment,
)

if __name__ == "__main__":
    configure_single_thread()

from experiments.paper_final.common.workers import run_frozen_worker, run_frozen_phase
from experiments.paper_final.common import (
    artifacts,
    frozen_policy as policy,
    policy_analysis as analysis,
)
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = artifacts.ROOT
PARENT = (
    ROOT
    / "results/paper_final/05_policy/20260927_run04_anchored26_joint_6seeds_100cases"
)
DEFAULT = (
    ROOT
    / "results/paper_final/05_policy/20260929_run05_minimax285_joint_6seeds_100cases"
)
OLD_POLICY = "periodic_2.60_1.00"
POLICY = "periodic_2.85_1.10"
REMOVED_POLICY = "periodic_1.00_3.00"
METHODS = ("reference", "fixed", "oracle", "periodic", "rl")
MAIN_METHODS = ("fixed", "oracle", "periodic", "rl")
LABELS = {
    "reference": "Fixed w=1",
    "fixed": "Stream-wide fixed*",
    "oracle": "Per-instance fixed*",
    "periodic": "Periodic (2.85, 1.10)",
    "rl": "Frozen RL",
}
THEORY_FILES = [
    "docs/theory/period_two_weighted_minimax_20260928.md",
    "docs/theory/anchored_schedule_verification_20260927.json",
    "docs/theory/period_two_minimax_verification_20260928.json",
]


def replace_schedule(jobs):
    """Retain parent jobs/order; substitute the free pair and remove (1,3)."""
    updated = copy.deepcopy(jobs)
    for j in updated:
        if j["method_policies"]["periodic"] != OLD_POLICY:
            raise ValueError("Unexpected parent schedule")
        if j["method_policies"].pop("periodic13") != REMOVED_POLICY:
            raise ValueError("Unexpected endpoint schedule")
        j["method_policies"]["periodic"] = POLICY
        j["policy_order"] = [
            POLICY if n == OLD_POLICY else n
            for n in j["policy_order"]
            if n != REMOVED_POLICY
        ]
        if set(j["method_policies"]) != set(METHODS):
            raise ValueError("Unexpected method roles")
        if set(j["policy_order"]) != set(j["method_policies"].values()):
            raise ValueError("Planned execution differs from method map")
    return updated


def verify(output):
    policy.verify_inputs(output)
    saved = artifacts.read(output / "prepared.json")
    if artifacts.file_hash(output / "selections.json") != saved["selections_sha256"]:
        raise RuntimeError("Prespecified selections changed")
    protocol = artifacts.read(output / "protocol.json")
    if protocol["run_number"] != 5 or protocol["methods"] != list(METHODS):
        raise RuntimeError("Incorrect Run 05 method roster")
    periodic = {
        k: v for k, v in protocol["policies"].items() if v["kind"] == "periodic"
    }
    if periodic != {POLICY: {"kind": "periodic", "pattern": [2.85, 1.10]}}:
        raise RuntimeError("Incorrect prescribed periodic policy")
    for job in artifacts.read(output / "jobs_test.json"):
        names = job["policy_order"]
        if (
            set(job["method_policies"]) != set(METHODS)
            or job["method_policies"]["periodic"] != POLICY
            or set(names) != set(job["method_policies"].values())
            or len(names) != len(set(names))
            or OLD_POLICY in names
            or REMOVED_POLICY in names
        ):
            raise RuntimeError("Incorrect evaluation job")


def prepare(output, parent=PARENT):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new Run 05 directory")
    policy.verify_inputs(parent, check_source=False)
    saved = artifacts.read(parent / "prepared.json")
    if artifacts.file_hash(parent / "selections.json") != saved["selections_sha256"]:
        raise RuntimeError("Parent selections changed")
    if artifacts.read(parent / "complete.json")["status"] != "complete":
        raise ValueError("Incomplete parent")
    math = artifacts.read(ROOT / THEORY_FILES[-1])
    implementation = artifacts.read(ROOT / THEORY_FILES[1])
    if (
        not math["passed"]
        or math["grid_minimizers_ordered"] != [["2.85", "1.10"], ["1.10", "2.85"]]
        or not implementation["implementation"]["passed"]
    ):
        raise ValueError("Free minimax and smoother-profile audits must pass")
    manifest = artifacts.read(parent / "source_manifest.json")
    differences = artifacts.source_differences(manifest)
    numerical = [
        p
        for p in differences
        if not any(s in p for s in ("plot_", "analyze_", "test_", "module05_figures/"))
    ]
    if numerical:
        raise RuntimeError(f"Parent execution sources changed: {numerical}")
    output.mkdir(parents=True)
    for name in ("analysis", "progress", "raw/test", "provenance/source", "theory"):
        (output / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(parent / "checkpoints", output / "checkpoints")
    for name in ("inputs.json", "jobs_base_test.json", "jobs_preflight.json"):
        shutil.copy2(parent / name, output / name)
    jobs = replace_schedule(artifacts.read(parent / "jobs_test.json"))
    chosen = artifacts.read(parent / "selections.json")
    chosen["chosen_policies"] = {
        k: v
        for k, v in chosen["chosen_policies"].items()
        if "/bandit_lstdq/" in k and k.rsplit("/", 1)[-1] in METHODS
    }
    for key, value in chosen["chosen_policies"].items():
        if key.endswith("/periodic"):
            if value != OLD_POLICY:
                raise ValueError("Parent selection mismatch")
            chosen["chosen_policies"][key] = POLICY
    chosen.update(
        run_number=5,
        origin_run=str(parent),
        selection_frozen_before_final_retiming=True,
        note="Run 04 fixed choices retained; (2.85,1.10) prescribed by the free 0.05-grid weighted-minimax rule before Run 05.",
    )
    old = artifacts.read(parent / "protocol.json")
    protocol = copy.deepcopy(old)
    for name in (
        "scan_execution_count",
        "scan_extra_repeat_case_ids",
        "changed_test_hierarchies_seed4",
        "conventional_references",
    ):
        protocol.pop(name, None)
    protocol["policies"].pop(OLD_POLICY)
    protocol["policies"].pop(REMOVED_POLICY)
    protocol["policies"][POLICY] = {"kind": "periodic", "pattern": [2.85, 1.10]}
    used = {n for j in jobs for n in j["policy_order"]}
    protocol["policies"] = {k: v for k, v in protocol["policies"].items() if k in used}
    protocol.update(
        run_number=5,
        comparison_run=4,
        methods=list(METHODS),
        created_at=artifacts.now(),
        origin_run=str(parent),
        scope="Matched repeat of Run 04 with a theory-prescribed (2.85,1.10) schedule",
        unchanged_checkpoint_seeds=list(range(1, 7)),
        main_heatmap_methods=list(MAIN_METHODS),
        labels=LABELS,
        execution_count=len(analysis.expected_cells(jobs)),
        selection_rule="All Run 04 fixed-weight choices reused, including its refreshed seed-4 scan; no new scan or outcome selection",
        periodic_rule="(2.85,1.10) is the joint free weighted-minimax choice on {1,1.05,...,3}; high first, reset each primary solve",
        test_schedule_rule="Only prescribed (2.85,1.10); (1,3) and (2.6,1) excluded; same full-cycle action and charged recovery",
        theory_scope="Normalized SPD smoothing surrogate and fixed exact two-grid bound; no multilevel runtime-optimality guarantee",
        ordering_rule="Exact Run 04 case/policy/repetition order, with the periodic pair replaced and endpoint policy removed",
        interpretation="Same accepted checkpoints and previously observed test cases; a prescribed policy follow-up, not new independent seeds or a fresh holdout",
        sorting="Run 04 baseline-difficulty case order retained; no sorting on Run 05 outcomes",
    )
    artifacts.dump(output / "protocol.json", protocol)
    artifacts.dump(output / "jobs_test.json", jobs)
    artifacts.dump(output / "selections.json", chosen)
    for relative in THEORY_FILES:
        shutil.copy2(ROOT / relative, output / "theory" / Path(relative).name)
    extras = [
        "run_05_policy_minimax.py",
        "analyze_05_policy_minimax.py",
        "test_05_policy_minimax.py",
        "verify_period_two_minimax.py",
    ]
    paths = (
        set(manifest)
        | artifacts.support_source_paths()
        | {"experiments/paper_final/" + p for p in extras}
        | set(THEORY_FILES)
    )
    manifest = {p: artifacts.file_hash(ROOT / p) for p in paths}
    for relative in manifest:
        target = output / "provenance/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    artifacts.dump(output / "source_manifest.json", manifest)
    artifacts.dump(
        output / "parent_audit.json",
        {
            "parent": str(parent),
            "checkpoint_and_input_hashes_verified": True,
            "unchanged_execution_sources": True,
            "historical_reporting_source_differences": differences,
            "parent_raw_complete_sha256": artifacts.file_hash(
                parent / "raw/test/complete.json"
            ),
            "parent_selections_sha256": artifacts.file_hash(parent / "selections.json"),
            "parent_summary_sha256": artifacts.file_hash(
                parent / "analysis/summary.json"
            ),
        },
    )
    artifacts.dump(
        output / "prepared.json",
        {
            "at": artifacts.now(),
            "protocol_sha256": artifacts.file_hash(output / "protocol.json"),
            "inputs_sha256": artifacts.file_hash(output / "inputs.json"),
            "selections_sha256": artifacts.file_hash(output / "selections.json"),
            "jobs_sha256": {
                p: artifacts.file_hash(output / f"jobs_{p}.json")
                for p in ("base_test", "preflight", "test")
            },
            "checkpoint_sha256": {
                str(p.relative_to(output)): artifacts.file_hash(p)
                for p in (output / "checkpoints").rglob("*")
                if p.is_file()
            },
        },
    )
    policy.environment_check(output)
    artifacts.emit(
        output, "Run 05 prepared", trials=protocol["execution_count"], pair=[2.85, 1.10]
    )


def preflight(output):
    verify(output)
    protocol = artifacts.read(output / "protocol.json")
    jobs = artifacts.read(output / "jobs_preflight.json")
    checks = []
    for seed in range(1, 7):
        bundle = policy.load_bundle(output, seed)
        before = policy.frozen_snapshot(bundle)
        job = next(j for j in jobs if j["seed"] == seed and j["case_id"] == 0)
        constant = artifacts.read(output / "selections.json")["chosen_policies"][
            f"{seed}/bandit_lstdq/fixed"
        ]
        names = ["fixed_1.00", constant, POLICY, "rl_frozen_lcb", POLICY, "fixed_1.00"]
        results = [
            policy.run_policy(job, n, protocol["policies"][n], bundle) for n in names
        ]
        for x, y in ((0, 5), (2, 4)):
            np.testing.assert_allclose(
                results[x]["outcome"]["cycle_residuals"],
                results[y]["outcome"]["cycle_residuals"],
                rtol=1e-12,
                atol=1e-15,
            )
        actions = results[2]["outcome"]["cycle_actions"]
        assert actions and actions == [[2.85, 1.10][i % 2] for i in range(len(actions))]
        policy.verify_frozen(bundle, before)
        checks.append(
            {
                "seed": seed,
                "repeated_reference_and_schedule_agree": True,
                "phase_resets": True,
                "controller_frozen": True,
            }
        )
    artifacts.dump(
        output / "preflight.json",
        {"at": artifacts.now(), "passed": True, "checks": checks},
    )


def run(output):
    if not (output / "prepared.json").exists():
        prepare(output)
    with (output / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        awake = prevent_sleep()

        def stop(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            verify(output)
            policy.environment_check(output)
            if not (output / "preflight.json").exists():
                preflight(output)
            if not (output / "raw/test/complete.json").exists():
                run_phase(output, "test")
            from experiments.paper_final.analyze_05_policy_minimax import analyze

            artifacts.dump(
                output / "status.json",
                {
                    "status": "running",
                    "phase": "analysis",
                    "run_number": 5,
                    "at": artifacts.now(),
                },
            )
            analyze(output)
            verify(output)
            artifacts.dump(
                output / "numerical_complete.json",
                {"status": "complete", "run_number": 5, "at": artifacts.now()},
            )
            final = {
                "status": "complete",
                "phase": "complete",
                "run_number": 5,
                "at": artifacts.now(),
                "elapsed_sec": time.monotonic() - start,
            }
            artifacts.dump(output / "complete.json", final)
            artifacts.dump(output / "status.json", final)
        except BaseException as exc:
            artifacts.dump(
                output / "status.json",
                {
                    "status": "failed",
                    "phase": "stopped",
                    "run_number": 5,
                    "at": artifacts.now(),
                    "error": str(exc),
                },
            )
            raise
        finally:
            stop_sleep_prevention(awake)


# The durable worker and phase supervisor below retain Run 04's execution,
# recovery, timing, and resume semantics; their run metadata is now 5.


def worker(output, phase, worker_id, workers):
    run_frozen_worker(
        output, phase, worker_id, workers, run_number=5, verify=verify, policy=policy
    )


def run_phase(output, phase, workers=3):
    run_frozen_phase(
        output, phase, runner_module=__spec__.name, run_number=5, workers=workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--phase", choices=("test",), default="test")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.output, args.phase, args.worker, 3)
    elif args.command == "prepare":
        prepare(args.output)
    else:
        run(args.output)
