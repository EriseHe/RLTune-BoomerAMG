"""Official Module 05 Run 05: six frozen checkpoints and free pair (2.85,1.10)."""

from __future__ import annotations

from experiments.runtime import (
    configure_single_thread,
    prevent_sleep,
    stop_sleep_prevention,
)

if __name__ == "__main__":
    configure_single_thread()

from experiments.paper_final.common.workers import run_frozen_worker, run_frozen_phase
from experiments.paper_final.common.frozen_inputs import (
    DEFAULT_BUNDLE,
    INPUT_FILES,
    execution_source_paths,
    verify_bundle,
)
from experiments.paper_final.common import (
    artifacts,
    frozen_policy as policy,
)
import argparse
import fcntl
from pathlib import Path
import shutil
import signal
import time
import numpy as np

ROOT = artifacts.ROOT
DEFAULT = ROOT / "results/paper_final/05_policy/run05_retiming"
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


def prepare(output, bundle=DEFAULT_BUNDLE):
    """Copy accepted inputs into a fresh, strictly tracked current-source run."""
    output, bundle = Path(output), Path(bundle).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new Run 05 directory")
    audit = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("analysis", "progress", "raw/test", "provenance/source"):
        (output / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle / "checkpoints", output / "checkpoints")
    shutil.copytree(bundle / "theory", output / "theory")
    shutil.copytree(bundle / "fixed_scans", output / "fixed_scans")
    for name in INPUT_FILES:
        shutil.copy2(bundle / name, output / name)
    for name in ("fixed_scan_evidence.json", "reference_environment.json"):
        shutil.copy2(bundle / name, output / name)
    for name in (
        "manifest.json",
        "accepted_prepared.json",
        "accepted_source_manifest.json",
    ):
        shutil.copy2(bundle / name, output / "provenance" / name)
    manifest = {
        name: artifacts.file_hash(ROOT / name)
        for name in sorted(execution_source_paths())
    }
    for name in manifest:
        target = output / "provenance/source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    artifacts.dump(output / "source_manifest.json", manifest)
    artifacts.dump(output / "bundle_audit.json", {**audit, "bundle": str(bundle)})
    environment = policy.environment_check(output)
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
            "bundle_sha256": audit["bundle_sha256"],
            "source_scope": "official_submission",
            "execution_environment": policy.environment_identity(environment),
            "evidence_sha256": {
                str(p.relative_to(output)): artifacts.file_hash(p)
                for p in (output / "fixed_scans").glob("*.gz")
            }
            | {
                name: artifacts.file_hash(output / name)
                for name in ("fixed_scan_evidence.json", "reference_environment.json")
            },
        },
    )
    verify(output)
    artifacts.emit(
        output,
        "Run 05 prepared from accepted frozen inputs",
        trials=artifacts.read(output / "protocol.json")["execution_count"],
        pair=[2.85, 1.10],
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


def run(output, bundle=DEFAULT_BUNDLE):
    if not (output / "prepared.json").exists():
        prepare(output, bundle)
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
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Verified accepted frozen-input bundle for a new run",
    )
    parser.add_argument("--phase", choices=("test",), default="test")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.output, args.phase, args.worker, 3)
    elif args.command == "prepare":
        prepare(args.output, args.bundle)
    else:
        run(args.output, args.bundle)
