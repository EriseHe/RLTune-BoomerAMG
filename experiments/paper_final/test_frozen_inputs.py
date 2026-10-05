"""Protect accepted input identity and independent fixed-grid reconstruction."""

import gzip
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.paper_final.common import artifacts, frozen_inputs, frozen_policy
from experiments.paper_final import run_05_policy_minimax as run
from experiments.paper_final.verify_05_policy_minimax import verify_fixed_choices


class FrozenInputTests(unittest.TestCase):
    def test_tracked_bundle_reconstructs_all_accepted_choices(self):
        result = frozen_inputs.verify_bundle(frozen_inputs.DEFAULT_BUNDLE)
        self.assertEqual(result["fixed_scan_records"], 29520)
        self.assertTrue(result["exact_accepted_input_hashes"])
        self.assertTrue(result["fixed_choices_reconstructed"])

    def test_corrupted_bundle_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            bundle = Path(name) / "bundle"
            shutil.copytree(frozen_inputs.DEFAULT_BUNDLE, bundle)
            with (bundle / "inputs.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch: inputs.json"):
                frozen_inputs.verify_bundle(bundle)

    def test_changed_manifest_cannot_relabel_accepted_input_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            bundle = Path(name) / "bundle"
            shutil.copytree(frozen_inputs.DEFAULT_BUNDLE, bundle)
            path = bundle / "jobs_test.json"
            with path.open("a") as handle:
                handle.write(" ")
            manifest = artifacts.read(bundle / "manifest.json")
            manifest["files_sha256"]["jobs_test.json"] = artifacts.file_hash(path)
            artifacts.dump(bundle / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Accepted input identity changed"):
                frozen_inputs.verify_bundle(bundle)

    def test_wrong_fixed_choice_rejected_from_independent_measurements(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "scan.jsonl.gz"
            rows = [
                {
                    "seed": 1,
                    "source": "bandit_lstdq",
                    "case_id": 0,
                    "input_id": "input",
                    "hierarchy_id": "hierarchy",
                    "kind": "fixed",
                    "policy": f"fixed_{1 + i * 0.05:.2f}",
                    "repeat": 0,
                    "native_continuation_sec": abs(i - 10) + 0.1,
                    "success": True,
                }
                for i in range(41)
            ]
            path.write_bytes(
                gzip.compress(
                    "".join(json.dumps(r) + "\n" for r in rows).encode(), mtime=0
                )
            )
            job = {
                "input_id": "input",
                "hierarchy_id": "hierarchy",
                "method_policies": {
                    "fixed": "fixed_1.50",
                    "oracle": "fixed_1.50",
                },
            }
            protocol = {"test_cases": 1, "scan_extra_repeat_case_ids": []}
            self.assertEqual(verify_fixed_choices(1, {(0, 0): job}, protocol, path), 41)
            job["method_policies"]["fixed"] = "fixed_1.00"
            with self.assertRaisesRegex(AssertionError, "Fixed choice changed"):
                verify_fixed_choices(1, {(0, 0): job}, protocol, path)

    def test_fresh_prepare_preserves_jobs_and_refuses_changed_inputs_and_source(self):
        environment = {
            key: None
            for key in (
                "python",
                "executable",
                "platform",
                "thread_environment",
                "native_hashes",
                "packages",
                "mpi",
                "mpi_world_size",
                "unexpected_setup_overrides",
            )
        }
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "run"
            with patch.object(
                frozen_policy, "environment_check", return_value=environment
            ):
                run.prepare(output)
            self.assertEqual(
                (output / "jobs_test.json").read_bytes(),
                (frozen_inputs.DEFAULT_BUNDLE / "jobs_test.json").read_bytes(),
            )
            run.verify(output)
            original = (output / "inputs.json").read_bytes()
            (output / "inputs.json").write_bytes(original + b" ")
            with self.assertRaisesRegex(RuntimeError, "Prepared input changed"):
                run.verify(output)
            (output / "inputs.json").write_bytes(original)
            manifest = artifacts.read(output / "source_manifest.json")
            manifest["experiments/paper_final/common/frozen_inputs.py"] = "changed"
            artifacts.dump(output / "source_manifest.json", manifest)
            with self.assertRaisesRegex(RuntimeError, "Experiment source changed"):
                run.verify(output)

    def test_changed_native_binary_refused_on_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "run"
            output.mkdir()
            shutil.copy2(
                frozen_inputs.DEFAULT_BUNDLE / "reference_environment.json", output
            )
            native = root / "hypre/interfaces/libamg_runtime.so"
            native.parent.mkdir(parents=True)
            native.write_bytes(b"native-one")
            hypre = root / "hypre/install/lib/libHYPRE.so"
            hypre.parent.mkdir(parents=True)
            hypre.write_bytes(b"hypre-one")
            mpi = SimpleNamespace(
                COMM_WORLD=SimpleNamespace(Get_size=lambda: 1),
                Get_library_version=lambda: "test MPI",
            )
            with (
                patch.object(frozen_policy, "ROOT", root),
                patch.object(frozen_policy, "AMG_RUNTIME_LIBRARY", native),
                patch.object(frozen_policy, "MPI", mpi),
                patch.object(
                    frozen_policy.importlib.metadata, "distributions", return_value=[]
                ),
            ):
                current = frozen_policy.environment_check(output)
                artifacts.dump(
                    output / "prepared.json",
                    {
                        "execution_environment": frozen_policy.environment_identity(
                            current
                        ),
                    },
                )
                frozen_policy.environment_check(output)
                native.write_bytes(b"native-two")
                with self.assertRaisesRegex(RuntimeError, "environment changed"):
                    frozen_policy.environment_check(output)


if __name__ == "__main__":
    unittest.main()
