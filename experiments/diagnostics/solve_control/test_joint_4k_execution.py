from __future__ import annotations

import _project_paths  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import joint_4k_execution as execution
from joint_method_spec import ComposableMethodSpec


class Joint4KExecutionTest(unittest.TestCase):
    def test_online_controller_solver_uses_bundle_lifecycle(self) -> None:
        captured = []
        bundle = Mock()
        bundle.run_case.side_effect = lambda case: (
            captured.append(case)
            or {
                "runtime": 0.30,
                "setup_runtime": 0.10,
                "solve_runtime": 0.20,
                "native_runtime": 0.29,
                "native_solve_runtime": 0.19,
                "infer_runtime": 0.01,
            }
        )
        spec = ComposableMethodSpec(
            name="bandit_rl",
            setup_kind="linucb",
            solve_kind="recursive_lstdq_v2",
        )
        solver = execution.make_method_solver(
            spec.name,
            solve=execution.SolveExecutionConfig(
                tolerance=1.0e-8,
                max_cycles=50,
            ),
            mkw={"nx": 40, "ny": 40, "nz": 40},
            case_progress=0.25,
            controller_bundles={spec.name: bundle},
            ppo_runner=None,
            composable_specs={spec.name: spec},
        )

        feedback = solver({"coarsen_type": 10})

        self.assertEqual(len(captured), 1)
        case = captured[0]
        self.assertEqual(case.params, {"coarsen_type": 10})
        self.assertTrue(case.learn)
        self.assertTrue(case.explore)
        self.assertTrue(case.record_action_metadata)
        self.assertIsNotNone(case.fallback_attempt)
        self.assertAlmostEqual(feedback["runtime"], 0.30)
        self.assertAlmostEqual(feedback["solve_runtime"], 0.20)

    def test_default_only_comparison_preserves_artifact_contract(self) -> None:
        outcome = {
            "runtime": 0.30,
            "setup_runtime": 0.10,
            "solve_runtime": 0.20,
            "infer_runtime": 0.0,
            "bandit_overhead_runtime": 0.0,
            "end_to_end_runtime": 0.30,
            "iterations": 2,
            "failed": False,
            "primary_status": "success",
            "fallback_status": "not_run",
            "fallback_used": False,
            "bandit_update_committed": False,
            "controller_update_committed": False,
        }
        native = {
            "runtime": 0.30,
            "setup_runtime": 0.10,
            "solve_runtime": 0.20,
            "iterations": 2,
            "failed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            trajectories_dir = output_dir / "trajectories"
            checkpoints_dir = output_dir / "checkpoints"
            trajectories_dir.mkdir()
            checkpoints_dir.mkdir()
            plan = execution.OnlineComparisonPlan(
                output_dir=output_dir,
                trajectories_dir=trajectories_dir,
                checkpoints_dir=checkpoints_dir,
                methods=("default_setup",),
                family_by_method={"default_setup": "default_setup"},
                bandit_methods=(),
                online_instances=[
                    (
                        {"nx": 4, "ny": 4, "nz": 4},
                        np.zeros(8, dtype=float),
                    )
                ],
                composable_specs={},
                branches={},
                controller_bundles={},
                ppo_runner=None,
                protocol={"schema": "test"},
                warmup=execution.WarmupArtifacts(
                    summary={"cases": 0},
                    trajectory="warmup_trajectory.jsonl",
                    state="warmup_state.npz",
                    record_count=0,
                ),
                solve=execution.SolveExecutionConfig(
                    tolerance=1.0e-8,
                    max_cycles=50,
                ),
                warmup_cases=0,
                online_cases=1,
                method_order_seed=17,
                progress_every=1,
                aot_enabled=False,
                aot_max_selections_per_case=3,
                default_setup_method="default_setup",
                include_solve_screen_report=False,
            )
            hooks = execution.OnlineComparisonHooks(
                method_solver=Mock(
                    side_effect=AssertionError(
                        "default-only execution built a method solver"
                    )
                ),
                run_default_setup_method=Mock(
                    side_effect=AssertionError(
                        "default-only execution used composable recovery"
                    )
                ),
                report_online_outcome=Mock(return_value=dict(outcome)),
            )

            with patch.object(
                execution,
                "solve_default_baseline_case",
                return_value=native,
            ):
                result = execution.run_online_comparison(
                    plan,
                    hooks=hooks,
                )

            trajectory = [
                json.loads(line)
                for line in (
                    trajectories_dir / "default_setup.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            method_order = [
                json.loads(line)
                for line in (
                    trajectories_dir / "method_order.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(trajectory), 1)
            self.assertEqual(
                method_order,
                [{"online_index": 0, "method_order": ["default_setup"]}],
            )
            self.assertEqual(
                trajectory[0]["outcome"]["end_to_end_runtime"],
                0.30,
            )
            self.assertEqual(
                result["artifacts"]["summary_csv"],
                str(output_dir / "summary_1.csv"),
            )
            self.assertNotIn("screen_report", result["artifacts"])
            self.assertEqual(result["recovery_audit"]["default_setup"]["valid"], True)
            self.assertTrue((output_dir / "progress.json").exists())
            self.assertTrue((output_dir / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
