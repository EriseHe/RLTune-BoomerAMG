from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import experiments.joint.solve_control.joint_4k_execution as execution
from experiments.joint.solve_control.joint_method_spec import ComposableMethodSpec


class Joint4KExecutionTest(unittest.TestCase):
    def test_v5_bandit_and_rl_receive_the_same_problem_context(self) -> None:
        spec = ComposableMethodSpec(
            name="aligned_v5_rl",
            setup_kind="linucb_v5",
            setup_space="recommended",
            candidate_sampling="structured512",
            solve_kind="recursive_lstdq_v3",
        )
        problem_context = np.asarray(
            [1.0, 0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.6],
            dtype=float,
        )
        native = {
            "runtime": 0.30,
            "setup_runtime": 0.10,
            "solve_runtime": 0.20,
            "infer_runtime": 0.01,
            "failed": False,
        }
        branch = Mock()
        branch.policy = object()
        branch.parameter_space = {}
        plan = execution.OnlineComparisonPlan(
            output_dir=Path("/tmp/aligned-v5-rl-test"),
            trajectories_dir=Path("/tmp/aligned-v5-rl-test/trajectories"),
            checkpoints_dir=Path("/tmp/aligned-v5-rl-test/checkpoints"),
            methods=(spec.name,),
            family_by_method={spec.name: spec.family},
            bandit_methods=(spec.name,),
            online_instances=[],
            composable_specs={spec.name: spec},
            branches={spec.name: branch},
            controller_bundles={},
            ppo_runner=None,
            protocol={},
            warmup=execution.WarmupArtifacts(
                summary={},
                trajectory="",
                state="",
                record_count=0,
            ),
            solve=execution.SolveExecutionConfig(
                tolerance=1.0e-6,
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
        solve_contexts = []
        hooks = execution.OnlineComparisonHooks(
            method_solver=lambda _method, **kwargs: (
                solve_contexts.append(
                    np.asarray(kwargs["problem_context"], dtype=float)
                )
                or Mock(return_value=native)
            ),
            run_default_setup_method=Mock(),
            report_online_outcome=lambda outcome, **_kwargs: dict(outcome),
            setup_context=lambda **kwargs: np.asarray(
                kwargs["context"],
                dtype=float,
            ),
        )

        with patch.object(
            execution,
            "run_bandit_step_test_final",
            return_value=(
                {"coarsen_type": 10},
                native,
                {},
                0,
                0.0,
            ),
        ) as run_bandit:
            execution._run_one_method(
                plan=plan,
                hooks=hooks,
                method=spec.name,
                online_index=0,
                execution_rank=0,
                mkw={"nx": 4, "ny": 4, "nz": 4},
                context=problem_context,
                progress_denom=1,
                previous_update={spec.name: 0.0},
                bandit_online_steps={spec.name: 0},
            )

        bandit_context = run_bandit.call_args.kwargs["problem_context"]
        np.testing.assert_array_equal(bandit_context, problem_context)
        np.testing.assert_array_equal(solve_contexts[0], problem_context)
        self.assertNotIn("context", run_bandit.call_args.kwargs)

    def test_delayed_solve_controller_uses_default_before_activation(self) -> None:
        bundle = Mock()
        bundle.run_case.return_value = {
            "runtime": 0.30,
            "setup_runtime": 0.10,
            "solve_runtime": 0.20,
            "native_runtime": 0.30,
            "native_solve_runtime": 0.20,
            "infer_runtime": 0.0,
        }
        spec = ComposableMethodSpec.from_mapping(
            {
                "id": "staged_joint",
                "setup": "linucb",
                "solve": "recursive_lstdq_v3",
                "solve_activation_case": 1000,
                "solve_tolerance": 1.0e-6,
            }
        )
        self.assertEqual(spec.solve_activation_case, 1000)
        self.assertEqual(
            ComposableMethodSpec.from_runner_token(spec.to_runner_token()),
            spec,
        )
        default_native = {
            "runtime": 0.40,
            "setup_runtime": 0.15,
            "solve_runtime": 0.25,
            "native_runtime": 0.40,
            "native_solve_runtime": 0.25,
            "infer_runtime": 0.0,
        }
        common = {
            "solve": execution.SolveExecutionConfig(
                tolerance=1.0e-8,
                max_cycles=50,
            ),
            "mkw": {"nx": 40, "ny": 40, "nz": 40},
            "case_progress": 0.25,
            "problem_context": np.asarray(
                [1.0, 0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.6]
            ),
            "controller_bundles": {spec.name: bundle},
            "ppo_runner": None,
            "composable_specs": {spec.name: spec},
        }

        with patch.object(
            execution,
            "solve_no_rl_case",
            return_value=default_native,
        ) as default_solve:
            before = execution.make_method_solver(
                spec.name,
                case_index=999,
                **common,
            )
            before({"coarsen_type": 10})

        default_solve.assert_called_once()
        self.assertEqual(
            default_solve.call_args.kwargs["solver_tol"],
            1.0e-6,
        )
        bundle.run_case.assert_not_called()

        at_boundary = execution.make_method_solver(
            spec.name,
            case_index=1000,
            **common,
        )
        at_boundary({"coarsen_type": 10})
        bundle.run_case.assert_called_once()
        self.assertEqual(
            bundle.run_case.call_args.args[0].solve_tol,
            1.0e-6,
        )
        self.assertEqual(
            bundle.run_case.call_args.args[0].problem_context,
            tuple(common["problem_context"]),
        )

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
            problem_context=np.asarray(
                [1.0, 0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.6]
            ),
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
        self.assertEqual(
            case.problem_context,
            (1.0, 0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.6),
        )
        self.assertIsNotNone(case.fallback_attempt)
        self.assertAlmostEqual(feedback["runtime"], 0.30)
        self.assertAlmostEqual(feedback["solve_runtime"], 0.20)

    def test_frozen_setup_replay_does_not_require_a_bandit_branch(self) -> None:
        spec = ComposableMethodSpec(
            name="frozen_setup_rl",
            setup_kind="linucb_v5",
            setup_space="recommended",
            candidate_sampling="structured512",
            solve_kind="recursive_lstdq_v3",
        )
        replay_params = {"coarsen_type": 10, "interp_type": 6}
        native = {
            "runtime": 0.30,
            "setup_runtime": 0.10,
            "solve_runtime": 0.20,
            "native_runtime": 0.30,
            "native_solve_runtime": 0.20,
            "infer_runtime": 0.0,
            "failed": False,
            "attempt_status": "success",
            "native_status": "converged",
            "residual_norm": 1.0e-7,
            "iterations": 2,
        }
        plan = execution.OnlineComparisonPlan(
            output_dir=Path("/tmp/frozen-setup-replay-test"),
            trajectories_dir=Path("/tmp/frozen-setup-replay-test/trajectories"),
            checkpoints_dir=Path("/tmp/frozen-setup-replay-test/checkpoints"),
            methods=(spec.name,),
            family_by_method={spec.name: spec.family},
            bandit_methods=(),
            online_instances=[],
            composable_specs={spec.name: spec},
            branches={},
            controller_bundles={},
            ppo_runner=None,
            protocol={},
            warmup=execution.WarmupArtifacts(
                summary={},
                trajectory="",
                state="",
                record_count=0,
            ),
            solve=execution.SolveExecutionConfig(
                tolerance=1.0e-6,
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
            setup_replay_rows=(
                {
                    "online_index": 0,
                    "params": replay_params,
                    "arm_index": 123,
                },
            ),
        )
        hooks = execution.OnlineComparisonHooks(
            method_solver=Mock(return_value=Mock(return_value=native)),
            run_default_setup_method=Mock(),
            report_online_outcome=execution._report_online_outcome,
        )

        row = execution._run_one_method(
            plan=plan,
            hooks=hooks,
            method=spec.name,
            online_index=0,
            execution_rank=0,
            mkw={"nx": 4, "ny": 4, "nz": 4},
            context=np.asarray(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=float,
            ),
            progress_denom=1,
            previous_update={},
            bandit_online_steps={},
        )

        self.assertEqual(row["params"], replay_params)
        self.assertEqual(row["arm_index"], 123)
        self.assertFalse(row["outcome"]["bandit_update_committed"])
        hooks.method_solver.return_value.assert_called_once_with(replay_params)

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
