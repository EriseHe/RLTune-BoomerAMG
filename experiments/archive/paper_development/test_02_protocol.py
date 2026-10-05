"""Verify the diagnostic's schedule execution and recovery accounting without PDE solves."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.archive.paper_development import run_02_diagnostics as diagnostic
from experiments.archive.paper_development.run_02_diagnostics import run_schedule, audit_outcome, np
from experiments.diagnostics.solve_control.run_lstdq_v3_stability import _run_case
from hypre.bindings import SolveStatus
from experiments.joint.solve_control.setup_aware_compare_common import solve_schedule_case


class ScheduleTests(unittest.TestCase):
    def test_periodic_schedule_keeps_cycle_metadata_and_timing(self):
        calls = []
        env = SimpleNamespace(r0=1., last_step=None)
        env.prepare_rl = lambda **kw: SimpleNamespace(setup_runtime_sec=.2)

        def step(**kw):
            calls.append(kw["relax_weight"])
            terminal = len(calls) == 3
            env.last_step = SimpleNamespace(status=SolveStatus.CONVERGED if terminal else SolveStatus.CONTINUE)
            return (1e-7 if terminal else .1, .01)

        env.step_rl = step
        with patch("experiments.joint.solve_control.native_evaluation.create_env", return_value=nullcontext(env)):
            result = run_schedule({}, {}, [2.9, 1.])
        self.assertEqual(calls, [2.9, 1., 2.9])
        self.assertEqual(result["cycle_actions"], calls)
        self.assertEqual(result["cycle_times"], [.01] * 3)
        self.assertAlmostEqual(result["solve_runtime"], .03)
        self.assertGreaterEqual(result["infer_runtime"], 0.)
        self.assertLess(result["completed_residual_norm"], 1e-6)
        audit_outcome(result)

    def test_failure_recovery_charges_both_attempts_and_reports_completed_residual(self):
        primary = {"runtime": .7, "setup_runtime": .2, "solve_runtime": .5,
                   "infer_runtime": .1, "native_solve_runtime": .4,
                   "failed": True, "attempt_status": "nonconvergence",
                   "residual_norm": .01, "iterations": 50}
        fallback = {"runtime": .5, "setup_runtime": .3, "solve_runtime": .2,
                    "failed": False, "attempt_status": "success",
                    "residual_norm": 1e-8, "iterations": 4}
        with patch("experiments.paper_final.common.frozen_policy.solve_schedule_case", return_value=primary), \
                patch("experiments.paper_final.common.frozen_policy.solve_no_rl_case", return_value=fallback):
            result = run_schedule({}, {}, [3.])
        self.assertTrue(result["recovered"])
        self.assertAlmostEqual(result["end_to_end_runtime"], 1.2)
        self.assertEqual(result["completed_residual_norm"], 1e-8)
        audit_outcome(result)

    def test_completed_fallback_residual_is_checked(self):
        outcome = {"setup_runtime": .1, "solve_runtime": .2, "infer_runtime": 0.,
                   "end_to_end_runtime": .3, "failed": False, "fallback_used": True,
                   "completed_residual_norm": 1e-3}
        with self.assertRaisesRegex(AssertionError, "Completed solve"):
            audit_outcome(outcome)

    def test_rl_helper_retains_the_actual_fallback_residual(self):
        fallback = {"residual_norm": 1e-8}

        def run_case(case):
            case.fallback_attempt()
            return {"runtime": .3, "native_runtime": .3, "setup_runtime": .1,
                    "solve_runtime": .2, "native_solve_runtime": .2, "infer_runtime": 0.,
                    "failed": False, "fallback_used": True, "residual_norm": .01}

        with patch("experiments.joint.solve_control.native_case.solve_no_rl_case", return_value=fallback), \
                patch("experiments.joint.solve_control.native_case.as_feedback", side_effect=lambda value, **kw: value):
            result = _run_case(bundle=SimpleNamespace(run_case=run_case),
                               setup_row={"mkw": {}, "params": {}}, args=diagnostic.NATIVE_ARGS,
                               learn=False, explore=False)
        self.assertEqual(result["completed_residual_norm"], 1e-8)
        audit_outcome(result)


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_original_rows_and_never_repeats_completed_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "results/paper_final/02_diagnostics"
            activation = root / "results/paper_final/03_activation/advection_80_s1"
            output.mkdir(parents=True)
            (activation / "checkpoints").mkdir(parents=True)
            (activation / "trajectories").mkdir()
            checkpoint = activation / "checkpoints/start_1000_final.npz"
            checkpoint.write_bytes(b"test-checkpoint")
            encoder = Path(str(checkpoint) + ".encoder.json")
            encoder.write_text("{}")
            diagnostic._write_json(activation.parent / "summary.json", {"complete": True, "completed_replicates": 1})
            diagnostic._write_json(activation.parent / "final_audit.json", {
                "prefix_and_cost_audit_valid": True, "checkpoint_numerics": {"start_1000": {
                    "checkpoint_sha256": diagnostic.file_hash(checkpoint),
                    "encoder_sha256": diagnostic.file_hash(encoder)}}})
            diagnostic._write_json(activation / "experiment_config.json", {"solve": {"smoother_profile": "test"}})
            (activation / "trajectories/setup_only.jsonl").write_text('{"mkw":{"id":-1}}\n')
            inputs = {"evaluation": {family: [{"mkw": {"id": i}} for i in range(12)]
                                     for family in ("diffusion", "advection")}}
            protocol = {"periodic_schedules": diagnostic.SCHEDULES, "tolerance": 1e-6, "cycle_cap": 50,
                        "learn": False, "explore": False, "selection_cases": list(range(1, 7)),
                        "evaluation_cases": list(range(7, 13)), "smoother_profile": "test",
                        "setups": {"reference": {}, "learned": {}}, "seeds": {"order": 9131615},
                        "fixed_weights": {"diffusion": [round(1 + i * .05, 2) for i in range(41)],
                                          "advection": [1., 1.5, 2., 2.5, 2.9, 3.]},
                        "fresh_input_sha256": diagnostic.hashlib.sha256(
                            json.dumps(inputs, sort_keys=True).encode()).hexdigest()}
            diagnostic._write_json(output / "protocol.json", protocol)
            diagnostic._write_json(output / "inputs.json", inputs)
            diagnostic._write_json(output / "summary.json", {"boundary_audits": {}, "frozen_policy_audit": {}})
            outcome = {"setup_runtime": .1, "solve_runtime": .2, "infer_runtime": 0., "iterations": 1,
                       "end_to_end_runtime": .3, "failed": False, "fallback_used": False,
                       "primary_status": "success", "residual_norm": 1e-8, "completed_residual_norm": 1e-8}
            names = [*(f"fixed_{w:.2f}" for w in protocol["fixed_weights"]["diffusion"]),
                     *(f"schedule_{name}" for name in diagnostic.SCHEDULES), "rl_lcb", "rl_mean"]
            cells = [(setup, policy) for setup in protocol["setups"] for policy in names]
            rng = np.random.default_rng(protocol["seeds"]["order"])
            rows = [{"family": "diffusion", "phase": "selection" if i < 6 else "evaluation", "case": i + 1,
                     "setup": cells[rank][0], "policy": cells[rank][1], "outcome": outcome}
                    for i in range(12) for rank in rng.permutation(len(cells))]
            rows.extend({"family": "advection", "phase": "training", "case": i + 1,
                         "setup": ("reference", "learned")[i % 2], "policy": "rl_train", "outcome": outcome}
                        for i in range(58))
            trajectory = output / "trajectories.jsonl"
            trajectory.write_text("".join(json.dumps(row) + "\n" for row in rows))
            original = trajectory.read_bytes()

            def bundle(*args):
                controller = SimpleNamespace(joint_dim=306, a_matrix=np.eye(306), a_inverse=np.eye(306),
                                             b=np.zeros(306), theta=np.zeros(306), inverse_is_valid=True,
                                             inverse_rebuild_count=0, rng=np.random.default_rng(1))
                controller.snapshot_learning_state = lambda: {"theta": controller.theta.copy(), "steps": 10}
                return SimpleNamespace(controller=controller, encoder=SimpleNamespace(
                    feature_dim=34, encoding_version="space_aware_v2", problem_context_mode="canonical_no_c_mean"))

            with patch.object(diagnostic, "ROOT", root), patch.object(diagnostic, "build_bundle", side_effect=bundle), \
                    patch.object(diagnostic, "configure_smoother_profile"), patch.object(diagnostic, "source_state", return_value={}), \
                    patch.object(diagnostic, "run_schedule", return_value=outcome) as schedule, \
                    patch.object(diagnostic, "_run_case", return_value=outcome) as rl, redirect_stdout(io.StringIO()):
                diagnostic.resume_advection(output, checkpoint)
                self.assertEqual(schedule.call_count, 288)
                self.assertEqual(rl.call_count, 48)
                for call in rl.call_args_list:
                    self.assertFalse(call.kwargs["learn"])
                    self.assertFalse(call.kwargs["explore"])
                self.assertTrue(trajectory.read_bytes().startswith(original))
                self.assertEqual(len(trajectory.read_text().splitlines()), 1570)
                completed_bytes = trajectory.read_bytes()
                schedule.reset_mock()
                rl.reset_mock()
                diagnostic.resume_advection(output, checkpoint)
                schedule.assert_not_called()
                rl.assert_not_called()
                self.assertEqual(trajectory.read_bytes(), completed_bytes)
                with trajectory.open("a") as handle:
                    handle.write(trajectory.read_text().splitlines()[-1] + "\n")
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    diagnostic.resume_advection(output, checkpoint)


if __name__ == "__main__":
    unittest.main()
