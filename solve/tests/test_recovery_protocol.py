from __future__ import annotations

import copy
import itertools
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hypre.bindings import (
    AMGNativeError,
    AttemptStatus,
    NativeCode,
    SolveStatus,
    run_with_default_fallback,
)
from hypre.bindings.recovery import AttemptOutcome, RecoveryOutcome, InvalidObservationError, execute_attempt
from solve.controllers.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    run_td_episode,
)
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqV3LcbController,
    RecursiveLstdqV3LcbSpec,
)


class _Encoder:
    feature_dim = 1

    def encode(self, **_kwargs):
        return np.ones(1, dtype=float)


class _FakeEnv:
    def __init__(self, residuals):
        self._residuals = iter(residuals)
        self._cycles = 0
        self.last_step = SimpleNamespace(status=SolveStatus.CONTINUE)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def prepare_rl(self, params):
        del params
        return SimpleNamespace(
            setup_runtime_sec=0.002,
            initial_residual_norm=1.0,
            initial_relax_weight=1.0,
        )

    def step_rl(self, **kwargs):
        residual = float(next(self._residuals))
        self.last_step.status = (
            SolveStatus.CONVERGED
            if residual <= float(kwargs["tol"])
            else SolveStatus.MAX_CYCLES
        )
        return residual, 0.001


class _SetupFailureEnv(_FakeEnv):
    def prepare_rl(self, params):
        del params
        raise AMGNativeError(
            operation="setup",
            code=NativeCode.SETUP_ERROR,
            setup_runtime_sec=0.003,
        )


class _SolveFailureEnv(_FakeEnv):
    def step_rl(self, **_kwargs):
        raise AMGNativeError(
            operation="solve",
            code=NativeCode.SOLVE_ERROR,
            solve_runtime_sec=0.004,
        )


class _SequentialEnv(_FakeEnv):
    def step_rl(self, **kwargs):
        residual = float(next(self._residuals))
        self._cycles += 1
        self.last_step.status = (
            SolveStatus.MAX_CYCLES
            if self._cycles >= int(kwargs["max_cycles"])
            else SolveStatus.CONVERGED
            if residual < float(kwargs["tol"])
            else SolveStatus.CONTINUE
        )
        return residual, 0.001


def _result(*, failed: bool, runtime: float = 0.01):
    return {
        "runtime": float(runtime),
        "setup_runtime": 0.004,
        "solve_runtime": float(runtime - 0.004),
        "infer_runtime": 0.0,
        "failed": bool(failed),
        "failure_reason": (
            "max_iter_reached_without_convergence" if failed else ""
        ),
        "failure_stage": "solve" if failed else "",
        "residual_norm": 1.0 if failed else 1.0e-8,
        "iterations": 10,
    }


def _controller(seed: int = 7):
    config = ExpectedSarsaLambdaConfig(
        weights=(1.0, 1.1),
        anchor_weight=1.0,
        alpha=0.1,
        gamma=1.0,
        trace_lambda=0.0,
        epsilon_start=1.0,
        epsilon_final=1.0,
        epsilon_decay_steps=1.0,
        initial_q_sec=0.0,
        td_algorithm="true_online_sarsa",
    )
    return ExpectedSarsaLambda(feature_dim=1, config=config, seed=seed)


def _v3_controller(seed: int = 7) -> RecursiveLstdqV3LcbController:
    base = _controller(seed=seed)
    return RecursiveLstdqV3LcbController(
        feature_dim=1,
        config=base.config,
        spec=RecursiveLstdqV3LcbSpec(
            ridge=1.0,
            uncertainty_beta=2.0,
            residual_floor_sec=1.0e-3,
        ),
        seed=seed,
    )


class _RecordingController(ExpectedSarsaLambda):
    def __init__(self, *, seed: int = 7):
        base = _controller(seed=seed)
        super().__init__(
            feature_dim=base.feature_dim,
            config=base.config,
            seed=seed,
        )
        self.physical_targets = []
        self.learning_costs = []

    def update(self, **kwargs):
        self.learning_costs.append(kwargs["cost"])
        self.physical_targets.append(
            (
                kwargs.get("native_cycle_cost"),
                kwargs.get("residual_ratio"),
            )
        )
        return super().update(**kwargs)


class RecoveryProtocolTests(unittest.TestCase):
    def test_budgeted_terminals_charge_recovery_and_penalty_once(self):
        cases = (
            ("success", [0.5, 1e-8], 3, None, [0.001, 0.001], 0.004),
            ("recovered", [0.5, 0.4], 2, False, [0.001, 0.021], 0.024),
            ("unrecovered", [0.5, 0.4], 2, True, [0.001, 0.521], 0.024),
            ("nonfinite", [0.5, float("nan")], 3, True, [0.001, 0.521], 0.024),
            ("nonfinite_recovered", [0.5, float("nan")], 3, False, [0.001, 0.021], 0.024),
        )
        for name, residuals, cap, fallback_failed, costs, native_cost in cases:
            with self.subTest(name=name):
                controller = _RecordingController()
                fallback = None if fallback_failed is None else lambda: _result(failed=fallback_failed, runtime=0.02)
                with patch("solve.controllers.sarsa.online_td_lambda.create_env", return_value=_SequentialEnv(residuals)), \
                        patch("time.perf_counter", side_effect=itertools.count(step=1.0)):
                    result = run_td_episode(
                        mkw={}, params={}, controller=controller, encoder=_Encoder(),
                        solve_tol=1e-6, solve_max_cycles=cap, learn=True, explore=False,
                        fallback_attempt=fallback, failure_penalty_sec=0.5,
                    )
                np.testing.assert_allclose(controller.learning_costs, costs)
                self.assertAlmostEqual(result.get("native_runtime", result["runtime"]), native_cost)
                self.assertTrue(result["controller_update_committed"])
                self.assertEqual(controller.episodes, 1)
                self.assertEqual(controller.steps, 2)
                self.assertEqual(result["failure_penalty_sec"], 0.5 if fallback_failed else 0.0)
                self.assertTrue(np.all(np.isfinite(controller.theta)))

    def test_valid_native_exception_retains_failed_terminal(self):
        controller = _RecordingController()
        with patch("solve.controllers.sarsa.online_td_lambda.create_env", return_value=_SolveFailureEnv([])), \
                patch("time.perf_counter", side_effect=itertools.count(step=1.0)):
            result = run_td_episode(
                mkw={}, params={}, controller=controller, encoder=_Encoder(),
                solve_tol=1e-6, solve_max_cycles=3, learn=True, explore=False,
                fallback_attempt=lambda: _result(failed=True, runtime=0.02), failure_penalty_sec=0.5,
            )
        self.assertEqual(controller.physical_targets, [(None, None)])
        np.testing.assert_allclose(controller.learning_costs, [0.524])
        self.assertTrue(result["controller_update_committed"])
        self.assertAlmostEqual(result["native_runtime"], 0.026)

    def test_failed_v3_episode_matches_batch_and_retains_coercivity(self):
        controller = _v3_controller()
        controller.config = replace(controller.config, trace_lambda=0.8)
        with patch("solve.controllers.sarsa.online_td_lambda.create_env", return_value=_SequentialEnv([0.8, 0.7, 0.6])), \
                patch("time.perf_counter", side_effect=itertools.count(step=1.0)), \
                patch.object(controller, "update", wraps=controller.update) as updates:
            result = run_td_episode(
                mkw={}, params={}, controller=controller, encoder=_Encoder(),
                solve_tol=1e-6, solve_max_cycles=3, learn=True, explore=False,
                fallback_attempt=lambda: _result(failed=True, runtime=0.02), failure_penalty_sec=0.5,
            )
        matrix = np.eye(controller.joint_dim)
        target = np.zeros(controller.joint_dim)
        trace = np.zeros(controller.joint_dim)
        for call in updates.call_args_list:
            kw = call.kwargs
            phi = controller.state_action_feature(kw["features"], kw["action_index"]).copy()
            following = np.zeros_like(phi) if kw["terminal"] else controller.state_action_feature(
                kw["next_features"], kw["next_action_index"]
            ).copy()
            trace = 0.8 * trace + phi
            matrix += np.outer(trace, phi - following)
            target += trace * kw["cost"]
        np.testing.assert_allclose(controller.a_matrix, matrix)
        np.testing.assert_allclose(controller.b, target)
        np.testing.assert_allclose(controller.theta, np.linalg.solve(matrix, target))
        self.assertGreaterEqual(np.linalg.eigvalsh((matrix + matrix.T) / 2).min(), 1.0 - 1e-12)
        self.assertEqual(controller.episode_moment_count, 1)
        self.assertTrue(result["unrecovered_failure"])
        self.assertTrue(result["controller_update_committed"])

    def test_corrupt_times_and_estimator_errors_abort_and_restore_v3(self):
        for invalid_kind in ("nan_time", "clock_jump", "estimator"):
            with self.subTest(invalid_kind=invalid_kind):
                controller = _v3_controller()
                before = controller.snapshot_learning_state()
                env = _SequentialEnv([0.5, 0.4])
                actual_step = env.step_rl

                def step(**kwargs):
                    residual, cost = actual_step(**kwargs)
                    if env._cycles == 2 and invalid_kind != "estimator":
                        cost = float("nan") if invalid_kind == "nan_time" else 300.0
                    return residual, cost

                def broken_update(**kwargs):
                    controller.b[0] = float("nan")
                    raise FloatingPointError("estimator arithmetic")

                with ExitStack() as stack:
                    stack.enter_context(patch("solve.controllers.sarsa.online_td_lambda.create_env", return_value=env))
                    stack.enter_context(patch.object(env, "step_rl", side_effect=step))
                    stack.enter_context(patch("time.perf_counter", side_effect=itertools.count(step=1.0)))
                    if invalid_kind == "estimator":
                        stack.enter_context(patch.object(controller, "update", side_effect=broken_update))
                    with self.assertRaises(FloatingPointError if invalid_kind == "estimator" else InvalidObservationError):
                        run_td_episode(
                            mkw={}, params={}, controller=controller, encoder=_Encoder(),
                            solve_tol=1e-6, solve_max_cycles=3, learn=True, explore=False,
                            failure_penalty_sec=0.5,
                        )
                after = controller.snapshot_learning_state()
                for key in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance", "trace"):
                    np.testing.assert_array_equal(before[key], after[key])
                self.assertEqual(controller.episodes, 0)

    def test_strict_attempt_rejects_execution_errors_and_sanitized_times(self):
        def broken():
            raise ValueError("invalid input/implementation")

        with self.assertRaises(ValueError):
            execute_attempt(broken, require_valid_observation=True)
        with patch("time.perf_counter", side_effect=itertools.count(step=1.0)):
            for bad in (float("nan"), float("inf"), -1.0):
                with self.subTest(bad=bad), self.assertRaises(InvalidObservationError):
                    execute_attempt(
                        lambda: {**_result(failed=True), "solve_runtime": bad},
                        require_valid_observation=True,
                    )

    def test_strict_attempt_checks_completed_residual_and_fallback_origin(self):
        primary = AttemptOutcome.from_mapping({**_result(failed=True), "residual_norm": float("nan")})
        success = AttemptOutcome.from_mapping(_result(failed=False))
        recovered = RecoveryOutcome(primary=primary, fallback=success).to_result()
        with patch("time.perf_counter", side_effect=itertools.count(step=1.0)):
            accepted = execute_attempt(lambda: recovered, require_valid_observation=True)
            self.assertEqual(accepted.result["completed_status"], "success")
            with self.assertRaises(InvalidObservationError):
                execute_attempt(lambda: {**recovered, "fallback_failure_origin":"execution"},
                                require_valid_observation=True)

    def test_completed_attempt_survives_recovery_serialization(self):
        success = AttemptOutcome.from_mapping(_result(failed=False))
        failure = AttemptOutcome.from_mapping(_result(failed=True))
        zero_cost_failure = AttemptOutcome.from_mapping({
            "failed":True, "failure_stage":"setup", "setup_runtime":0., "solve_runtime":0.,
        })
        cases = (
            RecoveryOutcome(primary_attempts=(success,)),
            RecoveryOutcome(primary_attempts=(failure, success)),
            RecoveryOutcome(primary_attempts=(failure,), fallback=success),
            RecoveryOutcome(primary_attempts=(failure,), fallback=failure),
            RecoveryOutcome(primary=zero_cost_failure, fallback=success),
            RecoveryOutcome(primary=zero_cost_failure, fallback=failure),
        )
        for recovery in cases:
            with self.subTest(attempts=recovery.primary_attempt_count, fallback=recovery.fallback_used,
                              failed=recovery.unrecovered_failure):
                completed = recovery.fallback or recovery.primary
                payload = recovery.to_result()
                restored = RecoveryOutcome.from_mapping(payload).to_result()
                for key in ("runtime", "native_runtime", "setup_runtime", "solve_runtime", "infer_runtime"):
                    self.assertAlmostEqual(restored[key], payload[key])
                for result in (payload, restored):
                    self.assertEqual(result["completed_residual_norm"], completed.residual_norm)
                    self.assertEqual(result["completed_cycles"], completed.cycles)
                    self.assertEqual(result["completed_status"], completed.status.value)
                    if recovery.fallback_used:
                        self.assertEqual(result["fallback_residual_norm"], completed.residual_norm)
                        self.assertEqual(result["fallback_cycles"], completed.cycles)

    def test_episode_passes_native_cycle_targets_separately_from_learning_cost(self):
        controller = _RecordingController()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_SequentialEnv([0.5, 1.0e-8]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=3,
                learn=True,
                explore=False,
            )

        self.assertFalse(outcome["failed"])
        self.assertEqual(len(controller.physical_targets), 2)
        self.assertEqual(controller.physical_targets[0], (0.001, 0.5))
        self.assertEqual(controller.physical_targets[1], (0.001, 2.0e-8))
        self.assertEqual(controller.learning_costs, [0.001, 0.001])
        self.assertEqual(outcome["cycle_times"], controller.learning_costs)
        self.assertAlmostEqual(outcome["setup_runtime"], 0.002)
        self.assertAlmostEqual(outcome["runtime"], 0.004)
        self.assertAlmostEqual(
            outcome["runtime"], outcome["setup_runtime"] + outcome["native_solve_runtime"]
        )

    def test_controller_lifecycle_is_charged_on_success_recovery_and_rollback(self):
        cases = (
            ("success", _SequentialEnv([1e-8]), None, 0.03, 0.003),
            ("recovered", _FakeEnv([0.5]), False, 0.03, 0.023),
            ("unrecovered", _FakeEnv([0.5]), True, 0.07, 0.023),
            ("setup_failure", _SetupFailureEnv([]), None, 0.05, 0.003),
        )
        for label, env, fallback_failed, expected_lifecycle, expected_native in cases:
            with self.subTest(case=label), ExitStack() as patches:
                controller = _RecordingController()
                clock = [0.0]

                def timed(operation, duration):
                    def call(*args, **kwargs):
                        try:
                            return operation(*args, **kwargs)
                        finally:
                            clock[0] += duration
                    return call

                for name, duration in (
                    ("snapshot_learning_state", 0.01),
                    ("start_episode", 0.02),
                    ("restore_learning_state", 0.04),
                    ("finish_episode", 0.08),
                ):
                    operation = getattr(controller, name)
                    patches.enter_context(patch.object(controller, name, side_effect=timed(operation, duration)))
                patches.enter_context(patch(
                    "solve.controllers.sarsa.online_td_lambda.time.perf_counter",
                    side_effect=lambda: clock[0],
                ))
                patches.enter_context(patch(
                    "solve.controllers.sarsa.online_td_lambda.create_env", return_value=env,
                ))
                fallback = (None if fallback_failed is None else
                            lambda: _result(failed=fallback_failed, runtime=0.02))
                outcome = run_td_episode(
                    mkw={}, params={}, controller=controller, encoder=_Encoder(),
                    solve_tol=1e-6, solve_max_cycles=2, learn=True, explore=False,
                    fallback_attempt=fallback,
                )
                self.assertAlmostEqual(outcome["lifecycle_runtime"], expected_lifecycle)
                self.assertAlmostEqual(outcome["infer_runtime"], sum(
                    outcome[key] for key in (
                        "feature_runtime", "decision_runtime", "update_runtime", "lifecycle_runtime"
                    )
                ))
                self.assertAlmostEqual(outcome.get("native_runtime", outcome["runtime"]), expected_native)
                if label in {"success", "recovered"}:
                    self.assertAlmostEqual(outcome["update_runtime"], 0.08)
                if label == "success":
                    self.assertEqual(controller.learning_costs, [0.001])

    def test_native_exception_does_not_fabricate_physical_cycle_target(self):
        controller = _RecordingController()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_SolveFailureEnv([]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=False,
                fallback_attempt=lambda: _result(failed=False, runtime=0.02),
            )

        self.assertTrue(outcome["recovered"])
        self.assertEqual(controller.physical_targets, [(None, None)])
        self.assertEqual(controller.learning_costs, [0.004 + 0.02])
        self.assertAlmostEqual(outcome["primary_solve_runtime"], 0.004)
        self.assertAlmostEqual(outcome["native_runtime"], 0.002 + 0.004 + 0.02)
        self.assertAlmostEqual(
            outcome["runtime"], outcome["native_runtime"] + outcome["infer_runtime"]
        )

    def test_primary_success_does_not_call_fallback(self):
        calls = []
        recovery = run_with_default_fallback(
            lambda: _result(failed=False),
            lambda: calls.append("fallback") or _result(failed=False),
        )
        self.assertEqual(recovery.primary.status, AttemptStatus.SUCCESS)
        self.assertFalse(recovery.fallback_used)
        self.assertEqual(calls, [])

    def test_setup_failure_runs_one_default_fallback(self):
        recovery = run_with_default_fallback(
            lambda: {
                **_result(failed=True),
                "failure_stage": "setup",
                "failure_reason": "setup construction error",
            },
            lambda: _result(failed=False),
        )
        self.assertEqual(recovery.primary.status, AttemptStatus.SETUP_FAILURE)
        self.assertTrue(recovery.fallback_used)
        self.assertTrue(recovery.recovered)

    def test_default_baseline_failure_is_not_retried(self):
        fallback_calls = []
        recovery = run_with_default_fallback(
            lambda: _result(failed=True),
            lambda: fallback_calls.append(True) or _result(failed=False),
            primary_is_default=True,
        )
        self.assertTrue(recovery.unrecovered_failure)
        self.assertFalse(recovery.fallback_used)
        self.assertEqual(fallback_calls, [])

    def test_nonfinite_result_has_typed_status(self):
        recovery = run_with_default_fallback(
            lambda: {
                **_result(failed=True),
                "failure_reason": "nonfinite residual",
                "residual_norm": float("nan"),
            },
            lambda: _result(failed=False),
        )
        self.assertIs(recovery.primary.status, AttemptStatus.NONFINITE_RESULT)
        self.assertTrue(recovery.recovered)

    def test_setup_failure_is_returned_for_bandit_reselection(self):
        controller = _controller()
        learning_before = controller.snapshot_learning_state()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_SetupFailureEnv([]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=False,
                fallback_attempt=lambda: _result(failed=False, runtime=0.02),
            )

        learning_after = controller.snapshot_learning_state()
        np.testing.assert_array_equal(
            learning_after["theta"], learning_before["theta"]
        )
        self.assertEqual(learning_after["episodes"], learning_before["episodes"])
        self.assertEqual(outcome["primary_status"], "setup_failure")
        self.assertFalse(outcome["fallback_used"])
        self.assertFalse(outcome["recovery_protocol_applied"])
        self.assertTrue(outcome["unrecovered_failure"])
        self.assertFalse(outcome["controller_update_committed"])

    def test_recovered_solve_error_commits_fallback_as_terminal_cost(self):
        controller = _controller()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_SolveFailureEnv([]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=False,
                fallback_attempt=lambda: _result(failed=False, runtime=0.02),
            )

        self.assertEqual(outcome["primary_status"], "solve_failure")
        self.assertTrue(outcome["recovered"])
        self.assertTrue(outcome["controller_update_committed"])
        self.assertEqual(controller.episodes, 1)
        self.assertEqual(controller.steps, 1)
        self.assertGreater(float(np.max(controller.theta)), 0.0)

    def test_double_failure_rolls_back_learning_but_not_rng(self):
        controller = _controller()
        learning_before = controller.snapshot_learning_state()
        rng_before = copy.deepcopy(controller.rng.bit_generator.state)

        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_FakeEnv([1.0]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=True,
                fallback_attempt=lambda: _result(failed=True),
            )

        learning_after = controller.snapshot_learning_state()
        np.testing.assert_array_equal(
            learning_after["theta"], learning_before["theta"]
        )
        np.testing.assert_array_equal(
            learning_after["td_counts"], learning_before["td_counts"]
        )
        self.assertEqual(learning_after["steps"], learning_before["steps"])
        self.assertEqual(learning_after["episodes"], learning_before["episodes"])
        self.assertNotEqual(controller.rng.bit_generator.state, rng_before)
        self.assertTrue(outcome["fallback_used"])
        self.assertTrue(outcome["unrecovered_failure"])
        self.assertFalse(outcome["controller_update_committed"])

    def test_recovered_nonconvergence_commits_one_episode(self):
        controller = _controller()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_FakeEnv([1.0]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=False,
                fallback_attempt=lambda: _result(failed=False, runtime=0.02),
            )

        self.assertTrue(outcome["fallback_used"])
        self.assertTrue(outcome["recovered"])
        self.assertFalse(outcome["unrecovered_failure"])
        self.assertTrue(outcome["controller_update_committed"])
        self.assertEqual(controller.episodes, 1)
        self.assertEqual(controller.steps, 1)
        self.assertGreater(outcome["runtime"], 0.02)

    def test_v3_recovered_nonconvergence_commits_one_cluster(self):
        controller = _v3_controller()
        covariance_before = controller.episode_moment_covariance.copy()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_FakeEnv([1.0]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=False,
                fallback_attempt=lambda: _result(failed=False, runtime=0.02),
            )

        self.assertTrue(outcome["controller_update_committed"])
        self.assertEqual(controller.episodes, 1)
        self.assertEqual(controller.episode_moment_count, 1)
        self.assertFalse(
            np.array_equal(
                controller.episode_moment_covariance,
                covariance_before,
            )
        )

    def test_v3_unrecovered_failure_restores_cluster_and_mean_state(self):
        controller = _v3_controller()
        state_before = controller.snapshot_learning_state()
        with patch(
            "solve.controllers.sarsa.online_td_lambda.create_env",
            return_value=_FakeEnv([1.0]),
        ):
            outcome = run_td_episode(
                mkw={},
                params={},
                controller=controller,
                encoder=_Encoder(),
                solve_tol=1.0e-6,
                solve_max_cycles=1,
                learn=True,
                explore=True,
                fallback_attempt=lambda: _result(failed=True),
            )

        state_after = controller.snapshot_learning_state()
        for key in (
            "a_matrix",
            "a_inverse",
            "b",
            "theta",
            "episode_moment_covariance",
            "trace",
        ):
            np.testing.assert_array_equal(state_after[key], state_before[key])
        for key in (
            "steps",
            "episodes",
            "sample_count",
            "episode_moment_count",
            "episode_active",
        ):
            self.assertEqual(state_after[key], state_before[key])
        self.assertTrue(outcome["unrecovered_failure"])
        self.assertFalse(outcome["controller_update_committed"])

    def test_v3_target_on_cycle_50_uses_recovery_and_transaction_rules(self):
        for fallback_failed in (False, True):
            with self.subTest(fallback_failed=fallback_failed):
                controller = _v3_controller()
                state_before = controller.snapshot_learning_state()
                calls = []

                def fallback():
                    calls.append(True)
                    return _result(failed=fallback_failed, runtime=0.02)

                with patch(
                    "solve.controllers.sarsa.online_td_lambda.create_env",
                    return_value=_SequentialEnv([0.5]*49 + [8e-7]),
                ):
                    outcome = run_td_episode(
                        mkw={}, params={}, controller=controller, encoder=_Encoder(),
                        solve_tol=1e-6, solve_max_cycles=50, learn=True, explore=False,
                        fallback_attempt=fallback,
                    )
                self.assertEqual(calls, [True])
                self.assertEqual(outcome["primary_status"], "nonconvergence")
                self.assertEqual(outcome["primary_cycles"], 50)
                self.assertLess(outcome["primary_residual_norm"], 1e-6)
                self.assertAlmostEqual(outcome["primary_solve_runtime"], 0.05)
                self.assertAlmostEqual(outcome["native_runtime"], 0.052 + 0.02)
                self.assertAlmostEqual(outcome["runtime"], outcome["native_runtime"] + outcome["infer_runtime"])
                self.assertEqual(outcome["controller_update_committed"], not fallback_failed)
                self.assertEqual(outcome["unrecovered_failure"], fallback_failed)
                if fallback_failed:
                    after = controller.snapshot_learning_state()
                    for key in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance"):
                        np.testing.assert_array_equal(after[key], state_before[key])
                    self.assertEqual(controller.episodes, 0)
                else:
                    self.assertEqual(controller.episodes, 1)
                    self.assertEqual(controller.steps, 50)
                    self.assertEqual(controller.episode_moment_count, 1)


if __name__ == "__main__":
    unittest.main()
