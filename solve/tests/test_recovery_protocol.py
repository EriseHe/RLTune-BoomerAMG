from __future__ import annotations

import copy
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
        self.last_step.status = (
            SolveStatus.CONVERGED
            if residual <= float(kwargs["tol"])
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

    def update(self, **kwargs):
        self.physical_targets.append(
            (
                kwargs.get("native_cycle_cost"),
                kwargs.get("residual_ratio"),
            )
        )
        return super().update(**kwargs)


class RecoveryProtocolTests(unittest.TestCase):
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
                solve_max_cycles=2,
                learn=True,
                explore=False,
            )

        self.assertFalse(outcome["failed"])
        self.assertEqual(len(controller.physical_targets), 2)
        self.assertEqual(controller.physical_targets[0], (0.001, 0.5))
        self.assertEqual(controller.physical_targets[1], (0.001, 2.0e-8))

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


if __name__ == "__main__":
    unittest.main()
