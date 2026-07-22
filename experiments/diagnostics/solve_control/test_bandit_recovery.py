from __future__ import annotations

import _project_paths  # noqa: F401

import unittest

import numpy as np

from setup_aware_compare_common import (
    run_bandit_step_test_final,
)


class _Policy:
    def __init__(self):
        self.updates = 0
        self.cancelled = 0
        self.model = self

    def select(self, *, context, parameter_space):
        return {"arm": 0}

    def update(self, **_kwargs):
        self.updates += 1

    def cancel_pending(self):
        self.cancelled += 1


class BanditRecoveryTimingTests(unittest.TestCase):
    def test_default_fallback_timings_and_single_update(self):
        outcomes = iter(
            [
                {
                    "runtime": 0.30,
                    "setup_runtime": 0.10,
                    "solve_runtime": 0.20,
                    "infer_runtime": 0.01,
                    "native_runtime": 0.29,
                    "native_solve_runtime": 0.19,
                    "failed": True,
                    "residual_norm": 1.0,
                    "iterations": 50,
                },
                {
                    "runtime": 0.70,
                    "setup_runtime": 0.30,
                    "solve_runtime": 0.40,
                    "infer_runtime": 0.02,
                    "native_runtime": 0.68,
                    "native_solve_runtime": 0.38,
                    "failed": False,
                    "residual_norm": 1.0e-8,
                    "iterations": 20,
                },
            ]
        )
        policy = _Policy()

        _params, outcome, _timing, fallback_used, _update_sec = (
            run_bandit_step_test_final(
                policy=policy,
                parameter_space={"actions": [{"arm": 0}]},
                context=np.zeros(2, dtype=float),
                solver_fn=lambda _params: next(outcomes),
                fallback_solver_fn=lambda _params: next(outcomes),
                prev_update_est=0.0,
            )
        )

        self.assertEqual(fallback_used, 1)
        self.assertFalse(outcome["failed"])
        self.assertAlmostEqual(outcome["runtime"], 1.00)
        self.assertAlmostEqual(outcome["setup_runtime"], 0.40)
        self.assertAlmostEqual(outcome["solve_runtime"], 0.60)
        self.assertAlmostEqual(outcome["infer_runtime"], 0.03)
        self.assertAlmostEqual(outcome["native_runtime"], 0.97)
        self.assertAlmostEqual(outcome["native_solve_runtime"], 0.57)
        self.assertTrue(outcome["fallback_used"])
        self.assertTrue(outcome["recovered"])
        self.assertTrue(outcome["bandit_update_committed"])
        self.assertEqual(policy.updates, 1)

    def test_double_failure_cancels_pending_update(self):
        policy = _Policy()
        failed = {
            "runtime": 0.2,
            "setup_runtime": 0.1,
            "solve_runtime": 0.1,
            "failed": True,
            "failure_reason": "max_iter_reached_without_convergence",
            "failure_stage": "solve",
            "residual_norm": 1.0,
            "iterations": 50,
        }

        _params, outcome, _timing, fallback_used, _update_sec = (
            run_bandit_step_test_final(
                policy=policy,
                parameter_space={"actions": [{"arm": 0}]},
                context=np.zeros(2, dtype=float),
                solver_fn=lambda _params: dict(failed),
                fallback_solver_fn=lambda _params: dict(failed),
                prev_update_est=0.0,
            )
        )

        self.assertEqual(fallback_used, 1)
        self.assertTrue(outcome["unrecovered_failure"])
        self.assertFalse(outcome["bandit_update_committed"])
        self.assertEqual(policy.updates, 0)
        self.assertEqual(policy.cancelled, 1)


if __name__ == "__main__":
    unittest.main()
