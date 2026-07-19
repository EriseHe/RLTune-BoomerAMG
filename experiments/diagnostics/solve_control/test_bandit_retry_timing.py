from __future__ import annotations

import _project_paths  # noqa: F401

import unittest
from collections import deque
from dataclasses import replace

import numpy as np

from setup_aware_compare_common import (
    default_test_final_bandit_config_from_env,
    run_bandit_step_test_final,
)


class _Policy:
    def select(self, *, context, parameter_space):
        return {"arm": 0}

    def update(self, **_kwargs):
        return None


class BanditRetryTimingTests(unittest.TestCase):
    def test_retry_timings_include_every_attempt(self):
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
        cfg = replace(
            default_test_final_bandit_config_from_env(),
            retry_max_attempts=2,
        )

        _params, outcome, _timing, failed_attempts, _update_sec = (
            run_bandit_step_test_final(
                policy=_Policy(),
                parameter_space={"actions": [{"arm": 0}]},
                context=np.zeros(2, dtype=float),
                solver_fn=lambda _params: next(outcomes),
                prev_update_est=0.0,
                success_runtime_history=deque(maxlen=10),
                b_min_runtime_sec=1.0e-3,
                solver_tol=1.0e-6,
                cfg=cfg,
            )
        )

        self.assertEqual(failed_attempts, 1)
        self.assertFalse(outcome["failed"])
        self.assertAlmostEqual(outcome["runtime"], 1.00)
        self.assertAlmostEqual(outcome["setup_runtime"], 0.40)
        self.assertAlmostEqual(outcome["solve_runtime"], 0.60)
        self.assertAlmostEqual(outcome["infer_runtime"], 0.03)
        self.assertAlmostEqual(outcome["native_runtime"], 0.97)
        self.assertAlmostEqual(outcome["native_solve_runtime"], 0.57)


if __name__ == "__main__":
    unittest.main()
