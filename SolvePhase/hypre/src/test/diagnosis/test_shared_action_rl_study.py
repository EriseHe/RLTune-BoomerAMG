from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from diagnosis.run_shared_action_rl_study import (
    CANDIDATE_METHODS,
    INDEPENDENT_METHOD,
    METHODS,
    _calibration_summary,
    _hierarchical_improvement,
    _make_controllers,
    _validate_pairing,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        tol=1.0e-6,
        max_cycles=2,
        c_max=1000.0,
        weights="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
        alpha=0.001,
        trace_lambda=0.8,
        epsilon_start=0.3,
        epsilon_final=0.03,
        epsilon_decay_steps=20_000.0,
        potential_scale_sec=0.0,
        failure_penalty_sec=0.1,
        action_rbf_sigma=0.2,
        action_rbf_centers="1.0,1.25,1.5,1.75,2.0",
        bootstrap_members=5,
        bootstrap_inclusion_probability=0.8,
        bootstrap_beta=1.0,
        lsvi_ridge=1.0,
        lsvi_beta=2.0,
        lsvi_residual_floor_sec=0.001,
    )


def _row(
    *,
    case: int,
    setup: str,
    solve: float = 1.0,
    eval_seed: int = 7,
) -> dict:
    return {
        "case_index": case,
        "eval_seed": eval_seed,
        "setup_hash": setup,
        "outcome": {
            "solve_runtime": solve,
            "end_to_end_runtime": solve,
            "failed": False,
        },
    }


class SharedActionStudyTests(unittest.TestCase):
    def test_factory_builds_exact_methods_and_compact_shared_basis(self) -> None:
        controllers, encoders = _make_controllers(_args(), seed=123)
        self.assertEqual(set(controllers), set(METHODS) - {"fixed_w1.6"})
        self.assertEqual(set(encoders), set(controllers))
        self.assertEqual(controllers[INDEPENDENT_METHOD].action_basis.shape, (11, 11))
        for method in CANDIDATE_METHODS:
            self.assertEqual(controllers[method].action_basis.shape, (11, 5))

    def test_pairing_requires_identical_case_and_setup(self) -> None:
        records = {
            method: [_row(case=0, setup="same")]
            for method in METHODS
        }
        self.assertEqual(_validate_pairing(records), 1.0)
        records[CANDIDATE_METHODS[0]][0]["setup_hash"] = "different"
        with self.assertRaises(ValueError):
            _validate_pairing(records)

    def test_calibration_uses_future_cycle_return(self) -> None:
        rows = [
            {
                "outcome": {
                    "cycle_selected_q_values": [0.002, 0.001],
                    "cycle_selected_uncertainties": [0.001, 0.001],
                    "cycle_times": [0.001, 0.001],
                    "failed": False,
                }
            }
        ]
        summary = _calibration_summary(rows, failure_penalty_sec=0.1)
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["mean_abs_error_sec"], 0.0)
        self.assertEqual(summary["coverage"]["95"], 1.0)

    def test_hierarchical_bootstrap_preserves_paired_improvement(self) -> None:
        seed_results = []
        for controller_seed in (1, 2):
            validation_records = {}
            for eval_seed in (10, 11):
                validation_records[eval_seed] = {
                    INDEPENDENT_METHOD: [
                        _row(case=index, setup="s", solve=0.009, eval_seed=eval_seed)
                        for index in range(5)
                    ],
                    CANDIDATE_METHODS[0]: [
                        _row(case=index, setup="s", solve=0.008, eval_seed=eval_seed)
                        for index in range(5)
                    ],
                }
            seed_results.append(
                {
                    "controller_seed": controller_seed,
                    "validation_records": validation_records,
                }
            )
        metric = _hierarchical_improvement(
            seed_results,
            candidate=CANDIDATE_METHODS[0],
            reference=INDEPENDENT_METHOD,
            metric="solve_runtime",
            bootstrap_samples=100,
            seed=99,
        )
        self.assertAlmostEqual(metric["improvement_pct"], 100.0 / 9.0)
        np.testing.assert_allclose(
            metric["hierarchical_bootstrap_95pct"],
            [100.0 / 9.0, 100.0 / 9.0],
        )


if __name__ == "__main__":
    unittest.main()
