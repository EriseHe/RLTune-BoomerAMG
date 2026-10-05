from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import unittest
from types import SimpleNamespace

from experiments.joint.solve_control.run_exp44_online_rl import make_controller
from experiments.joint.solve_control.run_true_online_sarsa_tuning import (
    CandidateSpec,
    DEFAULT_CONTROLLER_SEEDS,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    FINAL_STREAM_SEEDS,
    _candidate_aggregate,
    _controller_args,
    _native_solve_improvement_pct,
    candidate_grid,
    select_tuning_candidate,
    validate_seed_partition,
)


class TrueOnlineSarsaTuningTests(unittest.TestCase):
    def test_candidate_grid_matches_locked_protocol(self) -> None:
        grid = candidate_grid()
        self.assertEqual(len(grid), 6)
        self.assertEqual(
            {(candidate.alpha, candidate.trace_lambda) for candidate in grid},
            {
                (0.001, 0.2),
                (0.001, 0.5),
                (0.001, 0.8),
                (0.005, 0.2),
                (0.005, 0.5),
                (0.005, 0.8),
            },
        )

    def test_locked_controller_uses_true_online_uniform_and_setup_full(self) -> None:
        args = SimpleNamespace(
            weights="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
            anchor_weight=1.0,
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
        )
        controller_args = _controller_args(
            args,
            spec=CandidateSpec(0.005, 0.5),
            controller_seed=DEFAULT_CONTROLLER_SEEDS[0],
        )
        controller, encoder, config = make_controller(controller_args)
        self.assertEqual(config.td_algorithm, "true_online_sarsa")
        self.assertEqual(config.exploration_mode, "uniform")
        self.assertEqual(config.td_decay_power, 0.0)
        self.assertTrue(config.force_default_first_action)
        self.assertEqual(config.adaptive_cycles, None)
        self.assertEqual(encoder.mode, "setup_full")
        self.assertEqual(encoder.feature_dim, 35)
        self.assertEqual(controller.weights.size, 11)

    def test_seed_partitions_are_disjoint(self) -> None:
        validate_seed_partition(
            train_seeds=DEFAULT_TRAIN_SEEDS,
            eval_seeds=DEFAULT_EVAL_SEEDS,
            final_seeds=FINAL_STREAM_SEEDS,
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_seed_partition(
                train_seeds=DEFAULT_TRAIN_SEEDS,
                eval_seeds=(DEFAULT_TRAIN_SEEDS[0],),
                final_seeds=FINAL_STREAM_SEEDS,
            )

    def test_selection_applies_score_tolerance_then_tie_breaks(self) -> None:
        rows = [
            {
                "candidate": "best_score_but_slower_e2e",
                "alpha": 0.005,
                "trace_lambda": 0.8,
                "rejected": False,
                "score_mean_minus_std_pct": 1.00,
                "mean_end_to_end_runtime_sec": 0.100,
            },
            {
                "candidate": "within_tolerance_and_faster_e2e",
                "alpha": 0.005,
                "trace_lambda": 0.5,
                "rejected": False,
                "score_mean_minus_std_pct": 0.91,
                "mean_end_to_end_runtime_sec": 0.090,
            },
            {
                "candidate": "outside_tolerance",
                "alpha": 0.001,
                "trace_lambda": 0.2,
                "rejected": False,
                "score_mean_minus_std_pct": 0.89,
                "mean_end_to_end_runtime_sec": 0.080,
            },
        ]
        selected = select_tuning_candidate(rows)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate"], "within_tolerance_and_faster_e2e")

    def test_selection_returns_none_when_all_candidates_are_rejected(self) -> None:
        self.assertIsNone(
            select_tuning_candidate(
                [
                    {
                        "candidate": "rejected",
                        "alpha": 0.001,
                        "trace_lambda": 0.2,
                        "rejected": True,
                        "score_mean_minus_std_pct": 10.0,
                        "mean_end_to_end_runtime_sec": 0.01,
                    }
                ]
            )
        )

    def test_candidate_gate_rejects_failure_or_iteration_regression(self) -> None:
        spec = CandidateSpec(0.001, 0.2)
        seed_results = [
            {
                "controller_seed": DEFAULT_CONTROLLER_SEEDS[0],
                "native_solve_improvement_pct": 1.0,
                "native_solve_improvement_vs_default_pct": 1.0,
                "validation": {
                    "default_solve": {
                        "failed_count": 0,
                        "mean_iterations": 22.0,
                    },
                    "fixed_w1.6": {
                        "failed_count": 0,
                        "mean_iterations": 20.0,
                    },
                    "candidate": {
                        "failed_count": 1,
                        "mean_iterations": 22.0,
                        "mean_end_to_end_runtime_sec": 0.09,
                    },
                },
            }
        ]
        aggregate = _candidate_aggregate(spec=spec, seed_results=seed_results)
        self.assertTrue(aggregate["rejected"])
        self.assertEqual(len(aggregate["rejection_reasons"]), 2)

    def test_candidate_gate_requires_every_seed_to_exceed_default(self) -> None:
        spec = CandidateSpec(0.005, 0.8)
        seed_results = [
            {
                "controller_seed": DEFAULT_CONTROLLER_SEEDS[0],
                "native_solve_improvement_pct": -0.5,
                "native_solve_improvement_vs_default_pct": -0.1,
                "validation": {
                    "default_solve": {
                        "failed_count": 0,
                        "mean_iterations": 24.0,
                    },
                    "fixed_w1.6": {
                        "failed_count": 0,
                        "mean_iterations": 20.0,
                    },
                    "candidate": {
                        "failed_count": 0,
                        "mean_iterations": 20.5,
                        "mean_end_to_end_runtime_sec": 0.09,
                    },
                },
            }
        ]

        aggregate = _candidate_aggregate(spec=spec, seed_results=seed_results)

        self.assertTrue(aggregate["rejected"])
        self.assertIn("did not exceed default", aggregate["rejection_reasons"][0])

    def test_zero_runtime_baseline_is_reported_as_non_finite(self) -> None:
        improvement = _native_solve_improvement_pct(
            [{"solve_runtime": 0.0}],
            [{"solve_runtime": 0.1}],
        )
        self.assertTrue(improvement != improvement)


if __name__ == "__main__":
    unittest.main()
