from __future__ import annotations

import unittest
from types import SimpleNamespace

from run_joint_online_sarsa_4k import (
    BEHAVIOR_MODES,
    BATCHED_LSVI_METHOD,
    LSVI_METHOD,
    LSVI_METHODS,
    RECURSIVE_LCB_METHODS,
    RECURSIVE_LSTDQ_METHOD,
    RECURSIVE_MC_METHOD,
    REFERENCE_METHODS,
    _make_lsvi_controller,
    _validate_lsvi_protocol,
    candidate_grid,
)
from run_online_bandit_td_lambda import _build_paired_instance_stream


class JointOnlineSarsa4KTests(unittest.TestCase):
    @staticmethod
    def _lsvi_args() -> SimpleNamespace:
        return SimpleNamespace(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
            weights="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
            action_rbf_centers="1.0,1.25,1.5,1.75,2.0",
            action_rbf_sigma=0.2,
            epsilon_start=0.3,
            epsilon_final=0.03,
            epsilon_decay_steps=20_000.0,
            potential_scale_sec=0.0,
            failure_penalty_sec=0.1,
            lsvi_ridge=1.0,
            lsvi_beta=2.0,
            lsvi_residual_floor_sec=1.0e-3,
            lsvi_refit_interval_episodes=100,
            recursive_mc_ridge=1.0,
            recursive_mc_beta=2.0,
            recursive_lstdq_ridge=1.0,
            recursive_lstdq_beta=2.0,
            recursive_lstdq_lambda=0.8,
        )

    def test_two_behaviors_cross_six_alpha_lambda_candidates(self) -> None:
        candidates = candidate_grid(
            alphas=(0.001, 0.005),
            trace_lambdas=(0.2, 0.5, 0.8),
        )
        self.assertEqual(len(candidates), 12)
        self.assertEqual({candidate.behavior_mode for candidate in candidates}, set(BEHAVIOR_MODES))
        self.assertEqual(len({candidate.name for candidate in candidates}), 12)
        self.assertEqual(len(REFERENCE_METHODS) + len(candidates), 15)

    def test_selected_configuration_has_one_candidate_per_behavior(self) -> None:
        candidates = candidate_grid(
            alphas=(0.001,),
            trace_lambdas=(0.8,),
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {candidate.behavior_mode for candidate in candidates},
            set(BEHAVIOR_MODES),
        )
        self.assertTrue(
            all(candidate.alpha == 0.001 for candidate in candidates)
        )
        self.assertTrue(
            all(candidate.trace_lambda == 0.8 for candidate in candidates)
        )

    def test_locked_4k_stream_hash_and_partition(self) -> None:
        args = SimpleNamespace(
            train_seed_groups=(
                "39800039,39806039,39812039,39818039,"
                "39824039,39830039,39836039,39842039"
            ),
            train_shuffle_seeds="39848039",
            train_cases_per_seed=500,
            train_group_take=4000,
            train_cases=4000,
            instance_offset=0,
            grid_n=40,
            c_min=1.0,
            c_max=1000.0,
            seed=39860039,
        )
        stream, manifest = _build_paired_instance_stream(args)
        self.assertEqual(len(stream[:2000]), 2000)
        self.assertEqual(len(stream[2000:]), 2000)
        self.assertEqual(
            manifest["sha256"],
            "64b8b0d63e59c7addb835e671b4dbcf498effa9c79084e3218e52bc9a92ec03f",
        )
        self.assertEqual(manifest["segments"][0]["window"], [0, 500])

    def test_lsvi_mode_has_only_three_requested_methods(self) -> None:
        self.assertEqual(
            LSVI_METHODS,
            ("bandit_default", "bandit_fixed_w1.6", LSVI_METHOD),
        )

    def test_recursive_lcb_suite_has_exactly_five_requested_methods(self) -> None:
        self.assertEqual(
            RECURSIVE_LCB_METHODS,
            (
                "bandit_default",
                "bandit_fixed_w1.6",
                RECURSIVE_MC_METHOD,
                RECURSIVE_LSTDQ_METHOD,
                BATCHED_LSVI_METHOD,
            ),
        )

    def test_locked_lsvi_controller_matches_selected_screen(self) -> None:
        args = self._lsvi_args()
        _validate_lsvi_protocol(args)
        controller, encoder = _make_lsvi_controller(args, seed=39866039)
        self.assertEqual(controller.feature_dim, encoder.feature_dim)
        self.assertEqual(controller.weights.size, 11)
        for actual, expected in zip(
            controller.weights,
            [1.0 + 0.1 * i for i in range(11)],
        ):
            self.assertAlmostEqual(float(actual), expected)
        self.assertEqual(controller.action_basis.shape[0], 11)
        self.assertEqual(controller.spec.uncertainty_beta, 2.0)


if __name__ == "__main__":
    unittest.main()
