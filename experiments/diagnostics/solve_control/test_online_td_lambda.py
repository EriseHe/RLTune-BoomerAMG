from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from problems.registry import context_for_setup_method
from solve.controllers.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    OnlineFixedWeightIncumbent,
    SolveStateEncoder,
)
from solve.controllers.common import (
    LEGACY_DIFFUSION_ONLY_CONTEXT,
    PHYSICS_LINEAR_PROBLEM_CONTEXT,
)
from experiments.joint.solve_control.run_exp44_online_rl import _average_repeated_rows


class OnlineTDLambdaTests(unittest.TestCase):
    def test_setup_full_encoder_appends_reused_setup_features(self) -> None:
        class StubSetupEncoder:
            observed_keys = ("strong_threshold", "coarsen_type")

            @staticmethod
            def encode(params):
                return np.asarray(
                    [params["strong_threshold"], params["coarsen_type"]],
                    dtype=float,
                )

        encoder = SolveStateEncoder(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
            mode="setup_full",
            setup_obs_encoder=StubSetupEncoder(),
        )
        features = encoder.encode(
            mkw={"k": 1.0, "c": 1.0, "a0": 1.0},
            setup_params={"strong_threshold": 0.25, "coarsen_type": 1.5},
            initial_residual=1.0,
            residual=0.1,
            previous_residual=1.0,
            cycle=1,
            last_weight=1.6,
            last_cycle_time=1.0e-3,
        )

        self.assertEqual(encoder.feature_dim, 30)
        self.assertTrue(np.allclose(features[-2:], np.asarray([0.25, 1.5])))

    def test_action_rbf_shares_td_update_with_nearby_weights(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0, 1.1, 1.2, 1.8),
                anchor_weight=1.0,
                alpha=1.0,
                trace_lambda=0.0,
                initial_q_sec=0.0,
                action_rbf_sigma=0.1,
            ),
            seed=17,
        )
        features = np.asarray([1.0])

        learner.start_episode()
        learner.update(
            features=features,
            action_index=1,
            cost=1.0,
            next_features=features,
            terminal=True,
        )

        self.assertGreater(learner.theta[1, 0], learner.theta[0, 0])
        self.assertGreater(learner.theta[0, 0], 0.0)
        self.assertGreater(learner.theta[2, 0], 0.0)
        self.assertEqual(learner.theta[3, 0], 0.0)

    def test_repeated_evaluation_treats_missing_policy_overhead_as_zero(self) -> None:
        averaged = _average_repeated_rows(
            [
                [
                    {
                        "runtime": 0.08,
                        "setup_runtime": 0.03,
                        "solve_runtime": 0.05,
                        "failed": False,
                    }
                ],
                [
                    {
                        "runtime": 0.10,
                        "setup_runtime": 0.04,
                        "solve_runtime": 0.06,
                        "failed": False,
                    }
                ],
            ]
        )

        self.assertTrue(np.isclose(averaged[0]["runtime"], 0.09))
        self.assertEqual(averaged[0]["infer_runtime"], 0.0)

    def test_online_incumbent_balances_actions_and_uses_only_observed_costs(self) -> None:
        estimator = OnlineFixedWeightIncumbent(
            weights=(1.0, 1.5, 2.0),
            initial_weight=1.0,
            seed=12,
        )
        observed = {1.0: 0.08, 1.5: 0.05, 2.0: 0.07}

        for _ in range(6):
            action_index, weight = estimator.select_action()
            estimator.update(action_index, observed[weight])

        self.assertTrue(np.array_equal(estimator.counts, np.asarray([2, 2, 2])))
        self.assertEqual(estimator.incumbent_weight, 1.5)
        self.assertEqual(estimator.summary()["incumbent_weight"], 1.5)

    def test_state_encoder_is_bounded_and_has_declared_dimension(self) -> None:
        encoder = SolveStateEncoder(tol=1.0e-6, max_cycles=50, c_max=1000.0)
        features = encoder.encode(
            mkw={"k": 1000.0, "c": 10.0, "a0": 1.0},
            initial_residual=10.0,
            residual=0.1,
            previous_residual=1.0,
            cycle=2,
            last_weight=1.6,
            last_cycle_time=1.0e-3,
        )

        self.assertEqual(features.shape, (encoder.feature_dim,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_canonical_encoder_observes_advection_while_legacy_does_not(
        self,
    ) -> None:
        mkw = {
            "nx": 60,
            "ny": 60,
            "nz": 60,
            "k": 1000.0,
            "c": 10.0,
            "a0": 1.0,
            "a1": 3.0,
            "a2": 4.0,
            "a3": 5.0,
        }
        common = {
            "mkw": mkw,
            "initial_residual": 1.0,
            "residual": 0.1,
            "previous_residual": 1.0,
            "cycle": 2,
            "last_weight": 1.6,
            "last_cycle_time": 1.0e-3,
        }
        first_context = np.asarray(
            [1.0, 1.0, 0.5, 0.0, 0.5, 0.1, 0.2, 0.3],
        )
        second_context = first_context.copy()
        second_context[5:] = [0.8, 0.7, 0.6]

        canonical = SolveStateEncoder(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
        )
        canonical_first = canonical.encode(
            problem_context=first_context,
            **common,
        )
        canonical_second = canonical.encode(
            problem_context=second_context,
            **common,
        )
        np.testing.assert_array_equal(
            canonical_first[7:14],
            first_context[1:],
        )
        self.assertFalse(
            np.array_equal(canonical_first, canonical_second)
        )

        legacy = SolveStateEncoder(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
            problem_context_mode=LEGACY_DIFFUSION_ONLY_CONTEXT,
        )
        legacy_first = legacy.encode(
            problem_context=first_context,
            **common,
        )
        legacy_second = legacy.encode(
            problem_context=second_context,
            **common,
        )
        np.testing.assert_array_equal(legacy_first, legacy_second)
        self.assertEqual(canonical.feature_dim, legacy.feature_dim + 3)

    def test_physics_linear_encoder_reuses_setup_physics_projection(
        self,
    ) -> None:
        mkw = {
            "nx": 60,
            "ny": 60,
            "nz": 60,
            "k": 1000.0,
            "c": 10.0,
            "a0": 1.0,
            "a1": 300.0,
            "a2": -20.0,
            "a3": 5.0,
        }
        canonical_context = np.asarray(
            [1.0, 1.0, 0.5, 0.0, 0.5, 0.8, -0.4, 0.2],
        )
        encoder = SolveStateEncoder(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
            problem_context_mode=PHYSICS_LINEAR_PROBLEM_CONTEXT,
        )
        features = encoder.encode(
            mkw=mkw,
            problem_context=canonical_context,
            initial_residual=1.0,
            residual=0.1,
            previous_residual=1.0,
            cycle=2,
            last_weight=1.6,
            last_cycle_time=1.0e-3,
        )
        setup_context = context_for_setup_method(
            problem_kind="scalar_anisotropic_diffusion_advection",
            setup_kind="linucb",
            matrix_kwargs=mkw,
            stream_context=canonical_context,
            setup_context="physics_linear",
        )

        np.testing.assert_allclose(features[7:13], setup_context[1:])
        self.assertEqual(encoder.problem_context_fields[0], "bias")
        self.assertEqual(encoder.problem_feature_dim, 6)

    def test_terminal_cost_update_increases_selected_action_cost(self) -> None:
        config = ExpectedSarsaLambdaConfig(
            weights=(1.4, 1.6),
            anchor_weight=1.4,
            alpha=0.5,
            epsilon_start=0.0,
            epsilon_final=0.0,
            initial_q_sec=0.0,
        )
        learner = ExpectedSarsaLambda(feature_dim=2, config=config, seed=1)
        features = np.asarray([1.0, 0.5])

        learner.start_episode()
        td_error = learner.update(
            features=features,
            action_index=0,
            cost=0.01,
            next_features=features,
            terminal=True,
        )

        self.assertGreater(td_error, 0.0)
        self.assertGreater(learner.q_values(features)[0], 0.0)
        self.assertEqual(learner.q_values(features)[1], 0.0)

    def test_decayed_td_step_is_sample_average_for_terminal_tabular_state(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0,),
                anchor_weight=1.0,
                alpha=1.0,
                td_decay_power=1.0,
                trace_lambda=0.0,
                initial_q_sec=0.0,
            ),
            seed=13,
        )
        features = np.asarray([1.0])

        for cost in (0.01, 0.03):
            learner.start_episode()
            learner.update(
                features=features,
                action_index=0,
                cost=cost,
                next_features=features,
                terminal=True,
            )

        self.assertTrue(np.isclose(learner.q_values(features)[0], 0.02))
        self.assertEqual(learner.td_counts[0, 0], 2.0)

    def test_decayed_td_trace_uses_each_parameters_visit_count(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=2,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0,),
                anchor_weight=1.0,
                alpha=1.0,
                td_decay_power=1.0,
                gamma=1.0,
                trace_lambda=1.0,
                epsilon_start=0.0,
                epsilon_final=0.0,
                initial_q_sec=0.0,
            ),
            seed=14,
        )
        first = np.asarray([1.0, 0.0])
        second = np.asarray([0.0, 1.0])
        learner.td_counts[0, 0] = 3.0

        learner.start_episode()
        learner.update(
            features=first,
            action_index=0,
            cost=0.0,
            next_features=second,
            terminal=False,
        )
        learner.update(
            features=second,
            action_index=0,
            cost=2.0,
            next_features=second,
            terminal=True,
        )

        self.assertTrue(np.allclose(learner.theta[0], np.asarray([0.5, 2.0])))

    def test_cycle_tabular_encoder_separates_cycle_values(self) -> None:
        encoder = SolveStateEncoder(
            tol=1.0e-6,
            max_cycles=5,
            c_max=1000.0,
            mode="cycle_tabular",
        )
        common = {
            "mkw": {"k": 1.0, "c": 1.0, "a0": 1.0},
            "initial_residual": 1.0,
            "residual": 1.0,
            "previous_residual": 1.0,
            "last_weight": 1.4,
            "last_cycle_time": 0.0,
        }

        cycle_zero = encoder.encode(cycle=0, **common)
        cycle_one = encoder.encode(cycle=1, **common)

        self.assertEqual(encoder.feature_dim, 6)
        self.assertEqual(float(cycle_zero @ cycle_one), 0.0)
        self.assertEqual(float(np.sum(cycle_zero)), 1.0)

    def test_nonterminal_update_bootstraps_from_next_state(self) -> None:
        config = ExpectedSarsaLambdaConfig(
            weights=(1.4, 1.6),
            anchor_weight=1.4,
            alpha=0.1,
            gamma=1.0,
            epsilon_start=0.0,
            epsilon_final=0.0,
            initial_q_sec=0.0,
        )
        learner = ExpectedSarsaLambda(feature_dim=1, config=config, seed=2)
        learner.theta[0, 0] = 0.02
        learner.theta[1, 0] = 0.03
        features = np.asarray([1.0])

        learner.start_episode()
        td_error = learner.update(
            features=features,
            action_index=1,
            cost=0.005,
            next_features=features,
            terminal=False,
        )

        self.assertTrue(np.isclose(td_error, -0.005))

    def test_true_online_sarsa_matches_two_step_reference_update(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=2,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0,),
                anchor_weight=1.0,
                alpha=0.1,
                gamma=1.0,
                trace_lambda=0.8,
                epsilon_start=0.0,
                epsilon_final=0.0,
                initial_q_sec=0.0,
                td_algorithm="true_online_sarsa",
            ),
            seed=21,
        )
        learner.theta[0] = np.asarray([0.02, -0.01])
        first = np.asarray([1.0, 2.0])
        second = np.asarray([0.5, -1.0])

        learner.start_episode()
        learner.update(
            features=first,
            action_index=0,
            cost=0.01,
            next_features=second,
            next_action_index=0,
            terminal=False,
        )
        learner.update(
            features=second,
            action_index=0,
            cost=0.03,
            next_features=second,
            terminal=True,
        )

        self.assertTrue(
            np.allclose(learner.theta[0], np.asarray([0.024585, -0.00397]))
        )
        self.assertTrue(np.allclose(learner.eligibility[0], np.asarray([1.36, 0.48])))

    def test_accumulating_sarsa_uses_actual_next_behavior_action(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0, 1.1),
                anchor_weight=1.0,
                alpha=0.1,
                gamma=1.0,
                trace_lambda=0.8,
                epsilon_start=0.0,
                epsilon_final=0.0,
                initial_q_sec=0.0,
                td_algorithm="sarsa",
            ),
            seed=24,
        )
        learner.theta[:, 0] = np.asarray([0.02, 0.03])

        learner.start_episode()
        td_error = learner.update(
            features=np.asarray([1.0]),
            action_index=0,
            cost=0.005,
            next_features=np.asarray([1.0]),
            next_action_index=1,
            terminal=False,
        )

        self.assertTrue(np.isclose(td_error, 0.015))
        self.assertTrue(np.allclose(learner.theta[:, 0], np.asarray([0.0215, 0.03])))

    def test_accumulating_sarsa_requires_real_next_behavior_action(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0, 1.1),
                anchor_weight=1.0,
                td_algorithm="sarsa",
            ),
            seed=25,
        )

        learner.start_episode()
        with self.assertRaisesRegex(ValueError, "next behavior action"):
            learner.update(
                features=np.asarray([1.0]),
                action_index=0,
                cost=0.01,
                next_features=np.asarray([1.0]),
                terminal=False,
            )

    def test_true_online_sarsa_requires_real_next_behavior_action(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0, 1.1),
                anchor_weight=1.0,
                td_algorithm="true_online_sarsa",
            ),
            seed=22,
        )

        learner.start_episode()
        with self.assertRaisesRegex(ValueError, "next behavior action"):
            learner.update(
                features=np.asarray([1.0]),
                action_index=0,
                cost=0.01,
                next_features=np.asarray([1.0]),
                terminal=False,
            )

    def test_true_online_sarsa_rejects_per_parameter_step_decay(self) -> None:
        with self.assertRaisesRegex(ValueError, "constant scalar step size"):
            ExpectedSarsaLambda(
                feature_dim=1,
                config=ExpectedSarsaLambdaConfig(
                    weights=(1.0,),
                    anchor_weight=1.0,
                    td_algorithm="true_online_sarsa",
                    td_decay_power=0.6,
                ),
                seed=23,
            )

    def test_base_controller_forces_prepared_default_exactly_once(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.0, 1.1, 1.2),
                anchor_weight=1.0,
                epsilon_start=0.0,
                epsilon_final=0.0,
                initial_q_sec=0.0,
                td_algorithm="true_online_sarsa",
                force_default_first_action=True,
            ),
            seed=31,
        )
        learner.theta[:, 0] = np.asarray([3.0, 2.0, 0.0])
        features = np.asarray([1.0])

        learner.start_episode(initial_environment_weight=1.1)
        first, first_value, first_metadata = learner.select_action(
            features,
            explore=True,
        )
        learner.finish_episode(learned=True)
        learner.start_episode(initial_environment_weight=1.0)
        second, second_value, second_metadata = learner.select_action(
            features,
            explore=True,
        )

        self.assertEqual((first, first_value), (1, 1.1))
        self.assertTrue(first_metadata["forced_default_first_action"])
        self.assertEqual((second, second_value), (2, 1.2))
        self.assertFalse(second_metadata["forced_default_first_action"])
        self.assertEqual(learner.forced_initial_action_count, 1)

    def test_checkpoint_does_not_force_default_a_second_time(self) -> None:
        config = ExpectedSarsaLambdaConfig(
            weights=(1.0, 1.2),
            anchor_weight=1.0,
            epsilon_start=0.0,
            epsilon_final=0.0,
            initial_q_sec=0.0,
            td_algorithm="true_online_sarsa",
            force_default_first_action=True,
        )
        learner = ExpectedSarsaLambda(feature_dim=1, config=config, seed=32)
        learner.start_episode(initial_environment_weight=1.0)
        learner.select_action(np.asarray([1.0]), explore=True)
        learner.finish_episode(learned=True)
        learner.theta[:, 0] = np.asarray([1.0, 0.0])

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "controller.npz"
            learner.save(checkpoint)
            restored = ExpectedSarsaLambda(feature_dim=1, config=config, seed=33)
            metadata = restored.load(checkpoint)
            restored.start_episode(initial_environment_weight=1.0)
            action, value, selection = restored.select_action(
                np.asarray([1.0]),
                explore=True,
            )

        self.assertFalse(metadata["default_first_action_pending"])
        self.assertEqual(metadata["forced_initial_action_count"], 1)
        self.assertEqual((action, value), (1, 1.2))
        self.assertFalse(selection["forced_default_first_action"])

    def test_episode_start_resets_eligibility_without_resetting_parameters(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(weights=(1.4,), anchor_weight=1.4, initial_q_sec=0.0),
            seed=3,
        )
        learner.update(
            features=np.asarray([1.0]),
            action_index=0,
            cost=0.01,
            next_features=np.asarray([1.0]),
            terminal=True,
        )
        theta = learner.theta.copy()

        learner.start_episode()

        self.assertTrue(np.all(learner.eligibility == 0.0))
        self.assertTrue(np.array_equal(learner.theta, theta))

    def test_monte_carlo_correction_uses_future_cycle_costs(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=2,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4,),
                anchor_weight=1.4,
                initial_q_sec=0.0,
                monte_carlo_alpha=1.0,
                gamma=1.0,
            ),
            seed=4,
        )
        first = np.asarray([1.0, 0.0])
        second = np.asarray([0.0, 1.0])

        learner.monte_carlo_update([(first, 0, 0.01), (second, 0, 0.02)])

        self.assertTrue(np.isclose(learner.q_values(first)[0], 0.03))
        self.assertTrue(np.isclose(learner.q_values(second)[0], 0.02))

    def test_monte_carlo_control_variate_centers_full_future_returns(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=2,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4,),
                anchor_weight=1.4,
                initial_q_sec=0.0,
                monte_carlo_alpha=1.0,
                gamma=1.0,
            ),
            seed=11,
        )
        first = np.asarray([1.0, 0.0])
        second = np.asarray([0.0, 1.0])

        learner.monte_carlo_update(
            [(first, 0, 0.01), (second, 0, 0.02)],
            baseline_cost_sec=0.025,
        )

        self.assertTrue(np.isclose(learner.q_values(first)[0], 0.005))
        self.assertTrue(np.isclose(learner.q_values(second)[0], -0.005))

    def test_adaptive_cycle_limit_falls_back_to_anchor(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=3,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4, 1.6),
                anchor_weight=1.4,
                adaptive_cycles=2,
            ),
            seed=5,
        )
        features = np.asarray([1.0, 0.0, 0.0])
        learner.theta[1, 0] = -1.0

        early_index, early_weight, _ = learner.select_action(
            features,
            explore=False,
            cycle=1,
        )
        late_index, late_weight, selection = learner.select_action(
            features,
            explore=False,
            cycle=2,
        )

        self.assertEqual(early_index, 1)
        self.assertEqual(early_weight, 1.6)
        self.assertEqual(late_index, learner.anchor_index)
        self.assertEqual(late_weight, 1.4)
        self.assertEqual(selection["allowed_indices"], [learner.anchor_index])

    def test_bootstrap_respects_next_cycle_action_limit(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4, 1.6),
                anchor_weight=1.4,
                alpha=0.1,
                gamma=1.0,
                epsilon_start=0.0,
                epsilon_final=0.0,
                initial_q_sec=0.0,
                adaptive_cycles=1,
            ),
            seed=6,
        )
        learner.theta[:, 0] = np.asarray([0.02, -1.0])
        features = np.asarray([1.0])

        learner.start_episode()
        td_error = learner.update(
            features=features,
            action_index=1,
            cost=0.005,
            next_features=features,
            terminal=False,
            next_cycle=1,
        )

        self.assertTrue(np.isclose(td_error, 1.025))

    def test_checkpoint_load_restores_values_and_counters(self) -> None:
        config = ExpectedSarsaLambdaConfig(weights=(1.4, 1.6), anchor_weight=1.4)
        source = ExpectedSarsaLambda(feature_dim=2, config=config, seed=7)
        source.theta[:] = np.asarray([[0.1, 0.2], [0.3, 0.4]])
        source.steps = 12
        source.episodes = 3

        with TemporaryDirectory() as directory:
            path = Path(directory) / "controller.npz"
            source.save(path)
            restored = ExpectedSarsaLambda(feature_dim=2, config=config, seed=8)
            metadata = restored.load(path)

        self.assertTrue(np.array_equal(restored.theta, source.theta))
        self.assertEqual(restored.steps, 12)
        self.assertEqual(restored.episodes, 3)
        self.assertEqual(metadata["restored_steps"], 12)

    def test_decayed_monte_carlo_update_is_sample_average_for_tabular_state(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4,),
                anchor_weight=1.4,
                initial_q_sec=0.0,
                monte_carlo_alpha=1.0,
                monte_carlo_decay_power=1.0,
            ),
            seed=9,
        )
        features = np.asarray([1.0])

        learner.monte_carlo_update([(features, 0, 0.01)])
        learner.monte_carlo_update([(features, 0, 0.03)])

        self.assertTrue(np.isclose(learner.q_values(features)[0], 0.02))
        self.assertEqual(learner.monte_carlo_counts[0, 0], 2.0)

    def test_least_visited_exploration_balances_tabular_actions(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=2,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4, 1.6),
                anchor_weight=1.4,
                epsilon_start=1.0,
                epsilon_final=1.0,
                exploration_mode="least_visited",
            ),
            seed=10,
        )
        features = np.asarray([1.0, 0.0])
        learner.monte_carlo_counts[0, 0] = 2.0

        action_index, weight, _selection = learner.select_action(
            features,
            explore=True,
            cycle=0,
        )

        self.assertEqual(action_index, 1)
        self.assertEqual(weight, 1.6)

    def test_least_visited_exploration_uses_td_counts(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=2,
            config=ExpectedSarsaLambdaConfig(
                weights=(1.4, 1.6),
                anchor_weight=1.4,
                epsilon_start=1.0,
                epsilon_final=1.0,
                exploration_mode="least_visited",
            ),
            seed=11,
        )
        features = np.asarray([1.0, 0.0])
        learner.td_counts[0, 0] = 2.0

        action_index, weight, _selection = learner.select_action(
            features,
            explore=True,
            cycle=0,
        )

        self.assertEqual(action_index, 1)
        self.assertEqual(weight, 1.6)


if __name__ == "__main__":
    unittest.main()
