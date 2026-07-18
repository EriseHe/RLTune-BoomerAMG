from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from utils.setup_amg import build_actions_from_spec
from amg_setup_gym_env import build_setup_parameter_spec
from setup_aware_compare_common import (
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    build_grids_from_env,
    build_test_final_bandit_policy,
    default_test_final_bandit_config_from_env,
    validate_expected_setup_action_count,
)


class Exp44SetupResolutionTests(unittest.TestCase):
    def test_solve_runner_uses_setup_phase_tune7_candidate_defaults(self) -> None:
        parameter_spec = ParameterSpaceSpec(
            parameters=(
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
            )
        )
        actions = build_actions_from_spec(parameter_spec)

        with patch.dict(os.environ, {}, clear=True):
            cfg = default_test_final_bandit_config_from_env()

        expected = {
            "categorical": ("uniform", 512, "best_loss"),
            "agg_conditional": ("adaptive_local", 1024, "mean_loss"),
        }
        for variant, (strategy, pool_size, elite_metric) in expected.items():
            with self.subTest(variant=variant):
                policy = build_test_final_bandit_policy(
                    method="linucbv4",
                    tune_dim=7,
                    actions=actions,
                    context_dim=8,
                    seed=7,
                    default_params={"weight": 1.5},
                    default_arm_index=1,
                    parameter_spec=parameter_spec,
                    tune7_variant=variant,
                    cfg=cfg,
                )
                self.assertEqual(policy.model.candidate_strategy, strategy)
                self.assertEqual(policy.model._cand.candidate_pool_size, pool_size)
                self.assertEqual(policy.model._cand.elite_rank_metric, elite_metric)

    def test_linucb_v4_materializes_the_eager_action_feature_table(self) -> None:
        parameter_spec = ParameterSpaceSpec(
            parameters=(
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
            )
        )
        actions = build_actions_from_spec(parameter_spec)
        model = SharedLinUCB_AMG_v4(
            actions,
            context_dim=2,
            parameter_spec=parameter_spec,
            context_interaction_indices=(0, 1),
            candidate_pool_size=2,
            seed=7,
        )

        self.assertIsInstance(model.actions, list)
        self.assertIsInstance(model._g_actions, np.ndarray)
        self.assertEqual(model._g_actions.shape[0], len(actions))

    def test_matrix_grid_does_not_control_setup_parameter_resolution(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MATRIX_GRID_N": "40",
                "SETUP_PARAM_RESOLUTION": "20",
                "GRID_N": "3",
            },
            clear=True,
        ):
            resolution, threshold, max_row_sum, truncation = build_grids_from_env()
            parameter_spec, _fixed = build_setup_parameter_spec(
                tune_dim=7,
                tune7_variant="categorical",
            )

        self.assertEqual(resolution, EXP44_SETUP_PARAM_RESOLUTION)
        self.assertEqual(len(threshold), EXP44_SETUP_PARAM_RESOLUTION)
        self.assertEqual(len(max_row_sum), EXP44_SETUP_PARAM_RESOLUTION)
        self.assertEqual(len(truncation), EXP44_SETUP_PARAM_RESOLUTION)
        self.assertEqual(len(parameter_spec.parameters[0].values), 20)
        self.assertEqual(len(parameter_spec.parameters[1].values), 20)
        self.assertEqual(len(parameter_spec.parameters[2].values), 20)

        enumerated = int(np.prod([len(param.values) for param in parameter_spec.parameters]))
        default_is_represented = all(
            any(
                np.isclose(
                    float(param.default),
                    float(value),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for value in param.values
            )
            for param in parameter_spec.parameters
        )
        action_count = enumerated + int(not default_is_represented)
        self.assertEqual(action_count, EXP44_TUNE7_CATEGORICAL_ACTION_COUNT)

    def test_exp44_action_count_guard_rejects_coupled_grid_snapshot(self) -> None:
        valid = SimpleNamespace(
            policy=SimpleNamespace(
                model=SimpleNamespace(K=EXP44_TUNE7_CATEGORICAL_ACTION_COUNT)
            )
        )
        invalid = SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(K=23_040_001))
        )
        with patch.dict(
            os.environ,
            {"EXPECTED_SETUP_ACTION_COUNT": str(EXP44_TUNE7_CATEGORICAL_ACTION_COUNT)},
            clear=True,
        ):
            self.assertEqual(
                validate_expected_setup_action_count(valid),
                EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
            )
            with self.assertRaisesRegex(ValueError, "must remain independent"):
                validate_expected_setup_action_count(invalid)


if __name__ == "__main__":
    unittest.main()
