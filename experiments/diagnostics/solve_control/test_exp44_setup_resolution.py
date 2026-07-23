from __future__ import annotations

import _project_paths  # noqa: F401

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from learners.common import (
    AOTCandidateSchedule,
    CompactActionCatalog,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    ParameterSpec,
)
from utils.setup_amg import build_actions_from_spec
from setup_action_space import (
    SetupConfigurationSpace,
    build_setup_parameter_spec,
    build_setup_param_space,
)
from setup_aware_compare_common import (
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    build_grids_from_env,
    build_test_final_bandit_policy,
    default_test_final_bandit_config_from_env,
    validate_expected_setup_action_count,
)


class Exp44SetupResolutionTests(unittest.TestCase):
    @staticmethod
    def _expanded_configuration_space() -> SetupConfigurationSpace:
        return SetupConfigurationSpace(
            name="expanded",
            coarsen_types=(0, 3, 6, 8, 10, 21, 22),
            interp_types=(0, 2, 3, 4, 6, 7, 8, 12, 13, 14, 16, 17, 18),
            agg_interp_types=(1, 2, 3, 4, 5, 6, 7, 8),
        )

    def test_compact_catalog_and_factorized_features_match_explicit_space(self) -> None:
        kwargs = {
            "tune_dim": 7,
            "tune7_variant": "categorical",
            "parameter_resolution": 2,
            "configuration_space": self._expanded_configuration_space(),
        }
        explicit = build_setup_param_space(**kwargs, materialize=True)
        compact = build_setup_param_space(**kwargs, materialize=False)
        self.assertIsInstance(compact.actions, CompactActionCatalog)
        self.assertEqual(len(compact.actions), len(explicit.actions))
        self.assertEqual(compact.default_arm_index, explicit.default_arm_index)

        rng = np.random.default_rng(17)
        probes = np.unique(
            np.concatenate(
                [
                    np.asarray([0, compact.default_arm_index]),
                    rng.choice(len(compact.actions), size=128, replace=False),
                ]
            )
        )
        for arm in probes:
            self.assertEqual(
                compact.actions[int(arm)], explicit.actions[int(arm)]
            )

        encoder = GenericActionFeatureEncoder(compact.parameter_spec)
        cache = FactorizedActionFeatureCache(compact.actions, encoder)
        expected = encoder.encode_actions(
            [explicit.actions[int(arm)] for arm in probes]
        )
        np.testing.assert_array_equal(cache.features(probes), expected)

    def test_aot_schedule_stores_only_unique_candidate_ids(self) -> None:
        compact = build_setup_param_space(
            tune_dim=7,
            tune7_variant="categorical",
            parameter_resolution=1,
            configuration_space=self._expanded_configuration_space(),
            materialize=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            schedule = AOTCandidateSchedule(
                directory=Path(directory),
                catalog=compact.actions,
                rounds=4,
                pool_size=32,
                seed=19,
            )
            rows = [schedule.next_arms() for _ in range(4)]
            self.assertTrue(all(len(np.unique(row)) == 32 for row in rows))
            self.assertEqual(
                {path.name for path in Path(directory).iterdir()},
                {"candidate_ids.npy", "metadata.json"},
            )
            reused = AOTCandidateSchedule(
                directory=Path(directory),
                catalog=compact.actions,
                rounds=4,
                pool_size=32,
                seed=19,
            )
            np.testing.assert_array_equal(reused.next_arms(), rows[0])

    def test_named_setup_space_materializes_its_explicit_candidates(self) -> None:
        configuration_space = SetupConfigurationSpace(
            name="candidate_set",
            coarsen_types=(3, 10),
            interp_types=(6, 17),
            agg_interp_types=(1, 4, 8),
        )
        setup_space = build_setup_param_space(
            tune_dim=7,
            tune7_variant="categorical",
            parameter_resolution=1,
            configuration_space=configuration_space,
        )

        self.assertIsInstance(setup_space.actions, tuple)
        self.assertEqual(len(setup_space.actions), 385)
        self.assertEqual(
            {int(action["coarsen_type"]) for action in setup_space.actions},
            {3, 10},
        )
        self.assertEqual(
            {int(action["interp_type"]) for action in setup_space.actions},
            {6, 17},
        )
        active_agg_actions = [
            action
            for action in setup_space.actions
            if int(action["agg_num_levels"]) > 0
        ]
        self.assertEqual(
            {int(action["agg_interp_type"]) for action in active_agg_actions},
            {1, 4, 8},
        )
        self.assertTrue(
            all(
                int(action["agg_interp_type"]) == 4
                for action in setup_space.actions
                if int(action["agg_num_levels"]) == 0
            )
        )

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

        compact = build_setup_param_space(
            tune_dim=7,
            tune7_variant="categorical",
            parameter_resolution=20,
            configuration_space=SetupConfigurationSpace(
                name="original",
                coarsen_types=(0, 2, 6, 8, 10),
                interp_types=(6, 8),
                agg_interp_types=(4,),
            ),
            materialize=False,
        )
        self.assertEqual(
            len(compact.actions), EXP44_TUNE7_CATEGORICAL_ACTION_COUNT
        )
        self.assertFalse(compact.actions.has_appended_default)

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
