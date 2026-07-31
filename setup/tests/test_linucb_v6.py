from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import unittest

import numpy as np

from setup.learners.common import SharedSetupLearnerSpec
from setup.learners.linucb import (
    LINUCB_V6_CONTEXT_DIM,
    LINUCB_V6_CONTEXT_FIELDS,
    LINUCB_V6_CONTEXT_INTERACTION_INDICES,
    SharedLinUCB_AMG_v6,
    validate_linucb_v6_experimental_contract,
)
from setup.learners.linucb.SharedLinUCB_AMG_v6 import (
    LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES,
    LINUCB_V6_RECOMMENDED_COARSEN_TYPES,
    LINUCB_V6_RECOMMENDED_INTERP_TYPES,
)
from setup.registry import (
    build_online_setup_learner,
    make_setup_learner_spec,
)
from setup.space import SetupConfigurationSpace, build_setup_param_space


class SharedLinUCBV6Tests(unittest.TestCase):
    @staticmethod
    def _setup_space():
        configuration = SetupConfigurationSpace(
            name="recommended",
            coarsen_types=LINUCB_V6_RECOMMENDED_COARSEN_TYPES,
            interp_types=LINUCB_V6_RECOMMENDED_INTERP_TYPES,
            agg_interp_types=LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES,
        )
        return build_setup_param_space(
            tune_dim=7,
            tune7_variant="categorical",
            parameter_resolution=2,
            configuration_space=configuration,
            materialize=False,
        )

    @classmethod
    def _shared(cls) -> SharedSetupLearnerSpec:
        setup_space = cls._setup_space()
        default_action = dict(
            setup_space.actions[int(setup_space.default_arm_index)]
        )
        return SharedSetupLearnerSpec(
            actions=(default_action,),
            context_dim=LINUCB_V6_CONTEXT_DIM,
            parameter_spec=setup_space.parameter_spec,
            context_interaction_indices=(
                LINUCB_V6_CONTEXT_INTERACTION_INDICES
            ),
            candidate_pool_size=1,
            always_include_arms=(0,),
            seed=17,
        )

    def test_factory_builds_complete_context_action_tensor(self) -> None:
        model = build_online_setup_learner(
            make_setup_learner_spec(
                kind="linucb_v6",
                shared=self._shared(),
            )
        )

        self.assertIsInstance(model, SharedLinUCB_AMG_v6)
        self.assertEqual(len(LINUCB_V6_CONTEXT_FIELDS), 28)
        self.assertEqual(model.d_x, 28)
        self.assertEqual(model.g_dim, 80)
        self.assertEqual(model.d_phi, 2268)

        context = np.linspace(-0.5, 0.5, LINUCB_V6_CONTEXT_DIM)
        context[0] = 1.0
        action_features = model._feature_for_arm(0)
        expected = np.concatenate(
            (
                context,
                action_features,
                *(
                    context[index] * action_features
                    for index in LINUCB_V6_CONTEXT_INTERACTION_INDICES
                ),
            )
        )
        np.testing.assert_allclose(model._phi(context, 0), expected)

    def test_constructor_rejects_context_contract_drift(self) -> None:
        shared = self._shared()
        with self.assertRaisesRegex(ValueError, "context_dim=28"):
            SharedLinUCB_AMG_v6(
                shared.actions,
                context_dim=8,
                parameter_spec=shared.parameter_spec,
            )
        with self.assertRaisesRegex(ValueError, "context interactions"):
            SharedLinUCB_AMG_v6(
                shared.actions,
                context_dim=28,
                parameter_spec=shared.parameter_spec,
                context_interaction_indices=(1, 2, 3),
            )

    def test_runner_contract_reuses_recommended_tune7_space(self) -> None:
        validate_linucb_v6_experimental_contract(
            context_dim=LINUCB_V6_CONTEXT_DIM,
            context_interaction_indices=(
                LINUCB_V6_CONTEXT_INTERACTION_INDICES
            ),
            tune_dim=7,
            tune7_variant="categorical",
            action_space_mode="full_cartesian",
            setup_space_name="recommended",
            coarsen_types=LINUCB_V6_RECOMMENDED_COARSEN_TYPES,
            interp_types=LINUCB_V6_RECOMMENDED_INTERP_TYPES,
            agg_interp_types=LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES,
        )

        with self.assertRaisesRegex(ValueError, "28-D"):
            validate_linucb_v6_experimental_contract(
                context_dim=8,
                context_interaction_indices=tuple(range(1, 8)),
                tune_dim=7,
                tune7_variant="categorical",
                action_space_mode="full_cartesian",
                setup_space_name="recommended",
                coarsen_types=LINUCB_V6_RECOMMENDED_COARSEN_TYPES,
                interp_types=LINUCB_V6_RECOMMENDED_INTERP_TYPES,
                agg_interp_types=(
                    LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES
                ),
            )


if __name__ == "__main__":
    unittest.main()
