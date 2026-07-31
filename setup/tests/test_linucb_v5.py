from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import unittest

import numpy as np

from problems.scalar_anisotropic_diffusion_advection import (
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_FIELDS,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES,
)
from setup.learners.common import (
    FactorizedActionFeatureCache,
    RBFActionFeatureEncoder,
    SharedSetupLearnerSpec,
)
from setup.learners.linucb import (
    LINUCB_V5_CONTEXT_DIM,
    LINUCB_V5_CONTEXT_FIELDS,
    LINUCB_V5_CONTEXT_INTERACTION_INDICES,
    SharedLinUCB_AMG_v4,
    SharedLinUCB_AMG_v5,
    SharedLinUCB_AMG_v5_RBF,
)
from setup.learners.linucb.SharedLinUCB_AMG_v5 import (
    LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES,
    LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
    LINUCB_V5_RECOMMENDED_INTERP_TYPES,
    validate_linucb_v5_paper_contract,
)
from setup.registry import (
    build_online_setup_learner,
    make_setup_learner_spec,
)
from setup.space import (
    SetupConfigurationSpace,
    build_setup_param_space,
)


class SharedLinUCBV5Tests(unittest.TestCase):
    @staticmethod
    def _setup_space():
        configuration = SetupConfigurationSpace(
            name="recommended",
            coarsen_types=LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
            interp_types=LINUCB_V5_RECOMMENDED_INTERP_TYPES,
            agg_interp_types=LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES,
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
        return SharedSetupLearnerSpec(
            actions=setup_space.actions,
            context_dim=LINUCB_V5_CONTEXT_DIM,
            parameter_spec=setup_space.parameter_spec,
            context_interaction_indices=(
                LINUCB_V5_CONTEXT_INTERACTION_INDICES
            ),
            candidate_pool_size=8,
            always_include_arms=(setup_space.default_arm_index,),
            seed=17,
        )

    def test_context_contract_matches_diffusion_advection_problem(self) -> None:
        self.assertEqual(
            LINUCB_V5_CONTEXT_FIELDS,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_FIELDS,
        )
        self.assertEqual(
            LINUCB_V5_CONTEXT_DIM,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
        )
        self.assertEqual(
            LINUCB_V5_CONTEXT_INTERACTION_INDICES,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES,
        )

    def test_factory_builds_distinct_v5_with_v4_algorithm_state(self) -> None:
        shared = self._shared()
        v5 = build_online_setup_learner(
            make_setup_learner_spec(kind="linucb_v5", shared=shared)
        )
        v4 = SharedLinUCB_AMG_v4(
            shared.actions,
            context_dim=shared.context_dim,
            **shared.learner_kwargs(),
        )

        self.assertIsInstance(v5, SharedLinUCB_AMG_v5)
        self.assertNotEqual(type(v5), type(v4))
        self.assertEqual(v5.d_phi, v4.d_phi)
        np.testing.assert_array_equal(v5.A_inv, v4.A_inv)
        context = np.asarray(
            [1.0, 0.2, 0.3, 0.4, 0.3, 0.1, 0.2, 0.3],
            dtype=float,
        )
        self.assertEqual(v5.predict(context), v4.predict(context))

    def test_frozen_v5_keeps_the_legacy_action_feature_layout(self) -> None:
        shared = self._shared()
        v5 = build_online_setup_learner(
            make_setup_learner_spec(kind="linucb_v5", shared=shared)
        )
        action = shared.actions[0]
        numeric = []
        categorical_blocks = []
        for parameter in shared.parameter_spec.parameters:
            active = all(
                action[dependency] in allowed
                for dependency, allowed in parameter.active_if
            )
            if parameter.kind in {"continuous", "integer"}:
                numeric.append(
                    0.0
                    if not active
                    else (
                        float(action[parameter.name])
                        - float(parameter.center)
                    )
                    / float(parameter.scale)
                )
            else:
                levels = tuple(
                    value
                    for value in parameter.values
                    if value != parameter.default
                )
                categorical_blocks.append(
                    np.asarray(
                        [
                            float(
                                active
                                and action[parameter.name] == level
                            )
                            for level in levels
                        ],
                        dtype=float,
                    )
                )
        numeric_array = np.asarray(numeric, dtype=float)
        pairwise = np.asarray(
            [
                numeric_array[i] * numeric_array[j]
                for i in range(numeric_array.size)
                for j in range(i + 1, numeric_array.size)
            ],
            dtype=float,
        )
        categorical = np.concatenate(categorical_blocks)
        mixed = np.concatenate(
            [
                indicator * numeric_array
                for block in categorical_blocks
                for indicator in block
            ]
        )
        expected = np.concatenate(
            [
                numeric_array,
                numeric_array**2,
                pairwise,
                categorical,
                mixed,
            ]
        )
        np.testing.assert_allclose(v5._encoder.encode_action(action), expected)

    def test_v5_rejects_nonfinal_context_contract(self) -> None:
        shared = self._shared()
        kwargs = shared.learner_kwargs()
        with self.assertRaisesRegex(ValueError, "context_dim=8"):
            SharedLinUCB_AMG_v5(
                shared.actions,
                context_dim=11,
                **kwargs,
            )
        kwargs["context_interaction_indices"] = (1, 2, 3, 4)
        with self.assertRaisesRegex(ValueError, "context interactions"):
            SharedLinUCB_AMG_v5(
                shared.actions,
                context_dim=8,
                **kwargs,
            )

    def test_rbf_is_separate_and_preserves_frozen_v5(self) -> None:
        shared = self._shared()
        v5 = build_online_setup_learner(
            make_setup_learner_spec(kind="linucb_v5", shared=shared)
        )
        rbf = build_online_setup_learner(
            make_setup_learner_spec(kind="linucb_v5_rbf", shared=shared)
        )

        self.assertIs(type(v5), SharedLinUCB_AMG_v5)
        self.assertIs(type(rbf), SharedLinUCB_AMG_v5_RBF)
        self.assertNotIsInstance(v5._encoder, RBFActionFeatureEncoder)
        self.assertIsInstance(rbf._encoder, RBFActionFeatureEncoder)
        self.assertEqual(v5.g_dim, 80)
        self.assertEqual(rbf.g_dim, 92)
        self.assertEqual(v5.d_phi, 648)
        self.assertEqual(rbf.d_phi, 744)

        action = shared.actions[0]
        v5_features = v5._encoder.encode_action(action)
        rbf_features = rbf._encoder.encode_action(action)
        np.testing.assert_allclose(rbf_features[:5], v5_features[:5])
        for offset in (5, 10, 15):
            self.assertAlmostEqual(
                float(np.sum(rbf_features[offset : offset + 5])),
                1.0,
            )

        cache = FactorizedActionFeatureCache(
            shared.actions,
            rbf._encoder,
        )
        arm_indices = np.asarray(
            [0, shared.actions.default_arm_index, len(shared.actions) - 1],
            dtype=np.int64,
        )
        expected = np.vstack(
            [
                rbf._encoder.encode_action(shared.actions[int(index)])
                for index in arm_indices
            ]
        )
        np.testing.assert_allclose(cache.features(arm_indices), expected)

    def test_paper_contract_includes_coarsen_type_two(self) -> None:
        validate_linucb_v5_paper_contract(
            context_dim=8,
            context_interaction_indices=(1, 2, 3, 4, 5, 6, 7),
            tune_dim=7,
            tune7_variant="categorical",
            action_space_mode="full_cartesian",
            setup_space_name="recommended",
            coarsen_types=(0, 2, 3, 6, 8, 10),
            interp_types=(0, 2, 3, 6, 8, 17),
            agg_interp_types=(4,),
        )
        with self.assertRaisesRegex(ValueError, "coarsen_types"):
            validate_linucb_v5_paper_contract(
                context_dim=8,
                context_interaction_indices=(1, 2, 3, 4, 5, 6, 7),
                tune_dim=7,
                tune7_variant="categorical",
                action_space_mode="full_cartesian",
                setup_space_name="recommended",
                coarsen_types=(0, 3, 6, 8, 10),
                interp_types=(0, 2, 3, 6, 8, 17),
                agg_interp_types=(4,),
            )


if __name__ == "__main__":
    unittest.main()
