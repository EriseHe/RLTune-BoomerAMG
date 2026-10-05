from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import unittest
import numpy as np

from setup.learners.common import (
    SetupLearnerFactoryRequest,
    SharedSetupLearnerSpec,
    ParameterSpaceSpec,
    ParameterSpec,
)
from setup.learners.linucb import (
    SharedLinUCB_AMG_v4,
    LinUCBV4Spec,
    build_linucb_v4_learner,
)
from setup.registry import (
    COMPOSABLE_SETUP_KINDS,
    ONLINE_SETUP_KINDS,
    SetupLearnerBuildSpec,
    build_online_setup_learner,
    make_setup_learner_spec,
    setup_kind_registration,
)
from setup.space import build_actions_from_spec


class SetupRegistryTests(unittest.TestCase):
    @staticmethod
    def _shared(*, seed: int = 7) -> SharedSetupLearnerSpec:
        parameter_spec = ParameterSpaceSpec(
            (
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
        return SharedSetupLearnerSpec(
            actions=build_actions_from_spec(parameter_spec),
            context_dim=2,
            parameter_spec=parameter_spec,
            context_interaction_indices=(0, 1),
            candidate_pool_size=3,
            always_include_arms=(1,),
            elite_cache_size=1,
            initial_guess=(1.5,),
            initial_guess_rounds=1,
            seed=seed,
        )

    def test_registry_contains_only_paper_setup_kinds(self) -> None:
        self.assertEqual(COMPOSABLE_SETUP_KINDS, ("default", "linucb"))
        self.assertEqual(ONLINE_SETUP_KINDS, ("linucb",))
        self.assertIs(
            setup_kind_registration("linucbv4").learner_type, SharedLinUCB_AMG_v4
        )
        self.assertIs(
            setup_kind_registration("linucb").factory, build_linucb_v4_learner
        )
        for kind in ("random", "linucb_v5", "linucb_v6", "lints"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                setup_kind_registration(kind)

    def test_canonical_learner_class_path_is_stable(self) -> None:
        self.assertEqual(
            SharedLinUCB_AMG_v4.__module__, "setup.learners.linucb.SharedLinUCB_AMG_v4"
        )
        self.assertEqual(LinUCBV4Spec.__module__, "setup.learners.linucb.config")

    def test_linucb_factory_matches_direct_constructor(self) -> None:
        shared = self._shared()
        built = build_online_setup_learner(
            make_setup_learner_spec(kind="linucb", shared=shared)
        )
        direct = SharedLinUCB_AMG_v4(
            shared.actions,
            context_dim=shared.context_dim,
            **shared.learner_kwargs(),
        )

        context = np.asarray((1.0, 0.25), dtype=float)
        self.assertIsInstance(built, SharedLinUCB_AMG_v4)
        self.assertEqual(built.predict(context), direct.predict(context))
        np.testing.assert_allclose(built.A_inv, direct.A_inv)
        np.testing.assert_allclose(built.b, direct.b)

        family_built = build_linucb_v4_learner(
            SetupLearnerFactoryRequest(
                shared=shared,
                algorithm=LinUCBV4Spec(),
            )
        )
        self.assertIsInstance(family_built, SharedLinUCB_AMG_v4)
        np.testing.assert_allclose(family_built.A_inv, direct.A_inv)

    def test_factory_rejects_a_non_linucb_algorithm_spec(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires LinUCBV4Spec"):
            build_online_setup_learner(
                SetupLearnerBuildSpec(
                    kind="linucb", shared=self._shared(), algorithm=object()
                )
            )


if __name__ == "__main__":
    unittest.main()
