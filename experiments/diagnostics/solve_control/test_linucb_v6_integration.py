from __future__ import annotations

import _project_paths  # noqa: F401

import unittest

from joint_method_spec import ComposableMethodSpec
from setup.learners import SharedLinUCB_AMG_v6
from setup.learners.linucb.SharedLinUCB_AMG_v6 import (
    LINUCB_V6_CONTEXT_DIM,
    LINUCB_V6_CONTEXT_INTERACTION_INDICES,
    LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES,
    LINUCB_V6_RECOMMENDED_COARSEN_TYPES,
    LINUCB_V6_RECOMMENDED_INTERP_TYPES,
)
from setup.space import SetupConfigurationSpace
from setup_aware_compare_common import build_online_linucb_branch


class LinUCBV6IntegrationTests(unittest.TestCase):
    @staticmethod
    def _recommended_space() -> SetupConfigurationSpace:
        return SetupConfigurationSpace(
            name="recommended",
            coarsen_types=LINUCB_V6_RECOMMENDED_COARSEN_TYPES,
            interp_types=LINUCB_V6_RECOMMENDED_INTERP_TYPES,
            agg_interp_types=LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES,
        )

    def test_composable_method_and_runner_build_v6(self) -> None:
        method = ComposableMethodSpec.from_mapping(
            {
                "id": "internal_linucb_v6",
                "setup": "linucb_v6",
                "setup_space": "recommended",
                "candidate_sampling": "uniform512",
                "solve": "default",
            }
        )
        self.assertEqual(method.setup_kind, "linucb_v6")
        self.assertIn("LinUCB v6", method.label)

        branch, _config = build_online_linucb_branch(
            seed=101,
            learner_kind=method.setup_kind,
            tune_dim=7,
            tune7_variant="categorical",
            action_space_mode="full_cartesian",
            parameter_resolution=2,
            configuration_space=self._recommended_space(),
            context_dim=LINUCB_V6_CONTEXT_DIM,
            context_interaction_indices=(
                LINUCB_V6_CONTEXT_INTERACTION_INDICES
            ),
        )
        self.assertIsInstance(branch.policy.model, SharedLinUCB_AMG_v6)
        self.assertEqual(branch.family, "Shared LinUCB v6")
        self.assertEqual(branch.policy.model.d_phi, 2268)


if __name__ == "__main__":
    unittest.main()
