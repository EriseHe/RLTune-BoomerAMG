from __future__ import annotations

import _project_paths  # noqa: F401

import unittest

import numpy as np

from learners import SharedLinUCB_AMG_v4
from learners.common import ParameterSpaceSpec, ParameterSpec
from utils.setup_amg import build_actions_from_spec


class SharedLinUCBV4Tests(unittest.TestCase):
    def test_materializes_the_shared_action_feature_table(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
