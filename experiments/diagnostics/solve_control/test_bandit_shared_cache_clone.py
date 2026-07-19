from __future__ import annotations

import _project_paths  # noqa: F401

import unittest
import tempfile
from pathlib import Path

import numpy as np

from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from setup_aware_compare_common import (
    BranchRun,
    GenericBanditPolicy,
    clone_branch_for_independent_updates,
)


class BanditSharedCacheCloneTests(unittest.TestCase):
    def test_clone_shares_catalog_but_not_online_state(self) -> None:
        spec = ParameterSpaceSpec(
            (
                ParameterSpec(
                    name="strong_threshold",
                    kind="continuous",
                    values=(0.1, 0.2, 0.3),
                    default=0.2,
                    center=0.2,
                    scale=0.1,
                ),
            )
        )
        actions = [
            {"strong_threshold": value}
            for value in (0.1, 0.2, 0.3)
        ]
        model = SharedLinUCB_AMG_v4(
            actions,
            context_dim=5,
            parameter_spec=spec,
            candidate_pool_size=3,
            elite_cache_size=2,
            seed=7,
        )
        parameter_space = {"actions": actions, "context_dim": 5}
        source = BranchRun(
            label="source",
            family="Shared LinUCB v4",
            tune_set="tune7",
            seed=7,
            policy=GenericBanditPolicy(model),
            parameter_space=parameter_space,
            solver_tol=1.0e-6,
            solver_max_iter=50,
        )
        clone = clone_branch_for_independent_updates(source)
        cloned_model = clone.policy.model

        self.assertIs(clone.parameter_space, source.parameter_space)
        self.assertIs(cloned_model.actions, model.actions)
        self.assertIs(cloned_model._g_actions, model._g_actions)
        self.assertFalse(cloned_model._g_actions.flags.writeable)
        self.assertIsNot(cloned_model.A_inv, model.A_inv)
        self.assertIsNot(cloned_model.b, model.b)
        self.assertIsNot(cloned_model._cand, model._cand)

        original_inverse = model.A_inv.copy()
        clone.policy.select(np.zeros(5, dtype=float))
        clone.policy.update(0.25)
        np.testing.assert_allclose(model.A_inv, original_inverse)
        self.assertFalse(np.allclose(cloned_model.A_inv, original_inverse))

    def test_lightweight_checkpoint_restores_mutable_state(self) -> None:
        spec = ParameterSpaceSpec(
            (
                ParameterSpec(
                    name="strong_threshold",
                    kind="continuous",
                    values=(0.1, 0.2),
                    default=0.2,
                    center=0.2,
                    scale=0.1,
                ),
            )
        )
        actions = [{"strong_threshold": value} for value in (0.1, 0.2)]
        source = SharedLinUCB_AMG_v4(
            actions,
            context_dim=5,
            parameter_spec=spec,
            candidate_pool_size=2,
            elite_cache_size=1,
            seed=19,
        )
        source.predict(np.ones(5, dtype=float))
        source.update(0.125)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npz"
            source.save_mutable_state(path, metadata={"stream_hash": "abc"})
            restored = SharedLinUCB_AMG_v4(
                actions,
                context_dim=5,
                parameter_spec=spec,
                candidate_pool_size=2,
                elite_cache_size=1,
                seed=20,
            )
            metadata = restored.load_mutable_state(path)
        self.assertEqual(metadata["stream_hash"], "abc")
        self.assertEqual(restored.t, source.t)
        np.testing.assert_allclose(restored.A_inv, source.A_inv)
        np.testing.assert_allclose(restored.b, source.b)
        np.testing.assert_allclose(
            restored._cand._arm_obs_count,
            source._cand._arm_obs_count,
        )
        self.assertEqual(
            restored.rng.bit_generator.state,
            source.rng.bit_generator.state,
        )


if __name__ == "__main__":
    unittest.main()
