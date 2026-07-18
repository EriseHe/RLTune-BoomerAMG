from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from run_mature_bandit_rl_pipeline import (
    _initialize_continuous_policy_weight,
)


class AbsolutePpoTests(unittest.TestCase):
    def test_actor_mean_is_initialized_at_solver_default(self) -> None:
        action_net = torch.nn.Linear(4, 1)
        model = SimpleNamespace(
            policy=SimpleNamespace(action_net=action_net)
        )

        normalized = _initialize_continuous_policy_weight(
            model,
            initial_weight=1.0,
            w_center=1.5,
            w_scale=0.5,
        )

        self.assertEqual(normalized, -1.0)
        self.assertTrue(torch.equal(action_net.weight, torch.zeros_like(action_net.weight)))
        self.assertTrue(torch.equal(action_net.bias, torch.full_like(action_net.bias, -1.0)))

    def test_initial_weight_must_be_inside_absolute_action_range(self) -> None:
        model = SimpleNamespace(
            policy=SimpleNamespace(action_net=torch.nn.Linear(2, 1))
        )
        with self.assertRaises(ValueError):
            _initialize_continuous_policy_weight(
                model,
                initial_weight=0.9,
                w_center=1.5,
                w_scale=0.5,
            )


if __name__ == "__main__":
    unittest.main()
