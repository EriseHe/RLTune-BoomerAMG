from __future__ import annotations

import _project_paths  # noqa: F401

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_online_methods_2k as legacy_factory
import setup_aware_compare_common as legacy_runtime
from solve.controllers import ppo


class PpoFamilyOwnershipTests(unittest.TestCase):
    def test_historical_runtime_imports_reexport_solve_owned_types(self) -> None:
        self.assertIs(
            legacy_runtime.SetupAwareRLConfig,
            ppo.SetupAwareRLConfig,
        )
        self.assertIs(
            legacy_runtime.SetupAwareSolvePolicyRunner,
            ppo.SetupAwareSolvePolicyRunner,
        )
        self.assertIs(legacy_runtime._SpaceOnlyEnv, ppo._SpaceOnlyEnv)

    def test_historical_exp44_factory_is_a_thin_typed_delegate(self) -> None:
        args = SimpleNamespace(
            ppo_model=Path("/tmp/model.zip"),
            output_dir=Path("/tmp/output"),
            grid_n=60,
            c_min=1.0,
            c_max=1000.0,
            max_cycles=50,
            tol=1.0e-6,
            ppo_action_mode="continuous_absolute",
            ppo_w_center=1.5,
            ppo_w_scale=0.5,
            ppo_initial_observation_weight=1.0,
            ppo_force_default_first_action=True,
            ppo_default_first_weight=1.0,
        )
        runner = Mock()
        with patch.object(
            legacy_factory,
            "build_frozen_ppo_runner",
            return_value=runner,
        ) as delegate:
            result = legacy_factory.make_exp44_ppo_runner(args)

        self.assertIs(result, runner)
        delegate.assert_called_once()
        config = delegate.call_args.args[0]
        self.assertIsInstance(config, ppo.FrozenPpoConfig)
        self.assertEqual(config.ppo_model, args.ppo_model)
        self.assertEqual(config.output_dir, args.output_dir)
        self.assertEqual(config.grid_n, 60)
        self.assertTrue(config.ppo_force_default_first_action)


if __name__ == "__main__":
    unittest.main()
