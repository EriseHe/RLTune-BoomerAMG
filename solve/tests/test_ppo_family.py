from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

from hypre.bindings import SolveStatus
from setup.space import DEFAULT_SETUP_PARAMS
from solve.controllers.ppo import (
    FrozenPpoConfig,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    _SpaceOnlyEnv,
    build_frozen_ppo_runner,
)


def _model(*, action: float = 0.0) -> SimpleNamespace:
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(1,),
        dtype=np.float32,
    )
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(18,),
        dtype=np.float32,
    )
    return SimpleNamespace(
        policy=SimpleNamespace(action_space=action_space),
        action_space=action_space,
        observation_space=observation_space,
        predict=Mock(
            return_value=(
                np.asarray([action], dtype=np.float32),
                None,
            )
        ),
    )


def _runtime_config(
    directory: Path,
    **replacements: object,
) -> SetupAwareRLConfig:
    config = SetupAwareRLConfig(
        tune_dim=7,
        tune7_variant="categorical",
        algo="ppo",
        model_type="lstm",
        model_path=directory / "model.zip",
        vec_path=directory / "missing_vec.pkl",
        fixed_grid=(40, 40, 40),
        difconv_c_range=(1.0, 1000.0),
        w_only=True,
        w_center=1.5,
        w_scale=0.5,
        sweeps_min=1,
        sweeps_max=1,
        w_init=None,
        sweeps_init=None,
        solve_max_cycles=2,
        solve_tol=1.0e-6,
        default_setup_params=dict(DEFAULT_SETUP_PARAMS),
        obs_mode="cycle_action_setup",
        action_mode="discrete_w",
        discrete_w_values=(1.8,),
    )
    return replace(config, **replacements)


class _TwoCycleEnvironment:
    def __init__(self) -> None:
        self.r0 = 1.0
        self.last_step = SimpleNamespace(status=SolveStatus.CONTINUE)
        self.calls: list[dict[str, object]] = []

    def step_rl(self, **kwargs: object) -> tuple[float, float]:
        self.calls.append(dict(kwargs))
        status = (
            SolveStatus.CONTINUE
            if len(self.calls) == 1
            else SolveStatus.CONVERGED
        )
        self.last_step = SimpleNamespace(status=status)
        return 0.1 ** len(self.calls), 0.01


class PpoFamilyTests(unittest.TestCase):
    def test_model_loader_selection_is_unchanged(self) -> None:
        cases = (
            ("ppo", "lstm", "RecurrentPPO"),
            ("ppo", "mlp", "PPO"),
            ("dqn", "mlp", "DQN"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for algo, model_type, expected in cases:
                with self.subTest(algo=algo, model_type=model_type):
                    model = _model()
                    with (
                        patch(
                            "solve.controllers.ppo.runner.RecurrentPPO.load",
                            return_value=model,
                        ) as recurrent,
                        patch(
                            "solve.controllers.ppo.runner.PPO.load",
                            return_value=model,
                        ) as ppo,
                        patch(
                            "solve.controllers.ppo.runner.DQN.load",
                            return_value=model,
                        ) as dqn,
                    ):
                        runner = SetupAwareSolvePolicyRunner(
                            _runtime_config(
                                root,
                                algo=algo,
                                model_type=model_type,
                            )
                        )
                    loaders = {
                        "RecurrentPPO": recurrent,
                        "PPO": ppo,
                        "DQN": dqn,
                    }
                    loaders[expected].assert_called_once_with(
                        str(root / "model.zip")
                    )
                    self.assertTrue(
                        all(
                            loader.call_count == int(name == expected)
                            for name, loader in loaders.items()
                        )
                    )
                    self.assertEqual(
                        runner.use_lstm,
                        expected == "RecurrentPPO",
                    )

    def test_vecnormalize_and_space_host_contract_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vec_path = root / "vec.pkl"
            vec_path.touch()
            model = _model()
            vec_norm = SimpleNamespace(
                training=True,
                norm_reward=True,
            )
            with (
                patch(
                    "solve.controllers.ppo.runner.RecurrentPPO.load",
                    return_value=model,
                ),
                patch(
                    "solve.controllers.ppo.runner.VecNormalize.load",
                    return_value=vec_norm,
                ) as load_vec,
            ):
                runner = SetupAwareSolvePolicyRunner(
                    _runtime_config(root, vec_path=vec_path)
                )

            load_vec.assert_called_once()
            self.assertEqual(load_vec.call_args.args[0], str(vec_path))
            self.assertIs(runner.vec_norm, vec_norm)
            self.assertFalse(vec_norm.training)
            self.assertFalse(vec_norm.norm_reward)

            host = _SpaceOnlyEnv(
                observation_space=model.observation_space,
                action_space=model.action_space,
            )
            observation, info = host.reset(seed=7)
            np.testing.assert_array_equal(
                observation,
                np.zeros(18, dtype=np.float32),
            )
            self.assertEqual(info, {})
            with self.assertRaisesRegex(RuntimeError, "VecNormalize"):
                host.step(np.asarray([0.0]))

    def test_forced_default_action_remains_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _model()
            with patch(
                "solve.controllers.ppo.runner.RecurrentPPO.load",
                return_value=model,
            ):
                runner = SetupAwareSolvePolicyRunner(
                    _runtime_config(
                        root,
                        force_default_first_action=True,
                        default_first_weight=1.0,
                    )
                )

            environment = _TwoCycleEnvironment()
            result = runner.run(
                environment,
                mkw={
                    "k": 2.0,
                    "c": 3.0,
                    "a0": 4.0,
                    "nx": 40,
                    "ny": 40,
                    "nz": 40,
                },
                setup_params=dict(DEFAULT_SETUP_PARAMS),
            )
            self.assertEqual(result["cycle_actions"], [1.0, 1.8])
            self.assertEqual(
                result["cycle_forced_default_actions"],
                [True, False],
            )
            self.assertEqual(runner.forced_initial_action_count, 1)

            second_environment = _TwoCycleEnvironment()
            second = runner.run(
                second_environment,
                mkw={
                    "k": 2.0,
                    "c": 3.0,
                    "a0": 4.0,
                    "nx": 40,
                    "ny": 40,
                    "nz": 40,
                },
                setup_params=dict(DEFAULT_SETUP_PARAMS),
            )
            self.assertEqual(second["cycle_actions"], [1.8, 1.8])
            self.assertEqual(
                second["cycle_forced_default_actions"],
                [False, False],
            )
            self.assertEqual(runner.forced_initial_action_count, 1)

    def test_frozen_factory_preserves_exp44_runtime_contract(self) -> None:
        config = FrozenPpoConfig(
            ppo_model=Path("/tmp/model.zip"),
            output_dir=Path("/tmp/output"),
            grid_n=80,
            c_min=2.0,
            c_max=500.0,
            max_cycles=47,
            tol=2.0e-7,
            ppo_action_mode="Continuous_Absolute",
            ppo_w_center=1.4,
            ppo_w_scale=0.4,
            ppo_initial_observation_weight=1.1,
            ppo_force_default_first_action=True,
            ppo_default_first_weight=1.05,
        )
        built = Mock()
        with patch(
            "solve.controllers.ppo.factory.SetupAwareSolvePolicyRunner",
            return_value=built,
        ) as runner_type:
            result = build_frozen_ppo_runner(config)

        self.assertIs(result, built)
        policy_config = runner_type.call_args.args[0]
        self.assertEqual(policy_config.model_path, config.ppo_model)
        self.assertEqual(
            policy_config.vec_path,
            config.output_dir / ".unused_vecnormalize.pkl",
        )
        self.assertEqual(policy_config.fixed_grid, (80, 80, 80))
        self.assertEqual(policy_config.difconv_c_range, (2.0, 500.0))
        self.assertEqual(policy_config.solve_max_cycles, 47)
        self.assertEqual(policy_config.solve_tol, 2.0e-7)
        self.assertEqual(policy_config.obs_mode, "cycle_action_setup")
        self.assertEqual(policy_config.action_mode, "continuous_absolute")
        self.assertEqual(policy_config.w_center, 1.4)
        self.assertEqual(policy_config.w_scale, 0.4)
        self.assertEqual(policy_config.initial_observation_weight, 1.1)
        self.assertTrue(policy_config.force_default_first_action)
        self.assertEqual(policy_config.default_first_weight, 1.05)


if __name__ == "__main__":
    unittest.main()
