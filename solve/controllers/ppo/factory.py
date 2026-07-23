from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SetupAwareRLConfig
from .runner import SetupAwareSolvePolicyRunner


@dataclass(frozen=True)
class FrozenPpoConfig:
    """Typed experiment-selected inputs for the frozen Exp44 policy."""

    ppo_model: Path
    output_dir: Path
    grid_n: int
    c_min: float
    c_max: float
    max_cycles: int
    tol: float
    ppo_action_mode: str = "continuous_absolute"
    ppo_w_center: float = 1.5
    ppo_w_scale: float = 0.5
    ppo_initial_observation_weight: float = 1.0
    ppo_force_default_first_action: bool = False
    ppo_default_first_weight: float = 1.0

    @classmethod
    def from_runtime(cls, runtime: Any) -> "FrozenPpoConfig":
        """Decode the historical runner-shaped object at its boundary."""

        initial_weight = float(
            getattr(runtime, "ppo_initial_observation_weight", 1.0)
        )
        return cls(
            ppo_model=Path(runtime.ppo_model),
            output_dir=Path(runtime.output_dir),
            grid_n=int(runtime.grid_n),
            c_min=float(runtime.c_min),
            c_max=float(runtime.c_max),
            max_cycles=int(runtime.max_cycles),
            tol=float(runtime.tol),
            ppo_action_mode=str(
                getattr(
                    runtime,
                    "ppo_action_mode",
                    "continuous_absolute",
                )
            ),
            ppo_w_center=float(
                getattr(runtime, "ppo_w_center", 1.5)
            ),
            ppo_w_scale=float(
                getattr(runtime, "ppo_w_scale", 0.5)
            ),
            ppo_initial_observation_weight=initial_weight,
            ppo_force_default_first_action=bool(
                getattr(
                    runtime,
                    "ppo_force_default_first_action",
                    False,
                )
            ),
            ppo_default_first_weight=float(
                getattr(
                    runtime,
                    "ppo_default_first_weight",
                    initial_weight,
                )
            ),
        )


def build_frozen_ppo_runner(
    config: FrozenPpoConfig,
) -> SetupAwareSolvePolicyRunner:
    """Build the exact frozen recurrent PPO policy used by Exp44."""

    return SetupAwareSolvePolicyRunner(
        SetupAwareRLConfig(
            tune_dim=7,
            tune7_variant="categorical",
            algo="ppo",
            model_type="lstm",
            model_path=config.ppo_model,
            vec_path=(
                config.output_dir / ".unused_vecnormalize.pkl"
            ),
            fixed_grid=(int(config.grid_n),) * 3,
            difconv_c_range=(
                float(config.c_min),
                float(config.c_max),
            ),
            w_only=True,
            w_center=float(config.ppo_w_center),
            w_scale=float(config.ppo_w_scale),
            w_global_min=1.0,
            w_global_max=2.0,
            sweeps_min=1,
            sweeps_max=1,
            w_init=None,
            sweeps_init=None,
            solve_max_cycles=int(config.max_cycles),
            solve_tol=float(config.tol),
            default_setup_params={},
            obs_mode="cycle_action_setup",
            action_mode=str(config.ppo_action_mode).strip().lower(),
            initial_observation_weight=float(
                config.ppo_initial_observation_weight
            ),
            force_default_first_action=bool(
                config.ppo_force_default_first_action
            ),
            default_first_weight=float(
                config.ppo_default_first_weight
            ),
        )
    )


__all__ = ["FrozenPpoConfig", "build_frozen_ppo_runner"]
