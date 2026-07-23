from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from solve.controllers.common.config_decode import (
    config_value,
    reject_unknown_config_keys,
)


_SHARED_JSON_KEYS = frozenset(
    {
        "ridge",
        "beta",
        "residual_floor_sec",
        "refit_interval_episodes",
    }
)
_HIERARCHICAL_JSON_KEYS = frozenset(
    {
        "beta",
        "refit_sweeps",
        "shrinkage_samples",
    }
)
_JOINT_REFIT_INTERVAL_EPISODES = 100


@dataclass(frozen=True)
class StagewiseLsviLcbSpec:
    """Configuration for the finite-horizon stagewise LSVI controller."""

    horizon: int = 50
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    refit_interval_episodes: int = 1


@dataclass(frozen=True)
class HierarchicalLsviLcbSpec:
    """Stagewise LSVI configuration with a shared cross-stage ridge prior."""

    horizon: int = 50
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    refit_interval_episodes: int = 100
    refit_sweeps: int = 3
    residual_shrinkage_samples: float = 32.0


@dataclass(frozen=True)
class LsviFamilySpecs:
    """Decoded stagewise and hierarchical joint-experiment specs."""

    stagewise: StagewiseLsviLcbSpec
    hierarchical: HierarchicalLsviLcbSpec

    @classmethod
    def from_mappings(
        cls,
        shared_raw: Mapping[str, Any],
        hierarchical_raw: Mapping[str, Any],
        *,
        horizon: int,
        shared_name: str = "solve.lsvi",
        hierarchical_name: str = "solve.recalibrated_lsvi",
    ) -> "LsviFamilySpecs":
        """Decode base LSVI settings and hierarchical-only overrides."""

        reject_unknown_config_keys(
            shared_raw,
            _SHARED_JSON_KEYS,
            name=shared_name,
        )
        reject_unknown_config_keys(
            hierarchical_raw,
            _HIERARCHICAL_JSON_KEYS,
            name=hierarchical_name,
        )

        ridge = float(
            config_value(shared_raw, "ridge", StagewiseLsviLcbSpec.ridge)
        )
        residual_floor_sec = float(
            config_value(
                shared_raw,
                "residual_floor_sec",
                StagewiseLsviLcbSpec.residual_floor_sec,
            )
        )
        refit_interval_episodes = int(
            config_value(
                shared_raw,
                "refit_interval_episodes",
                _JOINT_REFIT_INTERVAL_EPISODES,
            )
        )
        return cls(
            stagewise=StagewiseLsviLcbSpec(
                horizon=int(horizon),
                ridge=ridge,
                uncertainty_beta=float(
                    config_value(
                        shared_raw,
                        "beta",
                        StagewiseLsviLcbSpec.uncertainty_beta,
                    )
                ),
                residual_floor_sec=residual_floor_sec,
                q_max_sec=float(StagewiseLsviLcbSpec.q_max_sec),
                refit_interval_episodes=refit_interval_episodes,
            ),
            hierarchical=HierarchicalLsviLcbSpec(
                horizon=int(horizon),
                ridge=ridge,
                uncertainty_beta=float(
                    config_value(
                        hierarchical_raw,
                        "beta",
                        HierarchicalLsviLcbSpec.uncertainty_beta,
                    )
                ),
                residual_floor_sec=residual_floor_sec,
                refit_interval_episodes=refit_interval_episodes,
                refit_sweeps=int(
                    config_value(
                        hierarchical_raw,
                        "refit_sweeps",
                        HierarchicalLsviLcbSpec.refit_sweeps,
                    )
                ),
                residual_shrinkage_samples=float(
                    config_value(
                        hierarchical_raw,
                        "shrinkage_samples",
                        HierarchicalLsviLcbSpec.residual_shrinkage_samples,
                    )
                ),
            ),
        )


__all__ = [
    "HierarchicalLsviLcbSpec",
    "LsviFamilySpecs",
    "StagewiseLsviLcbSpec",
]
