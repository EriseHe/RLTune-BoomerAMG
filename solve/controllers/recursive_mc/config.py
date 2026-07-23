from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from solve.controllers.common.config_decode import (
    config_value,
    reject_unknown_config_keys,
)


_JSON_KEYS = frozenset(
    {
        "ridge",
        "beta",
        "residual_floor_sec",
        "episode_half_life",
    }
)


@dataclass(frozen=True)
class RecursiveMonteCarloLcbSpec:
    """Configuration owned by the recursive Monte Carlo controller family."""

    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    episode_half_life: float = 500.0

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        name: str = "solve.recursive_mc",
    ) -> "RecursiveMonteCarloLcbSpec":
        """Decode the recursive-MC section of a joint experiment config."""

        reject_unknown_config_keys(raw, _JSON_KEYS, name=name)
        return cls(
            ridge=float(config_value(raw, "ridge", cls.ridge)),
            uncertainty_beta=float(
                config_value(raw, "beta", cls.uncertainty_beta)
            ),
            residual_floor_sec=float(
                config_value(
                    raw,
                    "residual_floor_sec",
                    cls.residual_floor_sec,
                )
            ),
            q_max_sec=float(cls.q_max_sec),
            episode_half_life=float(
                config_value(
                    raw,
                    "episode_half_life",
                    cls.episode_half_life,
                )
            ),
        )


__all__ = ["RecursiveMonteCarloLcbSpec"]
