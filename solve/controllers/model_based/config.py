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
        "min_samples",
        "scale_window",
    }
)


@dataclass(frozen=True)
class StructuredModelBasedSpec:
    """Shared RLS controls for physical cycle-cost/progress prediction."""

    ridge: float = 1.0
    minimum_samples: int = 32
    scale_window: int = 2048
    cost_floor_fraction: float = 0.1
    progress_floor_fraction: float = 0.1
    numerical_floor: float = 1.0e-12

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        name: str = "solve.structured_model",
    ) -> "StructuredModelBasedSpec":
        """Decode the structured-model section of a joint config."""

        reject_unknown_config_keys(raw, _JSON_KEYS, name=name)
        return cls(
            ridge=float(config_value(raw, "ridge", cls.ridge)),
            minimum_samples=int(
                config_value(raw, "min_samples", cls.minimum_samples)
            ),
            scale_window=int(
                config_value(raw, "scale_window", cls.scale_window)
            ),
            cost_floor_fraction=float(cls.cost_floor_fraction),
            progress_floor_fraction=float(cls.progress_floor_fraction),
            numerical_floor=float(cls.numerical_floor),
        )


__all__ = ["StructuredModelBasedSpec"]
