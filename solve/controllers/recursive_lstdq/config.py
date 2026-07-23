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
        "trace_lambda",
        "residual_floor_sec",
        "lcb_lower_bound_sec",
    }
)
_V2_JSON_KEYS = frozenset(
    {
        "beta",
        "coverage_ridge",
        "residual_window",
        "min_samples",
    }
)
_V3_JSON_KEYS = frozenset({"beta"})


@dataclass(frozen=True)
class RecursiveLstdqLcbSpec:
    """Configuration for recursive LSTDQ with sandwich uncertainty."""

    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    inverse_denominator_floor: float = 1.0e-10
    lcb_lower_bound_sec: float | None = 0.0


@dataclass(frozen=True)
class RecursiveLstdqV2LcbSpec(RecursiveLstdqLcbSpec):
    """Coverage-calibrated confidence controls for recursive LSTDQ."""

    coverage_ridge: float = 1.0
    residual_scale_window: int = 2048
    residual_scale_min_samples: int = 32


@dataclass(frozen=True)
class RecursiveLstdqV3LcbSpec(RecursiveLstdqLcbSpec):
    """Episode-cluster sandwich confidence controls for recursive LSTDQ."""


@dataclass(frozen=True)
class RecursiveLstdqFamilySpecs:
    """Decoded versioned specs and their shared trace configuration."""

    v1: RecursiveLstdqLcbSpec
    v2: RecursiveLstdqV2LcbSpec
    v3: RecursiveLstdqV3LcbSpec
    trace_lambda: float

    @classmethod
    def from_mappings(
        cls,
        shared_raw: Mapping[str, Any],
        v2_raw: Mapping[str, Any],
        v3_raw: Mapping[str, Any] | None = None,
        *,
        shared_name: str = "solve.lstdq",
        v2_name: str = "solve.lstdq_v2",
        v3_name: str = "solve.lstdq_v3",
    ) -> "RecursiveLstdqFamilySpecs":
        """Decode shared LSTDQ values, then apply version-only overrides."""

        resolved_v3_raw = {} if v3_raw is None else v3_raw

        reject_unknown_config_keys(
            shared_raw,
            _SHARED_JSON_KEYS,
            name=shared_name,
        )
        reject_unknown_config_keys(v2_raw, _V2_JSON_KEYS, name=v2_name)
        reject_unknown_config_keys(
            resolved_v3_raw,
            _V3_JSON_KEYS,
            name=v3_name,
        )

        shared_parameters = {
            "ridge": float(
                config_value(
                    shared_raw,
                    "ridge",
                    RecursiveLstdqLcbSpec.ridge,
                )
            ),
            "residual_floor_sec": float(
                config_value(
                    shared_raw,
                    "residual_floor_sec",
                    RecursiveLstdqLcbSpec.residual_floor_sec,
                )
            ),
            "q_max_sec": float(RecursiveLstdqLcbSpec.q_max_sec),
            "inverse_denominator_floor": float(
                RecursiveLstdqLcbSpec.inverse_denominator_floor
            ),
            "lcb_lower_bound_sec": float(
                config_value(
                    shared_raw,
                    "lcb_lower_bound_sec",
                    RecursiveLstdqLcbSpec.lcb_lower_bound_sec,
                )
            ),
        }
        return cls(
            v1=RecursiveLstdqLcbSpec(
                **shared_parameters,
                uncertainty_beta=float(
                    config_value(
                        shared_raw,
                        "beta",
                        RecursiveLstdqLcbSpec.uncertainty_beta,
                    )
                ),
            ),
            v2=RecursiveLstdqV2LcbSpec(
                **shared_parameters,
                uncertainty_beta=float(
                    config_value(
                        v2_raw,
                        "beta",
                        RecursiveLstdqV2LcbSpec.uncertainty_beta,
                    )
                ),
                coverage_ridge=float(
                    config_value(
                        v2_raw,
                        "coverage_ridge",
                        RecursiveLstdqV2LcbSpec.coverage_ridge,
                    )
                ),
                residual_scale_window=int(
                    config_value(
                        v2_raw,
                        "residual_window",
                        RecursiveLstdqV2LcbSpec.residual_scale_window,
                    )
                ),
                residual_scale_min_samples=int(
                    config_value(
                        v2_raw,
                        "min_samples",
                        RecursiveLstdqV2LcbSpec.residual_scale_min_samples,
                    )
                ),
            ),
            v3=RecursiveLstdqV3LcbSpec(
                **shared_parameters,
                uncertainty_beta=float(
                    config_value(
                        resolved_v3_raw,
                        "beta",
                        RecursiveLstdqV3LcbSpec.uncertainty_beta,
                    )
                ),
            ),
            trace_lambda=float(
                config_value(shared_raw, "trace_lambda", 0.8)
            ),
        )


__all__ = [
    "RecursiveLstdqFamilySpecs",
    "RecursiveLstdqLcbSpec",
    "RecursiveLstdqV2LcbSpec",
    "RecursiveLstdqV3LcbSpec",
]
