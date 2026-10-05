from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)
from .config import RecursiveLstdqV3LcbSpec
from .v3 import RecursiveLstdqV3LcbController


def build_recursive_lstdq_v3_controller(
    request: OnlineControllerFactoryRequest[RecursiveLstdqV3LcbSpec],
) -> ControllerBundle:
    """Build episode-cluster sandwich recursive LSTDQ v3."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="recursive_lstdq_v3",
        family="recursive_lstdq",
        controller_type=RecursiveLstdqV3LcbController,
        epsilon_enabled=True,
        protocol_details={
            "version": "v3",
            "target": "same recursive LSTDQ mean as v1/v2",
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "uncertainty": ("episode-cluster post-fit sandwich covariance"),
            "uncertainty_semantics": "parameter uncertainty",
            "cluster_unit": "complete committed AMG solve episode",
            "residual_floor_sec": float(spec.residual_floor_sec),
            "lcb_lower_bound_sec": spec.lcb_lower_bound_sec,
        },
    )


__all__ = ["build_recursive_lstdq_v3_controller"]
