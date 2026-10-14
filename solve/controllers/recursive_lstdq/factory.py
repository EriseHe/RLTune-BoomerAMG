from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)
from .config import RecursiveLstdqSpec
from .controller import RecursiveLstdqController


def build_recursive_lstdq_controller(
    request: OnlineControllerFactoryRequest[RecursiveLstdqSpec],
) -> ControllerBundle:
    """Build recursive LSTDQ with episode-cluster sandwich uncertainty."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="recursive_lstdq",
        family="recursive_lstdq",
        controller_type=RecursiveLstdqController,
        epsilon_enabled=True,
        protocol_details={
            "target": "completion cost with executed-next-action LSTDQ(lambda)",
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "uncertainty": ("episode-cluster post-fit sandwich covariance"),
            "uncertainty_semantics": "parameter uncertainty",
            "cluster_unit": "complete committed AMG solve episode",
            "residual_floor_sec": float(spec.residual_floor_sec),
            "lcb_lower_bound_sec": spec.lcb_lower_bound_sec,
        },
    )


__all__ = ["build_recursive_lstdq_controller"]
