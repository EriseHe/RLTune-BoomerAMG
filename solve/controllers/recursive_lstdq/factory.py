from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)

from .config import RecursiveLstdqLcbSpec, RecursiveLstdqV2LcbSpec
from .v1 import RecursiveLstdqLcbController
from .v2 import RecursiveLstdqV2LcbController


def build_recursive_lstdq_v1_controller(
    request: OnlineControllerFactoryRequest[RecursiveLstdqLcbSpec],
) -> ControllerBundle:
    """Build recursive LSTDQ v1 with its family-owned protocol metadata."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="recursive_lstdq_v1",
        family="recursive_lstdq",
        controller_type=RecursiveLstdqLcbController,
        epsilon_enabled=True,
        protocol_details={
            "version": "v1",
            "target": "on-policy LSTDQ(lambda) Bellman equation",
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "uncertainty": "unclipped post-fit sandwich covariance",
            "residual_floor_sec": float(spec.residual_floor_sec),
            "lcb_lower_bound_sec": spec.lcb_lower_bound_sec,
        },
    )


def build_recursive_lstdq_v2_controller(
    request: OnlineControllerFactoryRequest[RecursiveLstdqV2LcbSpec],
) -> ControllerBundle:
    """Build coverage-calibrated recursive LSTDQ v2."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="recursive_lstdq_v2",
        family="recursive_lstdq",
        controller_type=RecursiveLstdqV2LcbController,
        epsilon_enabled=True,
        protocol_details={
            "version": "v2",
            "target": "same recursive LSTDQ mean as v1",
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "uncertainty": "rolling-MAD-scaled feature coverage",
            "coverage_ridge": float(spec.coverage_ridge),
            "residual_window": int(spec.residual_scale_window),
            "minimum_scale_samples": int(spec.residual_scale_min_samples),
            "residual_floor_sec": float(spec.residual_floor_sec),
            "lcb_lower_bound_sec": spec.lcb_lower_bound_sec,
        },
    )


__all__ = [
    "build_recursive_lstdq_v1_controller",
    "build_recursive_lstdq_v2_controller",
]
