from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)

from .config import StructuredModelBasedSpec
from .controller import StructuredModelBasedController


def build_structured_model_based_controller(
    request: OnlineControllerFactoryRequest[StructuredModelBasedSpec],
) -> ControllerBundle:
    """Build the structured physical-model controller."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="structured_model_based",
        family="model_based",
        controller_type=StructuredModelBasedController,
        epsilon_enabled=True,
        protocol_details={
            "target": [
                "native_cycle_cost",
                "signed_log_residual_progress",
            ],
            "estimator": "shared recursive least squares",
            "planning": "receding time per log-residual reduction",
            "recovery_cost_in_physical_model": False,
            "ridge": float(spec.ridge),
            "minimum_samples": int(spec.minimum_samples),
            "scale_window": int(spec.scale_window),
        },
    )


__all__ = ["build_structured_model_based_controller"]
