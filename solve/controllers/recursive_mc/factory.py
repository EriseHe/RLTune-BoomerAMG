from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)

from .config import RecursiveMonteCarloLcbSpec
from .controller import RecursiveMonteCarloLcbController


def build_recursive_mc_controller(
    request: OnlineControllerFactoryRequest[RecursiveMonteCarloLcbSpec],
) -> ControllerBundle:
    """Build the recursive-MC family without exposing its constructor."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="recursive_mc",
        family="recursive_mc",
        controller_type=RecursiveMonteCarloLcbController,
        epsilon_enabled=True,
        protocol_details={
            "target": "undiscounted episodic cost-to-go",
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "residual_floor_sec": float(spec.residual_floor_sec),
            "episode_half_life": float(spec.episode_half_life),
            "uncertainty": "post-fit episode-cluster sandwich covariance",
        },
    )


__all__ = ["build_recursive_mc_controller"]
