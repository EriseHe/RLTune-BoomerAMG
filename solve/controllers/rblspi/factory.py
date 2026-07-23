from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)

from .config import RecursiveBlstdqSpec
from .controller import RecursiveBlstdqController


def build_rblspi_controller(
    request: OnlineControllerFactoryRequest[RecursiveBlstdqSpec],
) -> ControllerBundle:
    """Build recursive BLSTDQ with RBLSPI posterior sampling."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="rblspi",
        family="rblspi",
        controller_type=RecursiveBlstdqController,
        epsilon_enabled=False,
        protocol_details={
            "target": "off-policy BLSTDQ empirical Bellman equation",
            "posterior": "S^-1=alpha*I+beta*A.T*C^-1*A; m=beta*S*A.T*C^-1*b",
            "exploration": "one sampled Q-function per solve episode",
            "prior_precision": float(spec.prior_precision),
            "noise_precision": float(spec.noise_precision),
            "gram_ridge": float(spec.gram_ridge),
            "history_storage": "none; recursive A/C/b statistics only",
        },
    )


__all__ = ["build_rblspi_controller"]
