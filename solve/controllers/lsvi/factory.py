from __future__ import annotations

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)

from .config import HierarchicalLsviLcbSpec, StagewiseLsviLcbSpec
from .hierarchical import HierarchicalLsviLcbController
from .stagewise import StagewiseLsviLcbController


def build_stagewise_lsvi_controller(
    request: OnlineControllerFactoryRequest[StagewiseLsviLcbSpec],
) -> ControllerBundle:
    """Build finite-horizon stagewise LSVI-LCB."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="stagewise_lsvi",
        family="lsvi",
        controller_type=StagewiseLsviLcbController,
        epsilon_enabled=True,
        protocol_details={
            "target": "stagewise optimistic Bellman backup",
            "horizon": int(spec.horizon),
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "residual_floor_sec": float(spec.residual_floor_sec),
            "refit_interval_episodes": int(spec.refit_interval_episodes),
        },
    )


def build_hierarchical_lsvi_controller(
    request: OnlineControllerFactoryRequest[HierarchicalLsviLcbSpec],
) -> ControllerBundle:
    """Build hierarchical/recalibrated LSVI-LCB."""

    spec = request.algorithm
    return build_shared_action_controller_bundle(
        request=request,
        kind="recalibrated_lsvi",
        family="lsvi",
        controller_type=HierarchicalLsviLcbController,
        epsilon_enabled=True,
        protocol_details={
            "target": "stagewise optimistic Bellman backup",
            "horizon": int(spec.horizon),
            "ridge": float(spec.ridge),
            "uncertainty_beta": float(spec.uncertainty_beta),
            "residual_floor_sec": float(spec.residual_floor_sec),
            "batch_size_episodes": int(spec.refit_interval_episodes),
            "backward_shared_sweeps": int(spec.refit_sweeps),
            "residual_shrinkage_samples": float(spec.residual_shrinkage_samples),
            "value_cap": "maximum observed realized return",
        },
    )


__all__ = [
    "build_hierarchical_lsvi_controller",
    "build_stagewise_lsvi_controller",
]
