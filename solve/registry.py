from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from solve.controllers.common import (
    ControllerBundle,
    OnlineControllerFactoryRequest,
    SharedActionSpec,
    SolveStateSpec,
)
from solve.controllers.lsvi import (
    HierarchicalLsviLcbController,
    HierarchicalLsviLcbSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
    build_hierarchical_lsvi_controller,
    build_stagewise_lsvi_controller,
)
from solve.controllers.model_based import (
    StructuredModelBasedController,
    StructuredModelBasedSpec,
    build_structured_model_based_controller,
)
from solve.controllers.rblspi import (
    RecursiveBlstdqController,
    RecursiveBlstdqSpec,
    build_rblspi_controller,
)
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbController,
    RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbController,
    RecursiveLstdqV3LcbSpec,
    build_recursive_lstdq_v1_controller,
    build_recursive_lstdq_v2_controller,
    build_recursive_lstdq_v3_controller,
)
from solve.controllers.recursive_mc import (
    RecursiveMonteCarloLcbController,
    RecursiveMonteCarloLcbSpec,
    build_recursive_mc_controller,
)


OnlineSolveKind = Literal[
    "recursive_mc",
    "recursive_lstdq_v1",
    "recursive_lstdq_v2",
    "recursive_lstdq_v3",
    "rblspi",
    "stagewise_lsvi",
    "structured_model_based",
    "recalibrated_lsvi",
]

AlgorithmSpec = (
    RecursiveMonteCarloLcbSpec
    | RecursiveLstdqLcbSpec
    | RecursiveLstdqV2LcbSpec
    | RecursiveLstdqV3LcbSpec
    | RecursiveBlstdqSpec
    | StagewiseLsviLcbSpec
    | StructuredModelBasedSpec
    | HierarchicalLsviLcbSpec
)

__all__ = [
    "AlgorithmSpec",
    "COMPOSABLE_SOLVE_KINDS",
    "ONLINE_SOLVE_KINDS",
    "SOLVE_KIND_REGISTRY",
    "OnlineControllerBuildSpec",
    "OnlineSolveKind",
    "SolveKindRegistration",
    "build_online_solve_controller",
    "make_online_controller_spec",
    "solve_kind_registration",
]


@dataclass(frozen=True)
class SolveKindRegistration:
    kind: str
    family: str
    backend: str
    controller_type: type[Any] | None = None
    spec_type: type[Any] | None = None
    factory: (
        Callable[[OnlineControllerFactoryRequest[Any]], ControllerBundle] | None
    ) = None
    fixed_trace_lambda: float | None = None
    trace_lambda_from_request: bool = False

    @property
    def online(self) -> bool:
        return self.backend == "online_controller"


SOLVE_KIND_REGISTRY: dict[str, SolveKindRegistration] = {
    "default": SolveKindRegistration(
        kind="default",
        family="default",
        backend="native_default",
    ),
    "fixed": SolveKindRegistration(
        kind="fixed",
        family="fixed_weight",
        backend="fixed_weight",
    ),
    "ppo": SolveKindRegistration(
        kind="ppo",
        family="ppo",
        backend="frozen_policy",
    ),
    "recursive_mc": SolveKindRegistration(
        kind="recursive_mc",
        family="recursive_mc",
        backend="online_controller",
        controller_type=RecursiveMonteCarloLcbController,
        spec_type=RecursiveMonteCarloLcbSpec,
        factory=build_recursive_mc_controller,
        fixed_trace_lambda=1.0,
    ),
    "recursive_lstdq_v1": SolveKindRegistration(
        kind="recursive_lstdq_v1",
        family="recursive_lstdq",
        backend="online_controller",
        controller_type=RecursiveLstdqLcbController,
        spec_type=RecursiveLstdqLcbSpec,
        factory=build_recursive_lstdq_v1_controller,
        trace_lambda_from_request=True,
    ),
    "recursive_lstdq_v2": SolveKindRegistration(
        kind="recursive_lstdq_v2",
        family="recursive_lstdq",
        backend="online_controller",
        controller_type=RecursiveLstdqV2LcbController,
        spec_type=RecursiveLstdqV2LcbSpec,
        factory=build_recursive_lstdq_v2_controller,
        trace_lambda_from_request=True,
    ),
    "recursive_lstdq_v3": SolveKindRegistration(
        kind="recursive_lstdq_v3",
        family="recursive_lstdq",
        backend="online_controller",
        controller_type=RecursiveLstdqV3LcbController,
        spec_type=RecursiveLstdqV3LcbSpec,
        factory=build_recursive_lstdq_v3_controller,
        trace_lambda_from_request=True,
    ),
    "rblspi": SolveKindRegistration(
        kind="rblspi",
        family="rblspi",
        backend="online_controller",
        controller_type=RecursiveBlstdqController,
        spec_type=RecursiveBlstdqSpec,
        factory=build_rblspi_controller,
        fixed_trace_lambda=0.0,
    ),
    "stagewise_lsvi": SolveKindRegistration(
        kind="stagewise_lsvi",
        family="lsvi",
        backend="online_controller",
        controller_type=StagewiseLsviLcbController,
        spec_type=StagewiseLsviLcbSpec,
        factory=build_stagewise_lsvi_controller,
        fixed_trace_lambda=0.8,
    ),
    "structured_model_based": SolveKindRegistration(
        kind="structured_model_based",
        family="model_based",
        backend="online_controller",
        controller_type=StructuredModelBasedController,
        spec_type=StructuredModelBasedSpec,
        factory=build_structured_model_based_controller,
        fixed_trace_lambda=0.0,
    ),
    "recalibrated_lsvi": SolveKindRegistration(
        kind="recalibrated_lsvi",
        family="lsvi",
        backend="online_controller",
        controller_type=HierarchicalLsviLcbController,
        spec_type=HierarchicalLsviLcbSpec,
        factory=build_hierarchical_lsvi_controller,
        fixed_trace_lambda=0.8,
    ),
}

COMPOSABLE_SOLVE_KINDS = tuple(SOLVE_KIND_REGISTRY)
ONLINE_SOLVE_KINDS = tuple(
    kind for kind, registration in SOLVE_KIND_REGISTRY.items() if registration.online
)


@dataclass(frozen=True)
class OnlineControllerBuildSpec:
    kind: OnlineSolveKind
    state: SolveStateSpec
    actions: SharedActionSpec
    algorithm: AlgorithmSpec
    trace_lambda: float | None = None


def make_online_controller_spec(
    *,
    kind: OnlineSolveKind,
    state: SolveStateSpec,
    actions: SharedActionSpec,
    algorithm_parameters: Mapping[str, Any],
    trace_lambda: float | None = None,
) -> OnlineControllerBuildSpec:
    """Create a typed build spec from experiment-selected numeric values."""

    registration = _online_registration(kind)
    assert registration.spec_type is not None
    algorithm = registration.spec_type(**dict(algorithm_parameters))
    _resolve_trace_lambda(registration, trace_lambda)
    return OnlineControllerBuildSpec(
        kind=kind,
        state=state,
        actions=actions,
        algorithm=algorithm,
        trace_lambda=trace_lambda,
    )


def build_online_solve_controller(
    spec: OnlineControllerBuildSpec,
    *,
    setup_obs_encoder: Any,
    seed: int,
) -> ControllerBundle:
    """Build one setup-aware online controller from a single typed request."""

    registration = _online_registration(spec.kind)
    assert registration.spec_type is not None
    assert registration.factory is not None
    if type(spec.algorithm) is not registration.spec_type:
        raise TypeError(
            f"{spec.kind} requires {registration.spec_type.__name__}, "
            f"got {type(spec.algorithm).__name__}"
        )
    trace_lambda = _resolve_trace_lambda(registration, spec.trace_lambda)
    return registration.factory(
        OnlineControllerFactoryRequest(
            state=spec.state,
            actions=spec.actions,
            algorithm=spec.algorithm,
            trace_lambda=trace_lambda,
            setup_obs_encoder=setup_obs_encoder,
            seed=int(seed),
        )
    )


def solve_kind_registration(kind: str) -> SolveKindRegistration:
    try:
        return SOLVE_KIND_REGISTRY[str(kind)]
    except KeyError as exc:
        raise ValueError(f"Unknown solve-controller kind: {kind}") from exc


def _online_registration(kind: str) -> SolveKindRegistration:
    registration = solve_kind_registration(kind)
    if not registration.online:
        raise ValueError(
            f"{kind} is a {registration.backend} backend, not an online controller"
        )
    return registration


def _resolve_trace_lambda(
    registration: SolveKindRegistration,
    requested: float | None,
) -> float:
    if registration.trace_lambda_from_request:
        if requested is None:
            raise ValueError(f"{registration.kind} requires trace_lambda")
        value = float(requested)
        if not 0.0 <= value <= 1.0:
            raise ValueError("trace_lambda must lie in [0, 1]")
        return value
    if requested is not None:
        raise ValueError(f"{registration.kind} does not accept trace_lambda")
    assert registration.fixed_trace_lambda is not None
    return float(registration.fixed_trace_lambda)
