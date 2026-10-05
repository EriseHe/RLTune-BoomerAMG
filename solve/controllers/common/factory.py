from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

from .config import SharedActionSpec, SolveStateSpec
from .types import ControllerBundle


AlgorithmSpecT = TypeVar("AlgorithmSpecT")
ControllerT = TypeVar("ControllerT")


@dataclass(frozen=True)
class OnlineControllerFactoryRequest(Generic[AlgorithmSpecT]):
    """Fully typed input shared by family-owned controller factories."""

    state: SolveStateSpec
    actions: SharedActionSpec
    algorithm: AlgorithmSpecT
    trace_lambda: float
    setup_obs_encoder: Any
    seed: int


def build_shared_action_controller_bundle(
    *,
    request: OnlineControllerFactoryRequest[AlgorithmSpecT],
    kind: str,
    family: str,
    controller_type: Callable[..., ControllerT],
    epsilon_enabled: bool,
    protocol_details: Mapping[str, Any],
) -> ControllerBundle:
    """Apply shared encoding/action wiring around a family-owned constructor."""

    encoder = request.state.build_encoder(
        setup_obs_encoder=request.setup_obs_encoder,
        weights=request.actions.weights,
    )
    config = request.actions.to_td_config(
        trace_lambda=float(request.trace_lambda),
        epsilon_enabled=bool(epsilon_enabled),
    )
    controller = controller_type(
        feature_dim=encoder.feature_dim,
        config=config,
        spec=request.algorithm,
        seed=int(request.seed),
    )
    metadata = _shared_protocol_metadata(
        request=request,
        encoder=encoder,
        kind=kind,
        family=family,
        epsilon_enabled=epsilon_enabled,
    )
    metadata.update(dict(protocol_details))
    return ControllerBundle(
        controller=controller,
        encoder=encoder,
        _protocol_metadata=metadata,
    )


def _shared_protocol_metadata(
    *,
    request: OnlineControllerFactoryRequest[Any],
    encoder: Any,
    kind: str,
    family: str,
    epsilon_enabled: bool,
) -> dict[str, Any]:
    algorithm = asdict(request.algorithm)
    epsilon = request.actions.epsilon
    context_sources = {
        "canonical": "shared per-instance canonical setup/solve PDE context",
        "canonical_no_c_mean": (
            "shared three log diffusion and three signed-log advection "
            "coefficients; mean omitted and intercept retained separately"
        ),
        "legacy": "legacy mkw diffusion-only projection",
        "diffusion3d": (
            "shared three log diffusion coefficients; intercept retained separately"
        ),
        "diffusion4d": (
            "shared three log diffusion coefficients and their mean; "
            "intercept retained separately"
        ),
    }
    return {
        "kind": str(kind),
        "family": str(family),
        "state": str(request.state.mode),
        "state_encoder": {
            "encoding_version": encoder.encoding_version,
            "weight_bounds": encoder.weight_bounds,
            "setup_observed_keys": list(request.setup_obs_encoder.observed_keys),
            "setup_parameter_spec": (
                asdict(request.setup_obs_encoder.parameter_spec)
                if hasattr(request.setup_obs_encoder, "parameter_spec")
                else None
            ),
            "strict_setup_categories": bool(
                getattr(request.setup_obs_encoder, "strict_categories", False)
            ),
            "tol": float(request.state.tol),
            "max_cycles": int(request.state.max_cycles),
            "c_max": float(request.state.c_max),
            "time_scale_sec": float(request.state.time_scale_sec),
            "feature_dim": int(encoder.feature_dim),
            "problem_context_mode": str(request.state.problem_context_mode),
            "problem_context_fields": list(encoder.problem_context_fields),
            "problem_context_source": context_sources[
                str(request.state.problem_context_mode)
            ],
        },
        "action_profile": "explicit",
        "actions": [float(value) for value in request.actions.weights],
        "action_basis": {
            "mode": "compact_rbf",
            "centers": [float(value) for value in request.actions.rbf_centers],
            "sigma": float(request.actions.rbf_sigma),
        },
        "epsilon": (
            {
                "start": float(epsilon.start),
                "final": float(epsilon.final),
                "decay_steps": float(epsilon.decay_steps),
                "kind": "uniform epsilon-greedy floor",
            }
            if epsilon_enabled
            else {"kind": "none"}
        ),
        "trace_lambda": float(request.trace_lambda),
        "prepared_solver_default_forced_once": bool(
            request.actions.force_default_first_action
        ),
        "frozen": False,
        "algorithm": algorithm,
    }


__all__ = [
    "OnlineControllerFactoryRequest",
    "build_shared_action_controller_bundle",
]
