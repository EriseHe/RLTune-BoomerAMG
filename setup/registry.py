from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, cast

from .learners.common import SetupLearnerFactoryRequest, SharedSetupLearnerSpec
from .learners.linucb.config import LinUCBV4Spec
from .learners.linucb.factory import LINUCB_V4_LEARNER_TYPE, build_linucb_v4_learner

OnlineSetupKind = Literal["linucb"]
SetupAlgorithmSpec = LinUCBV4Spec


@dataclass(frozen=True)
class SetupKindRegistration:
    kind: str
    family: str
    backend: str
    learner_type: type[Any] | None = None
    spec_type: type[Any] | None = None
    factory: Callable[[SetupLearnerFactoryRequest[Any]], Any] | None = None

    @property
    def online(self) -> bool:
        return self.backend == "online_learner"


SETUP_KIND_REGISTRY: dict[str, SetupKindRegistration] = {
    "default": SetupKindRegistration(
        kind="default",
        family="default",
        backend="fixed_parameters",
    ),
    "linucb": SetupKindRegistration(
        kind="linucb",
        family="linucb",
        backend="online_learner",
        learner_type=LINUCB_V4_LEARNER_TYPE,
        spec_type=LinUCBV4Spec,
        factory=build_linucb_v4_learner,
    ),
}
ONLINE_SETUP_KINDS = tuple(
    kind for kind, registration in SETUP_KIND_REGISTRY.items() if registration.online
)
COMPOSABLE_SETUP_KINDS = ("default", *ONLINE_SETUP_KINDS)
_SETUP_KIND_ALIASES = {"linucbv4": "linucb", "sharedlinucbv4": "linucb"}


@dataclass(frozen=True)
class SetupLearnerBuildSpec:
    kind: OnlineSetupKind
    shared: SharedSetupLearnerSpec
    algorithm: SetupAlgorithmSpec


def normalize_setup_kind(kind: str) -> str:
    token = str(kind).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _SETUP_KIND_ALIASES.get(token, token)


def make_setup_learner_spec(
    *,
    kind: OnlineSetupKind | str,
    shared: SharedSetupLearnerSpec,
    algorithm_parameters: Mapping[str, Any] | None = None,
) -> SetupLearnerBuildSpec:
    """Create a typed setup learner request from experiment-selected values."""

    canonical_kind = normalize_setup_kind(kind)
    registration = _online_registration(canonical_kind)
    assert registration.spec_type is not None
    algorithm = registration.spec_type(**dict(algorithm_parameters or {}))
    return SetupLearnerBuildSpec(
        kind=cast(OnlineSetupKind, canonical_kind),
        shared=shared,
        algorithm=algorithm,
    )


def build_online_setup_learner(spec: SetupLearnerBuildSpec) -> Any:
    """Build an active setup learner without altering its persisted state."""

    registration = _online_registration(spec.kind)
    assert registration.spec_type is not None
    assert registration.factory is not None
    if type(spec.algorithm) is not registration.spec_type:
        raise TypeError(
            f"{spec.kind} requires {registration.spec_type.__name__}, "
            f"got {type(spec.algorithm).__name__}"
        )
    return registration.factory(
        SetupLearnerFactoryRequest(
            shared=spec.shared,
            algorithm=spec.algorithm,
        )
    )


def setup_kind_registration(kind: str) -> SetupKindRegistration:
    canonical_kind = normalize_setup_kind(kind)
    try:
        return SETUP_KIND_REGISTRY[canonical_kind]
    except KeyError as error:
        raise ValueError(f"Unsupported setup learner kind: {kind!r}") from error


def _online_registration(kind: str) -> SetupKindRegistration:
    registration = setup_kind_registration(kind)
    if not registration.online:
        raise ValueError(f"Setup kind {kind!r} is not an online setup learner")
    return registration


__all__ = [
    "COMPOSABLE_SETUP_KINDS",
    "ONLINE_SETUP_KINDS",
    "OnlineSetupKind",
    "SETUP_KIND_REGISTRY",
    "SetupAlgorithmSpec",
    "SetupKindRegistration",
    "SetupLearnerBuildSpec",
    "build_online_setup_learner",
    "make_setup_learner_spec",
    "normalize_setup_kind",
    "setup_kind_registration",
]
