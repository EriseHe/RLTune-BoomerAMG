from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

from .learners.common.config import (
    LinTSV2Spec,
    LinUCBV4Spec,
    SharedSetupLearnerSpec,
)
from .learners.linucb import SharedLinUCB_AMG_v4
from .learners.thompson import SharedLinTS_AMG_v2


OnlineSetupKind = Literal["linucb", "lints"]
SetupAlgorithmSpec = LinUCBV4Spec | LinTSV2Spec


@dataclass(frozen=True)
class SetupKindRegistration:
    kind: str
    family: str
    backend: str
    learner_type: type[Any] | None = None
    spec_type: type[Any] | None = None

    @property
    def online(self) -> bool:
        return self.backend == "online_learner"


SETUP_KIND_REGISTRY: dict[str, SetupKindRegistration] = {
    "default": SetupKindRegistration(
        kind="default",
        family="default",
        backend="fixed_parameters",
    ),
    "random": SetupKindRegistration(
        kind="random",
        family="random",
        backend="random_policy",
    ),
    "linucb": SetupKindRegistration(
        kind="linucb",
        family="linucb",
        backend="online_learner",
        learner_type=SharedLinUCB_AMG_v4,
        spec_type=LinUCBV4Spec,
    ),
    "lints": SetupKindRegistration(
        kind="lints",
        family="thompson",
        backend="online_learner",
        learner_type=SharedLinTS_AMG_v2,
        spec_type=LinTSV2Spec,
    ),
}

ONLINE_SETUP_KINDS = tuple(
    kind
    for kind, registration in SETUP_KIND_REGISTRY.items()
    if registration.online
)
COMPOSABLE_SETUP_KINDS = ("default", *ONLINE_SETUP_KINDS)

_SETUP_KIND_ALIASES = {
    "linucbv4": "linucb",
    "sharedlinucbv4": "linucb",
    "lintsv2": "lints",
    "sharedlintsv2": "lints",
}


@dataclass(frozen=True)
class SetupLearnerBuildSpec:
    kind: OnlineSetupKind
    shared: SharedSetupLearnerSpec
    algorithm: SetupAlgorithmSpec


def normalize_setup_kind(kind: str) -> str:
    token = (
        str(kind)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
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
    assert registration.learner_type is not None
    assert registration.spec_type is not None
    if type(spec.algorithm) is not registration.spec_type:
        raise TypeError(
            f"{spec.kind} requires {registration.spec_type.__name__}, "
            f"got {type(spec.algorithm).__name__}"
        )
    return registration.learner_type(
        spec.shared.actions,
        context_dim=int(spec.shared.context_dim),
        **spec.shared.learner_kwargs(),
        **spec.algorithm.learner_kwargs(),
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
        raise ValueError(
            f"Setup kind {kind!r} is not an online setup learner"
        )
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
