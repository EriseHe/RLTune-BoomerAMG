from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .config import SharedSetupLearnerSpec


AlgorithmSpecT = TypeVar("AlgorithmSpecT")


@dataclass(frozen=True)
class SetupLearnerFactoryRequest(Generic[AlgorithmSpecT]):
    """Shared typed input passed from the registry to one learner family."""

    shared: SharedSetupLearnerSpec
    algorithm: AlgorithmSpecT


__all__ = ["SetupLearnerFactoryRequest"]
