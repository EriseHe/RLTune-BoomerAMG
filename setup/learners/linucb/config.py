from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinUCBV4Spec:
    """Algorithm-specific configuration for shared LinUCB v4."""

    def learner_kwargs(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class LinUCBV5Spec:
    """Algorithm-specific configuration for paper-final shared LinUCB v5."""

    def learner_kwargs(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class LinUCBV5RBFSpec:
    """Algorithm-specific configuration for experimental LinUCB v5 RBF."""

    def learner_kwargs(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class LinUCBV6Spec:
    """Algorithm-specific configuration for internal shared LinUCB v6."""

    def learner_kwargs(self) -> dict[str, Any]:
        return {}


__all__ = [
    "LinUCBV4Spec",
    "LinUCBV5RBFSpec",
    "LinUCBV5Spec",
    "LinUCBV6Spec",
]
