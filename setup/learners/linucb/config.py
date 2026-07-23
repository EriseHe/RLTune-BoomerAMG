from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinUCBV4Spec:
    """Algorithm-specific configuration for shared LinUCB v4."""

    def learner_kwargs(self) -> dict[str, Any]:
        return {}


__all__ = ["LinUCBV4Spec"]
