from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinUCBSpec:
    """Algorithm-specific configuration for shared context-action LinUCB."""

    def learner_kwargs(self) -> dict[str, Any]:
        return {}


__all__ = ["LinUCBSpec"]
