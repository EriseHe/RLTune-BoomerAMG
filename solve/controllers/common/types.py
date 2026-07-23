from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solve.controllers.sarsa import SolveStateEncoder


@dataclass(frozen=True)
class ControllerBundle:
    """A solve controller and the exact state encoder it was built against."""

    controller: Any
    encoder: SolveStateEncoder

    def as_legacy_tuple(self) -> tuple[Any, SolveStateEncoder]:
        return self.controller, self.encoder
