from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from hypre.bindings import AttemptOutcome

from .state_encoder import SolveStateEncoder


FallbackAttempt = Callable[[], Mapping[str, Any] | AttemptOutcome]


@dataclass(frozen=True)
class OnlineSolveCase:
    """One controller episode, independent of experiment orchestration.

    Setup selection and fallback policy remain caller-owned.  A caller may
    inject the already-resolved setup parameters and an optional fallback
    callback, while the solve layer owns the controller/encoder transaction.
    """

    mkw: Mapping[str, Any]
    params: Mapping[str, Any]
    solve_tol: float
    solve_max_cycles: int
    learn: bool
    explore: bool
    epsilon: float | None = None
    defer_monte_carlo_update: bool = False
    record_action_metadata: bool = False
    initial_environment_weight_override: float | None = None
    fallback_attempt: FallbackAttempt | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.solve_tol)) or float(self.solve_tol) <= 0.0:
            raise ValueError("solve_tol must be finite and positive")
        if int(self.solve_max_cycles) <= 0:
            raise ValueError("solve_max_cycles must be positive")
        if self.epsilon is not None and not 0.0 <= float(self.epsilon) <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")


@dataclass(frozen=True)
class ControllerBundle:
    """Solve-layer facade for a controller and its exact state encoder."""

    controller: Any
    encoder: SolveStateEncoder
    _protocol_metadata: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def as_legacy_tuple(self) -> tuple[Any, SolveStateEncoder]:
        return self.controller, self.encoder

    def run_case(self, case: OnlineSolveCase) -> dict[str, Any]:
        """Run one solve episode through the shared transactional adapter."""

        from solve.controllers.sarsa.online_td_lambda import run_td_episode

        return run_td_episode(
            mkw=dict(case.mkw),
            params=dict(case.params),
            controller=self.controller,
            encoder=self.encoder,
            solve_tol=float(case.solve_tol),
            solve_max_cycles=int(case.solve_max_cycles),
            learn=bool(case.learn),
            explore=bool(case.explore),
            epsilon=case.epsilon,
            defer_monte_carlo_update=bool(case.defer_monte_carlo_update),
            record_action_metadata=bool(case.record_action_metadata),
            initial_environment_weight_override=(
                case.initial_environment_weight_override
            ),
            fallback_attempt=case.fallback_attempt,
        )

    def summary(self) -> dict[str, Any]:
        """Return the stable cross-family controller summary."""

        summary = {
            "steps": int(self.controller.steps),
            "episodes": int(self.controller.episodes),
            "epsilon": float(self.controller.epsilon),
        }
        if hasattr(self.controller, "behavior_summary"):
            family_summary = self.controller.behavior_summary()
        elif hasattr(self.controller, "summary"):
            family_summary = self.controller.summary()
        else:
            family_summary = {}
        if not isinstance(family_summary, Mapping):
            raise TypeError("controller summary must be a mapping")
        summary.update(dict(family_summary))
        return summary

    def save(self, path: Path) -> None:
        """Persist the native controller checkpoint without schema changes."""

        self.controller.save(path)

    def protocol_metadata(self) -> dict[str, Any]:
        """Return an isolated, JSON-ready description of the built controller."""

        return copy.deepcopy(dict(self._protocol_metadata))


__all__ = ["ControllerBundle", "FallbackAttempt", "OnlineSolveCase"]
