"""Window-free RL activation from first-attempt reliability evidence.

Implements equations (1)--(4) in docs/theory/dynamic_rl_start_and_theorem_framework.md.
The gate observes only the default-solve prefix, once per problem, before any
successful recovery can conceal a first-attempt failure. Crossing after problem
t enables RL on problem t+1. No fixed deadline or minimum window is imposed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ReliabilityActivationSpec:
    p_bad: float
    p_good: float
    delta: float
    kind: str = "reliability_mixture"

    def __post_init__(self) -> None:
        if self.kind != "reliability_mixture":
            raise ValueError("solve_activation.kind must be reliability_mixture")
        if not (0.0 < self.p_good < self.p_bad < 1.0):
            raise ValueError("solve_activation requires 0 < p_good < p_bad < 1")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("solve_activation.delta must lie strictly between 0 and 1")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ReliabilityActivationSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("solve_activation must be an object")
        unknown = set(raw) - {"kind", "p_bad", "p_good", "delta"}
        if unknown:
            raise ValueError(f"Unknown solve_activation keys: {sorted(unknown)}")
        return cls(
            kind=str(raw.get("kind", "reliability_mixture")),
            p_bad=float(raw["p_bad"]),
            p_good=float(raw["p_good"]),
            delta=float(raw["delta"]),
        )


class ReliabilityActivationGate:
    def __init__(self, spec: ReliabilityActivationSpec) -> None:
        self.spec = spec
        self.observations = 0
        self.failures = 0
        self.log_accumulator = -math.inf
        self.log_evalue = 0.0
        self.crossing_case: int | None = None
        self._log_success = math.log1p(-spec.p_good) - math.log1p(-spec.p_bad)
        self._log_failure = math.log(spec.p_good) - math.log(spec.p_bad)
        self.log_threshold = -math.log(spec.delta)

    @property
    def active(self) -> bool:
        return self.crossing_case is not None

    def observe(self, *, first_attempt_failed: bool) -> None:
        if self.active:
            raise RuntimeError("The activation gate must stop observing after crossing")
        self.observations += 1
        self.failures += int(first_attempt_failed)
        t = self.observations
        log_weight = -math.log(t) - math.log(t + 1)
        self.log_accumulator = float(np.logaddexp(self.log_accumulator, log_weight)) + (
            self._log_failure if first_attempt_failed else self._log_success
        )
        # Include the mass of all not-yet-started tests: sum_{k>t} omega_k=1/(t+1).
        self.log_evalue = float(np.logaddexp(self.log_accumulator, -math.log(t + 1)))
        if self.log_evalue >= self.log_threshold:
            self.crossing_case = t

    def summary(self) -> dict[str, Any]:
        return {
            **asdict(self.spec),
            "observations": self.observations,
            "first_attempt_failures": self.failures,
            "log_accumulator": (
                self.log_accumulator if self.observations else None
            ),
            "log_evalue": self.log_evalue,
            "log_threshold": self.log_threshold,
            "crossing_case": self.crossing_case,
            "first_rl_case": (
                None if self.crossing_case is None else self.crossing_case + 1
            ),
        }
