from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig

from .state_encoder import (
    CANONICAL_PROBLEM_CONTEXT,
    PROBLEM_CONTEXT_MODES,
    STATE_ENCODING_VERSIONS,
    SolveStateEncoder,
)


@dataclass(frozen=True)
class EpsilonScheduleSpec:
    """Shared epsilon schedule for discrete solve-action controllers."""

    start: float = 0.30
    final: float = 0.03
    decay_steps: float = 20_000.0

    def __post_init__(self) -> None:
        values = (float(self.start), float(self.final), float(self.decay_steps))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("epsilon schedule values must be finite")
        if not 0.0 <= float(self.final) <= float(self.start) <= 1.0:
            raise ValueError("epsilon schedule must satisfy 0 <= final <= start <= 1")
        if float(self.decay_steps) <= 0.0:
            raise ValueError("epsilon decay_steps must be positive")


@dataclass(frozen=True)
class SharedActionSpec:
    """Algorithm-independent discrete action representation and exploration."""

    weights: tuple[float, ...]
    rbf_centers: tuple[float, ...]
    rbf_sigma: float = 0.2
    epsilon: EpsilonScheduleSpec = field(default_factory=EpsilonScheduleSpec)
    anchor_weight: float | None = None
    force_default_first_action: bool = True

    def __post_init__(self) -> None:
        weights = tuple(float(value) for value in self.weights)
        centers = tuple(float(value) for value in self.rbf_centers)
        if not weights or not all(math.isfinite(value) for value in weights):
            raise ValueError("solve action weights must be non-empty and finite")
        if not centers or not all(math.isfinite(value) for value in centers):
            raise ValueError("action RBF centers must be non-empty and finite")
        if any(right <= left for left, right in zip(weights, weights[1:])):
            raise ValueError("solve action weights must be strictly increasing")
        if any(right <= left for left, right in zip(centers, centers[1:])):
            raise ValueError("action RBF centers must be strictly increasing")
        if not math.isfinite(float(self.rbf_sigma)) or float(self.rbf_sigma) <= 0.0:
            raise ValueError("action RBF sigma must be finite and positive")
        anchor = weights[0] if self.anchor_weight is None else float(self.anchor_weight)
        if not math.isfinite(anchor):
            raise ValueError("anchor weight must be finite")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "rbf_centers", centers)
        object.__setattr__(self, "anchor_weight", anchor)

    def to_td_config(
        self,
        *,
        trace_lambda: float,
        epsilon_enabled: bool = True,
    ) -> ExpectedSarsaLambdaConfig:
        """Build the legacy shared-action config without changing checkpoints."""

        return ExpectedSarsaLambdaConfig(
            weights=self.weights,
            anchor_weight=float(self.anchor_weight),
            alpha=0.001,
            td_decay_power=0.0,
            gamma=1.0,
            trace_lambda=float(trace_lambda),
            epsilon_start=float(self.epsilon.start) if epsilon_enabled else 0.0,
            epsilon_final=float(self.epsilon.final) if epsilon_enabled else 0.0,
            epsilon_decay_steps=float(self.epsilon.decay_steps),
            initial_q_sec=0.0,
            monte_carlo_alpha=0.0,
            adaptive_cycles=None,
            exploration_mode="uniform",
            action_rbf_sigma=float(self.rbf_sigma),
            action_basis_mode="compact_rbf",
            action_basis_centers=self.rbf_centers,
            td_algorithm="true_online_sarsa",
            force_default_first_action=bool(self.force_default_first_action),
        )


@dataclass(frozen=True)
class SolveStateSpec:
    """Setup-aware state representation shared by online solve controllers."""

    tol: float
    max_cycles: int
    c_max: float
    time_scale_sec: float = 2.0e-3
    mode: str = "setup_full"
    problem_context_mode: str = CANONICAL_PROBLEM_CONTEXT
    encoding_version: str = "legacy_v1"

    def __post_init__(self) -> None:
        if self.encoding_version not in STATE_ENCODING_VERSIONS:
            raise ValueError(f"encoding_version must be one of {STATE_ENCODING_VERSIONS}")
        if not math.isfinite(float(self.tol)) or float(self.tol) <= 0.0:
            raise ValueError("solve tolerance must be finite and positive")
        if int(self.max_cycles) <= 0:
            raise ValueError("max_cycles must be positive")
        if not math.isfinite(float(self.c_max)) or float(self.c_max) <= 0.0:
            raise ValueError("c_max must be finite and positive")
        if (
            not math.isfinite(float(self.time_scale_sec))
            or float(self.time_scale_sec) <= 0.0
        ):
            raise ValueError("time_scale_sec must be finite and positive")
        if str(self.mode).strip().lower() != "setup_full":
            raise ValueError("online solve controllers require setup_full state")
        if (
            str(self.problem_context_mode).strip().lower()
            not in PROBLEM_CONTEXT_MODES
        ):
            raise ValueError(
                "problem_context_mode must be one of "
                f"{PROBLEM_CONTEXT_MODES}"
            )

    def build_encoder(
        self, *, setup_obs_encoder: Any, weights: Sequence[float] | None = None,
    ) -> SolveStateEncoder:
        if setup_obs_encoder is None:
            raise ValueError("setup_full state requires a setup observation encoder")
        return SolveStateEncoder(
            tol=float(self.tol),
            max_cycles=int(self.max_cycles),
            c_max=float(self.c_max),
            time_scale_sec=float(self.time_scale_sec),
            mode=str(self.mode),
            setup_obs_encoder=setup_obs_encoder,
            problem_context_mode=str(self.problem_context_mode),
            encoding_version=self.encoding_version,
            weight_bounds=(
                (min(weights), max(weights))
                if self.encoding_version == "space_aware_v2" and weights is not None
                else None
            ),
        )
