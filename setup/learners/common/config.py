from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .action_features import ParameterSpaceSpec
from .aot_candidates import (
    AOTCandidateSchedule,
    FactorizedActionFeatureCache,
)


@dataclass(frozen=True)
class SharedSetupLearnerSpec:
    """Inputs shared by the active setup-phase linear learners.

    The fields intentionally mirror ``SharedLinUCB_AMG_v4``. Keeping this
    object free of experiment imports makes the construction boundary reusable
    by setup-only and joint runners.
    """

    actions: Sequence[dict[str, Any]]
    context_dim: int
    parameter_spec: ParameterSpaceSpec
    seed: int | None = None
    alpha: float = 1.0
    alpha_decay: bool = False
    l2_reg: float = 1.0
    context_interaction_indices: tuple[int, ...] = (1, 2, 3, 4)
    candidate_pool_size: int | None = None
    candidate_strategy: str = "uniform"
    candidate_pool_size_burnin: int | None = None
    candidate_pool_burnin_rounds: int = 0
    alpha_decay_burnin_rounds: int = 0
    elite_rank_metric: str = "best_loss"
    local_neighbor_radius: int = 1
    candidate_local_fraction: float = 0.0
    candidate_elite_fraction: float = 0.0
    always_include_arms: tuple[int, ...] | None = None
    elite_cache_size: int = 0
    failure_beta: float = 2.0
    failure_l2_reg: float = 1.0
    initial_guess: Sequence[Any] | Mapping[str, Any] | None = None
    initial_guess_rounds: int = 0
    candidate_schedule: AOTCandidateSchedule | None = None
    action_feature_cache: FactorizedActionFeatureCache | None = None

    def learner_kwargs(self) -> dict[str, Any]:
        """Return the legacy constructor arguments without changing defaults."""

        return {
            "parameter_spec": self.parameter_spec,
            "alpha": float(self.alpha),
            "alpha_decay": bool(self.alpha_decay),
            "l2_reg": float(self.l2_reg),
            "context_interaction_indices": tuple(
                int(index) for index in self.context_interaction_indices
            ),
            "candidate_pool_size": self.candidate_pool_size,
            "candidate_strategy": str(self.candidate_strategy),
            "candidate_pool_size_burnin": self.candidate_pool_size_burnin,
            "candidate_pool_burnin_rounds": int(
                self.candidate_pool_burnin_rounds
            ),
            "alpha_decay_burnin_rounds": int(
                self.alpha_decay_burnin_rounds
            ),
            "elite_rank_metric": str(self.elite_rank_metric),
            "local_neighbor_radius": int(self.local_neighbor_radius),
            "candidate_local_fraction": float(
                self.candidate_local_fraction
            ),
            "candidate_elite_fraction": float(
                self.candidate_elite_fraction
            ),
            "always_include_arms": self.always_include_arms,
            "elite_cache_size": int(self.elite_cache_size),
            "failure_beta": float(self.failure_beta),
            "failure_l2_reg": float(self.failure_l2_reg),
            "initial_guess": self.initial_guess,
            "initial_guess_rounds": int(self.initial_guess_rounds),
            "candidate_schedule": self.candidate_schedule,
            "action_feature_cache": self.action_feature_cache,
            "seed": self.seed,
        }

# Compatibility reexports. The active algorithm specs are family-owned; these
# imports stay below the shared spec definition so family factories can safely
# type their request during package initialization.
from ..linucb.config import (  # noqa: E402
    LinUCBV4Spec,
    LinUCBV5RBFSpec,
    LinUCBV5Spec,
    LinUCBV6Spec,
)
from ..thompson.config import LinTSV2Spec  # noqa: E402


__all__ = [
    "LinTSV2Spec",
    "LinUCBV4Spec",
    "LinUCBV5RBFSpec",
    "LinUCBV5Spec",
    "LinUCBV6Spec",
    "SharedSetupLearnerSpec",
]
