from __future__ import annotations

from dataclasses import dataclass


from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig


@dataclass(frozen=True)
class SarsaBehaviorSpec:
    """Behavior-policy controls for the exploratory SARSA adapter."""

    action_mode: str
    behavior_mode: str
    initial_weight: float = 1.0
    force_default_first_action: bool = True
    residual_min: float = 1.0
    residual_max: float = 2.0
    uncertainty_beta: float = 1.0
    uncertainty_ridge: float = 1.0
    uncertainty_td_floor_sec: float = 1.0e-3

    @property
    def name(self) -> str:
        return f"{self.action_mode}_{self.behavior_mode}"


__all__ = ["ExpectedSarsaLambdaConfig", "SarsaBehaviorSpec"]
