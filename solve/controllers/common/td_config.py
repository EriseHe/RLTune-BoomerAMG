"""Shared state/action and learning configuration for online TD controllers.

The established class name is retained for existing experiment configurations
and checkpoints. Its ownership is shared across controller families.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedSarsaLambdaConfig:
    """State/action and update configuration for online TD controllers."""

    weights: tuple[float, ...] = (1.4, 1.5, 1.6)
    anchor_weight: float = 1.4
    alpha: float = 0.02
    td_decay_power: float = 0.0
    gamma: float = 1.0
    trace_lambda: float = 0.8
    epsilon_start: float = 0.20
    epsilon_final: float = 0.01
    epsilon_decay_steps: float = 4000.0
    l2: float = 0.0
    initial_q_sec: float = 0.02
    monte_carlo_alpha: float = 0.0
    monte_carlo_decay_power: float = 0.0
    adaptive_cycles: int | None = None
    exploration_mode: str = "uniform"
    action_rbf_sigma: float = 0.0
    action_basis_mode: str = "legacy"
    action_basis_centers: tuple[float, ...] = ()
    td_algorithm: str = "expected_sarsa"
    force_default_first_action: bool = False


__all__ = ["ExpectedSarsaLambdaConfig"]
