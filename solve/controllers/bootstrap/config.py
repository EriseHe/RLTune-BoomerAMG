from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapSarsaSpec:
    """Configuration owned by the bootstrap SARSA controller family."""

    members: int = 5
    episode_inclusion_probability: float = 0.8
    uncertainty_beta: float = 1.0


__all__ = ["BootstrapSarsaSpec"]
