from __future__ import annotations

import _project_paths  # noqa: F401

from typing import Any

from SolvePhase.algorithms.sarsa import (
    BehaviorPolicySarsaController,
    SarsaBehaviorSpec,
)


ExplorationStudySpec = SarsaBehaviorSpec


class ExplorationStudyController(BehaviorPolicySarsaController):
    """Backward-compatible diagnosis alias for the shared controller."""

    def __init__(self, *, study_spec: SarsaBehaviorSpec, **kwargs: Any) -> None:
        super().__init__(behavior_spec=study_spec, **kwargs)

    @property
    def study_spec(self) -> SarsaBehaviorSpec:
        return self.behavior_spec

    def study_summary(self):
        return self.behavior_summary()
