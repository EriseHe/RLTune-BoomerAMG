from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class SetupAwareRLConfig:
    """Complete runtime contract for a frozen setup-aware solve policy."""

    tune_dim: int
    tune7_variant: str
    algo: str
    model_type: str
    model_path: Path
    vec_path: Path
    fixed_grid: Tuple[int, int, int]
    difconv_c_range: Tuple[float, float]
    w_only: bool
    w_center: float
    w_scale: float
    sweeps_min: int
    sweeps_max: int
    w_init: Optional[float]
    sweeps_init: Optional[int]
    solve_max_cycles: int
    solve_tol: float
    default_setup_params: Dict[str, Any]
    w_global_min: float = 1.0
    w_global_max: float = 2.0
    obs_mode: str = "full"
    obs_include_trace_progress: bool = False
    action_mode: str = "continuous"
    discrete_w_values: Tuple[float, ...] = ()
    discrete_joint_actions: Tuple[Tuple[float, int, int], ...] = ()
    discrete_extended_actions: Tuple[
        Tuple[float, int, int, int, int, int, float, float],
        ...,
    ] = ()
    discrete_blend_alphas: Tuple[float, ...] = ()
    blend_safe_action: Tuple[
        float,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
    ] = (1.6, 1, 1, 1, 1, 18, -1.0, -1.0)
    blend_aggr_action: Tuple[
        float,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
    ] = (1.6, 1, 1, 2, 1, 18, -1.0, 0.1)
    blend_decay_tau: float = 0.0
    blend_cutoff_cycles: int = -1
    blend_progress_start: float = 0.0
    blend_progress_end: float = 0.0
    initial_observation_weight: Optional[float] = None
    force_default_first_action: bool = False
    default_first_weight: Optional[float] = None


__all__ = ["SetupAwareRLConfig"]
