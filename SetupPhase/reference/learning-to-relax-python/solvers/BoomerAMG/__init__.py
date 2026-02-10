"""
BoomerAMG solver package for Learning to Relax.

Contains BoomerAMG solver bindings and utilities for parameter tuning
analogous to the SOR package structure.

Key tunable parameter: strong_threshold (range 0.0-1.0)
- Controls which connections are considered "strong" in coarsening
- Lower values: more aggressive coarsening
- Higher values: more conservative coarsening
- Typical values: 0.25 for 2D, 0.5-0.6 for 3D
"""

from .boomeramg import boomeramg
from .threshold_opt import threshold_opt
from .threshold_grid import threshold_grid
from .hypre_loader import load_hypre, resolve_hypre_library

__all__ = [
    'boomeramg',
    'threshold_opt',
    'threshold_grid',
    'load_hypre',
    'resolve_hypre_library',
]
