"""
Learners package for Learning to Relax.
Contains bandit algorithms for tuning solver parameters.

- TsallisINF: Original Tsallis-INF for SOR (loss = iteration count)
- TsallisINF_AMG: Tsallis-INF for BoomerAMG (loss = scalar derived from work units)
- TsallisINFCB: Contextual bandit wrapper
- ChebCB: Chebyshev contextual bandit
"""

from .TsallisINF_SOR import TsallisINF
from .TsallisINF_AMG import TsallisINF_AMG
from .ChebCB import ChebCB

# TsallisINFCB imports TsallisINF from TsallisINF_SOR
from .TsallisINFCB import TsallisINFCB

__all__ = ['TsallisINF', 'TsallisINF_AMG', 'TsallisINFCB', 'ChebCB']
