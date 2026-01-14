"""
Learners package for Learning to Relax.
Contains bandit algorithms for tuning solver parameters.
"""

from .TsallisINF import TsallisINF
from .TsallisINFCB import TsallisINFCB
from .ChebCB import ChebCB

__all__ = ['TsallisINF', 'TsallisINFCB', 'ChebCB']
