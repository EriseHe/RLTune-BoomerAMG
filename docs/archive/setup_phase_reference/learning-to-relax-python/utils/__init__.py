"""
Utils package for Learning to Relax.
Contains utility functions for sampling, optimization, and PDE simulation.
"""

from .truncated_normal import truncated_normal
from .bump import bump
from .golden_section import golden_section
from .Heat2D import Heat2D
from .delsq import delsq, numgrid

__all__ = ['truncated_normal', 'bump', 'golden_section', 'Heat2D', 'delsq', 'numgrid']

