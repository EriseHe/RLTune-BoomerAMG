"""
Solvers package for Learning to Relax.
Contains SOR, SSOR-PCG solvers and related utilities.
"""

from .sor import sor
from .omega_opt import omega_opt
from .omega_grid import omega_grid
from .rho_jacobi import rho_jacobi
from .cgbound import cgbound
from .energy_norm import energy_norm
from .ssor_pcg import ssor_pcg

__all__ = ['sor', 'omega_opt', 'omega_grid', 'rho_jacobi', 'cgbound', 'energy_norm', 'ssor_pcg']
