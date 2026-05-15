"""
Top-level solvers package.

- `solvers.SOR`: faithful MATLAB->Python translation of the original SOR/SSOR-PCG solvers.
- `solvers.BoomerAMG`: HYPRE BoomerAMG bindings (setup/in-progress).

The experiment scripts keep importing from `solvers` for backward compatibility.
"""

from .SOR import cgbound, energy_norm, omega_grid, omega_opt, rho_jacobi, sor, ssor_pcg

__all__ = [
    "sor",
    "omega_opt",
    "omega_grid",
    "rho_jacobi",
    "cgbound",
    "energy_norm",
    "ssor_pcg",
]

