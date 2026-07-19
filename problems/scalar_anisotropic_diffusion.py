from __future__ import annotations

"""Scalar anisotropic diffusion views of the shared DifConv problem."""

from typing import Any, Dict

import numpy as np

from .amg import (
    DIFCONV_CONTEXT_DIM,
    build_context_difconv,
    build_matrix_kwargs_difconv,
    stencil_0_difconv_rl,
)


SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM = DIFCONV_CONTEXT_DIM


def build_matrix_kwargs_scalar_anisotropic_diffusion(
    *,
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    ax: float,
    ay: float,
    az: float,
    rhs_seed: int,
    rhs_type: int = 1,
) -> Dict[str, Any]:
    """Map scalar-diffusion coefficients to the shared stencil-0 interface."""
    return build_matrix_kwargs_difconv(
        nx=nx,
        ny=ny,
        nz=nz,
        cx=cx,
        cy=cy,
        cz=cz,
        ax=ax,
        ay=ay,
        az=az,
        rhs_seed=rhs_seed,
        rhs_type=rhs_type,
    )


def build_context_scalar_anisotropic_diffusion(
    *,
    cx: float,
    cy: float,
    cz: float,
    nx: int,
    ny: int,
    nz: int,
    grid_norm_div: float,
    c_norm_div: float,
) -> np.ndarray:
    """Build the shared diffusion-convection context with zero advection."""
    return build_context_difconv(
        cx=cx,
        cy=cy,
        cz=cz,
        nx=nx,
        ny=ny,
        nz=nz,
        grid_norm_div=grid_norm_div,
        c_norm_div=c_norm_div,
    )


def stencil_0_scalar_anisotropic_diffusion_rl(
    *, rng: np.random.Generator, **extras: Any
):
    """Sample the scalar-diffusion stream through the canonical DifConv sampler."""
    return stencil_0_difconv_rl(rng=rng, **extras)
