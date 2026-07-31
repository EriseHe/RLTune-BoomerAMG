from __future__ import annotations

"""Scalar anisotropic diffusion views of the shared DifConv problem."""

from typing import Any, Dict

import numpy as np

from .amg import (
    DIFFUSION_ADVECTION_CONTEXT_DIM,
    DIFFUSION_ADVECTION_CONTEXT_FIELDS,
    build_context_diffusion_advection,
    build_context_diffusion_advection_from_matrix_kwargs,
    build_matrix_kwargs_difconv,
    stencil_0_difconv_rl,
)


SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_FIELDS = (
    DIFFUSION_ADVECTION_CONTEXT_FIELDS
)
SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM = (
    DIFFUSION_ADVECTION_CONTEXT_DIM
)


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
    """Build the canonical PDE context with three zero-advection fields."""
    return build_context_diffusion_advection(
        cx=cx,
        cy=cy,
        cz=cz,
        ax=0.0,
        ay=0.0,
        az=0.0,
        nx=nx,
        ny=ny,
        nz=nz,
        grid_norm_div=grid_norm_div,
        c_norm_div=c_norm_div,
        a_norm_div=1.0,
    )


def stencil_0_scalar_anisotropic_diffusion_rl(
    *, rng: np.random.Generator, **extras: Any
):
    """Sample scalar diffusion and expose the canonical zero-advection context."""

    scalar_extras = dict(extras)
    scalar_extras.update({"ax": 0.0, "ay": 0.0, "az": 0.0})
    matrix_kwargs, _legacy_context, metadata = stencil_0_difconv_rl(
        rng=rng,
        **scalar_extras,
    )
    context = build_context_diffusion_advection_from_matrix_kwargs(
        matrix_kwargs,
        grid_norm_div=float(
            scalar_extras.get("grid_norm_div", metadata["n_max"])
        ),
        c_norm_div=float(
            scalar_extras.get("c_norm_div", metadata["c_max"])
        ),
        a_norm_div=1.0,
    )
    return matrix_kwargs, context, metadata


__all__ = [
    "SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM",
    "SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_FIELDS",
    "build_context_scalar_anisotropic_diffusion",
    "build_matrix_kwargs_scalar_anisotropic_diffusion",
    "stencil_0_scalar_anisotropic_diffusion_rl",
]
