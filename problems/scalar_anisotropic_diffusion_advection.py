from __future__ import annotations

"""Scalar anisotropic diffusion-advection views of the shared DifConv problem."""

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


SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_FIELDS = (
    DIFFUSION_ADVECTION_CONTEXT_FIELDS
)
SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM = (
    DIFFUSION_ADVECTION_CONTEXT_DIM
)
SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
)


def build_matrix_kwargs_scalar_anisotropic_diffusion_advection(
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
    """Map diffusion-advection coefficients to the shared stencil-0 interface."""
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


def build_context_scalar_anisotropic_diffusion_advection(
    *,
    cx: float,
    cy: float,
    cz: float,
    ax: float,
    ay: float,
    az: float,
    nx: int,
    ny: int,
    nz: int,
    grid_norm_div: float,
    c_norm_div: float,
    a_norm_div: float,
) -> np.ndarray:
    """Build diffusion and advection descriptors for one fixed-grid matrix."""
    return build_context_diffusion_advection(
        cx=cx,
        cy=cy,
        cz=cz,
        ax=ax,
        ay=ay,
        az=az,
        nx=nx,
        ny=ny,
        nz=nz,
        grid_norm_div=grid_norm_div,
        c_norm_div=c_norm_div,
        a_norm_div=a_norm_div,
    )


def build_default_linucb_context(
    *,
    stream_context: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    grid_norm_div: float,
) -> np.ndarray:
    """Project the PDE stream onto the default LinUCB context."""
    context = np.asarray(stream_context, dtype=float).reshape(-1)
    if context.size != SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM:
        raise ValueError(
            "Canonical scalar PDE context must contain eight values"
        )
    denominator = float(grid_norm_div)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("grid_norm_div must be finite and positive")
    grid_context = np.asarray(
        [
            float(nx) / denominator,
            float(ny) / denominator,
            float(nz) / denominator,
        ],
        dtype=float,
    )
    return np.concatenate((context[:5], grid_context))


def stencil_0_scalar_anisotropic_diffusion_advection_rl(
    *,
    rng: np.random.Generator,
    **extras: Any,
):
    """Sample independently varying diffusion and advection coefficients."""
    a_min = float(extras.get("a_min", extras.get("c_min", 1.0)))
    a_max = float(extras.get("a_max", extras.get("c_max", 1000.0)))
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max < a_min:
        raise ValueError(
            "a_min/a_max must be finite and satisfy a_min <= a_max"
        )

    base_extras = dict(extras)
    for key in ("a_min", "a_max", "a_norm_div", "ax", "ay", "az"):
        base_extras.pop(key, None)
    _matrix_kwargs, _context, metadata = stencil_0_difconv_rl(
        rng=rng,
        **base_extras,
    )
    ax, ay, az = (
        float(value)
        for value in rng.uniform(a_min, a_max, size=3)
    )
    matrix_kwargs = build_matrix_kwargs_scalar_anisotropic_diffusion_advection(
        nx=int(metadata["nx"]),
        ny=int(metadata["ny"]),
        nz=int(metadata["nz"]),
        cx=float(metadata["cx"]),
        cy=float(metadata["cy"]),
        cz=float(metadata["cz"]),
        ax=ax,
        ay=ay,
        az=az,
        rhs_seed=int(metadata["rhs_seed"]),
        rhs_type=int(metadata["rhs_type"]),
    )

    a_norm_div = float(
        extras.get("a_norm_div", max(abs(a_min), abs(a_max)))
    )
    context = build_context_diffusion_advection_from_matrix_kwargs(
        matrix_kwargs,
        grid_norm_div=float(base_extras.get("grid_norm_div", metadata["n_max"])),
        c_norm_div=float(base_extras.get("c_norm_div", metadata["c_max"])),
        a_norm_div=a_norm_div,
    )
    metadata = {
        **metadata,
        "ax": ax,
        "ay": ay,
        "az": az,
        "a_min": a_min,
        "a_max": a_max,
        "a_norm_div": a_norm_div,
    }
    return matrix_kwargs, context, metadata


__all__ = [
    "SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM",
    "SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_FIELDS",
    "SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES",
    "build_context_scalar_anisotropic_diffusion_advection",
    "build_default_linucb_context",
    "build_matrix_kwargs_scalar_anisotropic_diffusion_advection",
    "stencil_0_scalar_anisotropic_diffusion_advection_rl",
]
