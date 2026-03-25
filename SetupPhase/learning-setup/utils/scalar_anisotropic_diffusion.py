from __future__ import annotations

"""Scalar anisotropic diffusion AMG problem stream utilities."""

from typing import Any, Dict

import numpy as np

SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM = 8


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
    """
    Build kwargs to pass to `solver.solve` for the scalar anisotropic diffusion matrix.

    The underlying stencil-0 interface matches HYPRE's `-difconv` coefficient mapping,
    but this sampler pins advection to zero and therefore represents pure anisotropic
    diffusion:

      -cx Dxx - cy Dyy - cz Dzz = f.
    """
    return dict(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        stencil=0,
        rhs_type=int(rhs_type),
        rhs_seed=int(rhs_seed),
        k=float(cx),
        c=float(cy),
        a0=float(cz),
        a1=float(ax),
        a2=float(ay),
        a3=float(az),
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
    """Build the scalar anisotropic diffusion bandit context."""
    eps = 1e-30
    c_denom = max(float(np.log(float(c_norm_div) + eps)), eps)
    s1 = float(np.log(float(cx) + eps) / c_denom)
    s2 = float(np.log(float(cy) + eps) / c_denom)
    s3 = float(np.log(float(cz) + eps) / c_denom)
    c_scale = float((s1 + s2 + s3) / 3.0)

    n_denom = max(float(grid_norm_div), eps)
    nx_norm = float(nx) / n_denom
    ny_norm = float(ny) / n_denom
    nz_norm = float(nz) / n_denom

    return np.array([1.0, s1, s2, s3, c_scale, nx_norm, ny_norm, nz_norm], dtype=float)


def stencil_0_scalar_anisotropic_diffusion_rl(*, rng: np.random.Generator, **extras: Any):
    """Scalar anisotropic diffusion problem stream with instance-varying diffusion only."""
    n_min = int(extras.get("n_min", 10))
    n_max = int(extras.get("n_max", 80))
    if n_min <= 0 or n_max < n_min:
        raise ValueError("n_min/n_max must satisfy 1 <= n_min <= n_max")

    c_min = float(extras.get("c_min", 1.0))
    c_max = float(extras.get("c_max", 1000.0))
    if c_min <= 0.0 or c_max < c_min:
        raise ValueError("c_min/c_max must satisfy 0 < c_min <= c_max")

    if "nx" in extras and "ny" in extras and "nz" in extras:
        nx = int(extras["nx"])
        ny = int(extras["ny"])
        nz = int(extras["nz"])
    else:
        n = int(rng.integers(n_min, n_max + 1))
        nx = ny = nz = n

    cx = float(rng.uniform(c_min, c_max))
    cy = float(rng.uniform(c_min, c_max))
    cz = float(rng.uniform(c_min, c_max))

    ax = float(extras.get("ax", 0.0))
    ay = float(extras.get("ay", 0.0))
    az = float(extras.get("az", 0.0))

    rhs_type = int(extras.get("rhs_type", 1))
    rhs_seed = int(rng.integers(0, 2**63 - 1))

    mkw = build_matrix_kwargs_scalar_anisotropic_diffusion(
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
    context = build_context_scalar_anisotropic_diffusion(
        cx=cx,
        cy=cy,
        cz=cz,
        nx=nx,
        ny=ny,
        nz=nz,
        grid_norm_div=float(extras.get("grid_norm_div", n_max)),
        c_norm_div=float(extras.get("c_norm_div", c_max)),
    )
    meta = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "cx": float(cx),
        "cy": float(cy),
        "cz": float(cz),
        "ax": float(ax),
        "ay": float(ay),
        "az": float(az),
        "rhs_seed": int(rhs_seed),
        "rhs_type": int(rhs_type),
        "n_min": int(n_min),
        "n_max": int(n_max),
        "c_min": float(c_min),
        "c_max": float(c_max),
    }
    return mkw, context, meta
