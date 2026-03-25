from __future__ import annotations

"""Scalar diffusion-convection AMG problem stream utilities."""

from typing import Any, Dict

import numpy as np

DIFCONV_CONTEXT_DIM = 11


def build_matrix_kwargs_difconv(
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
    Build kwargs to pass to `solver.solve` for the stencil-0 HYPRE `-difconv` operator.

    This follows HYPRE's coefficient structure:

      -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f.
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


def build_context_difconv(
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
) -> np.ndarray:
    """
    Build a diffusion-convection context using diffusion log-scales and
    dimensionless convection-to-diffusion ratios.
    """
    eps = 1e-30

    c_denom = max(float(np.log(float(c_norm_div) + eps)), eps)
    s1 = float(np.log(float(cx) + eps) / c_denom)
    s2 = float(np.log(float(cy) + eps) / c_denom)
    s3 = float(np.log(float(cz) + eps) / c_denom)
    c_scale = float((s1 + s2 + s3) / 3.0)

    hx = 1.0 / max(float(nx) + 1.0, 1.0)
    hy = 1.0 / max(float(ny) + 1.0, 1.0)
    hz = 1.0 / max(float(nz) + 1.0, 1.0)
    px = float(ax) * hx / max(float(cx), eps)
    py = float(ay) * hy / max(float(cy), eps)
    pz = float(az) * hz / max(float(cz), eps)

    n_denom = max(float(grid_norm_div), eps)
    nx_norm = float(nx) / n_denom
    ny_norm = float(ny) / n_denom
    nz_norm = float(nz) / n_denom

    return np.array([1.0, s1, s2, s3, px, py, pz, c_scale, nx_norm, ny_norm, nz_norm], dtype=float)


def stencil_0_difconv_rl(*, rng: np.random.Generator, **extras: Any):
    """
    3D scalar diffusion-convection problem stream with per-instance varying
    diffusion and convection coefficients.

    By default, convection is sampled through Peclet-like ratios `px,py,pz`
    and converted to `(ax, ay, az)` using the grid spacing so advection remains
    meaningfully coupled to the sampled diffusion strength.
    """
    n_min = int(extras.get("n_min", 10))
    n_max = int(extras.get("n_max", 80))
    if n_min <= 0 or n_max < n_min:
        raise ValueError("n_min/n_max must satisfy 1 <= n_min <= n_max")

    c_min = float(extras.get("c_min", 1.0))
    c_max = float(extras.get("c_max", 1000.0))
    if c_min <= 0.0 or c_max < c_min:
        raise ValueError("c_min/c_max must satisfy 0 < c_min <= c_max")

    pe_min = float(extras.get("pe_min", -2.0))
    pe_max = float(extras.get("pe_max", 2.0))
    if pe_max < pe_min:
        raise ValueError("pe_min/pe_max must satisfy pe_min <= pe_max")

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

    hx = 1.0 / max(float(nx) + 1.0, 1.0)
    hy = 1.0 / max(float(ny) + 1.0, 1.0)
    hz = 1.0 / max(float(nz) + 1.0, 1.0)

    if all(k in extras for k in ("ax", "ay", "az")):
        ax = float(extras["ax"])
        ay = float(extras["ay"])
        az = float(extras["az"])
    else:
        px = float(rng.uniform(pe_min, pe_max))
        py = float(rng.uniform(pe_min, pe_max))
        pz = float(rng.uniform(pe_min, pe_max))
        ax = float(px * cx / hx)
        ay = float(py * cy / hy)
        az = float(pz * cz / hz)

    px = float(ax) * hx / max(float(cx), 1e-30)
    py = float(ay) * hy / max(float(cy), 1e-30)
    pz = float(az) * hz / max(float(cz), 1e-30)

    rhs_type = int(extras.get("rhs_type", 1))
    rhs_seed = int(rng.integers(0, 2**63 - 1))

    mkw = build_matrix_kwargs_difconv(
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
    context = build_context_difconv(
        cx=cx,
        cy=cy,
        cz=cz,
        ax=ax,
        ay=ay,
        az=az,
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
        "px": float(px),
        "py": float(py),
        "pz": float(pz),
        "rhs_seed": int(rhs_seed),
        "rhs_type": int(rhs_type),
        "n_min": int(n_min),
        "n_max": int(n_max),
        "c_min": float(c_min),
        "c_max": float(c_max),
        "pe_min": float(pe_min),
        "pe_max": float(pe_max),
    }
    return mkw, context, meta
