from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

A1_BASE = -4.0
A2_BASE = -0.15
A3_BASE = -0.0125
NX, NY, NZ = 60, 60, 1
STENCIL = 27
CONTEXT_DIM = 5


def _validate_stencil(stencil: int) -> None:
    if int(stencil) != 27:
        raise ValueError("only 27pt stencil is supported")


def build_coefficients(*, s1: float, s2: float, s3: float, c_diag: float, nz: int = NZ) -> Tuple[float, float, float, float]:
    a1 = A1_BASE * float(s1)
    a2 = A2_BASE * float(s2)
    a3 = A3_BASE * float(s3)
    if int(nz) == 1:
        a0 = -(4.0 * a1 + 4.0 * a2) + float(c_diag)
    else:
        a0 = -(6.0 * a1 + 12.0 * a2 + 8.0 * a3) + float(c_diag)
    return float(a0), float(a1), float(a2), float(a3)


def build_context(*, s1: float, s2: float, s3: float, c_diag: float) -> np.ndarray:
    return np.array([1.0, float(s1), float(s2), float(s3), float(c_diag)], dtype=float)
def build_matrix_kwargs(
    *,
    s1: float,
    s2: float,
    s3: float,
    c_diag: float,
    rhs_seed: int,
    nx: int = NX,
    ny: int = NY,
    nz: int = NZ,
    stencil: int = STENCIL,
    ) -> Dict[str, Any]:
    _validate_stencil(int(stencil))
    a0, a1, a2, a3 = build_coefficients(s1=s1, s2=s2, s3=s3, c_diag=c_diag, nz=nz)
    return dict(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        stencil=int(stencil),
        rhs_type=1,
        rhs_seed=int(rhs_seed),
        k=1.0,
        c=0.0,
        a0=a0,
        a1=a1,
        a2=a2,
        a3=a3,
    )

def stencil_27_laplace(*, rng: np.random.Generator, **extras: Any):
    nx = int(extras.get("nx", NX))
    ny = int(extras.get("ny", NY))
    nz = int(extras.get("nz", NZ))
    stencil = int(extras.get("stencil", STENCIL))
    _validate_stencil(stencil)

    s1 = float(rng.uniform(0.5, 2.0))
    s2 = float(rng.uniform(0.5, 2.0))
    s3 = float(rng.uniform(0.5, 2.0))
    c_diag = float(rng.uniform(0.0, 5.0))
    rhs_seed = int(rng.integers(0, 2**63 - 1))
    mkw = build_matrix_kwargs(
        s1=s1,
        s2=s2,
        s3=s3,
        c_diag=c_diag,
        rhs_seed=rhs_seed,
        nx=nx,
        ny=ny,
        nz=nz,
        stencil=stencil,
    )
    context = build_context(s1=s1, s2=s2, s3=s3, c_diag=c_diag)
    meta = {"s1": s1, "s2": s2, "s3": s3, "c_diag": c_diag, "rhs_seed": rhs_seed}
    return mkw, context, meta


# ---------------------------------------------------------------------------
# DifConv sampler (stencil=0) - matches solve/core/amg_gym_env.py
# ---------------------------------------------------------------------------

DIFCONV_CONTEXT_DIM = 8


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
    Build kwargs to pass to `solver.solve` for the DifConv matrix (stencil=0).

    This uses the same arg mapping as solve/core/amg_gym_env.py:
      stencil=0, (k,c,a0..a3) <= (cx,cy,cz,ax,ay,az).
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
    nx: int,
    ny: int,
    nz: int,
    grid_norm_div: float,
    c_norm_div: float,
) -> np.ndarray:
    """
    Context that mirrors the solve-phase difconv descriptors:

    s1,s2,s3 = log(cx)/log(c_norm_div), log(cy)/log(c_norm_div), log(cz)/log(c_norm_div)
    nx_norm,ny_norm,nz_norm = log(nx)/log(grid_norm_div), ...

    We also include c_scale = mean([s1,s2,s3]) as a single "scale" feature so
    learners that use a c_diag-like interaction (index 4 by default) have a
    meaningful slot.
    """
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


def stencil_0_difconv_rl(*, rng: np.random.Generator, **extras: Any):
    """
    Problem stream matching solve-phase DifConv sampling:

    - If nx/ny/nz are not provided: sample n ~ UniformInt[n_min, n_max] and set (nx,ny,nz)=(n,n,n).
    - Sample diffusion coefficients cx,cy,cz ~ Uniform[c_min, c_max].
    - Set advection ax=ay=az=0 (matches current RL training default).
    - Sample rhs_seed uniformly (rhs_type=1 => deterministic Gaussian values in C).

    Returns (mkw, context, meta).
    """
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
