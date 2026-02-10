from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

A1_BASE = -4.0
A2_BASE = -0.15
A3_BASE = -0.0125
NX, NY, NZ = 60, 60, 1
STENCIL = 27


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
