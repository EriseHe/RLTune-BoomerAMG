from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


DIFCONV_CONTEXT_DIM = 8
DIFFUSION_ADVECTION_CONTEXT_FIELDS = (
    "bias",
    "c_x",
    "c_y",
    "c_z",
    "c_mean",
    "a_x",
    "a_y",
    "a_z",
)
DIFFUSION_ADVECTION_CONTEXT_DIM = len(DIFFUSION_ADVECTION_CONTEXT_FIELDS)
# The names count varying PDE descriptors; the intercept is retained separately.
COMPACT_DIFFUSION_CONTEXT_INDICES = {
    "diffusion3d": (0, 1, 2, 3),
    "diffusion4d": (0, 1, 2, 3, 4),
}
DIFFUSION_ADVECTION_NO_MEAN_CONTEXT = "canonical_no_c_mean"
DIFFUSION_ADVECTION_NO_MEAN_INDICES = (0, 1, 2, 3, 5, 6, 7)


def normalize_diffusion_advection_context(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Validate the canonical PDE context shared by setup and solve."""

    context = np.asarray(values, dtype=float).reshape(-1)
    if context.size != DIFFUSION_ADVECTION_CONTEXT_DIM:
        raise ValueError(
            "PDE context must contain "
            f"{DIFFUSION_ADVECTION_CONTEXT_DIM} values "
            f"({', '.join(DIFFUSION_ADVECTION_CONTEXT_FIELDS)})"
        )
    if not np.all(np.isfinite(context)):
        raise ValueError("PDE context values must be finite")
    if not np.isclose(context[0], 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("PDE context bias field must equal 1")
    return context


def compact_diffusion_context(
    values: Sequence[float] | np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Select diffusion coefficients, optionally their mean, plus the bias.

    The stream keeps its historical eight-field representation. Both learners
    use this same explicit view; nonzero advection must never be discarded.
    """
    context = normalize_diffusion_advection_context(values)
    if np.any(context[5:8] != 0.0):
        raise ValueError("Compact diffusion contexts require zero advection")
    return context[list(COMPACT_DIFFUSION_CONTEXT_INDICES[mode])].copy()


def diffusion_advection_context_without_mean(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Shared learner view: intercept and all six normalized coefficients."""
    context = normalize_diffusion_advection_context(values)
    return context[list(DIFFUSION_ADVECTION_NO_MEAN_INDICES)].copy()


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

    The native interface maps stencil=0 and (k,c,a0..a3) to
    (cx,cy,cz,ax,ay,az).
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


def _signed_log_normalize(value: float, *, scale: float) -> float:
    denominator = float(np.log1p(max(abs(float(scale)), 0.0)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.sign(float(value)) * np.log1p(abs(float(value))) / denominator)


def build_context_diffusion_advection(
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
    """Build the canonical PDE context shared by setup and solve learners."""

    legacy_context = build_context_difconv(
        cx=cx,
        cy=cy,
        cz=cz,
        nx=nx,
        ny=ny,
        nz=nz,
        grid_norm_div=grid_norm_div,
        c_norm_div=c_norm_div,
    )
    advection_context = np.asarray(
        [
            _signed_log_normalize(ax, scale=a_norm_div),
            _signed_log_normalize(ay, scale=a_norm_div),
            _signed_log_normalize(az, scale=a_norm_div),
        ],
        dtype=float,
    )
    return np.concatenate((legacy_context[:5], advection_context))


def build_context_diffusion_advection_from_matrix_kwargs(
    matrix_kwargs: Mapping[str, Any],
    *,
    grid_norm_div: float,
    c_norm_div: float,
    a_norm_div: float,
) -> np.ndarray:
    """Build the canonical learner context from the exact solver inputs."""

    return build_context_diffusion_advection(
        cx=float(matrix_kwargs["k"]),
        cy=float(matrix_kwargs["c"]),
        cz=float(matrix_kwargs["a0"]),
        ax=float(matrix_kwargs.get("a1", 0.0)),
        ay=float(matrix_kwargs.get("a2", 0.0)),
        az=float(matrix_kwargs.get("a3", 0.0)),
        nx=int(matrix_kwargs.get("nx", 1)),
        ny=int(matrix_kwargs.get("ny", 1)),
        nz=int(matrix_kwargs.get("nz", 1)),
        grid_norm_div=float(grid_norm_div),
        c_norm_div=float(c_norm_div),
        a_norm_div=float(a_norm_div),
    )


def stencil_0_difconv_rl(*, rng: np.random.Generator, **extras: Any):
    """
    Shared stencil-0 sampler for the scalar PDE streams:

    - If nx/ny/nz are not provided: sample n ~ UniformInt[n_min, n_max] and set (nx,ny,nz)=(n,n,n).
    - Sample diffusion coefficients cx,cy,cz ~ Uniform[c_min, c_max].
    - Set advection ax=ay=az=0.
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
