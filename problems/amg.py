from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

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
DIFFUSION_ADVECTION_CONTEXT_DIM = len(
    DIFFUSION_ADVECTION_CONTEXT_FIELDS
)
# The names count varying PDE descriptors; the intercept is retained separately.
COMPACT_DIFFUSION_CONTEXT_INDICES = {
    "diffusion3d": (0, 1, 2, 3),
    "diffusion4d": (0, 1, 2, 3, 4),
}
DIFFUSION_ADVECTION_NO_MEAN_CONTEXT = "canonical_no_c_mean"
DIFFUSION_ADVECTION_NO_MEAN_INDICES = (0, 1, 2, 3, 5, 6, 7)
DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS = (
    "diffusion_log_mean",
    "diffusion_log_contrast_xy",
    "diffusion_log_contrast_xyz",
    "cell_peclet_x",
    "cell_peclet_y",
    "cell_peclet_z",
)


def _complete_quadratic_field_names(
    fields: Sequence[str],
) -> tuple[str, ...]:
    linear = tuple(str(field) for field in fields)
    squares = tuple(f"{field}^2" for field in linear)
    pairs = tuple(
        f"{left}*{right}"
        for index, left in enumerate(linear)
        for right in linear[index + 1 :]
    )
    return ("bias", *linear, *squares, *pairs)


DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS = (
    _complete_quadratic_field_names(
        DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS
    )
)
DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM = len(
    DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS
)


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


def _signed_log_normalize(value: float, *, scale: float) -> float:
    denominator = float(np.log1p(max(abs(float(scale)), 0.0)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(
        np.sign(float(value))
        * np.log1p(abs(float(value)))
        / denominator
    )


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


def build_directional_cell_peclet_from_matrix_kwargs(
    matrix_kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Return the three signed cell Péclet ratios of the DifConv stencil.

    The native forward-difference stencil compares ``a_i / h_i`` against
    ``c_i / h_i**2`` in each direction, so its directional cell Péclet ratio
    is ``a_i * h_i / c_i`` with ``h_i = 1 / (n_i + 1)``.
    """

    diffusion = np.asarray(
        [
            matrix_kwargs["k"],
            matrix_kwargs["c"],
            matrix_kwargs["a0"],
        ],
        dtype=float,
    )
    advection = np.asarray(
        [
            matrix_kwargs.get("a1", 0.0),
            matrix_kwargs.get("a2", 0.0),
            matrix_kwargs.get("a3", 0.0),
        ],
        dtype=float,
    )
    grid_shape = np.asarray(
        [
            matrix_kwargs["nx"],
            matrix_kwargs["ny"],
            matrix_kwargs["nz"],
        ],
        dtype=float,
    )
    if not (
        np.all(np.isfinite(diffusion))
        and np.all(np.isfinite(advection))
        and np.all(np.isfinite(grid_shape))
    ):
        raise ValueError("Péclet inputs must be finite")
    if np.any(diffusion <= 0.0):
        raise ValueError("Péclet diffusion coefficients must be positive")
    if np.any(grid_shape < 1.0):
        raise ValueError("Péclet grid dimensions must be positive")

    directional_peclet = advection / diffusion / (grid_shape + 1.0)
    if not np.all(np.isfinite(directional_peclet)):
        raise ValueError("Péclet ratios must be finite")
    return directional_peclet


def build_normalized_cell_peclet_from_matrix_kwargs(
    matrix_kwargs: Mapping[str, Any],
) -> float:
    """Return the strongest absolute cell Péclet ratio mapped to ``[0, 1)``."""

    directional_peclet = np.abs(
        build_directional_cell_peclet_from_matrix_kwargs(matrix_kwargs)
    )
    cell_peclet = float(np.max(directional_peclet))
    if not np.isfinite(cell_peclet):
        raise ValueError("Péclet ratio must be finite")
    return float(cell_peclet / (1.0 + cell_peclet))


def build_diffusion_advection_physics_coordinates(
    *,
    canonical_context: Sequence[float] | np.ndarray,
    matrix_kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Build six non-redundant, bounded PDE coordinates.

    The three normalized log-diffusion coefficients are represented by one
    scale coordinate and two orthogonal anisotropy contrasts.  The remaining
    coordinates are signed directional cell Péclet ratios, smoothly bounded
    to ``(-1, 1)``.  This removes the redundant canonical ``c_mean`` field
    while preserving all three diffusion degrees of freedom.
    """

    canonical = normalize_diffusion_advection_context(canonical_context)
    diffusion = canonical[1:4]
    coordinates = np.asarray(
        [
            float(np.mean(diffusion)),
            float((diffusion[0] - diffusion[1]) / np.sqrt(2.0)),
            float(
                (diffusion[0] + diffusion[1] - 2.0 * diffusion[2])
                / np.sqrt(6.0)
            ),
        ],
        dtype=float,
    )
    directional_peclet = (
        build_directional_cell_peclet_from_matrix_kwargs(matrix_kwargs)
    )
    bounded_peclet = directional_peclet / (
        1.0 + np.abs(directional_peclet)
    )
    return np.concatenate((coordinates, bounded_peclet))


def build_diffusion_advection_physics_context(
    *,
    canonical_context: Sequence[float] | np.ndarray,
    matrix_kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Return bias plus the six non-redundant physical PDE coordinates."""

    coordinates = build_diffusion_advection_physics_coordinates(
        canonical_context=canonical_context,
        matrix_kwargs=matrix_kwargs,
    )
    return np.concatenate(([1.0], coordinates))


def build_diffusion_advection_quadratic_context(
    *,
    canonical_context: Sequence[float] | np.ndarray,
    matrix_kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Lift the six physical PDE coordinates into a complete Q2 basis."""

    linear_context = build_diffusion_advection_physics_context(
        canonical_context=canonical_context,
        matrix_kwargs=matrix_kwargs,
    )
    coordinates = linear_context[1:]
    squares = coordinates * coordinates
    pairs = np.asarray(
        [
            coordinates[left] * coordinates[right]
            for left in range(coordinates.size)
            for right in range(left + 1, coordinates.size)
        ],
        dtype=float,
    )
    context = np.concatenate((linear_context, squares, pairs))
    if context.size != DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM:
        raise AssertionError("quadratic PDE context dimension drifted")
    return context


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
