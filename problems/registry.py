from __future__ import annotations

"""Scalar PDE identities and the shared learning contexts used by SISC."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .amg import (
    COMPACT_DIFFUSION_CONTEXT_INDICES,
    DIFCONV_CONTEXT_DIM,
    DIFFUSION_ADVECTION_NO_MEAN_CONTEXT,
    compact_diffusion_context,
    diffusion_advection_context_without_mean,
    normalize_diffusion_advection_context,
)
from .scalar_anisotropic_diffusion_advection import (
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES,
    build_default_linucb_context,
)

SCALAR_ANISOTROPIC_DIFFUSION = "scalar_anisotropic_diffusion"
SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION = "scalar_anisotropic_diffusion_advection"
PROBLEM_KIND_ALIASES = {
    SCALAR_ANISOTROPIC_DIFFUSION: SCALAR_ANISOTROPIC_DIFFUSION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION: SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
}
SUPPORTED_PROBLEM_KINDS = tuple(PROBLEM_KIND_ALIASES)
DEFAULT_SETUP_CONTEXT = "default"
CANONICAL_NO_C_MEAN_SETUP_CONTEXT = DIFFUSION_ADVECTION_NO_MEAN_CONTEXT
SETUP_CONTEXT_MODES = (
    DEFAULT_SETUP_CONTEXT,
    CANONICAL_NO_C_MEAN_SETUP_CONTEXT,
    *COMPACT_DIFFUSION_CONTEXT_INDICES,
)


@dataclass(frozen=True)
class ProblemLearningContext:
    dimension: int
    interaction_indices: tuple[int, ...]


_DEFAULT_DIFCONV_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM, interaction_indices=(1, 2, 3, 4)
)
_SCALAR_SETUP_CONTEXT_CONTRACTS = {
    **{
        mode: ProblemLearningContext(
            dimension=len(indices), interaction_indices=tuple(range(1, len(indices)))
        )
        for mode, indices in COMPACT_DIFFUSION_CONTEXT_INDICES.items()
    },
    CANONICAL_NO_C_MEAN_SETUP_CONTEXT: ProblemLearningContext(
        dimension=DIFCONV_CONTEXT_DIM - 1,
        interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM - 1)),
    ),
}
_LEARNING_CONTEXTS = {
    SCALAR_ANISOTROPIC_DIFFUSION: ProblemLearningContext(
        dimension=DIFCONV_CONTEXT_DIM,
        interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM)),
    ),
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION: ProblemLearningContext(
        dimension=SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
        interaction_indices=SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES,
    ),
}


def normalize_problem_kind(kind: str) -> str:
    token = str(kind).strip().lower().replace("-", "_")
    try:
        return PROBLEM_KIND_ALIASES[token]
    except KeyError as exc:
        supported = ", ".join(sorted(PROBLEM_KIND_ALIASES))
        raise ValueError(
            f"Unsupported problem {kind!r}; expected one of: {supported}"
        ) from exc


def normalize_setup_context_mode(mode: str) -> str:
    token = str(mode).strip().lower().replace("-", "_")
    if token not in SETUP_CONTEXT_MODES:
        supported = ", ".join(SETUP_CONTEXT_MODES)
        raise ValueError(
            f"Unsupported setup context {mode!r}; expected one of: {supported}"
        )
    return token


def learning_context_for_problem(kind: str) -> ProblemLearningContext:
    return _LEARNING_CONTEXTS[normalize_problem_kind(kind)]


def learning_context_for_setup(
    problem_kind: str,
    setup_kind: str,
    setup_context: str = DEFAULT_SETUP_CONTEXT,
) -> ProblemLearningContext:
    """Resolve the context contract for the retained setup learner."""
    problem = normalize_problem_kind(problem_kind)
    context_mode = normalize_setup_context_mode(setup_context)
    if (
        context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES
        and problem != SCALAR_ANISOTROPIC_DIFFUSION
    ):
        raise ValueError(
            "Compact diffusion contexts require scalar_anisotropic_diffusion"
        )
    if context_mode in _SCALAR_SETUP_CONTEXT_CONTRACTS:
        return _SCALAR_SETUP_CONTEXT_CONTRACTS[context_mode]
    if str(setup_kind) == "linucb":
        return _DEFAULT_DIFCONV_CONTEXT
    return _LEARNING_CONTEXTS[problem]


def context_for_setup_method(
    *,
    problem_kind: str,
    setup_kind: str,
    matrix_kwargs: Mapping[str, Any],
    stream_context: np.ndarray,
    grid_norm_div: float | None = None,
    setup_context: str = DEFAULT_SETUP_CONTEXT,
) -> np.ndarray:
    """Return the learner-visible view of the shared scalar PDE context."""
    problem = normalize_problem_kind(problem_kind)
    context_mode = normalize_setup_context_mode(setup_context)
    context = np.asarray(stream_context, dtype=float).reshape(-1)
    if context_mode in _SCALAR_SETUP_CONTEXT_CONTRACTS:
        learning_context_for_setup(problem, setup_kind, setup_context=context_mode)
        canonical = normalize_diffusion_advection_context(context)
        if context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES:
            return compact_diffusion_context(canonical, mode=context_mode)
        return diffusion_advection_context_without_mean(canonical)
    if str(setup_kind) == "linucb":
        resolved_grid_norm = (
            max(
                int(matrix_kwargs["nx"]),
                int(matrix_kwargs["ny"]),
                int(matrix_kwargs["nz"]),
            )
            if grid_norm_div is None
            else float(grid_norm_div)
        )
        return build_default_linucb_context(
            stream_context=context,
            nx=int(matrix_kwargs["nx"]),
            ny=int(matrix_kwargs["ny"]),
            nz=int(matrix_kwargs["nz"]),
            grid_norm_div=float(resolved_grid_norm),
        )
    return context.copy()


__all__ = [
    "CANONICAL_NO_C_MEAN_SETUP_CONTEXT",
    "DEFAULT_SETUP_CONTEXT",
    "PROBLEM_KIND_ALIASES",
    "ProblemLearningContext",
    "SCALAR_ANISOTROPIC_DIFFUSION",
    "SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION",
    "SETUP_CONTEXT_MODES",
    "SUPPORTED_PROBLEM_KINDS",
    "context_for_setup_method",
    "learning_context_for_problem",
    "learning_context_for_setup",
    "normalize_problem_kind",
    "normalize_setup_context_mode",
]
