from __future__ import annotations

"""Problem identities and learning-context contracts."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .amg import (
    COMPACT_DIFFUSION_CONTEXT_INDICES,
    DIFCONV_CONTEXT_DIM,
    DIFFUSION_ADVECTION_NO_MEAN_CONTEXT,
    DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM,
    build_diffusion_advection_physics_context,
    build_diffusion_advection_quadratic_context,
    build_normalized_cell_peclet_from_matrix_kwargs,
    compact_diffusion_context,
    diffusion_advection_context_without_mean,
    normalize_diffusion_advection_context,
)
from .scalar_anisotropic_diffusion_advection import (
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES,
    build_legacy_linucb_v4_context,
)


SCALAR_ANISOTROPIC_DIFFUSION = "scalar_anisotropic_diffusion"
DIFFUSION_CONVECTION = "diffusion_convection"
SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION = (
    "scalar_anisotropic_diffusion_advection"
)

PROBLEM_KIND_ALIASES = {
    SCALAR_ANISOTROPIC_DIFFUSION: SCALAR_ANISOTROPIC_DIFFUSION,
    DIFFUSION_CONVECTION: DIFFUSION_CONVECTION,
    "difconv": DIFFUSION_CONVECTION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION: (
        SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION
    ),
}
SUPPORTED_PROBLEM_KINDS = tuple(PROBLEM_KIND_ALIASES)

DEFAULT_SETUP_CONTEXT = "default"
CANONICAL_NO_C_MEAN_SETUP_CONTEXT = DIFFUSION_ADVECTION_NO_MEAN_CONTEXT
CANONICAL_MEANS_ONLY_SETUP_CONTEXT = "canonical_means_only"
CANONICAL_WITH_A_MEAN_SETUP_CONTEXT = "canonical_with_a_mean"
CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT = (
    "canonical_with_means_and_peclet"
)
CANONICAL_PECLET_ONLY_SETUP_CONTEXT = "canonical_peclet_only"
PHYSICS_LINEAR_SETUP_CONTEXT = "physics_linear"
SETUP_CONTEXT_MODES = (
    DEFAULT_SETUP_CONTEXT,
    CANONICAL_NO_C_MEAN_SETUP_CONTEXT,
    CANONICAL_MEANS_ONLY_SETUP_CONTEXT,
    CANONICAL_WITH_A_MEAN_SETUP_CONTEXT,
    CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT,
    CANONICAL_PECLET_ONLY_SETUP_CONTEXT,
    PHYSICS_LINEAR_SETUP_CONTEXT,
    *COMPACT_DIFFUSION_CONTEXT_INDICES,
)


@dataclass(frozen=True)
class ProblemLearningContext:
    dimension: int
    interaction_indices: tuple[int, ...]


_DEFAULT_DIFCONV_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM,
    interaction_indices=(1, 2, 3, 4),
)
_LINUCB_V5_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM,
    interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM)),
)
_LINUCB_V6_CONTEXT = ProblemLearningContext(
    dimension=DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM,
    interaction_indices=tuple(
        range(1, DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM)
    ),
)
_LINUCB_NO_C_MEAN_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM - 1,
    interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM - 1)),
)
_LINUCB_MEANS_ONLY_CONTEXT = ProblemLearningContext(
    dimension=3,
    interaction_indices=(1, 2),
)
_LINUCB_WITH_A_MEAN_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM + 1,
    interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM + 1)),
)
_LINUCB_WITH_MEANS_AND_PECLET_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM + 2,
    interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM + 2)),
)
_LINUCB_PECLET_ONLY_CONTEXT = ProblemLearningContext(
    dimension=DIFCONV_CONTEXT_DIM,
    interaction_indices=tuple(range(1, DIFCONV_CONTEXT_DIM)),
)
_LINUCB_PHYSICS_LINEAR_CONTEXT = ProblemLearningContext(
    dimension=7,
    interaction_indices=tuple(range(1, 7)),
)
_SCALAR_SETUP_CONTEXT_CONTRACTS = {
    **{
        mode: ProblemLearningContext(
            dimension=len(indices),
            interaction_indices=tuple(range(1, len(indices))),
        )
        for mode, indices in COMPACT_DIFFUSION_CONTEXT_INDICES.items()
    },
    CANONICAL_NO_C_MEAN_SETUP_CONTEXT: _LINUCB_NO_C_MEAN_CONTEXT,
    CANONICAL_MEANS_ONLY_SETUP_CONTEXT: _LINUCB_MEANS_ONLY_CONTEXT,
    CANONICAL_WITH_A_MEAN_SETUP_CONTEXT: _LINUCB_WITH_A_MEAN_CONTEXT,
    CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT: (
        _LINUCB_WITH_MEANS_AND_PECLET_CONTEXT
    ),
    CANONICAL_PECLET_ONLY_SETUP_CONTEXT: _LINUCB_PECLET_ONLY_CONTEXT,
    PHYSICS_LINEAR_SETUP_CONTEXT: _LINUCB_PHYSICS_LINEAR_CONTEXT,
}
_LEARNING_CONTEXTS = {
    SCALAR_ANISOTROPIC_DIFFUSION: _LINUCB_V5_CONTEXT,
    DIFFUSION_CONVECTION: _DEFAULT_DIFCONV_CONTEXT,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION: ProblemLearningContext(
        dimension=SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
        interaction_indices=(
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_INTERACTION_INDICES
        ),
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
    """Resolve the context contract owned by one problem/setup pairing."""
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
        if problem not in {
            SCALAR_ANISOTROPIC_DIFFUSION,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
        }:
            raise ValueError(
                f"{context_mode} is only defined for scalar PDE streams"
            )
        if str(setup_kind) in {"linucb_v5", "linucb_v5_rbf"}:
            raise ValueError(
                "The frozen LinUCB v5 contract is 8-D; use the shared LinUCB "
                "algorithm for context-feature ablations"
            )
        if str(setup_kind) == "linucb_v6":
            raise ValueError(
                "The frozen LinUCB v6 contract is the 28-D quadratic physics "
                "context; use setup_context='default'"
            )
        return _SCALAR_SETUP_CONTEXT_CONTRACTS[context_mode]
    if str(setup_kind) == "linucb_v6":
        if problem not in {
            SCALAR_ANISOTROPIC_DIFFUSION,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
        }:
            raise ValueError(
                "LinUCB v6 is only defined for scalar diffusion and "
                "scalar diffusion-advection streams"
            )
        return _LINUCB_V6_CONTEXT
    if str(setup_kind) in {"linucb_v5", "linucb_v5_rbf"}:
        # v5 uses one semantic layout for every scalar PDE stream:
        # [1, cx, cy, cz, c_mean, ax, ay, az]. Scalar diffusion supplies
        # exact zeros for the final three fields.
        return _LINUCB_V5_CONTEXT
    if (
        problem
        in {
            SCALAR_ANISOTROPIC_DIFFUSION,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
        }
        and str(setup_kind) == "linucb"
    ):
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
    """Return the learner-visible view of a shared problem-stream context."""
    problem = normalize_problem_kind(problem_kind)
    context_mode = normalize_setup_context_mode(setup_context)
    context = np.asarray(stream_context, dtype=float).reshape(-1)
    if context_mode in _SCALAR_SETUP_CONTEXT_CONTRACTS:
        learning_context_for_setup(
            problem,
            setup_kind,
            setup_context=context_mode,
        )
        canonical = normalize_diffusion_advection_context(context)
        if context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES:
            return compact_diffusion_context(canonical, mode=context_mode)
        if context_mode == CANONICAL_NO_C_MEAN_SETUP_CONTEXT:
            return diffusion_advection_context_without_mean(canonical)

        advection_mean = float(np.mean(canonical[5:8]))
        if context_mode == CANONICAL_MEANS_ONLY_SETUP_CONTEXT:
            return np.asarray(
                [canonical[0], canonical[4], advection_mean],
                dtype=float,
            )
        if context_mode == CANONICAL_WITH_A_MEAN_SETUP_CONTEXT:
            return np.concatenate((canonical, [advection_mean]))

        peclet = build_normalized_cell_peclet_from_matrix_kwargs(
            matrix_kwargs
        )
        if context_mode == CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT:
            return np.concatenate(
                (canonical, [advection_mean, peclet])
            )
        if context_mode == CANONICAL_PECLET_ONLY_SETUP_CONTEXT:
            return np.concatenate(
                (canonical[[0, 1, 2, 3, 5, 6, 7]], [peclet])
            )
        if context_mode == PHYSICS_LINEAR_SETUP_CONTEXT:
            return build_diffusion_advection_physics_context(
                canonical_context=canonical,
                matrix_kwargs=matrix_kwargs,
            )
        raise AssertionError(f"Unhandled setup context: {context_mode}")
    if str(setup_kind) == "linucb_v6":
        learning_context_for_setup(problem, setup_kind)
        return build_diffusion_advection_quadratic_context(
            canonical_context=context,
            matrix_kwargs=matrix_kwargs,
        )
    if (
        problem
        in {
            SCALAR_ANISOTROPIC_DIFFUSION,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
        }
        and str(setup_kind) == "linucb"
    ):
        resolved_grid_norm = (
            max(
                int(matrix_kwargs["nx"]),
                int(matrix_kwargs["ny"]),
                int(matrix_kwargs["nz"]),
            )
            if grid_norm_div is None
            else float(grid_norm_div)
        )
        return build_legacy_linucb_v4_context(
            stream_context=context,
            nx=int(matrix_kwargs["nx"]),
            ny=int(matrix_kwargs["ny"]),
            nz=int(matrix_kwargs["nz"]),
            grid_norm_div=float(resolved_grid_norm),
        )
    return context.copy()


__all__ = [
    "CANONICAL_MEANS_ONLY_SETUP_CONTEXT",
    "CANONICAL_NO_C_MEAN_SETUP_CONTEXT",
    "CANONICAL_PECLET_ONLY_SETUP_CONTEXT",
    "CANONICAL_WITH_A_MEAN_SETUP_CONTEXT",
    "CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT",
    "DEFAULT_SETUP_CONTEXT",
    "DIFFUSION_CONVECTION",
    "PROBLEM_KIND_ALIASES",
    "PHYSICS_LINEAR_SETUP_CONTEXT",
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
