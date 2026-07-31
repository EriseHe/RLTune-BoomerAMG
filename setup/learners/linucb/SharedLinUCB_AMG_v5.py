"""Paper-final LinUCB setup learner for diffusion-advection experiments.

v5 deliberately reuses the stable v4 learning algorithm.  Its version boundary
is the experiment contract: an eight-dimensional diffusion-advection context
and the named Tune-7 ``recommended`` setup space.
"""

from __future__ import annotations

from typing import Any, Sequence

from problems.amg import DIFFUSION_ADVECTION_CONTEXT_FIELDS

from ..common.action_features import ParameterSpaceSpec
from .SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4


LINUCB_V5_CONTEXT_FIELDS = DIFFUSION_ADVECTION_CONTEXT_FIELDS
LINUCB_V5_CONTEXT_DIM = len(LINUCB_V5_CONTEXT_FIELDS)
LINUCB_V5_CONTEXT_INTERACTION_INDICES = (1, 2, 3, 4, 5, 6, 7)

LINUCB_V5_TUNE_DIM = 7
LINUCB_V5_TUNE7_VARIANT = "categorical"
LINUCB_V5_SETUP_SPACE = "recommended"
LINUCB_V5_RECOMMENDED_COARSEN_TYPES = (0, 2, 3, 6, 8, 10)
LINUCB_V5_RECOMMENDED_INTERP_TYPES = (0, 2, 3, 6, 8, 17)
LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES = (4,)
LINUCB_V5_PARAMETER_NAMES = (
    "strong_threshold",
    "max_row_sum",
    "trunc_factor",
    "P_max_elmts",
    "agg_num_levels",
    "coarsen_type",
    "interp_type",
)


def validate_linucb_v5_parameter_spec(
    parameter_spec: ParameterSpaceSpec,
) -> None:
    """Validate the Tune-7 recommended parameter-space identity."""

    names = tuple(parameter.name for parameter in parameter_spec.parameters)
    if names != LINUCB_V5_PARAMETER_NAMES:
        raise ValueError(
            "LinUCB v5 requires the Tune-7 parameters "
            f"{LINUCB_V5_PARAMETER_NAMES}"
        )
    parameters = {
        parameter.name: parameter for parameter in parameter_spec.parameters
    }
    categorical_contract = {
        "coarsen_type": LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
        "interp_type": LINUCB_V5_RECOMMENDED_INTERP_TYPES,
    }
    for name, expected_values in categorical_contract.items():
        parameter = parameters[name]
        actual_values = tuple(int(value) for value in parameter.values)
        if parameter.kind != "categorical" or actual_values != expected_values:
            raise ValueError(
                f"LinUCB v5 requires {name}={expected_values}"
            )


def validate_linucb_v5_paper_contract(
    *,
    context_dim: int,
    context_interaction_indices: Sequence[int],
    tune_dim: int,
    tune7_variant: str,
    action_space_mode: str,
    setup_space_name: str | None,
    coarsen_types: Sequence[int] | None,
    interp_types: Sequence[int] | None,
    agg_interp_types: Sequence[int] | None,
) -> None:
    """Reject runner configurations that drift from the paper v5 contract."""

    if int(context_dim) != LINUCB_V5_CONTEXT_DIM:
        raise ValueError(
            "LinUCB v5 requires the 8-D diffusion-advection context"
        )
    interactions = tuple(int(index) for index in context_interaction_indices)
    if interactions != LINUCB_V5_CONTEXT_INTERACTION_INDICES:
        raise ValueError(
            "LinUCB v5 requires context interactions "
            f"{LINUCB_V5_CONTEXT_INTERACTION_INDICES}"
        )
    if (
        int(tune_dim) != LINUCB_V5_TUNE_DIM
        or str(tune7_variant).strip().lower() != LINUCB_V5_TUNE7_VARIANT
    ):
        raise ValueError(
            "LinUCB v5 is fixed to Tune-7 categorical actions"
        )
    if str(action_space_mode).strip().lower() != "full_cartesian":
        raise ValueError(
            "LinUCB v5 requires the named full-cartesian recommended space"
        )
    if str(setup_space_name or "") != LINUCB_V5_SETUP_SPACE:
        raise ValueError(
            "LinUCB v5 requires setup_space='recommended'"
        )

    actual_spaces = (
        tuple(int(value) for value in (coarsen_types or ())),
        tuple(int(value) for value in (interp_types or ())),
        tuple(int(value) for value in (agg_interp_types or ())),
    )
    expected_spaces = (
        LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
        LINUCB_V5_RECOMMENDED_INTERP_TYPES,
        LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES,
    )
    if actual_spaces != expected_spaces:
        raise ValueError(
            "LinUCB v5 recommended setup space must use "
            f"coarsen_types={expected_spaces[0]}, "
            f"interp_types={expected_spaces[1]}, and "
            f"agg_interp_types={expected_spaces[2]}"
        )


class SharedLinUCB_AMG_v5(SharedLinUCB_AMG_v4):
    """v4 algorithm frozen to the paper-final v5 context contract."""

    context_fields = LINUCB_V5_CONTEXT_FIELDS

    def __init__(
        self,
        actions,
        context_dim: int = LINUCB_V5_CONTEXT_DIM,
        *,
        parameter_spec: ParameterSpaceSpec,
        context_interaction_indices: Sequence[int] = (
            LINUCB_V5_CONTEXT_INTERACTION_INDICES
        ),
        **kwargs: Any,
    ) -> None:
        if int(context_dim) != LINUCB_V5_CONTEXT_DIM:
            raise ValueError(
                "SharedLinUCB_AMG_v5 requires context_dim=8"
            )
        interactions = tuple(
            int(index) for index in context_interaction_indices
        )
        if interactions != LINUCB_V5_CONTEXT_INTERACTION_INDICES:
            raise ValueError(
                "SharedLinUCB_AMG_v5 requires context interactions "
                f"{LINUCB_V5_CONTEXT_INTERACTION_INDICES}"
            )
        validate_linucb_v5_parameter_spec(parameter_spec)
        super().__init__(
            actions,
            context_dim=LINUCB_V5_CONTEXT_DIM,
            parameter_spec=parameter_spec,
            context_interaction_indices=interactions,
            **kwargs,
        )


__all__ = [
    "LINUCB_V5_CONTEXT_DIM",
    "LINUCB_V5_CONTEXT_FIELDS",
    "LINUCB_V5_CONTEXT_INTERACTION_INDICES",
    "LINUCB_V5_PARAMETER_NAMES",
    "LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES",
    "LINUCB_V5_RECOMMENDED_COARSEN_TYPES",
    "LINUCB_V5_RECOMMENDED_INTERP_TYPES",
    "LINUCB_V5_SETUP_SPACE",
    "LINUCB_V5_TUNE7_VARIANT",
    "LINUCB_V5_TUNE_DIM",
    "SharedLinUCB_AMG_v5",
    "validate_linucb_v5_parameter_spec",
    "validate_linucb_v5_paper_contract",
]
