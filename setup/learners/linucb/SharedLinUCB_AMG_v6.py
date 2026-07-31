"""Internal LinUCB v6 with a complete quadratic PDE-context basis.

v6 keeps the stable v4 learning algorithm and the v5 recommended Tune-7
action space.  Its only experimental change is the setup context:

1. three non-redundant diffusion coordinates (scale plus two contrasts),
2. three signed directional cell Péclet coordinates, and
3. the complete degree-two basis of those six coordinates.

The resulting 28-D basis is crossed with the existing generic action features
by ``SharedLinUCB_AMG_v4``.  The model therefore remains linear in its fixed
feature map and is still a LinUCB model.
"""

from __future__ import annotations

from typing import Any, Sequence

from problems.amg import DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS

from ..common.action_features import ParameterSpaceSpec
from .SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from .SharedLinUCB_AMG_v5 import (
    LINUCB_V5_CONTEXT_DIM,
    LINUCB_V5_CONTEXT_INTERACTION_INDICES,
    LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES,
    LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
    LINUCB_V5_RECOMMENDED_INTERP_TYPES,
    LINUCB_V5_SETUP_SPACE,
    LINUCB_V5_TUNE7_VARIANT,
    LINUCB_V5_TUNE_DIM,
    validate_linucb_v5_paper_contract,
    validate_linucb_v5_parameter_spec,
)


LINUCB_V6_CONTEXT_FIELDS = DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS
LINUCB_V6_CONTEXT_DIM = len(LINUCB_V6_CONTEXT_FIELDS)
LINUCB_V6_CONTEXT_INTERACTION_INDICES = tuple(
    range(1, LINUCB_V6_CONTEXT_DIM)
)

LINUCB_V6_TUNE_DIM = LINUCB_V5_TUNE_DIM
LINUCB_V6_TUNE7_VARIANT = LINUCB_V5_TUNE7_VARIANT
LINUCB_V6_SETUP_SPACE = LINUCB_V5_SETUP_SPACE
LINUCB_V6_RECOMMENDED_COARSEN_TYPES = (
    LINUCB_V5_RECOMMENDED_COARSEN_TYPES
)
LINUCB_V6_RECOMMENDED_INTERP_TYPES = (
    LINUCB_V5_RECOMMENDED_INTERP_TYPES
)
LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES = (
    LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES
)


def _as_v6_contract_error(error: ValueError) -> ValueError:
    return ValueError(str(error).replace("LinUCB v5", "LinUCB v6"))


def validate_linucb_v6_parameter_spec(
    parameter_spec: ParameterSpaceSpec,
) -> None:
    """Validate the v5-compatible recommended Tune-7 action contract."""

    try:
        validate_linucb_v5_parameter_spec(parameter_spec)
    except ValueError as error:
        raise _as_v6_contract_error(error) from error


def validate_linucb_v6_experimental_contract(
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
    """Reject runner configurations that drift from the internal v6 design."""

    if int(context_dim) != LINUCB_V6_CONTEXT_DIM:
        raise ValueError(
            "LinUCB v6 requires the 28-D quadratic physics context"
        )
    interactions = tuple(int(index) for index in context_interaction_indices)
    if interactions != LINUCB_V6_CONTEXT_INTERACTION_INDICES:
        raise ValueError(
            "LinUCB v6 requires context interactions "
            f"{LINUCB_V6_CONTEXT_INTERACTION_INDICES}"
        )

    # v6 intentionally inherits every non-context part of the frozen v5
    # experiment contract.  Reuse that validator with its own fixed context
    # arguments so the two versions cannot silently diverge in action space.
    try:
        validate_linucb_v5_paper_contract(
            context_dim=LINUCB_V5_CONTEXT_DIM,
            context_interaction_indices=(
                LINUCB_V5_CONTEXT_INTERACTION_INDICES
            ),
            tune_dim=tune_dim,
            tune7_variant=tune7_variant,
            action_space_mode=action_space_mode,
            setup_space_name=setup_space_name,
            coarsen_types=coarsen_types,
            interp_types=interp_types,
            agg_interp_types=agg_interp_types,
        )
    except ValueError as error:
        raise _as_v6_contract_error(error) from error


class SharedLinUCB_AMG_v6(SharedLinUCB_AMG_v4):
    """v4 algorithm frozen to the internal v6 context/action contract."""

    context_fields = LINUCB_V6_CONTEXT_FIELDS

    def __init__(
        self,
        actions,
        context_dim: int = LINUCB_V6_CONTEXT_DIM,
        *,
        parameter_spec: ParameterSpaceSpec,
        context_interaction_indices: Sequence[int] = (
            LINUCB_V6_CONTEXT_INTERACTION_INDICES
        ),
        **kwargs: Any,
    ) -> None:
        if int(context_dim) != LINUCB_V6_CONTEXT_DIM:
            raise ValueError(
                "SharedLinUCB_AMG_v6 requires context_dim=28"
            )
        interactions = tuple(
            int(index) for index in context_interaction_indices
        )
        if interactions != LINUCB_V6_CONTEXT_INTERACTION_INDICES:
            raise ValueError(
                "SharedLinUCB_AMG_v6 requires context interactions "
                f"{LINUCB_V6_CONTEXT_INTERACTION_INDICES}"
            )
        validate_linucb_v6_parameter_spec(parameter_spec)
        super().__init__(
            actions,
            context_dim=LINUCB_V6_CONTEXT_DIM,
            parameter_spec=parameter_spec,
            context_interaction_indices=interactions,
            **kwargs,
        )


__all__ = [
    "LINUCB_V6_CONTEXT_DIM",
    "LINUCB_V6_CONTEXT_FIELDS",
    "LINUCB_V6_CONTEXT_INTERACTION_INDICES",
    "LINUCB_V6_RECOMMENDED_AGG_INTERP_TYPES",
    "LINUCB_V6_RECOMMENDED_COARSEN_TYPES",
    "LINUCB_V6_RECOMMENDED_INTERP_TYPES",
    "LINUCB_V6_SETUP_SPACE",
    "LINUCB_V6_TUNE7_VARIANT",
    "LINUCB_V6_TUNE_DIM",
    "SharedLinUCB_AMG_v6",
    "validate_linucb_v6_experimental_contract",
    "validate_linucb_v6_parameter_spec",
]
