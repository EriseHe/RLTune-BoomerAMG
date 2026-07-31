"""Experimental LinUCB v5 with local RBF setup-action features.

This variant keeps the paper-final v5 context, setup space, candidate schedule,
and LinUCB update rule. Only the representation of the three continuous setup
parameters changes; ``SharedLinUCB_AMG_v5`` remains untouched as the frozen
paper implementation.
"""

from __future__ import annotations

from typing import Any

from ..common.action_features import (
    ParameterSpaceSpec,
    RBFActionFeatureEncoder,
)
from ..common.aot_candidates import FactorizedActionFeatureCache
from .SharedLinUCB_AMG_v5 import (
    LINUCB_V5_CONTEXT_DIM,
    SharedLinUCB_AMG_v5,
)


LINUCB_V5_RBF_CENTERS_PER_PARAMETER = 5
LINUCB_V5_RBF_SIGMA_SPACING = 0.8
LINUCB_V5_RBF_CUTOFF_SIGMA = 2.5


class SharedLinUCB_AMG_v5_RBF(SharedLinUCB_AMG_v5):
    """Paper-final v5 contract with a hybrid local RBF action encoder."""

    def __init__(
        self,
        actions,
        context_dim: int = LINUCB_V5_CONTEXT_DIM,
        *,
        parameter_spec: ParameterSpaceSpec,
        action_feature_cache: FactorizedActionFeatureCache | None = None,
        **kwargs: Any,
    ) -> None:
        if action_feature_cache is None:
            encoder = RBFActionFeatureEncoder(
                parameter_spec,
                centers_per_parameter=(
                    LINUCB_V5_RBF_CENTERS_PER_PARAMETER
                ),
                sigma_spacing=LINUCB_V5_RBF_SIGMA_SPACING,
                cutoff_sigma=LINUCB_V5_RBF_CUTOFF_SIGMA,
            )
        else:
            encoder = action_feature_cache.encoder
            if not isinstance(encoder, RBFActionFeatureEncoder):
                raise ValueError(
                    "SharedLinUCB_AMG_v5_RBF requires an RBF action-feature "
                    "cache"
                )
        super().__init__(
            actions,
            context_dim=context_dim,
            parameter_spec=parameter_spec,
            action_feature_cache=action_feature_cache,
            action_feature_encoder=encoder,
            **kwargs,
        )


__all__ = [
    "LINUCB_V5_RBF_CENTERS_PER_PARAMETER",
    "LINUCB_V5_RBF_CUTOFF_SIGMA",
    "LINUCB_V5_RBF_SIGMA_SPACING",
    "SharedLinUCB_AMG_v5_RBF",
]
