from __future__ import annotations

from ..common.factory import SetupLearnerFactoryRequest

from .SharedLinTS_AMG_v2 import SharedLinTS_AMG_v2
from .config import LinTSV2Spec


LINTS_V2_LEARNER_TYPE = SharedLinTS_AMG_v2


def build_lints_v2_learner(
    request: SetupLearnerFactoryRequest[LinTSV2Spec],
) -> SharedLinTS_AMG_v2:
    """Construct the active LinTS learner from its typed family request."""

    return SharedLinTS_AMG_v2(
        request.shared.actions,
        context_dim=int(request.shared.context_dim),
        **request.shared.learner_kwargs(),
        **request.algorithm.learner_kwargs(),
    )


__all__ = [
    "LINTS_V2_LEARNER_TYPE",
    "build_lints_v2_learner",
]
