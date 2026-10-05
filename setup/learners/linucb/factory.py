from __future__ import annotations

from ..common.factory import SetupLearnerFactoryRequest
from .SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from .config import LinUCBV4Spec

LINUCB_V4_LEARNER_TYPE = SharedLinUCB_AMG_v4


def build_linucb_v4_learner(
    request: SetupLearnerFactoryRequest[LinUCBV4Spec],
) -> SharedLinUCB_AMG_v4:
    """Construct the active LinUCB learner from its typed family request."""

    return SharedLinUCB_AMG_v4(
        request.shared.actions,
        context_dim=int(request.shared.context_dim),
        **request.shared.learner_kwargs(),
        **request.algorithm.learner_kwargs(),
    )


__all__ = ["LINUCB_V4_LEARNER_TYPE", "build_linucb_v4_learner"]
