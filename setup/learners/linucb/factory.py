from __future__ import annotations

from ..common.factory import SetupLearnerFactoryRequest
from .learner import SharedLinUCB
from .config import LinUCBSpec

LINUCB_LEARNER_TYPE = SharedLinUCB


def build_linucb_learner(
    request: SetupLearnerFactoryRequest[LinUCBSpec],
) -> SharedLinUCB:
    """Construct the active LinUCB learner from its typed family request."""

    return SharedLinUCB(
        request.shared.actions,
        context_dim=int(request.shared.context_dim),
        **request.shared.learner_kwargs(),
        **request.algorithm.learner_kwargs(),
    )


__all__ = ["LINUCB_LEARNER_TYPE", "build_linucb_learner"]
