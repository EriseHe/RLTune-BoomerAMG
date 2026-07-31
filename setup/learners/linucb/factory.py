from __future__ import annotations

from ..common.factory import SetupLearnerFactoryRequest

from .SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from .SharedLinUCB_AMG_v5 import SharedLinUCB_AMG_v5
from .SharedLinUCB_AMG_v5_RBF import SharedLinUCB_AMG_v5_RBF
from .SharedLinUCB_AMG_v6 import SharedLinUCB_AMG_v6
from .config import (
    LinUCBV4Spec,
    LinUCBV5RBFSpec,
    LinUCBV5Spec,
    LinUCBV6Spec,
)


LINUCB_V4_LEARNER_TYPE = SharedLinUCB_AMG_v4
LINUCB_V5_LEARNER_TYPE = SharedLinUCB_AMG_v5
LINUCB_V5_RBF_LEARNER_TYPE = SharedLinUCB_AMG_v5_RBF
LINUCB_V6_LEARNER_TYPE = SharedLinUCB_AMG_v6


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


def build_linucb_v5_learner(
    request: SetupLearnerFactoryRequest[LinUCBV5Spec],
) -> SharedLinUCB_AMG_v5:
    """Construct the paper-final LinUCB v5 learner."""

    return SharedLinUCB_AMG_v5(
        request.shared.actions,
        context_dim=int(request.shared.context_dim),
        **request.shared.learner_kwargs(),
        **request.algorithm.learner_kwargs(),
    )


def build_linucb_v5_rbf_learner(
    request: SetupLearnerFactoryRequest[LinUCBV5RBFSpec],
) -> SharedLinUCB_AMG_v5_RBF:
    """Construct the experimental LinUCB v5 RBF learner."""

    return SharedLinUCB_AMG_v5_RBF(
        request.shared.actions,
        context_dim=int(request.shared.context_dim),
        **request.shared.learner_kwargs(),
        **request.algorithm.learner_kwargs(),
    )


def build_linucb_v6_learner(
    request: SetupLearnerFactoryRequest[LinUCBV6Spec],
) -> SharedLinUCB_AMG_v6:
    """Construct the internal quadratic-context LinUCB v6 learner."""

    return SharedLinUCB_AMG_v6(
        request.shared.actions,
        context_dim=int(request.shared.context_dim),
        **request.shared.learner_kwargs(),
        **request.algorithm.learner_kwargs(),
    )


__all__ = [
    "LINUCB_V4_LEARNER_TYPE",
    "LINUCB_V5_LEARNER_TYPE",
    "LINUCB_V5_RBF_LEARNER_TYPE",
    "LINUCB_V6_LEARNER_TYPE",
    "build_linucb_v4_learner",
    "build_linucb_v5_learner",
    "build_linucb_v5_rbf_learner",
    "build_linucb_v6_learner",
]
