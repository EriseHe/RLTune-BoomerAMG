"""Linear UCB setup-bandit variants."""

from .LinUCB_AMG import LinUCB_AMG
from .LinUCB_AMG_v2 import LinUCB_AMG_v2
from .SharedLinUCB_AMG import SharedLinUCB_AMG
from .SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from .SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from .SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from .SharedLinUCB_AMG_v5 import (
    LINUCB_V5_CONTEXT_DIM,
    LINUCB_V5_CONTEXT_FIELDS,
    LINUCB_V5_CONTEXT_INTERACTION_INDICES,
    SharedLinUCB_AMG_v5,
    validate_linucb_v5_paper_contract,
)
from .SharedLinUCB_AMG_v5_RBF import SharedLinUCB_AMG_v5_RBF
from .SharedLinUCB_AMG_v6 import (
    LINUCB_V6_CONTEXT_DIM,
    LINUCB_V6_CONTEXT_FIELDS,
    LINUCB_V6_CONTEXT_INTERACTION_INDICES,
    SharedLinUCB_AMG_v6,
    validate_linucb_v6_experimental_contract,
)
from .config import (
    LinUCBV4Spec,
    LinUCBV5RBFSpec,
    LinUCBV5Spec,
    LinUCBV6Spec,
)
from .factory import (
    build_linucb_v4_learner,
    build_linucb_v5_learner,
    build_linucb_v5_rbf_learner,
    build_linucb_v6_learner,
)
from .setup_reselection import (
    SetupReselectionResult,
    run_same_context_setup_reselection,
)

__all__ = [
    "LinUCB_AMG",
    "LinUCB_AMG_v2",
    "LinUCBV4Spec",
    "LinUCBV5RBFSpec",
    "LinUCBV5Spec",
    "LinUCBV6Spec",
    "LINUCB_V5_CONTEXT_DIM",
    "LINUCB_V5_CONTEXT_FIELDS",
    "LINUCB_V5_CONTEXT_INTERACTION_INDICES",
    "LINUCB_V6_CONTEXT_DIM",
    "LINUCB_V6_CONTEXT_FIELDS",
    "LINUCB_V6_CONTEXT_INTERACTION_INDICES",
    "SharedLinUCB_AMG",
    "SharedLinUCB_AMG_v2",
    "SharedLinUCB_AMG_v3",
    "SharedLinUCB_AMG_v4",
    "SharedLinUCB_AMG_v5",
    "SharedLinUCB_AMG_v5_RBF",
    "SharedLinUCB_AMG_v6",
    "SetupReselectionResult",
    "build_linucb_v4_learner",
    "build_linucb_v5_learner",
    "build_linucb_v5_rbf_learner",
    "build_linucb_v6_learner",
    "run_same_context_setup_reselection",
    "validate_linucb_v5_paper_contract",
    "validate_linucb_v6_experimental_contract",
]
