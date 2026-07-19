"""Linear UCB setup-bandit variants."""

from .LinUCB_AMG import LinUCB_AMG
from .LinUCB_AMG_v2 import LinUCB_AMG_v2
from .SharedLinUCB_AMG import SharedLinUCB_AMG
from .SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from .SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from .SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4

__all__ = [
    "LinUCB_AMG",
    "LinUCB_AMG_v2",
    "SharedLinUCB_AMG",
    "SharedLinUCB_AMG_v2",
    "SharedLinUCB_AMG_v3",
    "SharedLinUCB_AMG_v4",
]
