"""Backward-compatible imports for the former monolithic LCB module.

Implementations live in :mod:`solve.controllers`. This module remains so
existing imports and serialized references continue to resolve.
"""

from solve.controllers.bootstrap import (
    BootstrapLcbSarsaController,
    BootstrapSarsaSpec,
)
from solve.controllers.lsvi import (
    HierarchicalLsviLcbController,
    HierarchicalLsviLcbSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
)
from solve.controllers.model_based import (
    StructuredModelBasedController,
    StructuredModelBasedSpec,
)
from solve.controllers.rblspi import (
    RecursiveBlstdqController,
    RecursiveBlstdqSpec,
)
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbController,
    RecursiveLstdqV2LcbSpec,
)
from solve.controllers.recursive_mc import (
    RecursiveMonteCarloLcbController,
    RecursiveMonteCarloLcbSpec,
)

__all__ = [
    "BootstrapLcbSarsaController",
    "BootstrapSarsaSpec",
    "HierarchicalLsviLcbController",
    "HierarchicalLsviLcbSpec",
    "RecursiveBlstdqController",
    "RecursiveBlstdqSpec",
    "RecursiveLstdqLcbController",
    "RecursiveLstdqLcbSpec",
    "RecursiveLstdqV2LcbController",
    "RecursiveLstdqV2LcbSpec",
    "RecursiveMonteCarloLcbController",
    "RecursiveMonteCarloLcbSpec",
    "StagewiseLsviLcbController",
    "StagewiseLsviLcbSpec",
    "StructuredModelBasedController",
    "StructuredModelBasedSpec",
]
