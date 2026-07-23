"""Compatibility alias for shared linear-controller primitives."""

import sys as _sys

from solve.controllers.common import linear_lcb as _implementation

_sys.modules[__name__] = _implementation
