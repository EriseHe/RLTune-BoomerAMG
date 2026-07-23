"""Compatibility alias for the hierarchical LSVI controller."""

import sys as _sys

from solve.controllers.lsvi import hierarchical as _implementation

_sys.modules[__name__] = _implementation
