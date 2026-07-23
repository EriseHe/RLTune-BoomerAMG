"""Compatibility alias for the stagewise LSVI controller."""

import sys as _sys

from solve.controllers.lsvi import stagewise as _implementation

_sys.modules[__name__] = _implementation
