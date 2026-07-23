"""Compatibility alias for the recursive Monte Carlo controller."""

import sys as _sys

from solve.controllers.recursive_mc import controller as _implementation

_sys.modules[__name__] = _implementation
