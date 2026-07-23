"""Compatibility alias for the bootstrap controller."""

import sys as _sys

from solve.controllers.bootstrap import controller as _implementation

_sys.modules[__name__] = _implementation
