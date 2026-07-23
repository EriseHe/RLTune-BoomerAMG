"""Compatibility alias for the RBLSPI controller."""

import sys as _sys

from solve.controllers.rblspi import controller as _implementation

_sys.modules[__name__] = _implementation
