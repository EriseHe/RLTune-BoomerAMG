"""Compatibility alias for :mod:`solve.registry`."""

import sys as _sys

from solve import registry as _implementation

_sys.modules[__name__] = _implementation
