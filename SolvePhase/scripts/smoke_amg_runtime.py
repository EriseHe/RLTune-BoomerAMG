"""Compatibility entry point for :mod:`solve.scripts.smoke_amg_runtime`."""

import sys as _sys

from solve.scripts import smoke_amg_runtime as _implementation

_sys.modules[__name__] = _implementation
