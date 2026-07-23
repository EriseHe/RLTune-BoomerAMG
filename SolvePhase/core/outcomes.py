"""Compatibility alias for solve outcome classification."""

import sys as _sys

from solve.core import outcomes as _implementation

_sys.modules[__name__] = _implementation
