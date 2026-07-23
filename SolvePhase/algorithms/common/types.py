"""Compatibility alias for shared controller types."""

import sys as _sys

from solve.controllers.common import types as _implementation

_sys.modules[__name__] = _implementation
