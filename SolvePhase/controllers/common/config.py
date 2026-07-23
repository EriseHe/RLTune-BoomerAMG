"""Compatibility alias for shared controller configuration."""

import sys as _sys

from solve.controllers.common import config as _implementation

_sys.modules[__name__] = _implementation
