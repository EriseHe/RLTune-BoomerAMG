"""Compatibility alias for the structured model-based controller."""

import sys as _sys

from solve.controllers.model_based import controller as _implementation

_sys.modules[__name__] = _implementation
