"""Compatibility alias for the canonical setup parameter space module."""

import sys as _sys

from setup import space as _implementation

_sys.modules[__name__] = _implementation
