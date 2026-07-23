"""Compatibility alias for recursive LSTDQ-LCB v2."""

import sys as _sys

from solve.controllers.recursive_lstdq import v2 as _implementation

_sys.modules[__name__] = _implementation
