"""Compatibility alias for SARSA behavior policies."""

import sys as _sys

from solve.controllers.sarsa import exploration as _implementation

_sys.modules[__name__] = _implementation
